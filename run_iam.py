import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, NamedTuple, Union, List, NoReturn, Dict

import cv2
import editdistance
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tqdm
from tfaip import Sample
from tfaip.util.typing import AnyNumpy


class DataReader:
    def __init__(self, path: Path, limit=-1):
        self.path = path
        with open(self.path) as f:
            self.lines = list(map(lambda s: s.strip("\n"), f.readlines()))

        if limit > 0:
            self.lines = self.lines[:limit]

    def __len__(self):
        return len(self.lines)

    def yield_data(self) -> Iterable[Sample]:
        for line in self.lines:
            img_fn = line.strip("\n")
            txt_fn = img_fn + ".txt"

            with open(txt_fn) as gt_f:
                txt = gt_f.read().strip("\n")

            img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)

            yield Sample(inputs={'img': np.expand_dims(img.transpose(), axis=-1), "imgLen": np.asarray([img.shape[1]])},
                         targets={'tgt': txt, "tgtLen": np.asarray([len(txt)])})


class EvaluatorNames(NamedTuple):
    target_text: str = "tgt"
    decoder_text: str = "output/pred/text"
    encoder_text: str = "output/encoder/text"

    metric_wer_encoder: str = "wer/encoder"
    metric_cer_encoder: str = "cer/encoder"

    metric_wer_decoder: str = "wer/decoder"
    metric_cer_decoder: str = "cer/decoder"


class WERMetric:
    def __init__(self):
        self.total_errs, self.total_len = 0, 0

    def reset(self):
        self.total_errs, self.total_len = 0, 0

    def update_state(self, sentence_gt: Union[str, List[str]], sentence_pred: Union[str, List[str]]) -> NoReturn:
        if isinstance(sentence_gt, str):
            sentence_gt = sentence_gt.split()
        if isinstance(sentence_pred, str):
            sentence_pred = sentence_pred.split()

        self.total_errs += editdistance.eval(sentence_gt, sentence_pred)
        self.total_len += len(sentence_gt)

    def result(self):
        if self.total_len == 0:
            return 0
        return self.total_errs / self.total_len


class CERMetric:
    def __init__(self):
        self.total_errs, self.total_len = 0, 0

    def reset(self):
        self.total_errs, self.total_len = 0, 0

    def update_state(self, sentence_gt: Union[str, AnyNumpy], sentence_pred: Union[str, AnyNumpy]) -> NoReturn:
        self.total_errs += editdistance.eval(sentence_gt, sentence_pred)
        self.total_len += len(sentence_gt)

    def result(self):
        if self.total_len == 0:
            return 0
        return self.total_errs / self.total_len


class Evaluator:
    def __init__(self, names: EvaluatorNames, has_dec=True, has_enc=True):
        self._names = names
        self.has_dec = has_dec
        self.has_enc = has_enc

        if has_dec:
            self.wer_dec_metric = WERMetric()
            self.cer_dec_metric = CERMetric()

        if has_enc:
            self.wer_enc_metric = WERMetric()
            self.cer_enc_metric = CERMetric()

    def __enter__(self):
        if self.has_dec:
            self.wer_dec_metric.reset()
            self.cer_dec_metric.reset()

        if self.has_enc:
            self.wer_enc_metric.reset()
            self.cer_enc_metric.reset()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...

    def update_state(self, sample: Sample) -> NoReturn:
        tgt = sample.targets[self._names.target_text]
        if isinstance(tgt, (np.ndarray, list, tuple)):
            tgt = tgt[0]

        if isinstance(tgt, bytes):
            tgt = tgt.decode("utf-8")

        assert isinstance(tgt, str)

        if self._names.decoder_text not in sample.outputs:
            self.has_dec = False

        if self._names.encoder_text not in sample.outputs:
            self.has_enc = False

        print("GT:", tgt)
        if self.has_dec:
            pred_text = sample.outputs[self._names.decoder_text].decode("utf-8")
            self.wer_dec_metric.update_state(tgt, pred_text)
            self.cer_dec_metric.update_state(tgt, pred_text)
            print("DE:", pred_text)

        if self.has_enc:
            pred_text = sample.outputs[self._names.encoder_text].decode("utf-8")
            self.wer_enc_metric.update_state(tgt, pred_text)
            self.cer_enc_metric.update_state(tgt, pred_text)
            print("EN:", pred_text)

    def result(self) -> Dict[str, AnyNumpy]:
        res = {}
        if self.has_enc:
            res.update(
                {
                    self._names.metric_wer_encoder: self.wer_enc_metric.result(),
                    self._names.metric_cer_encoder: self.cer_enc_metric.result(),
                }
            )
        if self.has_dec:
            res.update(
                {
                    self._names.metric_wer_decoder: self.wer_dec_metric.result(),
                    self._names.metric_cer_decoder: self.cer_dec_metric.result(),
                }
            )
        return res


def sample_to_single_batch(sample: Sample) -> Sample:
    for k, v in sample.inputs.items():
        sample.inputs[k] = np.asarray([v])
    for k, v in sample.targets.items():
        sample.targets[k] = np.asarray([v])
    return sample


def to_batch(x):
    return {k: np.expand_dims(v, axis=0) for k, v in x.items()}


def from_batch(x):
    return {k: v[0] for k, v in x.items()}


if __name__ == '__main__':
    tfa.register_all()

    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--list", required=True)
    parser.add_argument("--limit", default=-1, type=int)
    args = parser.parse_args()

    data = DataReader(Path(args.list), limit=args.limit)
    samples = list(tqdm.tqdm(data.yield_data(), total=len(data), desc="Loading data"))
    model = tf.keras.models.load_model(args.model)
    with Evaluator(EvaluatorNames()) as evaluator:
        for sample in tqdm.tqdm(samples, total=len(data), desc="Predicting"):
            sample.outputs = from_batch(model.predict_on_batch(to_batch(sample.inputs)))
            evaluator.update_state(sample)

    print(json.dumps(evaluator.result(), indent=2))
