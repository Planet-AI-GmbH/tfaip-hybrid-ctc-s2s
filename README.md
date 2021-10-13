# Rescoring Sequence-to-Sequence Models for Text Line Recognition with CTC-Prefixes

## Setup

```shell
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run on IAM

```shell
source venv/bin/activate
python run_iam.py --model=model/best_lm=0.5_beams=5/serve --data=data/test.lst
```

## Shared Code

In `src` we share our implementations of the beam-search and the CTC-prefix-scores.
Note, that the full algorithm is build in the Tensorflow graph which is why a simple `model.predict` is sufficient to obtain the decoded sequence.

## Citing

Please cite

> Wick, C., Zöllner, J., and Grüning, T., *"Rescoring Sequence-to-Sequence Models for Text Line Recognition with CTC-Prefixes"*, arXiv e-prints, 2021.

```
@ARTICLE{2021arXiv211005909W,
       author = {Wick, Christoph and Zöllner, Jochen and Grüning, Tobias},
        title = "{Rescoring Sequence-to-Sequence Models for Text Line Recognition with CTC-Prefixes}",
      journal = {arXiv e-prints},
         year = 2021,
}
```


