from tensorflow_addons.seq2seq.basic_decoder import BasicDecoder, BasicDecoderOutput

import tensorflow as tf
from typeguard import typechecked
from typing import Dict

from scorer_base import ScorerBase


class BasicDecoder(BasicDecoder):
    @typechecked
    def __init__(
        self,
        scorer: Dict[str, ScorerBase],
        **kwargs,
    ):
        assert len(scorer) == 1
        self.key = next(iter(scorer.keys()))
        super().__init__(cell=scorer[self.key], **kwargs)
        self._static_inputs = None

    def initialize(self, inputs, static_inputs, initial_state=None, **kwargs):
        self._static_inputs = static_inputs
        s = super().initialize(inputs, initial_state, **kwargs)
        return s

    def step(self, time, inputs, state, training=None):
        cell_outputs, cell_state = self.cell(
            {"embeddings": inputs, "state": state[self.key], "static_inputs": self._static_inputs[self.key]},
            training=training,
        )
        cell_state = tf.nest.pack_sequence_as(state, tf.nest.flatten(cell_state))
        if self.output_layer is not None:
            cell_outputs = self.output_layer(cell_outputs)
        sample_ids = self.sampler.sample(time=time, outputs=cell_outputs, state=cell_state)
        (finished, next_inputs, next_state) = self.sampler.next_inputs(
            time=time, outputs=cell_outputs, state=cell_state, sample_ids=sample_ids
        )
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return outputs, next_state, next_inputs, finished

