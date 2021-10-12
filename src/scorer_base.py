from abc import ABC, abstractmethod

import tensorflow as tf


class ScorerBase(tf.keras.layers.AbstractRNNCell, ABC):
    """Scorer interface for beam search"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pre_beam_size = -1  # only required for partial scorers

    def setup(self, pre_beam_size):
        self.pre_beam_size = pre_beam_size

    @abstractmethod
    def select_state(self, state, i: tf.Tensor):
        # i: indices of next "beams" [BS, BW]
        raise NotImplementedError

    @abstractmethod
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def is_partial_scorer():
        raise NotImplementedError

