"""A decoder that performs beam search.

Adapted from tensorflow_addons/seq2seq/beam_search_decoder

"""
from functools import partial
from typing import Dict


from tensorflow_addons.seq2seq.beam_search_decoder import *
from tensorflow_addons.seq2seq.beam_search_decoder import (
    _as_shape,
    _check_ndims,
    _check_static_batch_beam_maybe,
    _check_batch_beam,
    _tensor_gather_helper,
    _maybe_tensor_gather_helper,
    _mask_probs,
)

from scorer_base import ScorerBase


def combined_static_and_dynamic_shape(tensor: AnyTensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape
    if not isinstance(static_tensor_shape, (list, tuple)):
        static_tensor_shape = static_tensor_shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


class BeamSearchDecoderOutput(
    collections.namedtuple("BeamSearchDecoderOutput", ("scores", "predicted_ids", "parent_ids", "finished_beams"))
):
    ...


class BeamSearchFinishedBeams(
    collections.namedtuple("BeamSearchFinishedBeams", ("scores", "predicted_ids", "parent_ids", "lengths"))
):
    ...


class BeamSearchDecoderState(
    collections.namedtuple(
        "BeamSearchDecoderState",
        (
            "cell_state",
            "log_probs",
            "finished",
            "lengths",
            "accumulated_attention_probs",
            "finished_beams",
        ),
    )
):
    ...


class BeamSearchDecoderMixin:
    """BeamSearchDecoderMixin contains the common methods for
    BeamSearchDecoder.

    It is expected to be used a base class for concrete
    BeamSearchDecoder. Since this is a mixin class, it is expected to be
    used together with other class as base.
    """

    @typechecked
    def __init__(
        self,
        scorer: Dict[str, ScorerBase],
        scorer_weights: Dict[str, float],
        beam_width: int,
        output_layers: Optional[Dict[str, tf.keras.layers.Layer]] = None,
        reorder_tensor_arrays: bool = True,
        output_all_scores: bool = False,
        pre_beam_ratio: float = 1.5,
        pre_beam_score_key: str = None,
        **kwargs,
    ):
        """Initialize the BeamSearchDecoderMixin.

        Args:
          cell: A layer that implements the `tf.keras.layers.AbstractRNNCell`
            interface.
          beam_width:  Python integer, the number of beams.
          output_layer: (Optional) An instance of `tf.keras.layers.Layer`,
            i.e., `tf.keras.layers.Dense`.  Optional layer to apply to the RNN
            output prior to storing the result or sampling.
          reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the
            cell state will be reordered according to the beam search path. If
            the `TensorArray` can be reordered, the stacked form will be
            returned. Otherwise, the `TensorArray` will be returned as is. Set
            this flag to `False` if the cell state contains `TensorArray`s that
            are not amenable to reordering.
          output_all_scores: If `True`, `BeamSearchDecoderOutput.scores` will
            contain scores for all token IDs and be of shape
            `[batch_size, beam_width, vocab_size]`. When `False` (default),
            only the top score corresponding to the predicted token will be
            output with shape `[batch_size, beam_width]`.
          **kwargs: Dict, other keyword arguments for parent class.
        """
        for cell in scorer.values():
            keras_utils.assert_like_rnncell("cell", cell)

        if output_layers:
            for k in output_layers.keys():
                assert k in scorer, f"Key {k} must be in scorers {list(scorer.keys())}"

        self._scorer = {}
        self._scorer_weights = scorer_weights
        self._output_layers = output_layers
        self._reorder_tensor_arrays = reorder_tensor_arrays
        self._output_all_scores = output_all_scores

        self._start_tokens = None
        self._end_token = None
        self._batch_size = None
        self._beam_width = beam_width
        super().__init__(**kwargs)

        self.pre_beam_size = int(pre_beam_ratio * beam_width)
        self._full_scorers = {}
        self._part_scorers = {}

        for k, v in scorer.items():
            w = scorer_weights.get(k, 0)
            if w == 0 or v is None:
                continue
            self._scorer[k] = v
            if v.is_partial_scorer():
                self._part_scorers[k] = v
            else:
                self._full_scorers[k] = v
        if (
            pre_beam_score_key is not None
            and pre_beam_score_key != "full"
            and pre_beam_score_key not in self._full_scorers
        ):
            raise KeyError(f"{pre_beam_score_key} is not found in {self._full_scorers}")
        self.pre_beam_score_key = pre_beam_score_key
        self.do_pre_beam = self.pre_beam_score_key is not None and len(self._part_scorers) > 0
        for v in self._scorer.values():
            v.setup(self.pre_beam_size if self.do_pre_beam else None)

    @property
    def batch_size(self):
        return self._batch_size

    def _rnn_output_size(self, scorer_id):
        """Get the output shape from the RNN layer."""
        size = self._scorer[scorer_id].output_size
        if self._output_layers.get(scorer_id, None) is None:
            return size
        else:
            # To use layer's compute_output_shape, we need to convert the
            # RNNCell's output_size entries into shapes with an unknown
            # batch size.  We then pass this through the layer's
            # compute_output_shape and read off all but the first (batch)
            # dimensions to get the output size of the rnn with the layer
            # applied to the top.
            output_shape_with_unknown_batch = tf.nest.map_structure(
                lambda s: tf.TensorShape([None]).concatenate(s), size
            )
            layer_output_shape = self._output_layers[scorer_id].compute_output_shape(output_shape_with_unknown_batch)
            return tf.nest.map_structure(lambda s: s[1:], layer_output_shape)

    @property
    def tracks_own_finished(self):
        """The BeamSearchDecoder shuffles its beams and their finished state.

        For this reason, it conflicts with the `dynamic_decode` function's
        tracking of finished states.  Setting this property to true avoids
        early stopping of decoding due to mismanagement of the finished state
        in `dynamic_decode`.

        Returns:
          `True`.
        """
        return True

    @property
    def output_size(self):
        # Return the cell output and the id
        score_size = tf.TensorShape([self._beam_width])
        return BeamSearchDecoderOutput(
            scores=score_size,
            predicted_ids=tf.TensorShape([self._beam_width]),
            parent_ids=tf.TensorShape([self._beam_width]),
            finished_beams=BeamSearchFinishedBeams(
                score_size,
                predicted_ids=tf.TensorShape([self._beam_width]),
                parent_ids=tf.TensorShape([self._beam_width]),
                lengths=tf.TensorShape([self._beam_width]),
            ),
        )

    def finalize(self, outputs, final_state, sequence_lengths):
        """Finalize and return the predicted_ids.

        Args:
          outputs: An instance of BeamSearchDecoderOutput.
          final_state: An instance of BeamSearchDecoderState. Passed through to
            the output.
          sequence_lengths: An `int64` tensor shaped
            `[batch_size, beam_width]`. The sequence lengths determined for
            each beam during decode. **NOTE** These are ignored; the updated
            sequence lengths are stored in `final_state.lengths`.

        Returns:
          outputs: An instance of `FinalBeamSearchDecoderOutput` where the
            predicted_ids are the result of calling _gather_tree.
          final_state: The same input instance of `BeamSearchDecoderState`.
        """
        del sequence_lengths
        # Get max_sequence_length across all beams for each batch.
        max_sequence_lengths = tf.cast(tf.reduce_max(final_state.lengths, axis=1), tf.int32)
        # ORIGINAL Code:
        # predicted_ids = gather_tree(
        #     outputs.predicted_ids, outputs.parent_ids,
        #     max_sequence_lengths=max_sequence_lengths,
        #     end_token=self._end_token,
        # )

        # Adapted Code to decode the finished beams
        # gather tree will use the parent ids to collect the predicted ids
        # IDEA: concatenate the finished beams at the end and shift the parent ids (offset) to the concatenated position
        # At "beam length" do the transition to the actual running beams in the first half
        # This function will compute the paths for both halves, so only keep the finished (2nd half)
        # NOTE: the beam_search_decoder_output of the FinalBeamSearchDecoderOutput is WRONG! (since it is related to
        # the running "non finished" beams. Only predicted ids can be used.
        offset_mask = 1 - tf.transpose(
            tf.sequence_mask(final_state.finished_beams.lengths, tf.shape(outputs.predicted_ids)[0], dtype="int32"),
            perm=[2, 0, 1],
        )
        offset_correction = offset_mask * (
            tf.range(self._beam_width, 2 * self._beam_width) - outputs.finished_beams.parent_ids
        )
        joined_pred_ids = tf.concat([outputs.predicted_ids, outputs.finished_beams.predicted_ids], axis=2)
        joined_parent_ids = tf.concat(
            [outputs.parent_ids, outputs.finished_beams.parent_ids + offset_correction], axis=2
        )
        predicted_ids = gather_tree(
            joined_pred_ids,
            joined_parent_ids,
            max_sequence_lengths=max_sequence_lengths,
            end_token=self._end_token,
        )
        predicted_ids = predicted_ids[:, :, self._beam_width :]

        if self._reorder_tensor_arrays:
            final_state = final_state._replace(
                cell_state=tf.nest.map_structure(
                    lambda t: self._maybe_sort_array_beams(t, outputs.parent_ids, final_state.lengths),
                    final_state.cell_state,
                )
            )
        outputs = FinalBeamSearchDecoderOutput(beam_search_decoder_output=outputs, predicted_ids=predicted_ids)
        return outputs, final_state

    def _merge_batch_beams(self, t, s=None):
        return _merge_batch_beams(self._batch_size, self._beam_width, t, s)

    def _split_batch_beams(self, t, s=None):
        return _split_batch_beams(self._batch_size, self._beam_width, t, s)

    def _maybe_split_batch_beams(self, t, s):
        return _maybe_split_batch_beams(self._batch_size, self._beam_width, t, s)

    def _maybe_merge_batch_beams(self, t, s):
        return _maybe_merge_batch_beams(self._batch_size, self._beam_width, t, s)

    def _maybe_sort_array_beams(self, t, parent_ids, sequence_length):
        return _maybe_sort_array_beams(self._batch_size, self._beam_width, t, parent_ids, sequence_length)

    def step(self, time, inputs, state, training=None, name=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          training: Python boolean. Indicates whether the layer should
              behave in training mode or in inference mode. Only relevant
              when `dropout` or `recurrent_dropout` is used.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        batch_size = self._batch_size
        beam_width = self._beam_width
        end_token = self._end_token

        with tf.name_scope(name or "BeamSearchDecoderStep"):
            beam_search_output, beam_search_state = _beam_search_step(
                scorer_weights=self._scorer_weights,
                time=time,
                beam_state=state,
                batch_size=batch_size,
                beam_width=beam_width,
                pre_beam_size=self.pre_beam_size if self.do_pre_beam else None,
                end_token=end_token,
                output_all_scores=self._output_all_scores,
                full_scorer=self._full_scorers,
                part_scorer=self._part_scorers,
                state=state,
                inputs=inputs,
                static_inputs=self._static_inputs,
                training=training,
            )

            finished = beam_search_state.finished
            sample_ids = beam_search_output.predicted_ids
            next_embeddings = tf.cond(
                tf.reduce_all(finished),
                lambda: self._start_inputs["embeddings"],
                lambda: self._embedding_fn(sample_ids),
            )

            next_inputs = {
                "embeddings": next_embeddings,
            }

        return beam_search_output, beam_search_state, next_inputs, finished


class BeamSearchDecoder(BeamSearchDecoderMixin, decoder.BaseDecoder):
    # Note that the inheritance hierarchy is important here. The Mixin has to be
    # the first parent class since we will use super().__init__(), and Mixin
    # which is a object will properly invoke the __init__ method of other parent
    # class.
    """Beam search decoder.

    **NOTE** If you are using the `BeamSearchDecoder` with a cell wrapped in
    `tfa.seq2seq.AttentionWrapper`, then you must ensure that:

    - The encoder output has been tiled to `beam_width` via
      `tfa.seq2seq.tile_batch` (NOT `tf.tile`).
    - The `batch_size` argument passed to the `get_initial_state` method of
      this wrapper is equal to `true_batch_size * beam_width`.
    - The initial state created with `get_initial_state` above contains a
      `cell_state` value containing properly tiled final state from the
      encoder.

    An example:

    ```
    tiled_encoder_outputs = tfa.seq2seq.tile_batch(
        encoder_outputs, multiplier=beam_width)
    tiled_encoder_final_state = tfa.seq2seq.tile_batch(
        encoder_final_state, multiplier=beam_width)
    tiled_sequence_length = tfa.seq2seq.tile_batch(
        sequence_length, multiplier=beam_width)
    attention_mechanism = MyFavoriteAttentionMechanism(
        num_units=attention_depth,
        memory=tiled_inputs,
        memory_sequence_length=tiled_sequence_length)
    attention_cell = AttentionWrapper(cell, attention_mechanism, ...)
    decoder_initial_state = attention_cell.get_initial_state(
        batch_size=true_batch_size * beam_width, dtype=dtype)
    decoder_initial_state = decoder_initial_state.clone(
        cell_state=tiled_encoder_final_state)
    ```

    Meanwhile, with `tfa.seq2seq.AttentionWrapper`, coverage penalty is suggested to use
    when computing scores (https://arxiv.org/pdf/1609.08144.pdf). It encourages
    the decoding to cover all inputs.
    """

    @typechecked
    def __init__(
        self,
        scorer: Dict[str, ScorerBase],
        beam_width: int,
        scorer_weights: Optional[Dict[str, float]] = None,
        embedding_fn: Optional[Callable] = None,
        reorder_tensor_arrays: bool = True,
        **kwargs,
    ):
        """Initialize the BeamSearchDecoder.

        Args:
          cell: A layer that implements the `tf.keras.layers.AbstractRNNCell`
            interface.
          beam_width:  Python integer, the number of beams.
          embedding_fn: A callable that takes a `int32` `Tensor` of token IDs
            and returns embedding tensors. If set, the `embedding` argument in
            the decoder call should be set to `None`.
          length_penalty_weight: Float weight to penalize length. Disabled with
            0.0.
          coverage_penalty_weight: Float weight to penalize the coverage of
            source sentence. Disabled with 0.0.
          reorder_tensor_arrays: If `True`, `TensorArray`s' elements within the
            cell state will be reordered according to the beam search path. If
            the `TensorArray` can be reordered, the stacked form will be
            returned. Otherwise, the `TensorArray` will be returned as is. Set
            this flag to `False` if the cell state contains `TensorArray`s that
            are not amenable to reordering.
          **kwargs: Dict, other keyword arguments for initialization.
        """
        super().__init__(
            scorer,
            scorer_weights or {},
            beam_width,
            reorder_tensor_arrays=reorder_tensor_arrays,
            **kwargs,
        )

        self._embedding_fn = embedding_fn

    def initialize(self, embedding, static_inputs, start_tokens, end_token, initial_state):
        """Initialize the decoder.

        Args:
          embedding: A `Tensor` (or `Variable`) to pass as the `params` argument
            for `tf.nn.embedding_lookup`. This overrides `embedding_fn` set in
            the constructor.
          start_tokens: Start the decoding from these tokens.
            A `int32` `Tensor` of shape `[batch_size]`.
          end_token: The token that marks the end of decoding.
            A `int32` scalar `Tensor`.
          initial_state: The initial cell state as a (possibly nested) structure
            of `Tensor` and `TensorArray`.

        Returns:
          `(finished, start_inputs, initial_state)`.

        Raises:
          ValueError: If `embedding` is `None` and `embedding_fn` was not set
            in the constructor.
          ValueError: If `start_tokens` is not a vector or `end_token` is not a
            scalar.
        """
        if embedding is not None:
            self._embedding_fn = lambda ids: tf.nn.embedding_lookup(embedding, ids)
        elif self._embedding_fn is None:
            raise ValueError(
                "You should either pass an embedding variable when calling the "
                "BeamSearchDecoder or set embedding_fn in the constructor."
            )

        self._start_tokens = tf.convert_to_tensor(start_tokens, dtype=tf.int32, name="start_tokens")
        if self._start_tokens.shape.ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._end_token = tf.convert_to_tensor(end_token, dtype=tf.int32, name="end_token")
        if self._end_token.shape.ndims != 0:
            raise ValueError("end_token must be a scalar")

        self._batch_size = tf.size(start_tokens)
        self._initial_cell_state = {
            k: tf.nest.map_structure(self._maybe_split_batch_beams, initial_state[k], cell.state_size)
            for k, cell in self._scorer.items()
        }
        self._start_tokens = tf.tile(tf.expand_dims(self._start_tokens, 1), [1, self._beam_width])
        self._start_inputs = {
            "embeddings": self._embedding_fn(self._start_tokens),
        }

        # Initialize static inputs: multiply with beam with, and reshape to [B * BW, ...]
        self._static_inputs = tf.nest.map_structure(
            lambda i: tf.tile(tf.expand_dims(i, axis=1), [1, self._beam_width] + [1] * (len(i.shape) - 1)),
            static_inputs,
        )
        self._static_inputs = tf.nest.map_structure(
            lambda i: tf.reshape(i, [-1] + combined_static_and_dynamic_shape(i)[2:]), self._static_inputs
        )

        self._finished = tf.one_hot(
            tf.zeros([self._batch_size], dtype=tf.int32),
            depth=self._beam_width,
            on_value=False,
            off_value=True,
            dtype=tf.bool,
        )

        finished, start_inputs = self._finished, self._start_inputs

        dtype = tf.nest.flatten(self._initial_cell_state)[0].dtype
        log_probs = tf.one_hot(  # shape(batch_sz, beam_sz)
            tf.zeros([self._batch_size], dtype=tf.int32),
            depth=self._beam_width,
            on_value=tf.convert_to_tensor(0.0, dtype=dtype),
            off_value=tf.convert_to_tensor(-np.Inf, dtype=dtype),
            dtype=dtype,
        )
        initial_state = BeamSearchDecoderState(
            cell_state=self._initial_cell_state,
            log_probs=log_probs,
            finished=finished,
            lengths=tf.zeros([self._batch_size, self._beam_width], dtype=tf.int64),
            accumulated_attention_probs={},
            finished_beams=BeamSearchFinishedBeams(
                predicted_ids=tf.zeros([self._batch_size, self._beam_width], dtype=tf.int32),
                parent_ids=tf.zeros([self._batch_size, self._beam_width], dtype=tf.int32),
                scores=tf.fill([self._batch_size, self._beam_width], value=-1.0e12),
                lengths=tf.zeros([self._batch_size, self._beam_width], dtype=tf.int64),
            ),
        )

        return finished, start_inputs, initial_state

    @property
    def output_dtype(self):
        # Assume the dtype of the cell is the output_size structure
        # containing the input_state's first component's dtype.
        # Return that structure and int32 (the id)
        dtype = tf.nest.flatten(self._initial_cell_state)[0].dtype
        return BeamSearchDecoderOutput(
            scores=tf.nest.map_structure(lambda _: dtype, self.output_size.scores),
            predicted_ids=tf.int32,
            parent_ids=tf.int32,
            finished_beams=BeamSearchFinishedBeams(
                scores=tf.float32,
                predicted_ids=tf.int32,
                parent_ids=tf.int32,
                lengths=tf.int64,
            ),
        )

    def call(self, embedding, start_tokens, end_token, initial_state, training=None, **kwargs):
        init_kwargs = kwargs
        init_kwargs["start_tokens"] = start_tokens
        init_kwargs["end_token"] = end_token
        init_kwargs["initial_state"] = initial_state
        return decoder.dynamic_decode(
            self,
            output_time_major=self.output_time_major,
            impute_finished=self.impute_finished,
            maximum_iterations=self.maximum_iterations,
            parallel_iterations=self.parallel_iterations,
            swap_memory=self.swap_memory,
            training=training,
            decoder_init_input=embedding,
            decoder_init_kwargs=init_kwargs,
        )


def _forward_scorer(
    batch_size,
    beam_width,
    scorer,
    cell_state,
    inputs,
    static_inputs,
    training,
    cs=None,
):
    inputs = tf.nest.map_structure(lambda inp: _merge_batch_beams(batch_size, beam_width, inp, s=inp.shape[2:]), inputs)
    cell_state = tf.nest.map_structure(
        partial(_maybe_merge_batch_beams, batch_size, beam_width), cell_state, scorer.state_size
    )
    r = scorer(
        {"embeddings": inputs["embeddings"], "state": cell_state, "cs": cs, "static_inputs": static_inputs},
        training=training,
    )
    cell_outputs, next_cell_state = r

    cell_outputs = tf.nest.map_structure(
        lambda out: _split_batch_beams(batch_size, beam_width, out, out.shape[1:]), cell_outputs
    )
    next_cell_state = tf.nest.pack_sequence_as(cell_state, tf.nest.flatten(next_cell_state))
    next_cell_state = tf.nest.map_structure(
        partial(_maybe_split_batch_beams, batch_size, beam_width), next_cell_state, scorer.state_size
    )

    return cell_outputs, next_cell_state


def _beam_score(
    inputs,
    static_inputs,
    beam_state,
    batch_size,
    beam_width,
    state,
    scorer,
    scorer_weights,
    training,
    end_token,
    part_ids=None,
):
    next_cell_state = {}
    previously_finished = beam_state.finished

    # Calculate the total log probs for the new hypotheses
    # Final Shape: [batch_size, beam_width, vocab_size]
    total_probs = None
    for k, s in scorer.items():
        cell_state = state.cell_state[k]
        cell_static_inputs = static_inputs.get(k, None)
        r = _forward_scorer(batch_size, beam_width, s, cell_state, inputs, cell_static_inputs, training, cs=part_ids)
        scorer_log_probs, next_cell_state[k] = r
        step_log_probs = scorer_log_probs
        step_log_probs = _mask_probs(step_log_probs, end_token, previously_finished)
        if total_probs is None:
            total_probs = step_log_probs * scorer_weights.get(k, 1)
        else:
            total_probs += step_log_probs * scorer_weights.get(k, 1)

    return next_cell_state, total_probs


def select_top_k(scores, beam_state, vocab_size, batch_size, beam_width, end_token, part_ids):
    if part_ids is not None:
        vocab_size = part_ids.shape[-1]

    # Calculate the current lengths of the predictions
    previously_finished = beam_state.finished
    static_batch_size = tf.get_static_value(batch_size)

    # During the first time step we only consider the initial beam
    scores_flat = tf.reshape(scores, [batch_size, -1])

    # Pick the next beams according to the specified successors function
    next_beam_size = tf.convert_to_tensor(beam_width, dtype=tf.int32, name="beam_width")
    next_beam_scores, word_indices = tf.math.top_k(scores_flat, k=next_beam_size)

    next_beam_scores.set_shape([static_batch_size, beam_width])
    word_indices.set_shape([static_batch_size, beam_width])

    # Pick out the probs, beam_ids, and states according to the chosen
    # predictions
    next_beam_probs = _tensor_gather_helper(
        gather_indices=word_indices,
        gather_from=scores,
        batch_size=batch_size,
        range_size=beam_width * vocab_size,
        gather_shape=[-1],
        name="next_beam_probs",
    )

    if part_ids is not None:
        # Map to true word ids
        part_word_ids = word_indices
        next_word_ids = tf.gather(tf.reshape(part_ids, (batch_size, -1)), word_indices, batch_dims=1)
        raw_next_word_ids = tf.math.floormod(word_indices, vocab_size, name="next_beam_word_ids")
        next_beam_ids = tf.cast(word_indices / vocab_size, tf.int32, name="next_beam_parent_ids")
        word_indices = tf.cast(raw_next_word_ids, tf.int32)
    else:
        # Note: just doing the following
        #   tf.to_int32(word_indices % vocab_size,
        #       name="next_beam_word_ids")
        # would be a lot cleaner but for reasons unclear, that hides the results of
        # the op which prevents capturing it with tfdbg debug ops.
        raw_next_word_ids = tf.math.floormod(word_indices, vocab_size, name="next_beam_word_ids")
        next_word_ids = tf.cast(raw_next_word_ids, tf.int32)
        next_beam_ids = tf.cast(word_indices / vocab_size, tf.int32, name="next_beam_parent_ids")
        part_word_ids = next_word_ids

    # Append new ids to current predictions
    previously_finished = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=previously_finished,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1],
    )
    next_finished = tf.logical_or(
        previously_finished,
        tf.equal(next_word_ids, end_token),
    )
    next_finished = tf.logical_or(
        next_finished,
        tf.less_equal(next_beam_scores, -1e8),
        name="next_beam_finished",
    )

    # Calculate the length of the next predictions.
    # 1. Finished beams remain unchanged.
    # 2. Beams that are now finished (EOS predicted) have their length
    #    increased by 1.
    # 3. Beams that are not yet finished have their length increased by 1.
    lengths_to_add = tf.cast(tf.logical_not(previously_finished), tf.int64)
    next_prediction_len = _tensor_gather_helper(
        gather_indices=next_beam_ids,
        gather_from=beam_state.lengths,
        batch_size=batch_size,
        range_size=beam_width,
        gather_shape=[-1],
    )
    next_prediction_len += lengths_to_add

    return (
        part_word_ids,
        next_word_ids,
        next_beam_ids,
        next_beam_probs,
        next_prediction_len,
        next_beam_scores,
        next_finished,
    )


def _beam_search_step(
    time,
    scorer_weights,
    beam_state,
    batch_size,
    beam_width,
    pre_beam_size: Optional[int],
    end_token,
    output_all_scores,
    full_scorer: Dict[str, ScorerBase],
    part_scorer: Dict[str, ScorerBase],
    state,
    inputs,
    static_inputs,
    training,
):
    """Performs a single step of Beam Search Decoding.

    Args:
      time: Beam search time step, should start at 0. At time 0 we assume
        that all beams are equal and consider only the first beam for
        continuations.
      beam_state: Current state of the beam search.
        An instance of `BeamSearchDecoderState`.
      batch_size: The batch size for this input.
      beam_width: Python int.  The size of the beams.
      end_token: The int32 end token.
      output_all_scores: Bool output scores for every token if True, else only
        output the top scores.

    Returns:
      A new beam state.
    """
    weighted_scores = tf.expand_dims(tf.zeros_like(beam_state.log_probs), 2)
    full_next_cell_state, full_scores = _beam_score(
        inputs,
        static_inputs,
        beam_state,
        batch_size,
        beam_width,
        state,
        full_scorer,
        scorer_weights,
        training,
        end_token,
    )
    weighted_scores += full_scores

    shape = combined_static_and_dynamic_shape(weighted_scores)
    vocab_size = shape[-1]

    part_ids = None
    if len(part_scorer) > 0:
        if pre_beam_size is not None:
            next_word_scores, next_word_ids = tf.math.top_k(weighted_scores, k=pre_beam_size)
            part_ids = tf.reshape(next_word_ids, shape=(-1, pre_beam_size))

        part_next_cell_state, part_scores = _beam_score(
            inputs,
            static_inputs,
            beam_state,
            batch_size,
            beam_width,
            state,
            part_scorer,
            scorer_weights,
            training,
            end_token,
            part_ids=part_ids,
        )

        next_cell_state = {**full_next_cell_state, **part_next_cell_state}
        if part_ids is None:
            weighted_scores += part_scores
        else:
            weighted_scores = tf.gather(weighted_scores, next_word_ids, batch_dims=-1)
            weighted_scores += part_scores
    else:
        next_cell_state = full_next_cell_state

    # add previous hyp score
    weighted_scores += tf.expand_dims(beam_state.log_probs, 2)

    # select new beams
    (
        part_word_ids,
        next_word_ids,
        next_beam_ids,
        next_beam_probs,
        next_prediction_len,
        next_beam_scores,
        next_finished,
    ) = select_top_k(
        vocab_size=vocab_size,
        scores=weighted_scores,
        beam_state=beam_state,
        batch_size=batch_size,
        beam_width=beam_width,
        end_token=end_token,
        part_ids=part_ids,
    )

    # Let scorer update their state based on the best indices
    for k, s in full_scorer.items():
        next_cell_state[k] = s.select_state(next_cell_state[k], next_word_ids)
        # Pick out the cell_states according to the next_beam_ids. We use a
        # different gather_shape here because the cell_state tensors, i.e.
        # the tensors that would be gathered from, all have dimension
        # greater than two and we need to preserve those dimensions.
        next_cell_state[k] = tf.nest.map_structure(
            lambda gather_from: _maybe_tensor_gather_helper(
                gather_indices=next_beam_ids,
                gather_from=gather_from,
                batch_size=batch_size,
                range_size=beam_width,
                gather_shape=[batch_size * beam_width, -1],
            ),
            next_cell_state[k],
        )

    for k, s in part_scorer.items():
        next_cell_state[k] = s.select_state(next_cell_state[k], part_word_ids)

    scores = weighted_scores if output_all_scores else next_beam_scores

    def extract_finished_beams_for_batch(inputs_):
        nf, sc, nwi, nbi, nl, old_finished_sc, old_finished_nwi, old_finished_nbi, old_finished_nl = inputs_
        finished_beam_ids = tf.squeeze(tf.where(tf.equal(nf, True)), axis=-1)
        f_nwi = tf.gather(nwi, finished_beam_ids)
        f_nbi = tf.gather(nbi, finished_beam_ids)
        f_sc, f_nl = tf.nest.map_structure(lambda x: tf.gather(x, finished_beam_ids), (sc, nl))

        result = tf.nn.top_k(tf.concat([f_sc, old_finished_sc], axis=0), k=beam_width)
        best_scores, indices = result.values, result.indices
        best_nwi = tf.gather(tf.concat([f_nwi, old_finished_nwi], axis=0), indices)
        best_nbi = tf.gather(tf.concat([f_nbi, old_finished_nbi], axis=0), indices)
        best_nl = tf.gather(tf.concat([f_nl, old_finished_nl], axis=0), indices)

        return best_scores, best_nwi, best_nbi, best_nl

    finished_beams = tf.map_fn(
        fn=extract_finished_beams_for_batch,
        elems=(next_finished, scores, next_word_ids, next_beam_ids, next_prediction_len, *state.finished_beams),
        fn_output_signature=(tf.float32, tf.int32, tf.int32, tf.int64),
    )

    next_finished = tf.logical_or(next_finished, next_beam_probs < tf.reduce_min(finished_beams[0]))

    next_state = BeamSearchDecoderState(
        cell_state=next_cell_state,
        log_probs=next_beam_probs,
        lengths=next_prediction_len,
        finished=next_finished,
        accumulated_attention_probs={},
        finished_beams=BeamSearchFinishedBeams(*finished_beams),
    )

    output = BeamSearchDecoderOutput(
        scores=scores,
        predicted_ids=next_word_ids,
        parent_ids=next_beam_ids,
        finished_beams=BeamSearchFinishedBeams(*finished_beams),
    )

    return output, next_state


def _merge_batch_beams(batch_size, beam_width, t, s=None):
    """Merges the tensor from a batch of beams into a batch by beams.

    More exactly, t is a tensor of dimension [batch_size, beam_width, s].
    We reshape this into [batch_size*beam_width, s]

    Args:
      t: Tensor of dimension [batch_size, beam_width, s]
      s: (Possibly known) depth shape.

    Returns:
      A reshaped version of t with dimension [batch_size * beam_width, s].
    """
    s = _as_shape(s)
    t_shape = tf.shape(t)
    static_batch_size = tf.get_static_value(batch_size)
    batch_size_beam_width = None if static_batch_size is None else static_batch_size * beam_width
    reshaped_t = tf.reshape(t, tf.concat(([batch_size * beam_width], t_shape[2:]), 0))
    reshaped_t.set_shape(tf.TensorShape([batch_size_beam_width]).concatenate(s))
    return reshaped_t


def _split_batch_beams(batch_size, beam_width, t, s=None):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, t is a tensor of dimension [batch_size*beam_width, s]. We
    reshape this into [batch_size, beam_width, s]

    Args:
      t: Tensor of dimension [batch_size*beam_width, s].
      s: (Possibly known) depth shape.

    Returns:
      A reshaped version of t with dimension [batch_size, beam_width, s].

    Raises:
      ValueError: If, after reshaping, the new tensor is not shaped
        `[batch_size, beam_width, s]` (assuming batch_size and beam_width
        are known statically).
    """
    s = _as_shape(s)
    t_shape = tf.shape(t)
    reshaped_t = tf.reshape(t, tf.concat(([batch_size, beam_width], t_shape[1:]), 0))
    static_batch_size = tf.get_static_value(batch_size)
    expected_reshaped_shape = tf.TensorShape([static_batch_size, beam_width]).concatenate(s)
    if not reshaped_t.shape.is_compatible_with(expected_reshaped_shape):
        raise ValueError(
            "Unexpected behavior when reshaping between beam width "
            "and batch size.  The reshaped tensor has shape: %s.  "
            "We expected it to have shape "
            "(batch_size, beam_width, depth) == %s.  Perhaps you "
            "forgot to call get_initial_state with "
            "batch_size=encoder_batch_size * beam_width?" % (reshaped_t.shape, expected_reshaped_shape)
        )
    reshaped_t.set_shape(expected_reshaped_shape)
    return reshaped_t


def _maybe_split_batch_beams(batch_size, beam_width, t, s):
    """Maybe splits the tensor from a batch by beams into a batch of beams.

    We do this so that we can use nest and not run into problems with
    shapes.

    Args:
      t: `Tensor`, either scalar or shaped `[batch_size * beam_width] + s`.
      s: `Tensor`, Python int, or `TensorShape`.

    Returns:
      If `t` is a matrix or higher order tensor, then the return value is
      `t` reshaped to `[batch_size, beam_width] + s`.  Otherwise `t` is
      returned unchanged.

    Raises:
      ValueError: If the rank of `t` is not statically known.
    """
    if isinstance(t, tf.TensorArray):
        return t
    _check_ndims(t)
    if t.shape.ndims >= 1:
        return _split_batch_beams(batch_size, beam_width, t, s)
    else:
        return t


def _maybe_merge_batch_beams(batch_size, beam_width, t, s):
    """Splits the tensor from a batch by beams into a batch of beams.

    More exactly, `t` is a tensor of dimension
    `[batch_size * beam_width] + s`, then we reshape it to
    `[batch_size, beam_width] + s`.

    Args:
      t: `Tensor` of dimension `[batch_size * beam_width] + s`.
      s: `Tensor`, Python int, or `TensorShape`.

    Returns:
      A reshaped version of t with shape `[batch_size, beam_width] + s`.

    Raises:
      ValueError:  If the rank of `t` is not statically known.
    """
    if isinstance(t, tf.TensorArray):
        return t
    _check_ndims(t)
    if t.shape.ndims >= 2:
        return _merge_batch_beams(batch_size, beam_width, t, s)
    else:
        return t


def _maybe_sort_array_beams(batch_size, beam_width, t, parent_ids, sequence_length):
    """Maybe sorts beams within a `TensorArray`.

    Args:
      t: A `TensorArray` of size `max_time` that contains `Tensor`s of
        shape `[batch_size, beam_width, s]` or
        `[batch_size * beam_width, s]` where `s` is the depth shape.
      parent_ids: The parent ids of shape
        `[max_time, batch_size, beam_width]`.
      sequence_length: The sequence length of shape
        `[batch_size, beam_width]`.

    Returns:
      A `TensorArray` where beams are sorted in each `Tensor` or `t` itself
        if it is not a `TensorArray` or does not meet shape requirements.
    """
    if not isinstance(t, tf.TensorArray):
        return t
    if t.element_shape.ndims is None or t.element_shape.ndims < 1:
        tf.get_logger().warn(
            "The TensorArray %s in the cell state is not amenable to "
            "sorting based on the beam search result. For a "
            "TensorArray to be sorted, its elements shape must be "
            "defined and have at least a rank of 1, but saw shape: %s" % (t.handle.name, t.element_shape)
        )
        return t
    if not _check_static_batch_beam_maybe(t.element_shape, tf.get_static_value(batch_size), beam_width):
        return t
    t = t.stack()
    with tf.control_dependencies([_check_batch_beam(t, batch_size, beam_width)]):
        return gather_tree_from_array(t, parent_ids, sequence_length)

