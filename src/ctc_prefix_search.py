# Based on Mitsubishi Electric Research Labs (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from functools import partial

import tensorflow as tf
import numpy as np

from scorer_base import ScorerBase

# Flag for debugging the prefix search in eager mode
debug = False


class CTCPrefixScorer(ScorerBase):
    """Decoder interface wrapper for CTCPrefixScore.

    Based on
    which is based on Algorithm 2 in WATANABE et al.
    "HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION,"
    """

    @staticmethod
    def is_partial_scorer():
        return True

    def call(self, inputs, **kwargs):
        # Variables
        # ==================================
        # B: batch size (or batch size * n_beams)
        # beams: n_beams
        # A: alphabet/vocab size (including blank)
        # y: last label [B]
        # log_p: log prob outputs of encoder [B, T, A]
        # last_gamma: last state of gamma = forward variables [B, 2, T]
        # cs: new tokens to select A* (subset of A), by default A* == A, [B, A*]
        # gamma: new state of gamma = forward variables for all cs [B, 2, T, A*]
        # log_psi: prefix probability for each beam [B, A*]

        y, states, cs = inputs["embeddings"], inputs["state"], inputs.get("cs", None)
        log_p, log_p_length = inputs["static_inputs"]

        batch_size = tf.shape(y)[0]  # incl. beams
        beams = log_p.shape[1]
        # log_p = tf.reshape(log_p, [batch_size, tf.shape(log_p)[2], log_p.shape[3]], name="flatten_log_p")
        log_p_length = tf.reshape(log_p_length, [batch_size, 1], name="flatten_log_p_length")

        # prepare state info
        last_gamma, last_log_psi, last_output_length = (
            states["gamma"],
            states["log_psi"],
            states["output_length"],
        )
        tf.assert_equal(batch_size, tf.shape(last_gamma)[0])
        last_gamma = tf.reshape(last_gamma, [batch_size, 2, self.input_length])

        tf.debugging.assert_shapes(
            [
                (y, [None]),
                (log_p, [None, None, self.vocab_size]),
                (last_gamma, [None, 2, None]),
                (last_log_psi, [None, 1]),
                (last_output_length, [None, 1]),
            ]
        )
        tf.debugging.assert_equal(True, tf.reduce_all(tf.equal(last_output_length, last_output_length[0])))

        # new CTC forward probs are prepared as a 2 x (BW x T x S) tensor
        # that corresponds to r_t^n(h) and r_t^b(h) in a batch.
        if cs is None:
            cs = tf.range(self.vocab_size, dtype=tf.int32)
            cs = tf.tile(tf.expand_dims(cs, axis=0), multiples=[batch_size, 1])
        tf.debugging.assert_equal(tf.rank(cs), 2)
        forward_batch_ = partial(forward_batch, eos=self.eos, blank=self.blank)
        if not debug:
            # If not debugging call in parallel
            log_psi, gamma = tf.map_fn(
                fn=forward_batch_,
                elems=(y, cs, last_gamma, log_p, log_p_length, last_output_length),
                fn_output_signature=(tf.float32, tf.float32),
                parallel_iterations=16,
            )
        else:
            # For debugging, call on first batch without map_fn
            rs = []
            for i in range(beams):
                rs.append(
                    forward_batch_((y[i], cs[i], last_gamma[i], log_p[i], log_p_length[i], last_output_length[i]))
                )

            log_psi, gamma = zip(*rs)
            log_psi = tf.stack(log_psi, axis=0)
            gamma = tf.stack(gamma, axis=0)

        tf.debugging.assert_equal(tf.shape(log_psi), tf.shape(cs))
        return (
            (log_psi - last_log_psi),
            {
                "gamma": tf.reshape(gamma, [batch_size, -1]),
                "log_psi": log_psi,
                "output_length": last_output_length + 1,  # Sequence is not started, always
            },
        )

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Get an initial state for decoding.
        Args:
            inputs (tf.Tensor, tf.Tensor): The encoded feature tensor, The lengths of the inputs
            batch_size
            dtype
        Returns: initial state
        """
        # initial CTC state is made of a frame x 2 tensor that corresponds to
        # r_t^n(<sos>) and r_t^b(<sos>), where 0 and 1 of axis=1 represent
        # superscripts n and b (non-blank and blank), respectively.
        x, xlens = inputs

        self.input_length = tf.shape(x)[1]  # Time dimension
        odim = x.shape[2]  # Feature dimension
        self.vocab_size = odim

        # initial state is <sos> (51) and (52)
        gamma_b = tf.cumsum(x[:, :, self.blank], axis=1)  # over Time [B * beams, T]
        gamma_n = tf.fill(dims=[batch_size, self.input_length], value=self.logzero)

        tf.debugging.assert_equal(tf.shape(gamma_b), [batch_size, self.input_length])
        tf.debugging.assert_equal(tf.shape(gamma_b), tf.shape(gamma_n))

        s = {
            "gamma": tf.reshape(
                tf.transpose(tf.stack([gamma_n, gamma_b]), perm=[1, 0, 2]), (batch_size, -1)
            ),  # [B, 2 * T]
            "log_psi": tf.reshape(tf.zeros((batch_size,)), shape=(batch_size, -1)),  # [B, 1]
            "output_length": tf.reshape(tf.fill((batch_size,), np.int32(0)), shape=(batch_size, 1)),  # [B, 1]
        }
        return s

    @property
    def state_size(self):
        return {
            "gamma": tf.TensorShape([None]),
            "log_psi": tf.TensorShape([None]),
            "output_length": tf.TensorShape([1]),  # Is the sequence starting
        }

    @property
    def output_size(self):
        return self.vocab_size

    def __init__(self, eos: int, blank: int, name: str = "ctc_prefix_scorer", **kwargs):
        """Initialize class.
        Args:
            ctc (torch.nn.Module): The CTC implementaiton.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.
        """
        super().__init__(name=name, **kwargs)
        self.vocab_size = -1
        self.input_length = -1
        self.eos = eos
        self.blank = blank
        self.logzero = np.float32(-1e10)
        assert self.blank >= 0, "this should be alphabet_size - 1"
        assert self.eos >= 0, "this should be 0"
        assert self.blank != self.eos, "Blank and EOS are at different locations!"

    def select_state(self, state, i: tf.Tensor):
        batch_size = tf.shape(i)[0]
        beams = i.shape[1]

        vocab_size = self.pre_beam_size if self.pre_beam_size else self.vocab_size

        def _select(v, idx):
            return tf.gather(
                tf.reshape(
                    tf.transpose(
                        tf.reshape(v, [batch_size, beams, -1, vocab_size]),
                        perm=[0, 1, 3, 2],
                    ),
                    shape=[batch_size, beams * vocab_size, -1],
                ),
                idx,
                batch_dims=1,
            )

        for key in {"log_psi", "gamma"}:
            state[key] = _select(state[key], i)
        return state


def forward_batch(inputs, blank=-1, eos=-1):
    # Variable Definitions
    # ==================================================
    # last: last label []
    # cs: array of all next labels [A*]  (where A is alphabet-size, or smaller)
    # r_prev: previous CTC state [2, T]  (gamma in paper)
    # x: encoder outputs [T, A]
    # xlen: length of x [1]
    # output_length: start of sequence decoding [1]
    # r: current CTC state [2, T, A*]  (incrementally appended for each  new T)
    # r_sum: precomputed transitions into new state [T] (forward probabilities of last label)
    # xs: encoder outputs, but only for the desired labels cs  [T, A*]  (where A* is the possibly smaller alphabet size)
    # log_phi: log space props of phi of paper [T, A*]  (transitions probs into new state for each char)
    # log_psi: prefix probability for each beam [A*]

    last, cs, r_prev, x, xlen, output_length = inputs

    output_length = tf.cast(output_length[0], xlen.dtype)  # [1] to scalar
    xlen = xlen[0]  # [1] to scalar
    alphabet_size = x.shape[-1]  # must be static
    alphabet_star_size = cs.shape[-1]  # must be static
    original_input_length = tf.cast(tf.shape(x)[0], xlen.dtype)

    tf.debugging.assert_shapes(
        [(last, []), (cs, [alphabet_star_size]), (r_prev, [2, None]), (x, [None, alphabet_size])]
    )

    logzero = -1e10

    # crop lengths
    x = x[:xlen]
    r_prev = r_prev[:, :xlen]

    # initialize CTC states
    # new CTC states are prepared as a frame x (n or b) x n_labels tensor
    # that corresponds to r_t^n(h) and r_t^b(h).
    xs = tf.gather(x, cs, axis=1)
    if output_length == 0:
        r = tf.expand_dims(tf.stack([xs[0], tf.tile([logzero], multiples=[alphabet_star_size])]), axis=1)
    else:
        r = tf.fill([2, 1, alphabet_star_size], logzero)

    tf.debugging.assert_shapes(
        [
            (xs, [None, alphabet_star_size]),
            (r, [2, None, alphabet_star_size]),
        ]
    )

    # prepare forward probabilities for the last label (line 10)
    r_sum = tf.expand_dims(tf.reduce_logsumexp(r_prev, axis=0), 1)  # log(r_t^n(g) + r_t^b(g)), [None, 1]
    log_phi = tf.broadcast_to(r_sum, [xlen, alphabet_star_size])
    if output_length > 0:
        equal_c_pos = tf.cast(tf.equal(last, cs), dtype=tf.float32)
        log_phi += equal_c_pos * (-log_phi + tf.expand_dims(r_prev[1], axis=1))

    tf.debugging.assert_shapes(
        [
            (r_sum, [None, 1]),
            (log_phi, [None, alphabet_star_size]),
        ]
    )

    # compute forward probabilities log(r_t^n(h)), log(r_t^b(h)),
    # and log prefix probabilities log(psi)
    # start at output_length since everything before has p = 0 (impossible to finish sequence)
    start = tf.maximum(output_length, 1)
    r = tf.concat([r, tf.fill([2, start - 1, alphabet_star_size], logzero)], axis=1)
    log_psi = r[0, start - 1]
    end_reached = tf.less_equal(xlen, tf.cast(tf.shape(r)[1], xlen.dtype))
    for t in range(start, xlen):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(r, tf.TensorShape((2, None, alphabet_star_size)))]
        )
        r_n_t = tf.reduce_logsumexp([r[0, -1], log_phi[t - 1]], axis=0) + xs[t]
        r_b_t = tf.reduce_logsumexp(r[:, -1], axis=0) + x[t, blank]  # note: use x not xs for correct blank index
        r = tf.concat([r, tf.expand_dims(tf.stack([r_n_t, r_b_t]), axis=1)], axis=1)
        log_psi = tf.reduce_logsumexp([log_psi, log_phi[t - 1] + xs[t]], axis=0)

    r = tf.concat(
        [
            r,
            tf.fill(
                [
                    2,
                    tf.maximum(tf.cast(0, xlen.dtype), original_input_length - tf.cast(tf.shape(r)[1], xlen.dtype)),
                    alphabet_star_size,
                ],
                logzero,
            ),
        ],
        axis=1,
    )  # remaining

    tf.debugging.assert_shapes([(log_psi, [alphabet_star_size])])

    # get P(...eos|X) that ends with the prefix itself
    eos_pos = tf.cast(tf.equal(cs, eos), log_psi.dtype)
    log_psi += eos_pos * (-log_psi + r_sum[-1])

    # exclude blank probs
    blank_pos = tf.cast(tf.equal(cs, blank), log_psi.dtype)
    log_psi += blank_pos * (-log_psi + logzero)

    # return the log prefix probability and CTC states, where the label axis
    # of the CTC states is moved to the first axis to slice it easily
    if end_reached:
        # reached the end, return logzero for all tokens except <eos>
        eos_pos = tf.cast(tf.not_equal(cs, eos), log_psi.dtype)
        return logzero * eos_pos, r
    return log_psi, r

