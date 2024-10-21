from typing import Callable, List, Optional

import numpy as np
from outlines_core.fsm.guide import RegexGuide
from pytest import approx
from scipy.stats import ks_2samp


def test_generate_length():
    class MockTokenizer:
        vocabulary = {"0": 1, "1": 2, "eos": 3}
        inverse_vocabulary = {1: "0", 2: "1", 3: ""}
        special_tokens = {"eos"}
        eos_token_id = 3

        def length(self):
            return len(self.vocabulary)

        def convert_token_to_string(self, token):
            return token

        def decode(self, token):
            return self.inverse_vocabulary[token]

    class NextToken:
        def __init__(
            self,
            _prob: Callable[[List[int]], np.array],
            _p0: np.array,
            _states: List[int],
            _seed: int,
        ):
            self.rng = np.random.default_rng(_seed)
            self.prob = _prob
            self.p0 = _p0
            self.states = _states

        def __call__(
            self, tokens: Optional[List[int]], *, mask: List[int]
        ) -> List[int]:
            prob = self.prob(tokens) if tokens is not None else self.p0
            prob = prob * np.array(mask)
            next_t = [self.rng.choice(self.states, p=prob / np.sum(prob))]
            return tokens + next_t if tokens is not None else next_t

    def generate(model, tokenizer, regex_str) -> Optional[List[int]]:
        n_tokens = tokenizer.length()

        fsm = RegexGuide.from_regex(regex_str, tokenizer)
        state: int = fsm.initial_state
        tokens = None
        while state != -1:
            allowed = fsm.get_next_instruction(state).tokens
            mask: List[int] = [1 if s in allowed else 0 for s in range(1, n_tokens + 1)]
            tokens = model(tokens, mask=mask)
            state = fsm.get_next_state(state, tokens[-1])
        return tokens

    def prob_non_markov(tokens: List[int]) -> np.array:
        n0 = np.sum([t == 1 for t in tokens])
        n1 = len(tokens) - n0
        p = np.array([1 + np.exp(n1 - n0), 1 + np.exp(n0 - n1), 0])
        p = p / np.sum(p)
        return 0.7 * p + 0.3 * np.array([0, 0, 1])

    def prob_markov(token: List[int]) -> np.array:
        probs: dict[int, np.array] = {
            1: np.array([0.2, 0.5, 0.3]),
            2: np.array([0.3, 0.4, 0.3]),
        }
        return probs[token[-1]]

    p0: np.array = np.array([0.2, 0.6, 0.2])
    states: List[int] = [1, 2, 3]

    n_samples: int = 250
    regex_str: str = r"11[01]+|0[01]*"
    tokenizer = MockTokenizer()
    model1 = NextToken(prob_markov, p0, states, 30127)
    model2 = NextToken(prob_non_markov, p0, states, 24601)

    lengths1: np.array = np.zeros((n_samples,))
    lengths2: np.array = np.zeros((n_samples,))
    for i in range(n_samples):
        out1: List[int] = generate(model1, tokenizer, regex_str)
        lengths1[i] = len(out1) - 1  # take off the eos token
        out2: List[int] = generate(model2, tokenizer, regex_str)
        lengths2[i] = len(out2) - 1  # take off the eos token

    # 2 sample KS test to check that lengths has the same distribution as
    # L = 1 + 2*X + Y, where X ~ Bern(0.75) and Y ~ Neg-Binom(1, 0.3)
    # E(L) = 2.5 + 0.7/0.3
    # Var(L) = 4*0.25*0.75 + 0.7/0.3^2 = 0.75 + 0.7 / 0.3^2

    # Test means
    # True value is 4.833333. Assert values correspond to the seed and n=250.
    # Large sample performance has been checked and is correct.
    assert np.mean(lengths1) == approx(4.624)
    assert np.mean(lengths2) == approx(4.64)

    # Test variances
    # True value is 8.5277777777. Assert values correspond to the seed and n=250
    # Large sample performance has been checked and is correct.
    assert np.var(lengths1) == approx(9.106624)
    assert np.var(lengths2) == approx(7.5344)

    # Test distributions
    # Value from test ~(>0.99, 0.97) with these seeds. In general p ~ Unif(0,1), so we
    # are just checking that it's not small enough to trigger a fairly loose test.
    # Exact asserts for p-values commented out to guard against scipy changes.
    rng = np.random.default_rng(525600)
    L = (
        1
        + 2 * rng.binomial(1, 0.75, n_samples)
        + rng.negative_binomial(1, 0.3, n_samples)
    )
    _, p_value1 = ks_2samp(lengths1, L)
    _, p_value2 = ks_2samp(lengths2, L)
    assert p_value1 > 0.1
    assert p_value2 > 0.1
