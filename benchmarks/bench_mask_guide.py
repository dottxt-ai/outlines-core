import time

import psutil
from outlines_core import (
    Guide,
    Index,
    Vocabulary,
    create_mask,
    first_token_id_from_mask,
    mask_to_list,
)

# Regex samples from bench_regex_guide.py
regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    "simple_phone": r"\+?[1-9][0-9]{7,14}",
    "complex_phone": r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    "permissive_any": r".{1000}",
    "permissive_words": r"[a-zA-Z]{100}",
    "complex_span": r"(['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?\s\|\s([^|\(\)\n]{1,})\s\|\s['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?(\s\|\s\(([^|\(\)\n]{1,})\s\|\s([^|\(\)\n]{1,})\))*\n)*",
}


class GuideBenchmark:
    params = [
        ("small", regex_samples["simple_phone"]),
        ("small", regex_samples["email"]),
        ("large", regex_samples["permissive_any"]),
        ("large", regex_samples["permissive_words"]),
        ("large", regex_samples["complex_span"]),
    ]
    param_names = ["vocab_size", "regex"]

    def setup(self, vocab_size, regex):
        if vocab_size == "small":
            self.vocab = Vocabulary(
                7, {"a": [1], "b": [2], "z": [3], "0": [4], "1": [5], "2": [6]}
            )
            self.vocab_size = 7  # Max TokenId (4) + 1
        else:  # large
            self.vocab = Vocabulary.from_pretrained("gpt2")
            self.vocab_size = 50257  # GPT-2

        self.index = Index(regex, self.vocab)
        self.guide_initial = Guide(self.index)
        self.guide_mask = Guide(self.index)
        self.mask = create_mask(self.vocab_size)
        self.process = psutil.Process()
        self.is_permissive = regex in ["permissive_any", "permissive_words"]
        assert (
            not self.guide_initial.is_finished()
        ), f"Initial state should not be finished for {regex}"

    def time_initial_advance(self, vocab_size, regex):
        guide = Guide(self.index)
        allowed = []
        iteration = 0

        if not guide.is_finished():
            iteration += 1
            allowed = guide.get_tokens()
        while not guide.is_finished():
            iteration += 1
            allowed = guide.advance(allowed[0])

        print(f"time_initial_advance iteration : : {iteration}")

    def time_mask_advance(self, vocab_size, regex):
        guide = Guide(self.index)
        token_id = 0
        iteration = 0

        if not guide.is_finished():
            iteration += 1
            guide.get_tokens_into_mask(self.mask)
            token_id = first_token_id_from_mask(self.mask)
        while not guide.is_finished():
            iteration += 1
            token_id = first_token_id_from_mask(self.mask)
            guide.advance_with_mask(token_id, self.mask)

        print(f"time_mask_advance iteration : : {iteration}")

    def peakmem_initial_advance(self, vocab_size, regex):
        guide = Guide(self.index)
        allowed = []
        if not guide.is_finished():
            allowed = guide.get_tokens()
        while not guide.is_finished():
            allowed = guide.advance(allowed[0])

    def peakmem_mask_advance(self, vocab_size, regex):
        guide = Guide(self.index)
        token_id = 0
        if not guide.is_finished():
            guide.get_tokens_into_mask(self.mask)
        while not guide.is_finished():
            token_id = first_token_id_from_mask(self.mask)
            guide.advance_with_mask(token_id, self.mask)

    def _memory_usage(self):
        return self.process.memory_info().rss / 1024**2

    # def run_benchmark(self, method_name, vocab_size, regex):
    #     guide = Guide(self.index)
    #     initial_mem = self._memory_usage()
    #     start = time.perf_counter()
    #     iteration = 0
    #     if method_name == "initial":
    #         allowed = []
    #         if not guide.is_finished():
    #             iteration += 1
    #             allowed = guide.get_tokens()
    #         while not guide.is_finished():
    #             iteration += 1
    #             allowed = guide.advance(allowed[0])
    #     else:  # mask
    #         token_id = 0
    #         if not guide.is_finished():
    #             iteration += 1
    #             guide.get_tokens_into_mask(self.mask)
    #         while not guide.is_finished():
    #             iteration += 1
    #             token_id = first_token_id_from_mask(self.mask)
    #             guide.advance_with_mask(token_id, self.mask)
    #     end = time.perf_counter()
    #     final_mem = self._memory_usage()
    #     print(
    #         f"{method_name} time ({vocab_size}, {regex}): {(end - start) * 1e6:.2f} µs, Memory: {final_mem - initial_mem:.2f} MB, Iteration : {iteration}"
    #     )

    def run_benchmark(self, vocab_size, regex):
        guide_initial = Guide(self.index)
        guide_mask = Guide(self.index)
        mask = self.mask

        iterations = 0
        initial_total_time = 0
        mask_total_time = 0

        if not guide_initial.is_finished():
            iterations += 1

            # time for initial method
            start_initial = time.perf_counter()
            initial_tokens = guide_initial.get_tokens()
            end_initial = time.perf_counter()
            initial_time = end_initial - start_initial
            initial_total_time += initial_time

            # time for mask method
            start_mask = time.perf_counter()
            guide_mask.get_tokens_into_mask(mask)
            end_mask = time.perf_counter()
            mask_time = end_mask - start_mask
            mask_total_time += mask_time

            token_id = initial_tokens[0]

            mask_tokens = mask_to_list(mask)
            assert (
                token_id in mask_tokens
            ), f"Token {token_id} from initial not found in mask mask"  # noqa: E713

        while not guide_initial.is_finished():
            iterations += 1

            # Avancer avec la méthode initiale et mesurer le temps
            start_initial = time.perf_counter()
            initial_tokens = guide_initial.advance(token_id)
            end_initial = time.perf_counter()
            initial_time = end_initial - start_initial
            initial_total_time += initial_time

            # Avancer avec la méthode mask et mesurer le temps
            start_mask = time.perf_counter()
            guide_mask.advance_with_mask(token_id, mask)
            end_mask = time.perf_counter()
            mask_time = end_mask - start_mask
            mask_total_time += mask_time

            # S'assurer que les deux méthodes sont dans le même état
            assert (
                guide_initial.is_finished() == guide_mask.is_finished()
            ), "Guides out of sync"

            if not guide_initial.is_finished():
                # Sélectionner le prochain token (hors du temps mesuré)
                token_id = initial_tokens[0]

                # Vérifier que ce token est également dans la mask
                mask_tokens = mask_to_list(mask)
                assert (
                    token_id in mask_tokens
                ), f"Token {token_id} from initial not found in mask mask at iteration {iterations}"  # noqa: E713

        # Conversion en microsecondes
        initial_total_time_us = initial_total_time * 1e6
        mask_total_time_us = mask_total_time * 1e6

        print(f"Synchronized benchmark results ({vocab_size}, {regex}):")
        print(f"  Total iterations: {iterations}")
        print(
            f"  Initial method: {initial_total_time_us:.2f} µs ({initial_total_time_us / iterations:.2f} µs per iteration)"
        )
        print(
            f"  mask method: {mask_total_time_us:.2f} µs ({mask_total_time_us / iterations:.2f} µs per iteration)"
        )
        print(f"  Speedup ratio: {initial_total_time_us / mask_total_time_us:.2f}x")


def test_benchmark_small():
    bench = GuideBenchmark()
    bench.setup("small", regex_samples["simple_phone"])
    bench.run_benchmark("small", "simple_phone")


def test_benchmark_large_permissive():
    bench = GuideBenchmark()
    bench.setup("large", regex_samples["permissive_any"])
    bench.run_benchmark("large", "permissive_any")
