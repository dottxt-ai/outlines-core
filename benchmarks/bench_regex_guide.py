import os
from concurrent.futures import ThreadPoolExecutor

import psutil

from outlines_core import Guide, Index, Vocabulary

regex_samples = {
    "email": r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
    "complex_phone": "\\+?\\d{1,4}?[-.\\s]?\\(?\\d{1,3}?\\)?[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,4}[-.\\s]?\\d{1,9}",
    "simple_phone": "\\+?[1-9][0-9]{7,14}",
    "date": r"([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])|([0-9][0-9]|19[0-9][0-9]|20[0-9][0-9])(\.|-|/)([1-9]|0[1-9]|1[0-2])(\.|-|/)([1-9]|0[1-9]|1[0-9]|2[0-9]|3[0-1])",
    "time": r"(0?[1-9]|1[0-2]):[0-5]\d\s?(am|pm)?",
    "ip": r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",
    "url": r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?",
    "ssn": r"\d{3}-\d{2}-\d{4}",
    "complex_span_constrained_relation_extraction": "(['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?\\s\\|\\s([^|\\(\\)\n]{1,})\\s\\|\\s['\"\\ ,]?((?:of|resulting|case|which|cultures|a|core|extreme|selflessness|spiritual|various|However|both|vary|in|other|secular|the|religious|among|moral|and|It|object|worldviews|altruism|traditional|material|aspect|or|life|beings|virtue|is|however|opposite|concern|an|practice|it|for|s|quality|religions|In|Altruism|animals|happiness|many|become|principle|human|selfishness|may|synonym)['\"\\ ,]?)+['\"\\ ,]?(\\s\\|\\s\\(([^|\\(\\)\n]{1,})\\s\\|\\s([^|\\(\\)\n]{1,})\\))*\\n)*",
}


class RegexIndexBenchmark:
    params = regex_samples.keys()

    def setup(self, pattern_name):
        self.vocabulary = Vocabulary.from_pretrained("gpt2")
        self.pattern = regex_samples[pattern_name]

    def time_regex_to_guide(self, pattern_name):
        Index(self.pattern, self.vocabulary)

    def time_regex_to_guide_threads(self, pattern_name):
        # Default GIL switch interval is 5ms (0.005), which isn't helpful for cpu heavy tasks,
        # this parallel case should be relatively close in runtime to one thread, but it is not,
        # because of the GIL.
        core_count = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            list(executor.map(self._from_regex, [pattern_name] * core_count))

    def time_regex_to_guide_threads_with_custom_switch_interval(self, pattern_name):
        # Note: after moving to full rust implementation for index and guide creation, this experiment
        # is no longer shows the drastic difference as it once showed when python was heavily involved,
        # due to average speedup ~10 times.

        # This test is to show, that if GIL's switch interval is set to be longer, then the parallel
        # test's runtime on physical cores will be much closer to the one-threaded case.
        import sys

        sys.setswitchinterval(5)

        core_count = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            list(executor.map(self._from_regex, [pattern_name] * core_count))

    def _from_regex(self, pattern_name):
        Index(self.pattern, self.vocabulary)


class MemoryRegexIndexBenchmark:
    params = ["simple_phone", "complex_span_constrained_relation_extraction"]

    def setup(self, pattern_name):
        self.vocabulary = Vocabulary.from_pretrained("gpt2")
        self.pattern = regex_samples[pattern_name]

    def peakmem_regex_to_index(self, pattern_name):
        Index(self.pattern, self.vocabulary)


class MemoryStabilityBenchmark:
    params = [1, 10_000]

    def setup(self, num):
        self.vocabulary = Vocabulary.from_pretrained("gpt2")
        self.index = Index(".*", self.vocabulary)
        self.process = psutil.Process(os.getpid())

    def _memory_usage(self):
        return self.process.memory_info().rss / 1024**2

    def peakmem_guides_per_index(self, num_guides):
        initial = self._memory_usage()
        objects = [Guide(self.index) for i in range(num_guides)]
        final = self._memory_usage()

        assert len(objects) == num_guides
        assert final - initial < 5


class WriteMaskIntoBenchmark:
    params = list(regex_samples.keys())
    param_names = ["regex_key"]

    def setup(self, regex_key):
        from outlines_core.kernels.torch import allocate_token_bitmask

        self.vocab = Vocabulary.from_pretrained("gpt2")
        self.mask = allocate_token_bitmask(len(self.vocab))
        self.index = Index(regex_samples[regex_key], self.vocab)
        self.guide = Guide(self.index)

    def time_write_mask_into(self, regex_key):
        self.guide.write_mask_into(
            self.mask.data_ptr(), self.mask.numel(), self.mask.element_size()
        )
