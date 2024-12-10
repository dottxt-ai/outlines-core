from concurrent.futures import ThreadPoolExecutor

import psutil
from outlines_core.fsm.guide import RegexGuide

from .common import setup_tokenizer

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


class RegexGuideBenchmark:
    params = regex_samples.keys()

    def setup(self, pattern_name):
        self.tokenizer = setup_tokenizer()
        self.pattern = regex_samples[pattern_name]

    def time_regex_to_guide(self, pattern_name):
        RegexGuide.from_regex(self.pattern, self.tokenizer)

    def time_regex_to_guide_parallel(self, pattern_name):
        # Default GIL switch interval is 5ms (0.005), which isn't helpful for cpu heavy tasks,
        # this parallel case should be relatively close in runtime to one thread, but it is not,
        # because of the GIL.
        core_count = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            list(executor.map(self._from_regex, [pattern_name] * core_count))

    def time_regex_to_guide_parallel_with_custom_switch_interval(self, pattern_name):
        # This test is to show, that if GIL's switch interval is set to be longer, then the parallel
        # test's runtime on physical cores will be much closer to the one-threaded case.
        import sys

        sys.setswitchinterval(5)

        core_count = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            list(executor.map(self._from_regex, [pattern_name] * core_count))

    def _from_regex(self, pattern_name):
        RegexGuide.from_regex(self.pattern, self.tokenizer)


class MemoryRegexGuideBenchmark:
    params = ["simple_phone", "complex_span_constrained_relation_extraction"]

    def setup(self, pattern_name):
        self.tokenizer = setup_tokenizer()
        self.pattern = regex_samples[pattern_name]

    def peakmem_regex_to_guide(self, pattern_name):
        RegexGuide.from_regex(self.pattern, self.tokenizer)
