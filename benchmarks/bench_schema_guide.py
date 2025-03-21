import os
from concurrent.futures import ThreadPoolExecutor

import psutil
from outlines_core import Guide, Index, Vocabulary, json_schema

schema_samples = {
    "schema_simple":r'{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}',
    "schema_simple_and_complex_phone" : r'{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}',
    "schema_complexe": """{
  "$schema": "http://json-schema.org/draft-04/schema#",
  "title": "Schema for a recording",
  "type": "object",
  "definitions": {
    "artist": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "functions": {
          "type": "array",
          "items": {"type": "string"}
        }
      },
      "required": ["id", "name", "functions"]
    }
  },
  "properties": {
    "id": {"type": "number"},
    "work": {
      "type": "object",
      "properties": {
        "id": {"type": "number"},
        "name": {"type": "string"},
        "composer": {"$ref": "#/definitions/artist"}
      }
    },
    "recording_artists": {
      "type": "array",
      "items": {"$ref": "#/definitions/artist"}
    }
  },
  "required": ["id", "work", "recording_artists"]
}""",
 "schema_curriculum":r'''{
                "$schema": "http://json-schema.org/draft-04/schema#",
                "title": "Schema for a Curriculum Vitae",
                "type": "object",
                "definitions": {
                    "experienceEntry": {
                    "type": "object",
                    "properties": {
                        "date": {
                        "type": "string",
                        "format": "date"
                        },
                        "position": {
                        "type": "string"
                        }
                    },
                    "required": ["date", "position"]
                    }
                },
                "properties": {
                    "name": {
                    "type": "string"
                    },
                    "surname": {
                    "type": "string"
                    },
                    "email": {
                    "type": "string",
                    "pattern": "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?"
                    },
                    "phone": {
                    "type": "string",
                    "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"
                    },
                    "website": {
                    "type": "string",
                    "pattern": "(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?"
                    },
                    "resume": {
                    "type": "array",
                    "items": {
                        "$ref": "#/definitions/experienceEntry"
                    }
                    }
                },
                "required": ["name", "surname", "email", "phone", "resume"]
                }'''
}


class SchemaIndexBenchmark:
    params = schema_samples.keys()

    def setup(self, pattern_name):
        self.vocabulary = Vocabulary.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
        self.pattern = json_schema.build_regex_from_schema(schema_samples[pattern_name])

    def time_schema_to_guide(self, pattern_name):
        Index(self.pattern, self.vocabulary)

    def time_schema_to_guide_threads(self, pattern_name):
        # Default GIL switch interval is 5ms (0.005), which isn't helpful for cpu heavy tasks,
        # this parallel case should be relatively close in runtime to one thread, but it is not,
        # because of the GIL.
        core_count = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            list(executor.map(self._from_schema, [pattern_name] * core_count))

    def time_schema_to_guide_threads_with_custom_switch_interval(self, pattern_name):
        # Note: after moving to full rust implementation for index and guide creation, this experiment
        # is no longer shows the drastic difference as it once showed when python was heavily involved,
        # due to average speedup ~10 times.

        # This test is to show, that if GIL's switch interval is set to be longer, then the parallel
        # test's runtime on physical cores will be much closer to the one-threaded case.
        import sys

        sys.setswitchinterval(5)

        core_count = psutil.cpu_count(logical=False)
        with ThreadPoolExecutor(max_workers=core_count) as executor:
            list(executor.map(self._from_schema, [pattern_name] * core_count))

    def _from_schema(self, pattern_name):
        Index(self.pattern, self.vocabulary)

