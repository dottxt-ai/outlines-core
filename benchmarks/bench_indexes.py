# flake8: noqa
# mypy: ignore-errors
import os
import random
import time

import psutil
from outlines_core import Guide, Index, Vocabulary, create_mask, mask_to_list
from outlines_core.json_schema import build_regex_from_schema

os.environ["RUST_LOG"] = "debug"


regexes = [
    {
        "name": "email",
        "regex": r"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]{1,63}(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]{1,63}){0,10})@(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,3}[a-z0-9](?:[a-z0-9-]{0,30}[a-z0-9])?",
    },
    {"name": "simple_phone", "regex": r"\+?[1-9][0-9]{7,14}"},
    {
        "name": "complex_phone",
        "regex": r"\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}",
    },
    {"name": "permissive_any", "regex": r".{255}$"},
    {"name": "permissive_words", "regex": r"[a-zA-Z]{100}"},
    {"name": "https", "regex" : r"(https?:\\/\\/)?([\\da-z\\.-]+)\\.([a-z\\.]{2,6})([\\/\\w \\.-]*)*\\/?"},
    {"name": "complexe", "regex" : r"""\{[ ]?"name"[ ]?:[ ]?"([^"\\\x00-\x1F\x7F-\x9F]|\\["\\])*"[ ]?,[ ]?"age"[ ]?:[ ]?(-)?(0|[1-9][0-9]*)[ ]?,[ ]?"complexe_phone"[ ]?:[ ]?"(\+?\d{1,4}?[-. ]?\(\d{1,3}\)?[-. ]?\d{1,4}[-. ]?\d{1,4}[-. ]?\d{1,9})"[ ]?\}"""}
]
schemas = [
    {
        "name": "schema_simple",
        "regex": r'{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}',
    },
    {
        "name": "schema_simple_phone",
        "regex": r'{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "complexe_phone": {"type": "string", "pattern": "\\+?\\d{1,4}?[-. ]?\\(\\d{1,3}\\)?[-. ]?\\d{1,4}[-. ]?\\d{1,4}[-. ]?\\d{1,9}"}}, "required": ["name", "age", "complexe_phone"]}',
    },
    {
        "name": "schema_complexe",
        "regex": """{
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
}"""
    },
    {
        "name" : "schema_curriculum",
        "regex" : r'''{
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
]


class V2IndexBenchmark:
    def setup(self, regex):
        self.vocab = Vocabulary.from_pretrained("unsloth/Llama-3.1-8B-Instruct")
        self.v2_index = Index(regex, self.vocab)

        self.v2_guide = Guide(self.v2_index)

        self.mask = create_mask(len(self.vocab) + 1)

        self.process = psutil.Process()

        assert (
            not self.v2_guide.is_finished()
        ), f"Compressed Guide should not be finished for {regex}"

    def run_benchmark(self):
        iterations = 0
        v2_total_time = 0

        self.current_token_id = -1

        if not self.v2_guide.is_finished():
            iterations += 1

            start_compressed = time.perf_counter()
            self.v2_guide.get_tokens(self.mask)
            end_compressed = time.perf_counter()

            v2_time = end_compressed - start_compressed
            v2_total_time += v2_time

        
            mask_tokens_list = mask_to_list(self.mask)
            random_idx = random.randrange(len(mask_tokens_list))
            self.current_token_id = mask_tokens_list[random_idx]


        while not self.v2_guide.is_finished():
            iterations += 1
            
            start_compressed = time.perf_counter()
            self.v2_guide.advance(self.current_token_id, self.mask)
            end_compressed = time.perf_counter()

            v2_time = end_compressed - start_compressed
            v2_total_time += v2_time

          
            if not self.v2_guide.is_finished():
                if iterations > 2000 :
                    break
                mask_tokens_list = mask_to_list(self.mask)
                random_idx = random.randrange(len(mask_tokens_list))
               
                self.current_token_id = mask_tokens_list[random_idx]
                

      
        v2_total_time_us = v2_total_time * 1e6

        print(f"  Total iterations (Number of tokens): {iterations}")
        print(
            f"  Guide with Compressed Index: {v2_total_time_us:.2f} µs ({v2_total_time_us / iterations:.2f} µs per iteration)"
        )
        


def test_benchmark_v2index():
    for r in regexes:
        name = r["name"]
        regex = r["regex"]

        print(f"> Regex : '{name}'")
        bench = V2IndexBenchmark()
        bench.setup(regex)
        bench.run_benchmark()

    for s in schemas:
        name = s["name"]
        schema = s["regex"]
        regex = build_regex_from_schema(schema, None)
        print(regex)
        print(f"> Schema : '{name}'")
        bench = V2IndexBenchmark()
        bench.setup(regex)
        bench.run_benchmark()


if __name__ == "__main__":
    print("Running main...")
    #test_benchmark_v2index()
    schema = schemas[3]['regex']
    regex = build_regex_from_schema(schema, None)
    print(regex)
    print(f"> Schema : curriculum")
    bench = V2IndexBenchmark()
    bench.setup(regex)
    bench.run_benchmark()
