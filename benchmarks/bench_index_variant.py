# flake8: noqa
# mypy: ignore-errors
import os
import random
import time

import psutil
from outlines_core import Guide, Index, Vocabulary, mask_bytearray_to_list
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
}""",
    },
]


class IndexVariantBenchmark:
    def setup(self, regex):
        self.vocab = Vocabulary.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct")
        self.standard_index = Index(regex, self.vocab)
        self.compressed_index = Index.with_compressed_index(regex, self.vocab)

        self.standard_guide = Guide(self.standard_index)
        self.compressed_guide = Guide(self.compressed_index)

        self.process = psutil.Process()
        assert (
            not self.standard_guide.is_finished()
        ), f"Standard Guide should not be finished for {regex}"

        assert (
            not self.compressed_guide.is_finished()
        ), f"Compressed Guide should not be finished for {regex}"

    def run_benchmark(self):
        iterations = 0
        standard_total_time = 0
        compressed_total_time = 0

        self.current_token_id = -1

        if not self.standard_guide.is_finished():
            iterations += 1

            start_standard = time.perf_counter()
            standard_tokens = self.standard_guide.get_tokens()
            end_standard = time.perf_counter()

            standard_time = end_standard - start_standard
            standard_total_time += standard_time

            start_compressed = time.perf_counter()
            compressed_tokens = self.compressed_guide.get_allowed_tokens_mask()
            end_compressed = time.perf_counter()

            compressed_time = end_compressed - start_compressed
            compressed_total_time += compressed_time

            random_idx = random.randrange(len(standard_tokens))
            self.current_token_id = standard_tokens[random_idx]

            mask_tokens_list = mask_bytearray_to_list(compressed_tokens)
            assert (
                self.current_token_id in mask_tokens_list,
                f"Token {self.current_token_id} from Standard not found in Compressed, Iteration {iterations}",
            )

            # print(f"Iteration {iterations}: Token chosen = {self.current_token_id}")
            # print(f"Standard state: {self.standard_guide.get_state()}")
            # print(f"Compressed state: {self.compressed_guide.get_state()}")

        while not self.standard_guide.is_finished():
            iterations += 1

            start_standard = time.perf_counter()
            standard_tokens = self.standard_guide.advance(self.current_token_id)
            end_standard = time.perf_counter()

            standard_time = end_standard - start_standard
            standard_total_time += standard_time  # noqa: E713

            start_compressed = time.perf_counter()
            compressed_tokens = self.compressed_guide.advance_compressed(
                self.current_token_id
            )
            end_compressed = time.perf_counter()

            compressed_time = end_compressed - start_compressed
            compressed_total_time += compressed_time

            # print(f"Iteration {iterations}: Token chosen = {self.current_token_id}")
            # print(f"Standard state: {self.standard_guide.get_state()}")
            # print(f"Compressed state: {self.compressed_guide.get_state()}")

            assert (
                self.standard_guide.is_finished() == self.compressed_guide.is_finished()
            ), f"Guides out of sync, Iteration {iterations}"

            if not self.standard_guide.is_finished():
                random_idx = random.randrange(len(standard_tokens))
                self.current_token_id = standard_tokens[random_idx]

                mask_tokens_list = mask_bytearray_to_list(compressed_tokens)
                assert (
                    self.current_token_id in mask_tokens_list,
                    f"Token {self.current_token_id} from Standard not found in Compressed, Iteration {iterations}",  # noqa: E731,F631
                )  # noqa: E731

        stantard_total_time_us = standard_total_time * 1e6
        compressed_total_time_us = compressed_total_time * 1e6

        print(f"  Total iterations (Number of tokens): {iterations}")
        print(
            f"  Guide with Standard Index: {stantard_total_time_us:.2f} µs ({stantard_total_time_us / iterations:.2f} µs per iteration)"
        )
        print(
            f"  Guide with Compressed Index: {compressed_total_time_us:.2f} µs ({compressed_total_time_us / iterations:.2f} µs per iteration)"
        )
        print(
            f"  Speedup ratio: {stantard_total_time_us / compressed_total_time_us:.2f}x"
        )


def test_benchmark_index_variant():
    for r in regexes:
        name = r["name"]
        regex = r["regex"]

        print(f"> Regex : '{name}'")
        bench = IndexVariantBenchmark()
        bench.setup(regex)
        bench.run_benchmark()

    for s in schemas:
        name = s["name"]
        schema = s["regex"]
        regex = build_regex_from_schema(schema, None)
        print(f"> Schema : '{name}'")
        bench = IndexVariantBenchmark()
        bench.setup(regex)
        bench.run_benchmark()


if __name__ == "__main__":
    print("Running main...")
    test_benchmark_index_variant()
