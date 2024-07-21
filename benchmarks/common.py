from outlines_core.models.transformers import TransformerTokenizer
from transformers import AutoTokenizer


def setup_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return TransformerTokenizer(tokenizer)
