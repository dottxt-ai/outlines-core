# Not convinced by name of file
import numpy as np
from typing import List, Optional
from outlines_core.fsm.guide import RegexGuide

def test_generate_length():
    class MockTokenizer:
        vocabulary = {"0": 1, "1": 2, "eos": 3}
        inverse_vocabulary = {1: "0", 2: "1", 3: ""}
        special_tokens = {"eos"}
        eos_token_id = 3
        
        def length(self):
            return len(self.vocabulary)

        def convert_token_to_string(self, token):
            return self.inverse_vocabulary[token]
    
    class NextToken:
        probs: dict[int, List[float]] = {
            1: [0.3, 0.4, 0.0],
            2: [0.4, 0.3, 0.1],
            3: [0, 0, 0]
        }
        p0: List[float] = [0.2, 0.8, 0.0]
        states: List[int] = [1, 2, 3]
        
        def __call__(self, token: Optional[int], *, mask: List[int]) -> int:
            if token is None:
                prob = [p * m for (p, m) in zip(self.p0, mask)]
            elif token in self.states:
                prob = [p * m for (p, m) in zip(self.probs[token], mask)]
            else:
                raise ValueError("Should not be reached")
            return np.random.sample(self.states, p = prob)
        
    def generate(model, tokenizer, regex_str) -> str:
        out_str: str = ""
        n_tokens = tokenizer.length()
        
        fsm = RegexGuide.from_regex(regex_str, tokenizer)
        
        state = 0
        token = None
        while not fsm.is_final_state(state):
            allowed = fsm.get_next_instruction(state)
            mask = [1 if s in allowed else 0 for s in range(1, n_tokens + 1)]
            token = model(token, mask = mask)
            out_str += tokenizer.inverse_vocabulary[token]
            state = fsm.get_next_state(state, token)
        return out_str
    
    n_samples: int = 1000
    regex_str: str = r"11[01]+|0[01]*"
    tokenizer = MockTokenizer()
    model = NextToken()
    
    tot = 0
    for i in range(n_samples):
        out = generate(model, tokenizer, regex_str)
        tot += len(out)
        
    mean = tot / n_samples
    print(mean)
        
        
        
if __name__ == "__main__":
    test_generate_length()
        
        
        
        
        
        
    
            
            
            
    