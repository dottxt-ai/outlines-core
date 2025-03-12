# V2Index and TokensDFA

> The current Index is a naive implementation. It means for a given DFA build from a regex it will 'bruteforce' 
> each state encountered during progression in the graph with all the tokens in order to build the tokens transitions table.
> This results in a complexity proportional to the size of the model vocabulary, the average size of the tokens in bytes and the complexity of the regex.
> The following is the will of  build an approach that takes the behaviors of DFA for regexes and extends them to the token scale in order to be less burdened by the complexity of regexes and the size of vocabularies.
> 
> At the end, the V2Index has much better compile-time performance than its predecessor, much better performance in serving the list of allowed tokens for each state, and takes up less memory in most cases.
 ---

 ## A. TokensDFA : Description

This new version of Index includes a TokensDFA object.
This TokenDFA can be seen as an extension of DFA in that it leverages DFA optimizations to reduce the computational complexity of constructing the tokens transitions table.
The trade-off that is made is to spend time upstream of the construction of the transition table in order to gain advantages during construction.

***Regex's world is such a childish world. Only 256 different values to manage, all of them with one byte size. 
Tokens world has no limit of different values with no limit of size. Dante described it as "Malebolge"***


```rust
pub struct TokensDFA
 {
    pub eos_token_id:u32,
    pub eos_class_id:u32,
    pub start_state: StateId,
    pub final_states: HashSet<StateId>,
    pub transitions_table: MasksTable,  
}
``` 
The structure of the TokensDFA is very similar to the current index. The difference lies in the initialization.
A series of five optimizations has been implemented:

### 1. Reduce Vocabulary size

A static analysis of the regex is made in order to make the list of the 'dead bytes'.
'dead bytes' are bytes that will not be allowed at any place in the regex.
It allows us to quickly discriminate all the tokens that have at least one of the dead bytes.
```rust
let byte_classes = dfa.byte_classes();
let mut dead_byte_classes:HashSet<u8> = compile_dead_byte_classes(&muted_regex, &byte_classes);
```
Before going further, one thing very important to know about DFA is, when it compile,  it tries to regroup bytes by class.
Bytes in the same class has same effect on the regex's graph.
```regex
"^[a-z]$"
```
In this example, all the char from 'a' to 'z' has the same class because they trigger the same behavior.
So, there are 2 states and only one transition.
Conversely, with the regex `"^a[a-z]$"` the char 'a' will have a different class than the chars 'b' to 'z'. 
Because only the 'a' is allowed as transition at state 0. Then, two classes are allowed. The one of 'a' and the one of [b-z].
It allows the DFA to reduce drastically the number of transitions by considering classes as transitions values.

We will use and abuse of these classes.

### 2. Tokens Classification

We take the ByteClasses of the DFA and we construct the class of each token by concating the classes of each of their byte.
In other world, if the range of bytes `[a-z]` has the class `[a]`, the token `'man'` will have the class `[a][a][a]` like all the 
tokens of 3 letters. 
So we put all the tokens behind their classes which allows us to only consider the classes for the construction of the transition table.

### 3. Prefix-Based Graph

After grouping tokens by their regex byte classes, we construct directed prefix-based graphs to efficiently represent token hierarchies and optimize pattern matching traversal.
```
[a]
  ↳ [a,b]
       ↳ [a,b,c]

[b]
  ↳ [b,c]
       ↳ [b,c,d]
       ↳ [b,c,e]
```
```rust
let eos_class_id = init_classes_and_graph_optimized(
                        vocabulary.tokens(), 
                        &additionnal_tokens,
                        &mut token_classes_graph, 
                        &mut transitions_table,
                        byte_classes, 
                        &mut dead_byte_classes,
                        eos_token_id);
```
By traversing the DFA transition table with each prefix-based graph, this allows us to quickly discriminate entire sections of tokens as soon as one of their prefixes encounters a dead state.

### 4. Good old Parallelization

The previous optimisation, a bunch of graphs which have no intersection, unlock the possibilities to to go through the DFA in parallel, with a thread by graph.
```rust
use rayon::prelude::*;
let roots = read_only_graph.get_roots();
 roots.par_iter()
        .for_each(|root| {
                       ...
        }
```

### 5. Ultima Optima : Mute Literals and coalescence

At this stage of optimization, the compilation times were already pretty good for sample regexes benchmark.
But it was weak  for JSON structure :


![image](https://github.com/user-attachments/assets/96269844-91df-4c33-9399-a9aa1be4cbb7)

After investigation it turns out that the problem comes from the literals !
Literals are worst nightmare for DFA (and by extension, TokensDFA).
It's easy to understand why. If we reconsidered our last regex `"^a[a-z]$"`, the char 'a' is a literal.
With classification, the char 'a' will not have the same class as the other letters.
By extension, every token for a given size, with a letter 'a' will not have the same classe as the other tokens with exact same size.
If we take two classes `'a' -> [a]` and `'b-z' -> [b]`, the words "hand", "five" and "ante" respectively have the classes 
'[b][a][b][b]' , '[b][b][b][b]' and '[a][b][b][b]'. It increases drastically the size of the alphabet, the number of transitions and the number of reached state.
And the big issue is that there is a lot of literals in JSON structures. (Every keys of attributes at least, every symboles {, ",}, etc...)

The best example is the 'HTTPS' regex.
| Regular Expression | V2Index Time | Index Time |
| ------------------ | ------------ | ---------- |
| `(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?` | 27.683738s | 22.3142975s |

Here, 'https' is a literal but also 'http', 'h', 't' and 'p'. It a huge stab in the previous optimisation.
Now, if we transform the 'https' determinist sequence by two 'ghost' symbols. (one for 'http', the other for 's' because 's' is optionnal with '?') :

| Regular Expression | V2Index Time | Index Time |
| ------------------ | ------------ | ---------- |
| `(∟1(∟2)?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?` | 1.41s | 22.3142975s |

Yes, it's a huge improvment. Again, literals are the worst nightmare of Regexes.

So, at the beginning, we add an other static analysis of the regex to extract every literals (or 'determinist sequence') with alphanumeric chars.
```rust
let (muted_regex, muted_list) = mute_literals(regex, vocabulary, &mut additionnal_tokens);
```

For each of them, we will find the best combination of tokens to express them. This is where **coalescence** takes place.
If we extract the literal 'filename', we can express it with tokens 'file', 'name', 'f', 'i', 'l', 'e', 'n', 'a', 'm', 'e'.
Then, we find the smallest combination, here, the tokens 'file' and 'name'. For these tokens, we create two 'ghost' symbols.
'Ghost' tokens are choosen with char which have  small probabilities to appear in the regex and zero probabilities to be a prefix of real tokens.

So, every 'Ghost' tokens begins by the char "\x1C" which is the File separator (Very Rare) then we concate with iteration index.
In our example, 'file' will be [28, 49] (byte values for "\x1C1") and 'name' will be [28,50] (byte values for "\x1C2").
We affect to 'ghost' tokens same ids than their respective real token and we create new regex with ghost tokens combination instead of the literals.



### 6 Minimize Transitions Table

We use the same structure as the CompressIndex here : https://github.com/agourdel/outlines-core/tree/opt/new-index-struct 
to reduce the index size on average after compilation and increase the performance to serve the allowed tokens.
When we reduce, we replace the ghost tokens by the real tokens.

```rust
transitions_table.reduce(muted_list);
``` 

Bitset Masks of allowed tokens are already initiate for every state.


## B - Compilations Benchmark (From Rust)

![image](https://github.com/user-attachments/assets/f18aaaa7-40da-48f6-9ab1-ed06d7fc6142)

## C - Memory Sizes Benchmark (From Rust)

![image](https://github.com/user-attachments/assets/b9f3fdf8-bb7a-4799-be61-7bf2da8f778a)

## D - Average Time to Inner Mask (From Python)

*Using mask reference as parameter*

![image](https://github.com/user-attachments/assets/338b09f9-828d-4963-8373-8449c734b2e7)

## E - Ready-To-Use 

With this branch, the V2Index is directly integrated into the Index python class without any breaking changes.
It's ready to use.
```python
class Guide:
   [...]
    def get_tokens(self, mask:Optional[array.array]) -> List[int]:
        """Gets the list of allowed tokens for the current state."""
        ...
    def advance(self, token_id: int, mask: Optional[array.array]) -> List[int]:
        """Guide moves to the next state provided by the token id and returns a list of allowed tokens."""
    [...]
```
The 'get_tokens()' and 'advance()' functions can be used as previous version.

```python
from outlines_core import Guide, Index, Vocabulary

v2_index = Index(regex, vocab)
v2_guide = Guide(v2_index)

list_tokens = v2_guide.get_tokens()
new_list_tokens = v2_guide.advance(list_tokens[0])

```

Or, they can be used with a reference to a mask. (Much faster)

```python
from outlines_core import Guide, Index, Vocabulary

v2_index = Index(regex, vocab)
v2_guide = Guide(v2_index)
mask : array.array = create_mask(vocab.size())
v2_guide.get_tokens(mask)
v2_guide.advance(mask)

```

## TODO 


1. Cleaning code and remove debug lines
2. More tests for the feature "Mute Literals" with tricky regexes
3. Some legacy python tests will not passed anymore because they implies number of transaction and this number has changed (dûe to coalescence).
4. Make tests of the end to end inference process. (Some undiscloded behavior can still be possible with complex structure regexes)
5. Buy coffee

    


 
 

