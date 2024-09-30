# TODO: THIS IS A WORK IN PROGRESS AND WILL BE COMPLETELY REFACTORED BEFORE MERGING
from interegular.fsm import anything_else
from outlines_core.fsm.regex import parse_pattern_to_fsm

import interegular


class InteregularFSMInfo:
    def __init__(self, initial, finals, states, map, symbol_mapping, by_transition):
        self.initial = initial
        self.finals = finals
        self.states = states
        self.map = map
        self.symbol_mapping = symbol_mapping
        self.by_transition = by_transition


def map_states_with_symbols(state_map, symbol_mapping):
    inv_symbol_mapping = {v: k for k, v in symbol_mapping.items()}

    mapped_states = {}
    for state, transitions in state_map.items():
        mapped_transitions = {}
        for symbol, next_state in transitions.items():
            mapped_symbol = inv_symbol_mapping.get(symbol, symbol)
            mapped_transitions[mapped_symbol] = next_state
        mapped_states[state] = mapped_transitions

    return mapped_states


def make_fsm_comparable(fsm):
    # Create a new symbol mapping
    new_symbol_mapping = {}
    for symbol, value in fsm.symbol_mapping.items():
        if symbol == "\x00":
            new_symbol_mapping[anything_else] = value
        else:
            new_symbol_mapping[symbol] = value

    # Create a new map
    new_map = {}
    for state, transitions in fsm.map.items():
        new_transitions = {}
        for symbol, next_state in transitions.items():
            if symbol == b"\x00":
                new_transitions[anything_else] = next_state
            else:
                new_transitions[symbol] = next_state
        new_map[state] = new_transitions

    new_fsm = InteregularFSMInfo(
        states=fsm.states,
        initial=fsm.initial,
        finals=fsm.finals,
        map=new_map,
        symbol_mapping=new_symbol_mapping,
        by_transition=fsm.by_transition,
    )

    return new_fsm


def compare_sets(set1, set2):
    # ensure that the sets are equal
    return frozenset(set1) == frozenset(set2)


def sort_map(map):
    for key in map:
        if isinstance(map[key], dict):
            map[key] = sort_map(map[key])
    return dict(sorted(map.items()))


def test_parse_pattern_to_fsm(pattern):
    fsm = parse_pattern_to_fsm(pattern)
    fsm = make_fsm_comparable(fsm)

    ref_pattern = interegular.parse_pattern(pattern)

    # # interegulat alphabet
    # symbol_map = {
    #     "z": 0,
    #     "a": 1,
    #     "i": 2,
    #     "t": 3,
    #     anything_else: 4,
    #     "d": 5,
    #     "v": 6,
    #     "h": 7,
    #     "l": 8,
    #     "o": 9,
    # }
    # my_alphabet = Alphabet(symbol_map)

    my_alphabet = None

    ref_fsm = ref_pattern.to_fsm(my_alphabet)

    # TODO: prefer asserts once fsm building is implemented
    # Compare FSMs
    # assert fsm.states == ref_fsm.states
    # assert fsm.initial == ref_fsm.initial
    # assert fsm.finals == ref_fsm.finals
    # assert fsm.map == ref_fsm.map

    # make maps deterministic (sort by key)
    fsm_map = sort_map(fsm.map)
    ref_map = sort_map(ref_fsm.map)

    equal_states = frozenset(fsm.states) == frozenset(ref_fsm.states)
    equal_initial = fsm.initial == ref_fsm.initial
    equal_finals = frozenset(fsm.finals) == frozenset(ref_fsm.finals)
    equal_map = map_states_with_symbols(
        fsm.map, fsm.symbol_mapping
    ) == map_states_with_symbols(ref_fsm.map, ref_fsm.alphabet._symbol_mapping)

    print()
    if equal_states and equal_initial and equal_finals and equal_map:
        print(f"✅ Test passed for pattern: {pattern}")
    else:
        print(f"❌ Test failed for pattern: {pattern}")

    print("fsm: symbol_mapping\n", fsm.symbol_mapping)
    print("fsm: by_transition\n", fsm.by_transition)

    print("ref: symbol_mapping\n", ref_fsm.alphabet._symbol_mapping)
    print("ref: by_transition\n", ref_fsm.alphabet.by_transition)

    print("States")
    print(f"  fsm: {frozenset(fsm.states)}")
    print(f"  ref: {ref_fsm.states}")

    print("Initial")
    print(f"  fsm: {fsm.initial}")
    print(f"  ref: {ref_fsm.initial}")

    print("Finals")
    print(f"  fsm: {frozenset(fsm.finals)}")
    print(f"  ref: {ref_fsm.finals}")

    print("Map")

    print(f"  fsm: {fsm_map}")
    print(f"  ref: {ref_map}")

    print("Map with symbols")
    fsm_map_with_symbols = map_states_with_symbols(fsm_map, fsm.symbol_mapping)
    print(f"  fsm: {sort_map(fsm_map_with_symbols)}")

    ref_map_with_symbols = map_states_with_symbols(
        ref_map, ref_fsm.alphabet._symbol_mapping
    )
    print(f"  ref: {sort_map(ref_map_with_symbols)}")

    return True


# TODO: remove if not needed
# tests copied so they can be run as a standalone script
if __name__ == "__main__":
    test_cases = [
        # "a",
        # "ab",
        # "a|b",
        "[ab]",
        # TODO: long simple patterns (should work)
        # "aaaaa",
        # "davidholtz",
        # TODO: revisit these cases
        # "a*b+c?",
        # "(ab|cd)*",
        # "[a-z0-9]+",
        # "foo(bar|baz)*qux",
        # "(a|b|c){1,3}",
        # "[^aeiou]{2,4}"
    ]

    all_passed = all(test_parse_pattern_to_fsm(case) for case in test_cases)
    # print(f"All tests passed: {all_passed}")
