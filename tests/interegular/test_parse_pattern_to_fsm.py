# TODO: THIS IS A WORK IN PROGRESS AND WILL BE COMPLETELY REFACTORED BEFORE MERGING
from outlines_core.fsm.regex import parse_pattern_to_fsm

import interegular


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

    equal_states = frozenset(fsm.states) == frozenset(ref_fsm.states)
    equal_initial = fsm.initial == ref_fsm.initial
    equal_finals = frozenset(fsm.finals) == frozenset(ref_fsm.finals)
    # equal_map = fsm.map == ref_fsm.map

    print()
    if equal_states and equal_initial and equal_finals:  # and equal_map:
        print(f"✅ Test passed for pattern: {pattern}")
    else:
        print(f"❌ Test failed for pattern: {pattern}")

    print("_symbol_mapping\n", ref_fsm.alphabet._symbol_mapping)
    print("by_transition\n", ref_fsm.alphabet.by_transition)

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

    # make maps deterministic (sort by key)
    fsm_map = sort_map(fsm.map)
    ref_map = sort_map(ref_fsm.map)

    print(f"  fsm: {fsm_map}")
    print(f"  ref: {ref_map}")

    return True


# TODO: remove if not needed
# tests copied so they can be run as a standalone script
if __name__ == "__main__":
    test_cases = [
        "a",
        # "ab",
        # "a|b",
        # "[ab]",
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
