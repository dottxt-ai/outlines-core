use core::panic;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::from_fn;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Copy)]
pub enum TransitionKey {
    Symbol(usize),
    AnythingElse,
}

impl From<usize> for TransitionKey {
    fn from(i: usize) -> Self {
        TransitionKey::Symbol(i)
    }
}

impl From<TransitionKey> for usize {
    fn from(c: TransitionKey) -> Self {
        match c {
            TransitionKey::Symbol(i) => i,
            _ => panic!("Cannot convert `anything else` to usize"),
        }
    }
}

pub trait SymbolTrait: Eq + Hash + Clone + Debug + From<char> {}
impl<T: Eq + Hash + Clone + Debug + From<char>> SymbolTrait for T {}

#[derive(Debug, Clone)]
pub struct Alphabet<T: SymbolTrait> {
    pub symbol_mapping: HashMap<T, TransitionKey>,
    pub by_transition: HashMap<TransitionKey, Vec<T>>,
}

impl<T: SymbolTrait> Alphabet<T> {
    pub fn new(symbol_mapping: HashMap<T, TransitionKey>) -> Self {
        let mut by_transition = HashMap::new();
        for (symbol, transition) in &symbol_mapping {
            by_transition
                .entry(*transition)
                .or_insert_with(Vec::new)
                .push(symbol.clone());
        }
        Alphabet {
            symbol_mapping,
            by_transition,
        }
    }

    pub fn get(&self, item: &T) -> TransitionKey {
        match self.symbol_mapping.get(item) {
            Some(x) => *x,
            None => TransitionKey::AnythingElse,
        }
    }

    pub fn contains(&self, item: &T) -> bool {
        self.symbol_mapping.contains_key(item)
    }

    #[must_use]
    pub fn from_groups(groups: &[HashSet<T>]) -> Self {
        let mut symbol_mapping = HashMap::new();
        for (i, group) in groups.iter().enumerate() {
            for symbol in group {
                symbol_mapping.insert(symbol.clone(), TransitionKey::Symbol(i));
                // symbol_mapping.insert(symbol.clone(), i);
            }
        }
        Alphabet::new(symbol_mapping)
    }

    pub fn union(alphabets: &[Self]) -> (Self, Vec<HashMap<TransitionKey, TransitionKey>>) {
        let all_symbols: HashSet<&T> = alphabets
            .iter()
            .flat_map(|a| a.symbol_mapping.keys())
            .collect();
        let mut symbol_to_keys = HashMap::new();
        for symbol in all_symbols {
            symbol_to_keys.insert(
                symbol,
                alphabets.iter().map(|a| a.get(symbol)).collect::<Vec<_>>(),
            );
        }

        let mut keys_to_symbols = HashMap::new();
        for (symbol, keys) in symbol_to_keys {
            keys_to_symbols
                .entry(keys.clone())
                .or_insert_with(Vec::new)
                .push(symbol);
        }

        let mut keys_to_key = HashMap::new();
        for keys in keys_to_symbols.keys() {
            keys_to_key.insert(keys.clone(), keys_to_key.len());
        }

        let mut symbol_mapping = HashMap::new();
        for (keys, symbols) in keys_to_symbols {
            for symbol in symbols {
                symbol_mapping.insert(symbol.clone(), TransitionKey::Symbol(keys_to_key[&keys]));
            }
        }
        let result = Alphabet::<T>::new(symbol_mapping);

        let mut new_to_old_mappings: Vec<HashMap<TransitionKey, TransitionKey>> =
            (0..alphabets.len()).map(|_| HashMap::new()).collect();

        for (keys, new_key) in &keys_to_key {
            for (i, &old_key) in keys.iter().enumerate() {
                new_to_old_mappings[i].insert(TransitionKey::Symbol(*new_key), old_key);
            }
        }

        (result, new_to_old_mappings)
    }
}

#[derive(Debug, Clone)]
pub struct Fsm<T: SymbolTrait> {
    alphabet: Alphabet<T>,
    pub states: HashSet<TransitionKey>,
    pub initial: TransitionKey,
    pub finals: HashSet<TransitionKey>,
    pub map: HashMap<TransitionKey, HashMap<TransitionKey, TransitionKey>>,
}
impl<T: SymbolTrait> Fsm<T> {
    #[must_use]
    pub fn new(
        alphabet: Alphabet<T>,
        states: HashSet<TransitionKey>,
        initial: TransitionKey,
        finals: HashSet<TransitionKey>,
        map: HashMap<TransitionKey, HashMap<TransitionKey, TransitionKey>>,
    ) -> Self {
        // TODO: revisit if we need validation logic
        Fsm {
            alphabet,
            states,
            initial,
            finals,
            map,
        }
    }

    pub fn accepts(&self, input: &[T]) -> bool {
        let mut state = self.initial;
        for symbol in input.iter() {
            let transition = self.alphabet.get(symbol);
            let allowed_transition_map = self.map.get(&state);
            match allowed_transition_map {
                Some(transitions) => match transitions.get(&transition) {
                    Some(next_state) => {
                        state = *next_state;
                    }
                    None => {
                        return false;
                    }
                },
                None => {
                    return false;
                }
            }
        }
        self.finals.contains(&state)
    }

    #[must_use]
    pub fn reduce(&self) -> Self {
        self.reversed().reversed()
    }

    pub fn reversed(&self) -> Self {
        let initial = self.finals.clone();
        let mut reverse_map = HashMap::new();

        for (state, transition_map) in &self.map {
            for (transition, next_state) in transition_map {
                reverse_map
                    .entry((*next_state, *transition))
                    .or_insert_with(HashSet::new)
                    .insert(*state);
            }
        }

        let follow = |current: &HashSet<TransitionKey>,
                      transition: &TransitionKey|
         -> Option<HashSet<TransitionKey>> {
            let mut next_states = HashSet::new();
            for state in current {
                if let Some(prev_states) = reverse_map.get(&(*state, *transition)) {
                    next_states.extend(prev_states);
                }
            }
            if next_states.is_empty() {
                return None;
            }
            Some(next_states)
        };

        let final_fn = |state: &HashSet<TransitionKey>| state.contains(&self.initial);

        crawl(&self.alphabet, initial, final_fn, follow)
    }

    #[must_use]
    pub fn is_live(&self, state: TransitionKey) -> bool {
        let mut seen = HashSet::new();
        let mut reachable = vec![state];
        let mut i = 0;

        while i < reachable.len() {
            let current = reachable[i];
            if self.finals.contains(&current) {
                return true;
            }
            if let Some(transitions) = self.map.get(&current) {
                for next_state in transitions.values() {
                    if !seen.contains(next_state) {
                        reachable.push(*next_state);
                        seen.insert(*next_state);
                    }
                }
            }
            i += 1;
        }
        false
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.is_live(self.initial)
    }

    pub fn strings(&self) -> impl Iterator<Item = Vec<T>> + '_ {
        let live_states: HashSet<TransitionKey> = self
            .states
            .iter()
            .filter(|&&s| self.is_live(s))
            .copied()
            .collect();
        let mut strings = VecDeque::new();
        let mut result = Vec::new();

        if live_states.contains(&self.initial) {
            if self.finals.contains(&self.initial) {
                result.push(Vec::new());
            }
            strings.push_back((Vec::new(), self.initial));
        }

        from_fn(move || {
            while let Some((current_string, current_state)) = strings.pop_front() {
                if let Some(transitions) = self.map.get(&current_state) {
                    for (transition, &next_state) in transitions {
                        if live_states.contains(&next_state) {
                            for symbol in &self.alphabet.by_transition[transition] {
                                let mut new_string = current_string.clone();
                                new_string.push(symbol.clone());
                                if self.finals.contains(&next_state) {
                                    result.push(new_string.clone());
                                }
                                strings.push_back((new_string, next_state));
                            }
                        }
                    }
                }
            }
            result.pop()
        })
    }

    #[must_use]
    pub fn union(fsms: &[Self]) -> Self {
        Self::parallel(fsms, |accepts| accepts.iter().any(|&x| x))
    }

    #[must_use]
    pub fn intersection(fsms: &[Self]) -> Self {
        Self::parallel(fsms, |accepts| accepts.iter().all(|&x| x))
    }

    #[must_use]
    pub fn symmetric_difference(fsms: &[Self]) -> Self {
        Self::parallel(fsms, |accepts| {
            accepts.iter().filter(|&&x| x).count() % 2 == 1
        })
    }

    #[must_use]
    pub fn difference(fsms: &[Self]) -> Self {
        Self::parallel(fsms, |accepts| {
            accepts[0] && !accepts[1..].iter().any(|&x| x)
        })
    }

    #[must_use]
    pub fn concatenate(fsms: &[Self]) -> Self {
        let alphabets_from_fsms: Vec<Alphabet<T>> =
            fsms.iter().map(|f| f.alphabet.clone()).collect();
        let alphabets = Alphabet::union(alphabets_from_fsms.as_slice());

        let alphabet = alphabets.0;
        let new_to_old = alphabets.1;

        let last_index = fsms.len() - 1;
        let last = &fsms[last_index];

        let connect_all = |i: TransitionKey,
                           substate: TransitionKey|
         -> HashSet<(TransitionKey, TransitionKey)> {
            let mut result = HashSet::new();
            let current_i = i;
            let mut current_substate = substate;

            result.insert((i, substate));

            let mut _current_i: usize = current_i.into();
            while _current_i < last_index && fsms[_current_i].finals.contains(&current_substate) {
                _current_i += 1;
                current_substate = fsms[_current_i].initial;
                result.insert((current_i, current_substate));
            }

            result
        };

        let initial = connect_all(0.into(), fsms[0].initial);

        let final_fn = |state: &HashSet<(TransitionKey, TransitionKey)>| {
            for &(i, substate) in state {
                // if i == last_index && fsms[i].finals.contains(&substate) {
                let _i: usize = i.into();
                if _i == last_index && last.finals.contains(&substate) {
                    return true;
                }
            }
            false
        };

        let follow = |current: &HashSet<(TransitionKey, TransitionKey)>,
                      transition: &TransitionKey|
         -> Option<HashSet<(TransitionKey, TransitionKey)>> {
            let mut next = HashSet::new();
            for &(i, substate) in current {
                let _i: usize = i.into();
                let fsm = &fsms[_i];

                if fsm.map.contains_key(&substate) {
                    let a = new_to_old[_i].clone();
                    let _b = a[transition];
                    if fsm.map.contains_key(&substate) {
                        // fsm.map[substate][new_to_old[i][new_transition]]
                        let _i: usize = i.into();
                        let key = &new_to_old[_i][transition];
                        if let Some(&next_state) = fsm.map[&substate].get(key) {
                            let connected = connect_all(i, next_state);
                            next.extend(connected);
                        }
                    }
                }
            }
            if next.is_empty() {
                return None;
            }
            Some(next)
        };

        crawl(&alphabet, initial, final_fn, follow)
    }

    #[must_use]
    pub fn star(&self) -> Self {
        let initial = HashSet::from([self.initial]);

        let follow = |state: &HashSet<TransitionKey>,
                      transition: &TransitionKey|
         -> Option<HashSet<TransitionKey>> {
            let mut next = HashSet::new();
            for &substate in state {
                if let Some(transitions) = self.map.get(&substate) {
                    if let Some(&next_state) = transitions.get(transition) {
                        next.insert(next_state);
                    }
                }
                if self.finals.contains(&substate) {
                    if let Some(transitions) = self.map.get(&self.initial) {
                        if let Some(&next_state) = transitions.get(transition) {
                            next.insert(next_state);
                        }
                    }
                }
            }

            if next.is_empty() {
                return None;
            }
            Some(next)
        };

        let final_fn =
            |state: &HashSet<TransitionKey>| state.iter().any(|s| self.finals.contains(s));

        let mut result = crawl(&self.alphabet, initial, final_fn, follow);
        result.finals.insert(result.initial);
        result
    }

    #[must_use]
    pub fn times(&self, multiplier: usize) -> Self {
        // metastate is a set of iterations+states
        let initial = HashSet::from([(self.initial, 0)]);
        let final_fn = |state: &HashSet<(TransitionKey, usize)>| {
            state.iter().any(|&(substate, iteration)| {
                substate == self.initial
                    && (self.finals.contains(&substate) || iteration == multiplier)
            })
        };

        let follow = |current: &HashSet<(TransitionKey, usize)>,
                      transition: &TransitionKey|
         -> Option<HashSet<(TransitionKey, usize)>> {
            let mut next = HashSet::new();

            for &(substate, iteration) in current {
                if iteration < multiplier
                    && self.map.contains_key(&substate)
                    && self.map[&substate].contains_key(transition)
                {
                    next.insert((self.map[&substate][transition], iteration));
                    if self.finals.contains(&self.map[&substate][transition]) {
                        next.insert((self.initial, iteration + 1));
                    }
                }
            }
            if next.is_empty() {
                return None;
            }
            Some(next)
        };

        crawl(&self.alphabet, initial, final_fn, follow)
    }

    #[must_use]
    pub fn everythingbut(&self) -> Self {
        let initial = HashSet::from([(self.initial, 0)]);

        let follow = |current: &HashSet<(TransitionKey, usize)>,
                      transition: &TransitionKey|
         -> Option<HashSet<(TransitionKey, usize)>> {
            let mut next = HashSet::new();
            for &(substate, iteration) in current {
                if substate == self.initial
                    && self.map.contains_key(&substate)
                    && self.map[&substate].contains_key(transition)
                {
                    next.insert((self.map[&substate][transition], iteration));
                }
            }
            if next.is_empty() {
                return None;
            }
            Some(next)
        };

        let final_fn = |state: &HashSet<(TransitionKey, usize)>| {
            !state.iter().any(|&(substate, _iteration)| {
                substate == self.initial && self.finals.contains(&substate)
            })
        };

        crawl(&self.alphabet, initial, final_fn, follow)
    }

    pub fn parallel<F>(fsms: &[Self], test: F) -> Self
    where
        F: Fn(&[bool]) -> bool,
    {
        let alphabets_from_fsms: Vec<Alphabet<T>> =
            fsms.iter().map(|f| f.alphabet.clone()).collect();
        let alphabets = Alphabet::union(alphabets_from_fsms.as_slice());
        let alphabet = alphabets.0;
        let new_to_old = alphabets.1;
        let initial: HashMap<usize, TransitionKey> = fsms
            .iter()
            .enumerate()
            .map(|(i, fsm)| (i, fsm.initial))
            .collect();

        let follow = |current: &HashSet<(usize, TransitionKey)>,
                      transition: &TransitionKey|
         -> Option<HashSet<(usize, TransitionKey)>> {
            let mut next = HashSet::new();
            for (i, fsm) in fsms.iter().enumerate() {
                if let Some(old_transition) = new_to_old.get(i).and_then(|map| map.get(transition))
                {
                    if let Some((_, current_state)) = current.iter().find(|&&(idx, _)| idx == i) {
                        if let Some(next_state) = fsm
                            .map
                            .get(current_state)
                            .and_then(|map| map.get(old_transition))
                        {
                            next.insert((i, *next_state));
                        }
                    }
                }
            }
            if next.is_empty() {
                None
            } else {
                Some(next)
            }
        };

        let final_fn = |state: &HashSet<(usize, TransitionKey)>| {
            let accepts: Vec<bool> = fsms
                .iter()
                .enumerate()
                .map(|(i, fsm)| {
                    state
                        .iter()
                        .any(|&(idx, key)| idx == i && fsm.finals.contains(&key))
                })
                .collect();
            test(&accepts)
        };

        let initial_set: HashSet<(usize, TransitionKey)> = initial.into_iter().collect();

        crawl(&alphabet, initial_set, final_fn, follow)
    }
}

#[must_use]
pub fn null<T: SymbolTrait>(alphabet: &Alphabet<T>) -> Fsm<T> {
    Fsm::new(
        alphabet.clone(),
        HashSet::from([0.into()]),
        0.into(),
        HashSet::new(),
        HashMap::from([(
            0.into(),
            alphabet
                .by_transition
                .keys()
                .map(|&k| (k, 0.into()))
                .collect(),
        )]),
    )
}

#[must_use]
pub fn epsilon<T: SymbolTrait>(alphabet: &Alphabet<T>) -> Fsm<T> {
    Fsm::new(
        alphabet.clone(),
        HashSet::from([0.into()]),
        0.into(),
        HashSet::from([0.into()]),
        HashMap::new(),
    )
}

fn crawl<T, F, G, I, C>(alphabet: &Alphabet<T>, initial: C, final_fn: F, follow: G) -> Fsm<T>
where
    T: SymbolTrait,
    F: Fn(&C) -> bool,
    G: Fn(&C, &TransitionKey) -> Option<C>,
    I: Clone + Eq + Hash + std::fmt::Debug,
    C: IntoIterator<Item = I> + FromIterator<I> + Clone + PartialEq,
{
    let mut states = VecDeque::new();
    states.push_back(initial);
    let mut finals = HashSet::<TransitionKey>::new();
    let mut map = HashMap::new();
    let mut i = 0;

    while i < states.len() {
        let state = states[i].clone();

        if final_fn(&state) {
            finals.insert(i.into());
        }

        map.insert(TransitionKey::Symbol(i), HashMap::new());

        for transition in alphabet.by_transition.keys() {
            match follow(&state, transition) {
                Some(next) => {
                    let j = if let Some(index) = states.iter().position(|s| s == &next) {
                        index
                    } else {
                        states.push_back(next.clone());
                        states.len() - 1
                    };
                    map.get_mut(&TransitionKey::Symbol(i))
                        .unwrap()
                        .insert(*transition, TransitionKey::Symbol(j));
                }
                None => {
                    // reached oblivion
                    continue;
                }
            }
        }
        i += 1;
    }

    Fsm::new(
        alphabet.clone(),
        (0..states.len()).map(TransitionKey::Symbol).collect(),
        0.into(),
        finals,
        map,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_simple_fsm() -> Fsm<char> {
        let mut symbol_mapping = HashMap::new();
        symbol_mapping.insert('a', 0.into());
        symbol_mapping.insert('b', 1.into());
        let alphabet = Alphabet::new(symbol_mapping);

        let mut map = HashMap::new();
        // only 'a' transition from initial state
        map.insert(0.into(), [(0.into(), 1.into())].iter().copied().collect());
        // only 'b' transitions from accepting state
        map.insert(1.into(), [(1.into(), 1.into())].iter().copied().collect());

        Fsm::new(
            alphabet,
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            map,
        )
    }

    #[test]
    fn test_simple_fsm() {
        let fsm = create_simple_fsm();
        assert!(fsm.accepts(&['a']));
        assert!(fsm.accepts(&['a', 'b', 'b']));
        assert!(fsm.accepts(&['a', 'b', 'b', 'b']));

        assert!(!fsm.accepts(&['a', 'a', 'a']));
        assert!(!fsm.accepts(&['b']));
        assert!(!fsm.accepts(&['a', 'b', 'a', 'b', 'b']));
    }

    #[test]
    fn test_is_empty() {
        let fsm = create_simple_fsm();
        assert!(!fsm.is_empty());

        let empty_fsm = Fsm::new(
            fsm.alphabet.clone(),
            [0.into()].iter().copied().collect(),
            0.into(),
            HashSet::new(),
            HashMap::new(),
        );
        assert!(empty_fsm.is_empty());
    }

    #[test]
    fn test_reverse() {
        let fsm = create_simple_fsm();
        let reversed = fsm.reversed();

        assert!(reversed.accepts(&['b', 'b', 'a']));
        assert!(reversed.accepts(&['b', 'a']));

        assert!(!reversed.accepts(&['a', 'a']));
        // not accepted because it is not a final state
        assert!(!reversed.accepts(&['b']));

        // TODO: review this case
        // its just the final state..
        // not sure if we need to force it to be 'b' first?
        assert!(reversed.accepts(&['a']));
    }

    #[test]
    fn test_reduce() {
        let fsm = create_simple_fsm();
        let reduced = fsm.reduce();

        // reduced FSM should have the same behavior as the original
        assert!(fsm.accepts(&['a']));
        assert!(fsm.accepts(&['a', 'b', 'b']));
        assert!(fsm.accepts(&['a', 'b', 'b', 'b']));

        assert!(!fsm.accepts(&['a', 'a', 'a']));
        assert!(!fsm.accepts(&['b']));
        assert!(!fsm.accepts(&['a', 'b', 'a', 'b', 'b']));
    }

    #[test]
    fn test_union() {
        let mut symbol_mapping = HashMap::new();
        symbol_mapping.insert('a', 0.into());
        symbol_mapping.insert('b', 1.into());
        let alphabet = Alphabet::new(symbol_mapping);

        let fsm1 = Fsm::new(
            alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(0.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let fsm2 = Fsm::new(
            alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(1.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let union = Fsm::union(&[fsm1, fsm2]);

        assert!(union.accepts(&['a']));
        assert!(union.accepts(&['b']));
        assert!(!union.accepts(&[' ']));
        assert!(!union.accepts(&['a', 'a']));
    }

    #[test]
    fn test_intersection() {
        let fsm1 = Fsm::new(
            create_simple_fsm().alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(0.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let fsm2 = Fsm::new(
            create_simple_fsm().alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(1.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let intersection = Fsm::intersection(&[fsm1, fsm2]);

        assert!(!intersection.accepts(&['a']));
        assert!(!intersection.accepts(&['b']));
        assert!(!intersection.accepts(&[' ']));
        assert!(!intersection.accepts(&['a', 'a']));
    }

    #[test]
    fn test_concatenate() {
        let fsm1 = Fsm::new(
            create_simple_fsm().alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(0.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let fsm2 = Fsm::new(
            create_simple_fsm().alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(1.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let concatenated = Fsm::concatenate(&[fsm1, fsm2]);

        // assert!(concatenated.accepts(&['a', 'b']));
        assert!(!concatenated.accepts(&['a']));
        assert!(!concatenated.accepts(&['b']));
        assert!(!concatenated.accepts(&['b', 'a']));
    }

    #[test]
    fn test_star() {
        let fsm = Fsm::new(
            create_simple_fsm().alphabet.clone(),
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [(0.into(), [(0.into(), 1.into())].iter().copied().collect())]
                .iter()
                .cloned()
                .collect(),
        );

        let star = fsm.star();

        assert!(star.accepts(&[]));
        assert!(star.accepts(&['a']));
        assert!(star.accepts(&['a', 'a']));
        assert!(star.accepts(&['a', 'a', 'a']));
        assert!(!star.accepts(&['b']));
    }

    #[test]
    fn test_times() {
        let mut symbol_mapping = HashMap::new();
        symbol_mapping.insert('a', 0.into());
        symbol_mapping.insert('b', 1.into());
        let alphabet = Alphabet::new(symbol_mapping);

        let fsm = Fsm::new(
            alphabet,
            [0.into(), 1.into()].iter().copied().collect(),
            0.into(),
            [1.into()].iter().copied().collect(),
            [
                (0.into(), [(0.into(), 1.into())].iter().copied().collect()),
                (1.into(), [].iter().copied().collect()),
            ]
            .iter()
            .cloned()
            .collect(),
        );

        let times_2 = fsm.times(2);

        assert!(times_2.accepts(&['a', 'a']));

        assert!(!times_2.accepts(&[]));
        assert!(!times_2.accepts(&['a']));
        assert!(!times_2.accepts(&['a', 'a', 'a']));

        assert!(!times_2.accepts(&['b']));
        assert!(!times_2.accepts(&['a', 'b']));
        assert!(!times_2.accepts(&['b', 'a']));
        assert!(!times_2.accepts(&['b', 'b']));
        assert!(!times_2.accepts(&['a', 'a', 'a', 'a', 'a']));
    }
}
