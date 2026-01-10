# Index Binary Format Specification

This document describes the binary format used for serializing and deserializing the `Index` structure.

## Overview

The Index is saved as a compressed binary file using gzip compression. The uncompressed data follows a structured format with fixed-size fields for efficient storage and retrieval.

## Binary Format Structure

All multi-byte integers are stored in **little-endian** format.

### Header Section

| Offset | Size (bits) | Field | Description |
|--------|-------------|-------|-------------|
| 0 | 32 | vocab_size | Size of the vocabulary used to build the index |
| 4 | 32 | eos_token_id | Token ID reserved for the end-of-sequence token |
| 8 | 32 | initial_state_id | ID of the initial state in the automaton |
| 12 | 32 | num_final_states | Number of final (accepting) states |

### Final States Section

Starting at offset 16, this section contains the IDs of all final states.

| Size (bits) | Field | Description |
|-------------|-------|-------------|
| 32 × num_final_states | final_state_ids | Array of final state IDs |

### Index Type

| Size (bits) | Field | Description |
|-------------|-------|-------------|
| 8 | index_type | Type identifier for the index format (currently only type 1 is supported) |

### Transitions Section (Type 1)

The format of this section depends on the index type. For type 1:

#### States Header

| Size (bits) | Field | Description |
|-------------|-------|-------------|
| 32 | num_states | Number of states with transitions |

#### For Each State

For each of the `num_states` states:

| Size (bits) | Field | Description |
|-------------|-------|-------------|
| 32 | state_id | ID of the current state |
| 32 | num_transitions | Number of transitions from this state |

#### For Each Transition

For each of the `num_transitions` transitions in a state:

| Size (bits) | Field | Description |
|-------------|-------|-------------|
| 32 | token_id | Token ID that triggers this transition |
| 32 | next_state_id | Destination state ID for this transition |

## Compression

The entire binary structure described above is compressed using gzip compression (flate2) with default compression level before being written to disk.

## Example Layout

```
┌─────────────────────────────────────────────────────────┐
│ Compressed File (gzip)                                  │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Uncompressed Binary Data                            │ │
│ │ ┌───────────────────────────────────────────────┐   │ │
│ │ │ Header (16 bytes)                             │   │ │
│ │ │ - vocab_size (4 bytes)                        │   │ │
│ │ │ - eos_token_id (4 bytes)                      │   │ │
│ │ │ - initial_state_id (4 bytes)                  │   │ │
│ │ │ - num_final_states (4 bytes)                  │   │ │
│ │ └───────────────────────────────────────────────┘   │ │
│ │ ┌───────────────────────────────────────────────┐   │ │
│ │ │ Final States (4 bytes × num_final_states)     │   │ │
│ │ └───────────────────────────────────────────────┘   │ │
│ │ ┌───────────────────────────────────────────────┐   │ │
│ │ │ Index Type (1 byte)                           │   │ │
│ │ └───────────────────────────────────────────────┘   │ │
│ │ ┌───────────────────────────────────────────────┐   │ │
│ │ │ Transitions Section                           │   │ │
│ │ │ - num_states (4 bytes)                        │   │ │
│ │ │ - For each state:                             │   │ │
│ │ │   - state_id (4 bytes)                        │   │ │
│ │ │   - num_transitions (4 bytes)                 │   │ │
│ │ │   - For each transition:                      │   │ │
│ │ │     - token_id (4 bytes)                      │   │ │
│ │ │     - next_state_id (4 bytes)                 │   │ │
│ │ └───────────────────────────────────────────────┘   │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Version History

- **Type 1**: Initial format supporting basic state transitions with token-to-state mappings.

## Future Extensions

The index type field allows for future extensions of the format. New index types can be added to support:
- Optimized storage formats for sparse or dense transition tables
- Compressed transition representations
- Alternative state machine encodings
