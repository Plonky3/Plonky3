# p3-fri

An implementation of the FRI low-degree test (LDT) and a FRI-based
polynomial commitment scheme.

Key items:

- `TwoAdicFriPcs` — the `p3_commit::Pcs` instantiation over two-adic multiplicative cosets
- `HidingFriPcs` — a zero-knowledge variant that pads traces with caller-supplied randomness
- `FriParameters` — blowup, query count, proof-of-work and arity configuration
- `prover` / `verifier` — the underlying FRI folding protocol with configurable per-round arities

Soundness depends on the chosen parameters; see `FriParameters` docs for
how blowup, query count and grinding bits combine into the security level.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
