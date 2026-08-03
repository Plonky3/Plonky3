# p3-uni-stark

A minimal univariate STARK framework: proving and verification of a single
AIR over a two-adic field, generic over the polynomial commitment scheme.

Key items:

- `prove` / `verify` (and `*_with_preprocessed` variants) — the prover and verifier entry points
- `StarkConfig` / `StarkGenericConfig` — ties together field, PCS and challenger
- `SymbolicAirBuilder` — symbolic constraint evaluation for degree inference
- `Proof`, `VerificationError` — proof object and typed verifier errors

The verifier is designed to reject malformed proofs with a typed error, but
panics on adversarial inputs are not yet ruled out; see the repository
README's known-issues section.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
