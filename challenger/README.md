# p3-challenger

A Fiat–Shamir transcript and challenger framework, used to derive random
challenges from an IOP's transcript.

Key items:

- `CanObserve`, `CanSample`, `CanSampleBits`, `FieldChallenger` — the observe/sample trait family
- `DuplexChallenger` — sponge-based challenger over a cryptographic permutation
- `HashChallenger` — challenger built from a generic hash function
- `MultiField32Challenger`, `SerializingChallenger32/64` — challengers bridging fields of different sizes
- `GrindingChallenger` — proof-of-work witness generation and verification

Prover and verifier must observe and sample in the exact same order; every
proof object is bound to the transcript before any challenge it influences
is sampled.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
