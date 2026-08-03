# p3-lookup

Lookup arguments for STARKs, implementing the
[LogUp](https://eprint.iacr.org/2022/1530) protocol for intra-AIR (local)
and cross-AIR (global) lookup arguments.

Key items:

- `LogUpGadget` / `LookupProtocol` — permutation-column generation and constraint enforcement
- `LookupBus`, `PermutationCheckBus` — domain-separated buses for cross-AIR interactions
- `InteractionBuilder`, `InteractionSymbolicBuilder` — AIR-builder integration for declaring sends and receives
- `debug_util` — out-of-circuit multiset balance checks for debugging

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
