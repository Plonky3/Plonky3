# p3-lookup

Lookup arguments for STARKs, implementing the
[LogUp](https://eprint.iacr.org/2022/1530) protocol for intra-AIR (local)
and cross-AIR (global) lookup arguments.

Key items:

- `LogUpGadget` / `LookupProtocol` — permutation-column generation and constraint enforcement
- `LookupBus`, `PermutationCheckBus` — domain-separated buses for cross-AIR interactions
- `InteractionBuilder`, `InteractionSymbolicBuilder` — AIR-builder integration for declaring sends and receives
- `debug_util` — out-of-circuit multiset balance checks for debugging

## Migration from STARK configuration parameters

Lookup trace generation now takes the base field `F` and extension field `EF`
directly. Replace `LookupTraceBuilder<'a, SC>` and `RowEvalContext<'a, SC>` with
`LookupTraceBuilder<'a, F, EF>` and `RowEvalContext<'a, F, EF>`. In a STARK caller,
replace `generate_permutation::<SC>` with
`generate_permutation::<Val<SC>, SC::Challenge>`.

The STARK-specific constraint folders moved from `p3_lookup::folder` to
`p3_batch_stark::folder`. Shared lookup code no longer depends on `p3-uni-stark`.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
