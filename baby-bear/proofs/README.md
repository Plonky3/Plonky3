# p3-baby-bear proofs

This directory houses the formal verification of this crate. The extraction is
scalar-only (`thumbv7em-none-eabi`): AVX2, AVX-512 and NEON are out of scope.
Production on x86-64 and aarch64 runs those backends, not the portable
`no_packing` path modelled here.

The active backend is [`legacy-lean/`](legacy-lean/) — Hax's `legacy-lean`
output, as a Lake package (extraction, patches, theorems). See
[`legacy-lean/README.md`](legacy-lean/README.md) for the layout and how to build.

Other extractors exist — [Aeneas](https://github.com/AeneasVerif/aeneas), and
Hax's current Lean backend — and could replace this later. They are not used
now because their extraction of this crate is not clean enough to work against.
