# p3-commit

A framework for cryptographic commitment schemes, including non-hiding
variants. This crate defines the traits that connect proof systems to
their commitment backends.

Key items:

- `Pcs` / `MultilinearPcs` — polynomial commitment scheme interfaces used by the STARK provers and verifiers
- `Mmcs` — "Mixed Matrix Commitment Scheme", a vector-commitment abstraction over batches of matrices of differing heights
- `PolynomialSpace` and `TwoAdicMultiplicativeCoset` — evaluation-domain abstractions
- `periodic` — periodic-column evaluation helpers
- `testing` (the opt-in `test-utils` feature) — mock instantiations and shared PCS contract checks for downstream tests

Implementations live in `p3-merkle-tree` (Mmcs), `p3-fri`, `p3-circle` and
`p3-whir` (Pcs).

`testing::assert_pcs_opening_contract` exercises real transparent backends with
caller-supplied matrices and independently computed expected values. It checks
batched opening order, honest verification, transcript agreement, and rejection
of modified values and missing matrix/column claims. FRI, Circle, and STIR use
this helper alongside their backend-specific tests.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.

## Univariate PCS API migration

`Pcs` covers domains, commitments, prover data, and opening/verification. Implementers
that serve univariate STARKs also implement `UnivariateStarkPcs`: move
`EvaluationsOnDomain`, `ZK`, LDE/evaluation/quotient/periodic helpers and preprocessing
adaptation to that implementation. Generic commitment clients can continue to bound
only `Pcs`; STARK configurations require `UnivariateStarkPcs`. Import both traits when
calling both sets of methods, and change qualified references to moved members to
`UnivariateStarkPcs`. `MultilinearPcs` is unchanged.

Opening batches now use named requests and claims:

```rust,ignore
let requests = vec![OpeningRequest {
    prover_data: &data,
    points: vec![vec![zeta]], // one point vector per committed matrix
}];
let claims = vec![CommitmentOpening {
    commitment,
    matrices: vec![MatrixOpening {
        domain,
        points: vec![PointOpening { point: zeta, values }],
    }],
}];
```

Requests borrow prover data and own their point vectors. Claims own commitments,
matrices and column values, as before. All ordering remains significant: requests,
matrices, points and columns are used in caller order. The existing
`CommitmentWithOpeningPoints` name aliases `CommitmentOpening`. Explicit `From`
conversions accept the former tuples, including nested matrix/point claims, so a
caller can migrate a batch with `old_batch.into_iter().map(Into::into).collect()`.
The FRI `ProverDataWithOpeningPoints` alias likewise names `OpeningRequest`.

PCS traits no longer define trace, quotient or preprocessing commitment indices.
Univariate STARK owns these positions in `p3_uni_stark::StarkOpeningLayout`.
`open_with_preprocessing` now takes `Option<usize>` identifying the preprocessing
**commitment request**, replacing the old boolean. Use `None` for no preprocessing;
STARK callers pass `Some(layout.preprocessed)` when it is present. Other clients may
place preprocessing at any commitment index.

This is a Rust API change only; commitment/proof serialization, transcript absorption
order and evaluation-view ownership remain unchanged. No version bump is included.
