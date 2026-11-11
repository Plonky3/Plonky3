# p3-merkle-tree

A Merkle-tree implementation of the `Mmcs` commitment interface, used to
commit to batches of trace and LDE matrices.

Key items:

- `MerkleTree` — an N-ary Merkle tree (with binary bridge levels) over rows of multiple matrices of differing heights
- `MerkleTreeMmcs` — the `p3_commit::Mmcs` instantiation, with `verify_batch` for opening verification
- `MerkleTreeHidingMmcs` — a hiding variant that salts leaves with caller-supplied randomness
- `PrunedMerklePaths` / `PrunedBatchOpening` — de-duplicated multi-opening proofs

The tree is generic over the hash and compression functions from
`p3-symmetric`, and digests can be truncated via `MerkleCap`.

Every node is compressed at its level, so a `p3_symmetric::MixedCompression` can hash
the levels near the leaves with a fast byte hash and the levels near the root with an
arithmetic one. `p3_symmetric::SerializingCompression` gives the byte hash field-element
digests, so both halves of the tree share one digest type. The `mixed_hash` benchmark
compares such trees against single-hash ones.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
