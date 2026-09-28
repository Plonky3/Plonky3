# p3-merkle-tree

A Merkle-tree implementation of the `Mmcs` commitment interface, used to
commit to batches of trace and LDE matrices.

Key items:

- `MerkleTree` — an N-ary Merkle tree (with binary bridge levels) over rows of multiple matrices of differing heights; `N = 5` with a `p3_symmetric::T5` node gives the T5 tree of [2021/373](https://eprint.iacr.org/2021/373)
- `MerkleTreeMmcs` — the `p3_commit::Mmcs` instantiation, with `verify_batch` for opening verification
- `MerkleTreeHidingMmcs` — a hiding variant that salts leaves with caller-supplied randomness
- `PrunedMerklePaths` / `PrunedBatchOpening` — de-duplicated multi-opening proofs

The tree is generic over the hash and compression functions from
`p3-symmetric`, and digests can be truncated via `MerkleCap`.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
