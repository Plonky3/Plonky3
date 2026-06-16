# p3-merkle-tree

A Merkle-tree implementation of the `Mmcs` commitment interface, used to
commit to batches of trace and LDE matrices.

Key items:

- `MerkleTree` — a binary Merkle tree over rows of multiple matrices of differing heights
- `MerkleTreeMmcs` — the `p3_commit::Mmcs` instantiation, with `verify_batch` for opening verification
- `MerkleTreeHidingMmcs` — a hiding variant that salts leaves with caller-supplied randomness
- `PrunedMerklePaths` / `PrunedBatchOpening` — de-duplicated multi-opening proofs

The tree is generic over the hash and compression functions from
`p3-symmetric`, and digests can be truncated via `MerkleCap`.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
