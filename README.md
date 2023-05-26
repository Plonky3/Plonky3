# Plonky3

Plonky3 is a toolkit for implementing polynomial IOPs (PIOPs), such as PLONK and STARKs. It aims to support several polynomial commitment schemes, such as Brakedown.

This is the "core" repo, but the plan is to move each crate into its own repo once APIs stabilize.


## Status

Fields:
- [x] Mersenne31
  - [ ] common extension fields
  - [ ] AVX2
  - [ ] AVX-512
  - [ ] NEON
- [x] Goldilocks
  - [ ] common extension fields
  - [ ] AVX2
  - [ ] AVX-512
  - [ ] NEON

Vector-ish commitment schemes
- [x] generalized Merkle tree

Polynomial commitment schemes
- [ ] FRI-based PCS
- [ ] tensor PCS
- [ ] univariate-to-multivariate adapter
- [ ] multivariate-to-univariate adapter

PIOPs
- [ ] STARK
  - [ ] univariate
  - [ ] multivariate
- [ ] PLONK

Codes
- [x] Brakedown
- [ ] Reed-Solomon

Algorithms
- [ ] Barycentric interpolation
- [ ] basic FFT
- [ ] Mersenne FFT

Hashes
- [ ] Rescue
- [x] Poseidon
- [ ] Poseidon2
- [x] Keccak-256
- [ ] BLAKE3-modified


## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
