# p3-binary-dft

An additive NTT over `F_2`-linear subspaces of the Plonky3 binary tower fields, the
characteristic-2 analogue of a multiplicative-coset DFT. The domain `S_ℓ` is spanned by the
first `ℓ` vectors of the Cantor basis added to `p3-binary-field`, and evaluation over it is
computed by the Lin–Chung–Han (LCH) transform.

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
