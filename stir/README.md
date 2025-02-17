
# An implementation of the STIR low-degree test

This crate provides an implementation of the protocol proposed in [STIR: Reed–Solomon Proximity Testing with Fewer Queries](https://eprint.iacr.org/2024/390). It partially follows the [implementation](https://github.com/WizardOfMenlo/stir) by the article's co-author Giacomo Fenzi.

Broadly speaking, STIR allows a prover to convince a verifier that a certain codeword is close to a concrete Reed-Solomon code - that is, it is the evaluation over a certain domain of a polynomial satisfying a known degree bound. Much like FRI, it iteratively folds the original codeword throughout various rounds until it reaches the encoding of a polynomial of a pre-established stopping degree, which is then sent in plain and manually tested. However, the rate of the codes tested in each round progressively decreases instead of remaining constant, which leads to fewer queries and therefore shorter proofs in exchange for a slightly costlier prover.

## How to use

The workflow of the STIR protocol can roughly be outlined as follows:

 1. Create a configuration of type `StirParameters`, which among other fields includes the log2 of the degree bound (plus one) that shoud be proved.  configuration, which among other 
 
 1. Encode the polynomial of interest and commit to the codeword using the `commit` method, which produces a `StirWitness` for the prover and an MMCS commitment. If desired,  the commitment can be shared with the verifier at this stage, although this is often unnecessary in non-interactive contexts as the commitment is simply observed by the transcript.

 2. Prove the low-degreeness of the polynomial using `prove`, which outputs a `StirProof`. Share the proof with the verifier and, if not done earlier, the commitment as well.

 3. As the verifier, call `verify` on the commitment and proof, which will return `Ok(())` if verification was successful and `Err(VerificationError)` otherwise.

STIR can also be composed with other protocols, in which case the commitment from step 1 might have been produced elsewhere.

## Tests

Thorough tests have been created of both the low-level, core functionality as well as the end-to-end prover and verifier APIs. Due to the considerable size of the polynomials used, it is recommended to run these in `release` mode:

```cargo test --release```

## Benchmarks

A benchmark is available which checks the performance of `commit`, `prove` and `verify` on two different fields: a quintic extension of `BabyBear` and a quadratic extension of `GoldiLocks`. Run with:

```cargo bench --bench stir --all-features```

## Features

## Implementation notes

* Thorough FS with domain separation and vec sizes
* Absorption of public parameters done
* Custom verifier error class

* Thorough tests, including every error

* Benchmark

* Verifier optimisations 2.5-10%

* Tests should be run with —release

* Benches

* SecurityAssumptions

* Doc very careful with indices, rounds...

* Proof serialisation (can be done to files too using std or their alternatives)

* Verifier optimisations
