![Plonky3-powered-by-polygon](https://github.com/Plonky3/Plonky3/assets/86010/7ec356ad-b0f3-4c4c-aa1d-3a151c1065e7)

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/Plonky3/Plonky3/blob/main/LICENSE-MIT)
[![License](https://img.shields.io/github/license/Plonky3/Plonky3)](https://github.com/Plonky3/Plonky3/blob/main/LICENSE-APACHE)

Plonky3 is a toolkit which provides a set of primitives, such as polynomial commitment schemes, for implementing polynomial IOPs (PIOPs). It is mainly used to power STARK-based zkVMs, though in principle it may be used for PLONK-based circuits or other PIOPs.

For questions or discussions, please use the Telegram group, [t.me/plonky3](https://t.me/plonky3).


## Status

Fields:
- [x] Mersenne31
  - [x] "complex" extension field
  - [x] ~128 bit extension field
  - [x] AVX2
  - [x] AVX-512
  - [x] NEON
- [x] General 31 bit fields (BabyBear and KoalaBear)
  - [x] ~128 bit extension field
  - [x] AVX2
  - [x] AVX-512
  - [x] NEON
- [x] Goldilocks
  - [x] ~128 bit extension field

Generalized vector commitment schemes
- [x] generalized Merkle tree

Polynomial commitment schemes
- [x] FRI-based PCS
- [ ] tensor PCS
- [ ] univariate-to-multivariate adapter
- [ ] multivariate-to-univariate adapter

PIOPs
- [x] univariate STARK
- [ ] multivariate STARK
- [ ] PLONK

Codes
- [x] Brakedown
- [x] Reed-Solomon

Interpolation
- [x] Barycentric interpolation
- [x] radix-2 DIT FFT
- [x] radix-2 Bowers FFT
- [ ] four-step FFT
- [x] Mersenne circle group FFT

Hashes
- [x] Rescue
- [x] Poseidon
- [x] Poseidon2
- [x] BLAKE3
  - [ ] modifications to tune BLAKE3 for hashing small leaves
- [x] Keccak-256
- [x] Monolith


## Benchmarks

Many variations are possible, with different fields, hashes, and so forth, which can be controlled through the command line.

For example, to prove 2^20 Poseidon2 permutations of width 16, using the `KoalaBear` field, `Radix2DitParallel` DFT, and `KeccakF` as the Merkle tree hash:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo run --example prove_prime_field_31 --release --features parallel -- --field koala-bear --objective poseidon-2-permutations --log-trace-length 17 --discrete-fourier-transform radix-2-dit-parallel --merkle-hash keccak-f
```

Currently the options for the command line arguments are:
- `--field` (`-f`): `mersenne-31` or `koala-bear` or `baby-bear`.
- `--objective` (`-o`): `blake-3-permutations, poseidon-2-permutations, keccak-f-permutations`.
- `--log-trace-length` (`-l`): Accepts any integer between `0` and `255`. The number of permutations proven is `trace_length, 8*trace_length` and `trace_length/24` for `blake3, poseidon2` and `keccakf` respectively. 
- `--discrete-fourier-transform` (`-d`): `radix-2-dit-parallel, recursive-dft`. This option should be omitted if the field choice is `mersenne-31` as the circle stark currently only supports a single discrete fourier transform.
- `--merkle-hash` (`-m`): `poseidon-2, keccak-f`.

Extra speedups may be possible with some configuration changes:
- `JEMALLOC_SYS_WITH_MALLOC_CONF=retain:true,dirty_decay_ms:-1,muzzy_decay_ms:-1` will cause jemalloc to hang on to virtual memory. This may not affect the very first proof much, but can help significantly with subsequent proofs as fewer pages (if any) will need to be newly assigned by the OS. These settings might not be suitable for all production environments, e.g. if the process' virtual memory is limited by `ulimit` or `max_map_count`.
- Adding `lto = "fat"` in the top-level `Cargo.toml` may improve performance slightly, at the cost of longer compilation times.

## CPU features

Plonky3 contains optimizations that rely on newer CPU instructions unavailable in older processors. These instruction sets include x86's [BMI1 and 2](https://en.wikipedia.org/wiki/X86_Bit_manipulation_instruction_set), [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions#Advanced_Vector_Extensions_2), and [AVX-512](https://en.wikipedia.org/wiki/AVX-512). Rustc does not emit those instructions by default; they must be explicitly enabled through the `target-feature` compiler option (or implicitly by setting `target-cpu`). To enable all features that are supported on your machine, you can set `target-cpu` to `native`. For example, to run the tests:
```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo test
```

Support for some instructions, such as AVX-512, is still experimental. They are only available in the nightly build of Rustc and are enabled by the [`nightly-features` feature flag](#nightly-only-optimizations). To use them, you must enable the flag in Rustc (e.g. by setting `target-feature`) and you must also enable the `nightly-features` feature.


## Nightly-only optimizations

Some optimizations (in particular, AVX-512-optimized math) rely on features that are currently available only in the nightly build of Rustc. To use them, you need to enable the `nightly-features` feature. For example, to run the tests:
```bash
cargo test --features nightly-features
```


## Known issues

The verifier might panic upon receiving certain invalid proofs.


## License

Licensed under either of

* Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


## Guidance for external contributors

Do you feel keen and able to help with Plonky3? That's great! We
encourage external contributions!

We want to make it easy for you to contribute, but at the same time, we
must manage the burden of reviewing external contributions. We are a
small team, and the time we spend reviewing external contributions is
time we are not developing ourselves.

We also want to help you avoid inadvertently duplicating work that
is already underway, or building something that we will not
want to incorporate.

First and foremost, please keep in mind that this is a highly
technical piece of software and contributing is only suitable for
experienced mathematicians, cryptographers, and software engineers.

The Polygon Zero Team reserves the right to accept or reject any
external contribution for any reason, including a simple lack of time
to maintain it (now or in the future); we may even decline to review
something that is not considered a sufficiently high priority for us.

To avoid disappointment, please communicate your intention to
contribute openly, while respecting the limited time and availability
we have to review and provide guidance for external contributions. It
is a good idea to drop a note in our public Discord #development
channel about your intention to work on something, whether an issue, a
new feature, or a performance improvement. This is probably all that's
really required to avoid duplication of work with other contributors.

What follows are some more specific requests for how to write PRs in a
way that will make them easy for us to review. Deviating from these
guidelines may result in your PR being rejected, ignored, or forgotten.


### General guidance for your PR

Obviously, PRs will not be considered unless they pass our Github
CI. The GitHub CI is not executed for PRs from forks, but you can
simulate the GitHub CI by running the commands in
`.github/workflows/ci.yml`.

Under no circumstances should a single PR mix different purposes: Your
PR is either a bug fix, a new feature, or a performance improvement,
never a combination. Nor should you include, for example, two
unrelated performance improvements in one PR. Please just submit
separate PRs. The goal is to make reviewing your PR as simple as
possible, and you should be thinking about how to compose the PR to
minimize the burden on the reviewer.

Plonky3 uses stable Rust, so any PR that depends on unstable features
is likely to be rejected. It's possible that we may relax this policy
in the future, but we aim to minimize the use of unstable features;
please discuss with us before enabling any.

Please do not submit any PRs consisting of merely minor fixes to documentation.
They will not be accepted. If you find something that bothers you particularly,
make an issue and we will fix it when we find the time.

Here are a few specific guidelines for the three main categories of
PRs that we expect:


#### The PR fixes a bug

In the PR description, please clearly but briefly describe

1. the bug (could be a reference to a GH issue; if it is from a
   discussion (on Discord/email/etc. for example), please copy in the
   relevant parts of the discussion);
2. what turned out to cause the bug; and
3. how the PR fixes the bug.

Wherever possible, PRs that fix bugs should include additional tests
that (i) trigger the original bug and (ii) pass after applying the PR.


#### The PR implements a new feature

If you plan to contribute the implementation of a new feature, please
double-check with the Polygon Zero team that it is a sufficient
priority for us that it will be reviewed and integrated.

In the PR description, please clearly but briefly describe

1. what the feature does
2. the approach taken to implement it

All PRs for new features must include a suitable test suite.


#### The PR improves performance

Performance improvements are particularly welcome! Please note that it
can be quite difficult to establish true improvements for the
workloads we care about. To help filter out false positives, the PR
description for a performance improvement must clearly identify

1. the target bottleneck (only one per PR to avoid confusing things!)
2. how performance is measured
3. characteristics of the machine used (CPU, OS, #threads if appropriate)
4. performance before and after the PR


### Licensing

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
