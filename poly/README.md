# p3-poly

A high-performance Rust crate providing dense univariate polynomial operations over finite fields. The crate is inspired by the [`ark-poly`](https://github.com/arkworks-rs/algebra/tree/master/poly) crate in the arkworks ecosystem.

## Features

- **Unique Representation**: A `Polynomial` is a wrapper around a `Vec<F>`. The API ensures that the polynomial is always stored in canonical form, i.e. that the leading coefficient is non-zero, with the zero polynomial being represented as an empty `Vec<F>`.

- **Basic Operations**:
  - Standard arithmetic operator overloading for polynomials and constants 
  - Two multiplication algorithms on `TwoAdicField`s
    - Naive multiplication for small degrees
    - FFT-based multiplication for larger polynomials
  - Degree computation that returns `Option<usize>` for better null case handling

- **Advanced Operations**:
  - Polynomial evaluation at points using Horner's method
  - Generation of vanishing polynomials
  - Lagrange interpolation
  - Optimized algorithms for common operations:
    - Division by linear terms (`x - a`)
    - Multiplication by power polynomials (`1 + rx + r^2x^2 + ... + r^nx^n`)

## Testing

The crate contains a comprehensive test suite that covers all public methods, with a short description of the test case in the doc comments. To run the tests, use the command `cargo test`.

## Adding to your project

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
p3-poly = "0.1"
```

The crate is `no_std` by default. To access test utility functions like `rand_poly`, enable the `test-utils` feature:

```toml
[dependencies]
p3-poly = { version = "0.1", features = ["test-utils"] }
```

## Performance Considerations

- All polynomial operations are implemented for references only. This design choice follows ark-poly's approach, as implementing operations for both owned values and references can lead to inconsistencies and unnecessary cloning. 

- The overloaded `*` operator is only implemented for polynomials over `TwoAdicField`s. Depending on the degrees of the input polynomials, the operation may be performed using the naive or FFT-based multiplication algorithm.

- Given a set of n points-evaluation pairs, the method `Polynomial::lagrange_interpolate` performs Lagrange interpolation in O(n^2) time. For more efficient evaluation/interpolation over cosets that runs in O(nlog n) time, use the `TwoAdicCoset` methods in the `p3-coset` crate.

## Attribution

Some of the polynomial algorithms in this crate are inspired by or adapted from the arkworks ecosystem. Specific attributions are provided in the relevant code sections.