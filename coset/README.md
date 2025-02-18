# p3-coset

A Rust crate for manipulating two-adic cosets over finite fields. The crate provides a `TwoAdicCoset` struct that represents a coset of size 2^n (where n is some non-negative integer) in a generic `TwoAdicField` F.

## Features

- **Construction**: Create cosets of any power-of-two size
- **Manipulation**: 
  - Shrink cosets or their subgroups by power-of-two factors
  - Apply arbitrary field element shifts
- **Polynomial Operations**: Fast evaluation and interpolation using `Radix2Dit`
- **Element Access**:
  - Index-based lookup with optional memoization
  - Iterator interface for sequential access
- **Membership and Equality**: 
  - Coset membership verification
  - Structural equality testing to detect if cosets are equal even when represented with different shift/generator combinations

## Testing

The crate contains a comprehensive test suite that covers all public methods, with a short description of the test case in the doc comments. To run the tests, use the command `cargo test`.

## Adding to your project

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
p3-coset = "0.1"
```

The crate is `no_std` and does not have any feature flags that affect the functionality.

## Benchmarking

The crate includes a single benchmark file `benches/coset.rs` that compares different methods for evaluating polynomials on two-adic cosets, specifically when the coset shift equals the subgroup generator. Three approaches are compared:

1. `TwoAdicSubgroupDft::coset_dft`: Transforms the polynomial by multiplying coefficients with powers of the shift before DFT
2. `TwoAdicSubgroupDft::dft` followed by `rotate_left`: Performs standard DFT, then shifts the result vector one position to the left in a circular manner
3. `TwoAdicCoset::evaluate_polynomial`: A wrapper method that internally uses the more efficient approach

The benchmarks are run across multiple field types:
- BabyBear (base field)
- BabyBearExt (degree-5 extension of BabyBear)
- Goldilocks (base field)
- GoldilocksExt (degree-2 extension of Goldilocks)

The benchmark tests various sizes ranging from 2^16 to 2^22 elements (in steps of 2 powers).

Results show that the `dft + rotate` method is typically 10-20% faster than the `coset_dft` approach, and `TwoAdicCoset::evaluate_polynomial` performs comparably to the faster method.

To run the benchmarks, use the command `cargo bench --bench coset`.

## Performance Considerations

The crate provides two methods for accessing coset elements by index:

```rust
coset.element(i)         // mutable reference required
coset.element_immutable(i)  // immutable reference
```
The `element()` method requires mutable access to the coset because it memoizes intermediate computations, specifically the iterated squares of the generator (g^(2^0), g^(2^1), etc.). This makes it more efficient when performing frequent element queries on the same coset. In contrast, `element_immutable()` works with just an immutable reference and is better suited for single element lookups or situations where mutable access isn't available.
