use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// Comprehensive tests for the BasedVectorSpace trait
pub fn test_based_vector_space_all<V, B>()
where
    V: PrimeCharacteristicRing + Eq + BasedVectorSpace<B>,
    B: PrimeCharacteristicRing + Copy,
    StandardUniform: Distribution<V>,
{
    let mut rng = SmallRng::seed_from_u64(7);

    // Roundtrip via coefficients
    let v: V = rng.random();
    let v_binding = v.clone();
    let coeffs = v_binding.as_basis_coefficients_slice();
    let rebuilt = V::from_basis_coefficients_fn(|i| coeffs[i]);
    assert_eq!(v, rebuilt);

    // Batch roundtrip and chunking
    let vec_v: alloc::vec::Vec<V> = (0..5).map(|_| rng.random()).collect();
    let mut flat = alloc::vec::Vec::new();
    let dim = <V as BasedVectorSpace<B>>::DIMENSION;
    for e in &vec_v {
        flat.extend_from_slice(e.as_basis_coefficients_slice());
    }
    let rebuilt_vec: alloc::vec::Vec<V> = flat
        .chunks_exact(dim)
        .map(|chunk| V::from_basis_coefficients_fn(|i| chunk[i]))
        .collect();
    assert_eq!(vec_v, rebuilt_vec);

    // from_basis_coefficients_iter success
    let ok = V::from_basis_coefficients_iter(
        (0..dim)
            .map(|i| coeffs[i])
            .collect::<alloc::vec::Vec<B>>()
            .into_iter(),
    );
    assert_eq!(ok, Some(v));

    // from_basis_coefficients_iter failure (wrong length)
    let bad = V::from_basis_coefficients_iter((0..(dim - 1)).map(|i| coeffs[i]));
    assert!(bad.is_none());
}
