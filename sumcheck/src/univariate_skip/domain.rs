//! The evaluation domain of a univariate-skip round: an `F_2`-subspace inside a larger one.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use thiserror::Error;

/// Reasons a skip domain cannot be built.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum SkipDomainError {
    /// The extension is not strictly larger than the subspace it extends.
    ///
    /// A polynomial already known to vanish on the subspace carries no information there.
    ///
    /// Its transmitted points must therefore lie outside it.
    #[error("the extension dimension {log_extended} must exceed the subspace dimension {log_size}")]
    EmptyExtension {
        /// Dimension of the subspace the round polynomial vanishes on.
        log_size: usize,
        /// Dimension of the subspace the round polynomial is transmitted on.
        log_extended: usize,
    },
    /// The extension needs more dimensions than the field or the machine word can carry.
    ///
    /// The field supplies one independent basis vector per dimension.
    ///
    /// The point list is indexed by a machine word, so the tighter of the two bounds applies.
    #[error("a dimension-{log_extended} subspace is above the limit of {limit}")]
    ExtensionTooWide {
        /// Requested dimension of the extension.
        log_extended: usize,
        /// Widest extension this field and this platform admit.
        limit: usize,
    },
}

/// A skip round's domain: where the round polynomial vanishes, and where it is transmitted.
///
/// # Overview
///
/// A univariate-skip round collapses `k` sumcheck rounds into one.
///
/// The `2^k` skipped Boolean coordinates become the points of an `F_2`-linear subspace `S`.
///
/// The round polynomial is transmitted on the points of a larger subspace `T` lying outside `S`.
///
/// One point list covers both sets, because `S` is the index prefix of `T`:
///
/// ```text
///     point(j) = sum_r bit_r(j) * v_r
///
///     j in [0, 2^k)        ->  S, where the round polynomial vanishes
///     j in [2^k, 2^(k+e))  ->  T \ S, where its values are transmitted
/// ```
///
/// - The `v_r` are Cantor basis vectors, so `T` is the additive-NTT domain of that dimension.
/// - An index below `2^k` selects only vectors spanning `S`, which is why the prefix is `S`.
///
/// # Why the indexing rule matters
///
/// The index-to-point map is `F_2`-linear, so `point(a xor b) = point(a) + point(b)`.
///
/// That identity is what lets the compressed extension trade a chunk offset for an index offset.
///
/// Any other indexing would make it wrong.
///
/// # Algorithm
///
/// The vanishing polynomial of `S` comes from the recursion
///
/// ```text
///     W_0(x) = x
///     W_{j+1}(x) = W_j(x)^2 + W_j(x)
/// ```
///
/// - It is `F_2`-linear and monic of degree `2^j`.
/// - It vanishes exactly on the dimension-`j` subspace.
/// - Monic of the right degree with the right roots makes it the product of `x + s` over `S`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkipDomain<F> {
    /// Dimension of the subspace the round polynomial vanishes on.
    log_size: usize,
    /// Dimension of the subspace the round polynomial is transmitted on.
    log_extended: usize,
    /// Every point of the larger subspace in index order, the smaller one first.
    points: Vec<F>,
    /// The vanishing polynomial of the smaller subspace, evaluated on the transmitted points.
    ///
    /// Being `F_2`-linear and zero on the smaller subspace makes it constant on each coset.
    ///
    /// One value per point rather than one per coset keeps the hot loops flat.
    vanishing_on_extension: Vec<F>,
    /// The value the formal derivative of the vanishing polynomial takes on the smaller subspace.
    ///
    /// The derivative at a subspace point is that point's product of differences to the others.
    ///
    /// Reindexing turns every one of those products into the product of the nonzero points.
    derivative_on_subspace: F,
}

impl<F: TowerLevel> SkipDomain<F> {
    /// Build the domain that vanishes on dimension `log_size` and transmits on `log_extended`.
    ///
    /// # Errors
    ///
    /// - When the extension is not strictly larger than the subspace it extends.
    /// - When the extension needs more independent directions than the field has.
    pub fn new(log_size: usize, log_extended: usize) -> Result<Self, SkipDomainError> {
        // Transmitting only where the polynomial is already known to vanish says nothing.
        if log_extended <= log_size {
            return Err(SkipDomainError::EmptyExtension {
                log_size,
                log_extended,
            });
        }

        // The extension needs one independent basis vector per dimension.
        //
        // Its point list is indexed by a machine word, so that is the other bound.
        let limit = (1usize << F::LOG_BITS).min(usize::BITS as usize - 1);
        if log_extended > limit {
            return Err(SkipDomainError::ExtensionTooWide {
                log_extended,
                limit,
            });
        }

        // Walk the index in natural order, mapping each to the basis vectors its bits select.
        let points = (0..1usize << log_extended)
            .map(Self::point)
            .collect::<Vec<_>>();

        // The smaller subspace occupies the index prefix, so the transmitted points are the rest.
        let vanishing_on_extension = points[1 << log_size..]
            .iter()
            .map(|&x| Self::vanishing_poly(log_size, x))
            .collect::<Vec<_>>();

        // The derivative is the product of the nonzero points of the smaller subspace.
        let derivative_on_subspace = points[1..1 << log_size].iter().copied().product::<F>();

        Ok(Self {
            log_size,
            log_extended,
            points,
            vanishing_on_extension,
            derivative_on_subspace,
        })
    }

    /// Build the smallest domain able to transmit a round polynomial of the given degree.
    ///
    /// # Overview
    ///
    /// Vanishing on the subspace makes the round polynomial a multiple of the vanishing one.
    ///
    /// Only that quotient has to be transmitted:
    ///
    /// ```text
    ///     deg P  <= degree * (2^k - 1)    the round polynomial
    ///     deg Q  =  deg P - 2^k           the quotient by the vanishing polynomial
    /// ```
    ///
    /// The extension must hold at least `deg Q + 1` points.
    ///
    /// Its dimension is therefore the smallest `k+e` with `2^e` reaching the degree.
    ///
    /// # Errors
    ///
    /// - When the degree is zero, which describes no round polynomial at all.
    /// - When the resulting extension needs more directions than the field has.
    pub fn for_degree(log_size: usize, degree: usize) -> Result<Self, SkipDomainError> {
        // A degree-zero composition has nothing to transmit.
        let Some(extra) = Self::extra_dimensions(degree) else {
            return Err(SkipDomainError::EmptyExtension {
                log_size,
                log_extended: log_size,
            });
        };
        Self::new(log_size, log_size + extra)
    }

    /// Dimensions above the subspace a constraint of the given degree needs.
    ///
    /// # Algorithm
    ///
    /// A degree-`d` constraint over multilinears reads as a univariate here.
    /// Its degree there is `d * (2^k - 1)` for a dimension-`k` subspace.
    ///
    /// Reconstructing it needs `d` rounded up to a power of two:
    ///
    /// ```text
    ///     d = 1, 2  ->  one extra dimension
    ///     d = 3, 4  ->  two
    ///     d = 5..8  ->  three
    /// ```
    ///
    /// # Returns
    ///
    /// Nothing for degree zero, which is not a degree a round can carry.
    #[must_use]
    pub const fn extra_dimensions(degree: usize) -> Option<usize> {
        if degree == 0 {
            return None;
        }
        let extra = usize::BITS as usize - (degree - 1).leading_zeros() as usize;

        // A degree-one constraint still needs a coset to be read on.
        Some(if extra == 0 { 1 } else { extra })
    }

    /// Whether this domain is wide enough for a constraint of that degree.
    ///
    /// A domain that is too narrow does not make the round unsound.
    ///
    /// It makes it incomplete.
    /// The honest round polynomial no longer fits what is sent, so it fails.
    #[must_use]
    pub const fn admits_degree(&self, degree: usize) -> bool {
        match Self::extra_dimensions(degree) {
            Some(extra) => self.log_size + extra <= self.log_extended,
            None => false,
        }
    }

    /// The point an index maps to, as the sum of the basis vectors its set bits select.
    fn point(index: usize) -> F {
        // Accumulate one basis vector per set bit, lowest bit first.
        let mut point = F::ZERO;
        let mut remaining = index;
        let mut position = 0;
        while remaining != 0 {
            if remaining & 1 == 1 {
                point += F::cantor_basis(position);
            }
            remaining >>= 1;
            position += 1;
        }
        point
    }

    /// The normalised subspace polynomial of the dimension-`log_size` subspace, at `x`.
    fn vanishing_poly(log_size: usize, x: F) -> F {
        // Square-and-add the recursion once per dimension, starting from the identity map.
        let mut value = x;
        for _ in 0..log_size {
            value = value.square() + value;
        }
        value
    }

    /// Dimension of the subspace the round polynomial vanishes on.
    #[must_use]
    pub const fn log_size(&self) -> usize {
        self.log_size
    }

    /// Number of points the round polynomial vanishes on, two raised to the skipped-round count.
    #[must_use]
    pub const fn size(&self) -> usize {
        1 << self.log_size
    }

    /// Dimension of the subspace the round polynomial is transmitted on.
    #[must_use]
    pub const fn log_extended(&self) -> usize {
        self.log_extended
    }

    /// Number of transmitted evaluations, which is the extension size minus the subspace size.
    #[must_use]
    pub const fn num_transmitted(&self) -> usize {
        (1 << self.log_extended) - (1 << self.log_size)
    }

    /// The points the round polynomial vanishes on, in index order.
    #[must_use]
    pub fn subspace(&self) -> &[F] {
        &self.points[..self.size()]
    }

    /// The points the round polynomial is transmitted on, in index order.
    #[must_use]
    pub fn transmitted(&self) -> &[F] {
        &self.points[self.size()..]
    }

    /// Evaluate the subspace vanishing polynomial at an arbitrary point.
    ///
    /// The result is zero exactly on the subspace.
    #[must_use]
    pub fn vanishing(&self, x: F) -> F {
        Self::vanishing_poly(self.log_size, x)
    }

    /// The vanishing polynomial of the subspace, evaluated on the transmitted points.
    #[must_use]
    pub fn vanishing_on_transmitted(&self) -> &[F] {
        &self.vanishing_on_extension
    }

    /// The value the vanishing polynomial's derivative takes on every subspace point.
    #[must_use]
    pub const fn derivative_on_subspace(&self) -> F {
        self.derivative_on_subspace
    }

    /// The matrix that resamples a polynomial from its values on the subspace to the extension.
    ///
    /// One row of `2^k` entries per transmitted point, in row-major order.
    ///
    /// # Algorithm
    ///
    /// Lagrange interpolation on the subspace, evaluated at a transmitted point, is
    ///
    /// ```text
    ///     M[i][c] = Z(t_i) / ((t_i + s_c) * Z'(s_c))
    /// ```
    ///
    /// The derivative is constant on the subspace, and the numerator is constant on each coset.
    ///
    /// Every entry is therefore one scalar over a difference:
    ///
    /// ```text
    ///     M[i][c] = Z(t_i) / Z'(S) / (t_i + s_c)
    /// ```
    ///
    /// - That is a Cauchy matrix, scaled row by row.
    /// - Every difference is nonzero, because the transmitted points lie outside the subspace.
    #[must_use]
    pub fn resampling_matrix(&self) -> Vec<F> {
        self.resampling_prefix(self.size())
    }

    /// The first `columns` columns of the resampling matrix, still one row per transmitted point.
    ///
    /// A caller that reaches the remaining columns by another route pays only for these.
    ///
    /// # Panics
    ///
    /// Panics if more columns are asked for than the subspace has points.
    #[must_use]
    pub fn resampling_prefix(&self, columns: usize) -> Vec<F> {
        assert!(columns <= self.size(), "one column per subspace point");

        // The derivative is the same at every subspace point, so it leaves the inner loop.
        let inverse_derivative = self.derivative_on_subspace.inverse();

        // Fill row by row over the transmitted points, column by column over the subspace.
        let mut matrix = F::zero_vec(self.num_transmitted() * columns);
        let rows = matrix.chunks_exact_mut(columns);
        for ((row, &t), &vanishing) in rows
            .zip(self.transmitted())
            .zip(&self.vanishing_on_extension)
        {
            // One scale per row, since the numerator only depends on which coset the row is in.
            let scale = vanishing * inverse_derivative;
            for (entry, &s) in row.iter_mut().zip(self.subspace()) {
                // The two point sets are disjoint, so the difference is always invertible.
                *entry = scale * (t + s).inverse();
            }
        }
        matrix
    }

    /// Evaluate the polynomial that takes the given values on the subspace, at an arbitrary point.
    ///
    /// This is the textbook Lagrange formula, at one inversion per subspace point.
    ///
    /// It is the reference the fast extension paths are tested against.
    ///
    /// # Panics
    ///
    /// Panics if the number of values differs from the number of subspace points.
    #[must_use]
    pub fn interpolate(&self, values: &[F], at: F) -> F {
        assert_eq!(values.len(), self.size(), "one value per subspace point");

        // A query landing on a subspace point reads the value there rather than dividing by zero.
        if let Some(index) = self.subspace().iter().position(|&s| s == at) {
            return values[index];
        }

        // Off the subspace every difference is invertible, so the barycentric form applies.
        //
        //     p(at) = Z(at) / Z'(S) * sum_c values[c] / (at + s_c)
        let scale = self.vanishing(at) * self.derivative_on_subspace.inverse();
        scale
            * values
                .iter()
                .zip(self.subspace())
                .map(|(&value, &s)| value * (at + s).inverse())
                .sum::<F>()
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField16, Ghash128};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    // A dimension-6 subspace inside a dimension-7 one is 128 of a byte field's 256 elements.
    //
    // That is the shape a degree-2 zerocheck with six skipped rounds runs.
    const LOG_SIZE: usize = 6;

    #[test]
    fn the_subspace_is_the_index_prefix_of_the_extension() {
        // Fixture state: S has 64 points, the extension adds 64 more, all distinct.
        let domain = SkipDomain::<BinaryField8>::new(LOG_SIZE, LOG_SIZE + 1).unwrap();
        assert_eq!(domain.size(), 64);
        assert_eq!(domain.num_transmitted(), 64);

        // Invariant: the two point sets never meet, so every Cauchy difference is invertible.
        for &t in domain.transmitted() {
            assert!(!domain.subspace().contains(&t));
        }

        // Invariant: all 128 points are distinct, so no row of the resampling matrix collapses.
        let mut seen = domain.subspace().to_vec();
        seen.extend_from_slice(domain.transmitted());
        for (index, &point) in seen.iter().enumerate() {
            assert!(!seen[index + 1..].contains(&point), "index={index}");
        }
    }

    #[test]
    fn index_zero_is_the_origin_and_single_bits_are_basis_vectors() {
        // The index-to-point map must be the F_2-linear one, not merely some bijection.
        // Everything the compressed extension does rests on this.
        let domain = SkipDomain::<BinaryField8>::new(LOG_SIZE, LOG_SIZE + 1).unwrap();
        assert_eq!(domain.subspace()[0], BinaryField8::ZERO);
        for position in 0..LOG_SIZE {
            assert_eq!(
                domain.subspace()[1 << position],
                BinaryField8::cantor_basis(position)
            );
        }
    }

    #[test]
    fn indexing_is_additive_over_index_xor() {
        // Invariant: point(a xor b) = point(a) + point(b) across the whole extension.
        //
        // The chunk-offset trick depends on exactly this.
        //
        // Moving a chunk from position 0 to position b shifts the output index by 8 times b.
        let domain = SkipDomain::<BinaryField8>::new(3, 7).unwrap();
        let points = domain
            .subspace()
            .iter()
            .chain(domain.transmitted())
            .copied()
            .collect::<Vec<_>>();
        for a in 0..points.len() {
            for b in 0..points.len() {
                assert_eq!(points[a ^ b], points[a] + points[b], "a={a} b={b}");
            }
        }
    }

    #[test]
    fn vanishing_polynomial_matches_the_product_over_the_subspace() {
        // The recursion costs log(|S|) squarings where the definition costs |S| products.
        // Pin the fast form to the definition it stands in for.
        let domain = SkipDomain::<BinaryField8>::new(LOG_SIZE, LOG_SIZE + 1).unwrap();
        for x in 0..=u8::MAX {
            let x = BinaryField8::from_repr(x);
            let product = domain
                .subspace()
                .iter()
                .map(|&s| x + s)
                .product::<BinaryField8>();
            assert_eq!(domain.vanishing(x), product, "x={x:?}");
        }
    }

    #[test]
    fn vanishing_polynomial_is_zero_exactly_on_the_subspace() {
        // Each coset of the subspace carries one nonzero value.
        //
        // That is what lets the verifier divide the vanishing factor out of what it receives.
        let domain = SkipDomain::<BinaryField8>::new(LOG_SIZE, LOG_SIZE + 1).unwrap();
        for &s in domain.subspace() {
            assert_eq!(domain.vanishing(s), BinaryField8::ZERO);
        }
        for (&t, &cached) in domain
            .transmitted()
            .iter()
            .zip(domain.vanishing_on_transmitted())
        {
            assert_eq!(domain.vanishing(t), cached);
            assert_ne!(cached, BinaryField8::ZERO);
        }
    }

    #[test]
    fn the_vanishing_polynomial_is_constant_on_each_coset() {
        // Fixture state: dimension 3 inside dimension 6, so 7 cosets of 8 points each.
        //
        // transmitted index i  ->  coset i / 8 all 8 points of a coset share one vanishing value
        let domain = SkipDomain::<BinaryField8>::new(3, 6).unwrap();
        for coset in domain
            .vanishing_on_transmitted()
            .chunks_exact(domain.size())
        {
            assert!(coset.iter().all(|&value| value == coset[0]));
        }
    }

    #[test]
    fn canonical_normalisation_makes_the_derivative_one() {
        // The Cantor basis normalises the subspace polynomial to derivative one on the subspace.
        //
        // The code computes the constant rather than assuming it.
        //
        // A basis change that broke this would move every resampling coefficient, so pin it.
        for log_size in 1..7 {
            let domain = SkipDomain::<BinaryField8>::new(log_size, log_size + 1).unwrap();
            assert_eq!(domain.derivative_on_subspace(), BinaryField8::ONE);
        }
    }

    #[test]
    fn resampling_matrix_agrees_with_lagrange_interpolation() {
        // Fixture state: random values on S, resampled onto the extension two different ways.
        //
        //     matrix row i . values     ->  transmitted value at index i
        //     interpolate(values, t_i)  ->  the same value
        let mut rng = SmallRng::seed_from_u64(7);
        let domain = SkipDomain::<BinaryField8>::new(LOG_SIZE, LOG_SIZE + 1).unwrap();
        let values = (0..domain.size())
            .map(|_| rng.random::<BinaryField8>())
            .collect::<Vec<_>>();
        let matrix = domain.resampling_matrix();

        for (index, row) in matrix.chunks_exact(domain.size()).enumerate() {
            let by_matrix = row
                .iter()
                .zip(&values)
                .map(|(&m, &v)| m * v)
                .sum::<BinaryField8>();
            assert_eq!(
                by_matrix,
                domain.interpolate(&values, domain.transmitted()[index]),
                "index={index}"
            );
        }
    }

    #[test]
    fn interpolation_reproduces_the_values_it_was_given() {
        // Values on the subspace pin a polynomial of degree below the subspace size.
        //
        // Querying a subspace point must return the stored value and never divide by zero.
        let mut rng = SmallRng::seed_from_u64(11);
        let domain = SkipDomain::<BinaryField16>::new(5, 6).unwrap();
        let values = (0..domain.size())
            .map(|_| rng.random::<BinaryField16>())
            .collect::<Vec<_>>();
        for (index, &s) in domain.subspace().iter().enumerate() {
            assert_eq!(domain.interpolate(&values, s), values[index]);
        }
    }

    #[test]
    fn degree_sizing_leaves_room_for_the_quotient() {
        // Invariant: the extension must hold every coefficient of the quotient.
        //
        // Otherwise the transmitted values would not determine the round polynomial.
        //
        //     deg P = degree * (2^k - 1)
        //     deg Q = deg P - 2^k
        //     need    num_transmitted >= deg Q + 1
        for degree in 1..=8usize {
            let domain = SkipDomain::<BinaryField16>::for_degree(4, degree).unwrap();
            let deg_p = degree * (domain.size() - 1);
            let deg_q = deg_p.saturating_sub(domain.size());
            let needed = deg_q + 1;
            assert!(
                domain.num_transmitted() >= needed,
                "degree={degree} transmitted={} needs={needed}",
                domain.num_transmitted(),
            );
        }

        // A degree-2 composition is the product form a zerocheck of R1CS shape uses.
        // It must cost exactly one extra dimension, which is what keeps the table in cache.
        let domain = SkipDomain::<BinaryField8>::for_degree(6, 2).unwrap();
        assert_eq!(domain.log_extended(), 7);
    }

    #[test]
    fn rejects_an_extension_that_adds_nothing() {
        // Mutation: ask for an extension no larger than the subspace, leaving nothing to send.
        assert_eq!(
            SkipDomain::<BinaryField8>::new(4, 4),
            Err(SkipDomainError::EmptyExtension {
                log_size: 4,
                log_extended: 4,
            })
        );
        assert_eq!(
            SkipDomain::<BinaryField8>::for_degree(4, 0),
            Err(SkipDomainError::EmptyExtension {
                log_size: 4,
                log_extended: 4,
            })
        );
    }

    #[test]
    fn rejects_an_extension_wider_than_the_field() {
        // A byte field has 8 independent directions, so dimension 9 has no basis to span it.
        assert_eq!(
            SkipDomain::<BinaryField8>::new(8, 9),
            Err(SkipDomainError::ExtensionTooWide {
                log_extended: 9,
                limit: 8,
            })
        );
        // Dimension 8 is the whole byte field, so it is the widest extension that fits.
        assert!(SkipDomain::<BinaryField8>::new(7, 8).is_ok());
    }

    #[test]
    fn rejects_an_extension_wider_than_a_machine_word() {
        // A 128-bit field has enough directions for any dimension up to 128.
        //
        // The point list is indexed by a machine word, though, so the word is the tighter bound.
        //
        //     2^100 points cannot be indexed, let alone allocated
        //
        // Without this the index arithmetic would wrap before the field bound ever applied.
        let limit = usize::BITS as usize - 1;
        assert_eq!(
            SkipDomain::<Ghash128>::new(99, 100),
            Err(SkipDomainError::ExtensionTooWide {
                log_extended: 100,
                limit,
            })
        );
        assert!(SkipDomain::<Ghash128>::new(3, 4).is_ok());
    }

    proptest! {
        #[test]
        fn interpolation_is_linear_in_the_values(seed: u64) {
            // Invariant: interpolation is an F_2-linear map on the value vector.
            // The compressed extension exploits exactly this to split a row into chunks.
            let mut rng = SmallRng::seed_from_u64(seed);
            let domain = SkipDomain::<Ghash128>::new(4, 5).unwrap();

            // Two independent value vectors and the point they are both read at.
            let left = (0..domain.size()).map(|_| rng.random::<Ghash128>()).collect::<Vec<_>>();
            let right = (0..domain.size()).map(|_| rng.random::<Ghash128>()).collect::<Vec<_>>();
            let at = rng.random::<Ghash128>();

            // Summing the inputs must equal summing the outputs.
            let summed = left.iter().zip(&right).map(|(&a, &b)| a + b).collect::<Vec<_>>();
            prop_assert_eq!(
                domain.interpolate(&summed, at),
                domain.interpolate(&left, at) + domain.interpolate(&right, at)
            );
        }
    }
}
