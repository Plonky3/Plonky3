use alloc::vec;
use alloc::vec::Vec;

use itertools::{Itertools, iterate};
use p3_commit::{LagrangeSelectors, PolynomialSpace};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, batch_multiplicative_inverse};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize};
use tracing::instrument;

use crate::cfft::CircleEvaluations;
use crate::point::Point;

/// A twin-coset of the circle group on F. It has a power-of-two size and an arbitrary shift.
///
/// X is generator, O is the first coset, goes counterclockwise
/// ```text
///   O X .
///  .     .
/// .       O <- start = shift
/// .   .   - (1,0)
/// O       .
///  .     .
///   . . O
/// ```
///
/// For ordering reasons, the other half will start at gen / shift:
/// ```text
///   . X O  <- start = gen/shift
///  .     .
/// O       .
/// .   .   - (1,0)
/// .       O
///  .     .
///   O . .
/// ```
///
/// The full domain is the interleaving of these two cosets
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CircleDomain<F> {
    // log_n corresponds to the log size of the WHOLE domain
    pub(crate) log_n: usize,
    pub(crate) shift: Point<F>,
}

impl<F: ComplexExtendable> CircleDomain<F> {
    pub const fn new(log_n: usize, shift: Point<F>) -> Self {
        Self { log_n, shift }
    }
    pub fn standard(log_n: usize) -> Self {
        Self {
            log_n,
            shift: Point::generator(log_n + 1),
        }
    }
    fn is_standard(&self) -> bool {
        self.shift == Point::generator(self.log_n + 1)
    }
    pub(crate) fn subgroup_generator(&self) -> Point<F> {
        Point::generator(self.log_n - 1)
    }
    pub(crate) fn coset0(&self) -> impl Iterator<Item = Point<F>> {
        let g = self.subgroup_generator();
        iterate(self.shift, move |&p| p + g).take(1 << (self.log_n - 1))
    }
    fn coset1(&self) -> impl Iterator<Item = Point<F>> {
        let g = self.subgroup_generator();
        iterate(g - self.shift, move |&p| p + g).take(1 << (self.log_n - 1))
    }
    pub(crate) fn points(&self) -> impl Iterator<Item = Point<F>> {
        self.coset0().interleave(self.coset1())
    }
    /// Same points as [`Self::points`], materialized eagerly. Each half-coset's sequential
    /// point-addition chain is split into chunks (reseeded via a scalar multiplication) that
    /// run in parallel, instead of walking the whole chain on a single thread.
    pub(crate) fn points_vec(&self) -> Vec<Point<F>> {
        let half = 1usize << (self.log_n - 1);
        let g = self.subgroup_generator();
        let c0 = parallel_point_chain(self.shift, g, half);
        let c1 = parallel_point_chain(g - self.shift, g, half);
        c0.into_iter().interleave(c1).collect()
    }
    pub(crate) fn nth_point(&self, idx: usize) -> Point<F> {
        let (idx, lsb) = (idx >> 1, idx & 1);
        if lsb == 0 {
            self.shift + self.subgroup_generator() * idx
        } else {
            -self.shift + self.subgroup_generator() * (idx + 1)
        }
    }

    pub(crate) fn vanishing_poly<EF: ExtensionField<F>>(&self, at: Point<EF>) -> EF {
        at.v_n(self.log_n) - self.shift.v_n(self.log_n)
    }

    /// `log_period`, `log_repetitions` with `trace_len / period = 2^log_repetitions`.
    fn periodic_column_fold_params(&self, period: usize) -> (usize, usize) {
        debug_assert!(period.is_power_of_two());
        let trace_len = self.size();
        assert_eq!(
            trace_len % period,
            0,
            "trace length must be divisible by periodic column length"
        );
        let log_period = log2_strict_usize(period);
        let log_repetitions = log2_strict_usize(trace_len / period);
        (log_period, log_repetitions)
    }
}

/// Below this length, chunking overhead (each chunk reseeds via a scalar multiplication costing
/// `O(log len)` point operations) outweighs the benefit of splitting the chain across threads.
const PARALLEL_THRESHOLD: usize = 1 << 10;

/// Materialize `len` points of the sequential chain `iterate(seed, |&p| p + g)`, splitting it
/// into chunks that run in parallel. Each chunk reseeds itself with one scalar multiplication
/// (`g * chunk_start`) instead of walking the prefix of the chain that precedes it.
fn parallel_point_chain<F: ComplexExtendable>(
    seed: Point<F>,
    g: Point<F>,
    len: usize,
) -> Vec<Point<F>> {
    if len < PARALLEL_THRESHOLD {
        return iterate(seed, move |&p| p + g).take(len).collect_vec();
    }
    let num_chunks = current_num_threads().max(1).min(len);
    let chunk_len = len.div_ceil(num_chunks);
    (0..len.div_ceil(chunk_len))
        .into_par_iter()
        .map(|c| {
            let chunk_start = c * chunk_len;
            let this_len = chunk_len.min(len - chunk_start);
            let chunk_seed = seed + g * chunk_start;
            iterate(chunk_seed, move |&p| p + g)
                .take(this_len)
                .collect_vec()
        })
        .collect::<Vec<_>>()
        .concat()
}

impl<F: ComplexExtendable> PolynomialSpace for CircleDomain<F> {
    type Val = F;

    fn size(&self) -> usize {
        1 << self.log_n
    }

    fn first_point(&self) -> Self::Val {
        self.shift.to_projective_line().unwrap()
    }

    fn next_point<Ext: ExtensionField<Self::Val>>(&self, x: Ext) -> Option<Ext> {
        // Only in standard position do we have an algebraic expression to access the next point.
        if self.is_standard() {
            (Point::from_projective_line(x) + Point::generator(self.log_n)).to_projective_line()
        } else {
            None
        }
    }

    fn try_create_disjoint_domain(&self, min_size: usize) -> Option<Self> {
        // Right now we simply guarantee the domain is disjoint by returning a
        // larger standard position coset, which is fine because we always ask for a larger
        // domain. If we wanted good performance for a disjoint domain of the same size,
        // we could change the shift. Also we could support nonstandard twin cosets.
        if !self.is_standard() {
            return None;
        }
        let log_n = log2_ceil_usize(min_size);
        // Any standard position coset that is not the same size as us will be disjoint.
        Some(Self::standard(if log_n == self.log_n {
            log_n + 1
        } else {
            log_n
        }))
    }

    /// Decompose a domain into disjoint twin-cosets.
    fn split_domains(&self, num_chunks: usize) -> Vec<Self> {
        assert!(self.is_standard());
        let log_chunks = log2_strict_usize(num_chunks);
        assert!(log_chunks <= self.log_n);
        self.points()
            .take(num_chunks)
            .map(|shift| Self {
                log_n: self.log_n - log_chunks,
                shift,
            })
            .collect()
    }

    /// The tangent-functional transition selector (Remark 17) puts the overall quotient `Q` in
    /// `L_{(d-1)*N+2}` rather than `L_{(d-1)*N}` (Remark 22). This is invisible whenever the
    /// `num_regular_chunks`-way decomposition already has slack, i.e. whenever rounding
    /// `num_regular_chunks` up to a power of two would add a chunk. It is only exposed when
    /// `num_regular_chunks` is itself an exact power of two greater than one (the degenerate
    /// `num_regular_chunks == 1` case has no cross-chunk decomposition to exceed), in which case
    /// an extra disjoint twin-coset carries the excess.
    ///
    /// That excess is the residual `q_0 = (Q - Q') / v` on the extra coset, where `Q'` is the
    /// plain `num_regular_chunks`-way reconstruction (in the FFT space over `quotient_domain`) and
    /// `v` is `quotient_domain`'s vanishing polynomial. Since `Q` overshoots `Q'` by two degrees
    /// and `v` has degree `(d-1)*N/2`, `q_0` has degree at most `1`: it lies in `L_2`, i.e.
    /// `span{1, x, y}` (dimension `3`). A size-2 twin-coset's interpolation space is only
    /// `span{1, y}` and would silently drop `q_0`'s `x` component; the size-4 FFT space
    /// `span{1, y, x, x*y}` contains all of `L_2`, so a size-4 twin-coset is the smallest that
    /// represents `q_0` exactly.
    fn quotient_extension_size(&self, num_regular_chunks: usize) -> Option<usize> {
        (num_regular_chunks > 1 && num_regular_chunks.is_power_of_two()).then_some(4)
    }

    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Self::Val>,
    ) -> Vec<RowMajorMatrix<Self::Val>> {
        let log_chunks = log2_strict_usize(num_chunks);
        assert!(evals.height() >> (log_chunks + 1) >= 1);
        let width = evals.width();
        let mut values: Vec<Vec<Self::Val>> = vec![vec![]; num_chunks];
        evals
            .rows()
            .enumerate()
            .for_each(|(i, row)| values[forward_backward_index(i, num_chunks)].extend(row));
        values
            .into_iter()
            .map(|v| RowMajorMatrix::new(v, width))
            .collect()
    }

    fn vanishing_poly_at_point<Ext: ExtensionField<Self::Val>>(&self, point: Ext) -> Ext {
        self.vanishing_poly(Point::from_projective_line(point))
    }

    fn selectors_at_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
    ) -> LagrangeSelectors<Ext> {
        let point = Point::from_projective_line(point);

        // Single-point specialization of the fused pass in `selectors_on_coset`: one
        // shared `vanishing_poly` evaluation and one batch inversion instead of four
        // separate `log_n`-step squaring chains and three separate inversions.
        let neg_shift = -self.shift;
        let k = neg_shift.s_p_at_p(self.log_n);
        let z = self.vanishing_poly(point);
        let den_shift = self.shift.v_tilde_p(point);
        let den_negshift_k = neg_shift.v_tilde_p(point) * k;

        let inv = batch_multiplicative_inverse(&[den_shift, den_negshift_k, z]);
        let (inv_den_shift, inv_den_negshift_k, inv_z) = (inv[0], inv[1], inv[2]);

        let z_inv_dk = z * inv_den_negshift_k;
        LagrangeSelectors {
            is_first_row: z * inv_den_shift,
            is_last_row: z_inv_dk * k,
            // Tangent functional at the last point `P = -shift` (eprint 2024/278, Remark 17):
            // `s_P(x, y) = x_P * x + y_P * y - 1`, a double zero at `P` and no other zeros on the
            // circle. As a selector in `L_2` it lets transition constraints carry the full AIR
            // degree with no selector degree penalty.
            is_transition: point.x * neg_shift.x + point.y * neg_shift.y - Ext::ONE,
            inv_vanishing: inv_z,
        }
    }

    /*
    chunks=2:

          1 . 1
         .     .
        0       0 <-- start
        .   .   - (1,0)
        0       0
         .     .
          1 . 1


    idx -> which chunk to put it in:
    chunks=2: 0 1 1 0 0 1 1 0 0 1 1 0 0 1 1 0
    chunks=4: 0 1 2 3 3 2 1 0 0 1 2 3 3 2 1 0
    */
    #[instrument(skip_all, fields(log_n = %coset.log_n))]
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Self::Val>> {
        let pts = coset.points().collect_vec();
        let n = pts.len();

        let neg_shift = -self.shift;
        let k = neg_shift.s_p_at_p(self.log_n);
        // `vanishing_poly(at) = at.v_n(log_n) - shift.v_n(log_n)`; the second term is the
        // same constant for every point in the coset, so it is hoisted out of the loop
        // below instead of being recomputed (as a `log_n`-step squaring chain) per point.
        let shift_v_n = self.shift.v_n(self.log_n);

        // Fused parallel pass over the coset points: `vanishing_poly`,
        // `shift.v_tilde_p` and `(-shift).v_tilde_p * k` are independent per
        // point. Computing them side-by-side reads `pts` once and writes the
        // three outputs in parallel.
        let mut z_vals = Self::Val::zero_vec(n);
        let mut den_shift = Self::Val::zero_vec(n);
        let mut den_negshift_k = Self::Val::zero_vec(n);
        z_vals
            .par_iter_mut()
            .zip(den_shift.par_iter_mut())
            .zip(den_negshift_k.par_iter_mut())
            .zip(pts.par_iter())
            .for_each(|(((z, ds), dnk), &at)| {
                *z = at.v_n(self.log_n) - shift_v_n;
                *ds = self.shift.v_tilde_p(at);
                *dnk = neg_shift.v_tilde_p(at) * k;
            });

        // Batch inverses (already internally parallel).
        let inv_vanishing = batch_multiplicative_inverse(&z_vals);
        let inv_den_shift = batch_multiplicative_inverse(&den_shift);
        let inv_den_negshift_k = batch_multiplicative_inverse(&den_negshift_k);

        // Fused parallel selector build for the two Lagrangian selectors:
        let mut is_first_row = Self::Val::zero_vec(n);
        let mut is_last_row = Self::Val::zero_vec(n);
        is_first_row
            .par_iter_mut()
            .zip(is_last_row.par_iter_mut())
            .zip(z_vals.par_iter())
            .zip(inv_den_shift.par_iter())
            .zip(inv_den_negshift_k.par_iter())
            .for_each(|((((ifr, ilr), &z), &inv_d), &inv_dk)| {
                *ifr = z * inv_d;
                *ilr = z * inv_dk * k;
            });

        // Tangent functional at the last point `P = -shift` (eprint 2024/278, Remark 17):
        // `s_P(x, y) = x_P * x + y_P * y - 1`, evaluated over the coset points.
        let mut is_transition = Self::Val::zero_vec(n);
        is_transition
            .par_iter_mut()
            .zip(pts.par_iter())
            .for_each(|(itr, &at)| {
                *itr = neg_shift.x * at.x + neg_shift.y * at.y - Self::Val::ONE;
            });

        LagrangeSelectors {
            is_first_row,
            is_last_row,
            is_transition,
            inv_vanishing,
        }
    }

    fn evaluate_polynomial_at<Ext: ExtensionField<F>>(&self, evals: &[F], point: Ext) -> Ext {
        assert!(
            self.is_standard(),
            "evaluate_polynomial_at requires standard position"
        );
        assert_eq!(evals.len(), self.size());
        let values = RowMajorMatrix::new(evals.to_vec(), 1);
        let circle_evals = CircleEvaluations::from_natural_order(*self, values);
        let circle_point = Point::from_projective_line(point);
        circle_evals.evaluate_at_point(circle_point)[0]
    }

    fn evaluate_periodic_column_at<Ext: ExtensionField<F>>(&self, col: &[F], point: Ext) -> Ext {
        if col.is_empty() {
            return Ext::ZERO;
        }
        assert!(
            self.is_standard(),
            "evaluate_periodic_column_at requires standard position"
        );
        assert!(
            col.len().is_power_of_two(),
            "periodic column length must be a power of 2"
        );

        let (log_period, log_repetitions) = self.periodic_column_fold_params(col.len());
        let periodic_domain = Self::standard(log_period);

        let evals = CircleEvaluations::from_natural_order(
            periodic_domain,
            RowMajorMatrix::new_col(col.to_vec()),
        );

        let query_point = Point::<Ext>::from_projective_line(point);
        let periodic_point = query_point.repeated_double(log_repetitions);
        evals.evaluate_at_point(periodic_point)[0]
    }
}

// 0 1 2 .. len-1 len len len-1 .. 1 0 0 1 ..
const fn forward_backward_index(mut i: usize, len: usize) -> usize {
    i %= 2 * len;
    if i < len { i } else { 2 * len - 1 - i }
}

#[cfg(test)]
mod tests {
    use core::iter;

    use hashbrown::HashSet;
    use itertools::izip;
    use p3_field::{PrimeCharacteristicRing, batch_multiplicative_inverse};
    use p3_mersenne_31::Mersenne31;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;
    use crate::CircleEvaluations;

    fn assert_is_twin_coset<F: ComplexExtendable>(d: CircleDomain<F>) {
        let pts = d.points().collect_vec();
        let half_n = pts.len() >> 1;
        for (&l, &r) in izip!(&pts[..half_n], pts[half_n..].iter().rev()) {
            assert_eq!(l, -r);
        }
    }

    fn do_test_circle_domain(log_n: usize, width: usize) {
        let n = 1 << log_n;

        type F = Mersenne31;
        let d = CircleDomain::<F>::standard(log_n);

        // we can move around the circle and end up where we started
        let p0 = d.first_point();
        let mut p1 = p0;
        for i in 0..(n - 1) {
            // nth_point is correct
            assert_eq!(Point::from_projective_line(p1), d.nth_point(i));
            p1 = d.next_point(p1).unwrap();
            assert_ne!(p1, p0);
        }
        assert_eq!(d.next_point(p1).unwrap(), p0);

        // .points() is the same as first_point -> next_point
        let mut uni_point = d.first_point();
        for p in d.points() {
            assert_eq!(Point::from_projective_line(uni_point), p);
            uni_point = d.next_point(uni_point).unwrap();
        }

        // disjoint domain is actually disjoint, and large enough
        let seen: HashSet<Point<F>> = d.points().collect();
        for disjoint_size in [10, 100, n - 5, n + 15] {
            let dd = d.create_disjoint_domain(disjoint_size);
            assert!(dd.size() >= disjoint_size);
            for pt in dd.points() {
                assert!(!seen.contains(&pt));
            }
        }

        // zp is zero
        for p in d.points() {
            assert_eq!(
                d.vanishing_poly_at_point(p.to_projective_line().unwrap()),
                F::ZERO
            );
        }

        let mut rng = SmallRng::seed_from_u64(1);

        // split domains
        let evals = RowMajorMatrix::rand(&mut rng, n, width);
        let orig: Vec<(Point<F>, Vec<F>)> = d
            .points()
            .zip(evals.rows().map(|r| r.collect_vec()))
            .collect();
        for num_chunks in [1, 2, 4, 8] {
            let mut combined = vec![];

            let sds = d.split_domains(num_chunks);
            assert_eq!(sds.len(), num_chunks);
            let ses = d.split_evals(num_chunks, evals.clone());
            assert_eq!(ses.len(), num_chunks);
            for (sd, se) in izip!(sds, ses) {
                // Split domains are twin cosets
                assert_is_twin_coset(sd);
                // Split domains have correct size wrt original domain
                assert_eq!(sd.size() * num_chunks, d.size());
                assert_eq!(se.width(), evals.width());
                assert_eq!(se.height() * num_chunks, d.size());
                combined.extend(sd.points().zip(se.rows().map(|r| r.collect_vec())));
            }
            // Union of split domains and evals is the original domain and evals
            assert_eq!(
                orig.iter().map(|x| x.0).collect::<HashSet<_>>(),
                combined.iter().map(|x| x.0).collect::<HashSet<_>>(),
                "union of split domains is orig domain"
            );
            assert_eq!(
                orig.iter().map(|x| &x.1).collect::<HashSet<_>>(),
                combined.iter().map(|x| &x.1).collect::<HashSet<_>>(),
                "union of split evals is orig evals"
            );
            assert_eq!(
                orig.iter().collect::<HashSet<_>>(),
                combined.iter().collect::<HashSet<_>>(),
                "split domains and evals correspond to orig domains and evals"
            );
        }
    }

    #[test]
    fn selectors() {
        type F = Mersenne31;
        let log_n = 8;
        let n = 1 << log_n;

        let d = CircleDomain::<F>::standard(log_n);
        let coset = d.create_disjoint_domain(n);
        let sels = d.selectors_on_coset(coset);

        // selectors_on_coset matches selectors_at_point
        let mut pt = coset.first_point();
        for i in 0..coset.size() {
            let pt_sels = d.selectors_at_point(pt);
            assert_eq!(sels.is_first_row[i], pt_sels.is_first_row);
            assert_eq!(sels.is_last_row[i], pt_sels.is_last_row);
            assert_eq!(sels.is_transition[i], pt_sels.is_transition);
            assert_eq!(sels.inv_vanishing[i], pt_sels.inv_vanishing);
            pt = coset.next_point(pt).unwrap();
        }

        let coset_to_d = |evals: &[F]| {
            let evals = CircleEvaluations::from_natural_order(
                coset,
                RowMajorMatrix::new_col(evals.to_vec()),
            );
            let coeffs = evals.interpolate().to_row_major_matrix();
            let (lo, hi) = coeffs.split_rows(n);
            assert_eq!(hi.values, F::zero_vec(n));
            CircleEvaluations::evaluate(d, lo.to_row_major_matrix())
                .to_natural_order()
                .to_row_major_matrix()
                .values
        };

        // Nonzero at first point, zero everywhere else on domain
        let is_first_row = coset_to_d(&sels.is_first_row);
        assert_ne!(is_first_row[0], F::ZERO);
        assert_eq!(&is_first_row[1..], &F::zero_vec(n - 1));

        // Nonzero at last point, zero everywhere else on domain
        let is_last_row = coset_to_d(&sels.is_last_row);
        assert_eq!(&is_last_row[..n - 1], &F::zero_vec(n - 1));
        assert_ne!(is_last_row[n - 1], F::ZERO);

        // Nonzero everywhere on domain but last point
        let is_transition = coset_to_d(&sels.is_transition);
        assert_ne!(&is_transition[..n - 1], &F::zero_vec(n - 1));
        assert_eq!(is_transition[n - 1], F::ZERO);

        // Vanishing polynomial coefficients look like [0.. (n times), 1, 0.. (n-1 times)]
        let z_coeffs = CircleEvaluations::from_natural_order(
            coset,
            RowMajorMatrix::new_col(batch_multiplicative_inverse(&sels.inv_vanishing)),
        )
        .interpolate()
        .to_row_major_matrix()
        .values;
        assert_eq!(
            z_coeffs,
            iter::empty()
                .chain(iter::repeat_n(F::ZERO, n))
                .chain(iter::once(F::ONE))
                .chain(iter::repeat_n(F::ZERO, n - 1))
                .collect_vec()
        );
    }

    #[test]
    fn test_circle_domain() {
        do_test_circle_domain(4, 8);
        do_test_circle_domain(10, 32);
    }

    #[test]
    fn points_vec_matches_points() {
        type F = Mersenne31;
        for log_n in 1..8 {
            let d = CircleDomain::<F>::standard(log_n);
            assert_eq!(d.points_vec(), d.points().collect_vec());
        }
    }

    /// Sanity check for the Lemma-12 chunk reconstruction math used by
    /// `recompose_quotient_from_chunks`: splitting a random array over `quotient_domain`
    /// into `num_chunks` equal chunks, then recombining via
    /// `sum_k q_k(zeta) * prod_{j!=k} v_Hj(zeta)/v_Hj(rep_k)`, should equal directly
    /// interpolating the whole array and evaluating at `zeta`.
    #[test]
    fn chunk_recompose_matches_direct_interpolation() {
        use p3_field::Field;
        use p3_field::extension::BinomialExtensionField;
        use rand::RngExt;

        type F = Mersenne31;
        type EF = BinomialExtensionField<Mersenne31, 3>;

        let mut rng = SmallRng::seed_from_u64(123);
        let log_n = 3;
        let num_chunks = 4;
        let quotient_domain = CircleDomain::<F>::standard(log_n);
        let values: Vec<F> = (0..quotient_domain.size()).map(|_| rng.random()).collect();

        let zeta: EF = rng.random();

        // Direct interpolation of the whole array.
        let direct = quotient_domain.evaluate_polynomial_at(&values, zeta);

        // Chunk-based reconstruction, mirroring `recompose_quotient_from_chunks`.
        let chunk_domains = quotient_domain.split_domains(num_chunks);
        let evals_matrix = RowMajorMatrix::new(values.clone(), 1);
        let chunk_evals = quotient_domain.split_evals(num_chunks, evals_matrix);
        let chunk_vals_at_zeta: Vec<EF> = chunk_domains
            .iter()
            .zip(&chunk_evals)
            .map(|(&d, m)| {
                CircleEvaluations::from_natural_order(d, m.as_view())
                    .evaluate_at_point(Point::from_projective_line(zeta))[0]
            })
            .collect();

        let zps: Vec<EF> = chunk_domains
            .iter()
            .enumerate()
            .map(|(i, d)| {
                chunk_domains
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, other)| {
                        other.vanishing_poly_at_point(zeta)
                            * other.vanishing_poly_at_point(d.first_point()).inverse()
                    })
                    .product::<EF>()
            })
            .collect_vec();

        let reconstructed: EF = chunk_vals_at_zeta
            .iter()
            .zip(&zps)
            .map(|(&v, &zp)| v * zp)
            .sum();

        assert_eq!(direct, reconstructed);
    }

    /// The Remark 22 extension chunk carries the residual `q_0 = (Q - Q') / v`, which lies in
    /// `L_2 = span{1, x, y}`. This locks in why [`CircleDomain::quotient_extension_size`] reports a
    /// size-4 (not size-2) twin-coset: the size-2 FFT space is only `span{1, y}` and drops the `x`
    /// component of a residual, whereas the size-4 FFT space `span{1, y, x, x*y}` reproduces every
    /// element of `L_2` exactly.
    #[test]
    fn size_four_extension_reproduces_l2_but_size_two_drops_x() {
        use p3_field::extension::BinomialExtensionField;
        use rand::RngExt;

        type F = Mersenne31;
        type EF = BinomialExtensionField<Mersenne31, 3>;

        let mut rng = SmallRng::seed_from_u64(99);
        // Mirror how the extension domain is actually built: a bigger standard domain's own
        // `try_create_disjoint_domain`, at both candidate sizes.
        let big = CircleDomain::<F>::standard(8);
        let size_two: CircleDomain<F> = big.try_create_disjoint_domain(2).unwrap();
        let size_four: CircleDomain<F> = big.try_create_disjoint_domain(4).unwrap();
        assert_eq!(size_two.size(), 2);
        assert_eq!(size_four.size(), 4);

        // A residual `q_0(x, y) = a + b*x + c*y in L_2` with a nonzero `x` component.
        let (a, b, c): (F, F, F) = (rng.random(), rng.random(), rng.random());
        assert_ne!(b, F::ZERO);
        let q0 = |p: Point<F>| a + b * p.x + c * p.y;

        let target: EF = rng.random();
        let target_point = Point::<EF>::from_projective_line(target);
        let expected = EF::from(a) + EF::from(b) * target_point.x + EF::from(c) * target_point.y;

        let interpolate_at = |domain: CircleDomain<F>| {
            let values: Vec<F> = domain.points().map(&q0).collect();
            CircleEvaluations::from_natural_order(domain, RowMajorMatrix::new(values, 1))
                .evaluate_at_point(target_point)[0]
        };

        assert_eq!(
            interpolate_at(size_four),
            expected,
            "size-4 interpolation must reproduce every element of L_2"
        );
        assert_ne!(
            interpolate_at(size_two),
            expected,
            "size-2 interpolation drops the x component, so it cannot carry the residual"
        );
    }
}
