use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use itertools::{Itertools, iterate, izip};
use p3_commit::PolynomialSpace;
use p3_dft::{Butterfly, DifButterfly, DitButterfly};
use p3_field::extension::ComplexExtendable;
use p3_field::{
    ExtensionField, Field, FieldArray, PackedValue, batch_multiplicative_inverse,
};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_util::{log2_ceil_usize, log2_strict_usize, reverse_slice_index_bits};
use tracing::{debug_span, instrument};

use crate::domain::CircleDomain;
use crate::point::{Point, compute_lagrange_den_batched};
use crate::{CfftPermutable, CfftView, cfft_permute_index, cfft_permute_slice};

#[derive(Clone)]
pub struct CircleEvaluations<F, M = RowMajorMatrix<F>> {
    pub(crate) domain: CircleDomain<F>,
    pub(crate) values: M,
}

impl<F: Copy + Send + Sync, M: Matrix<F>> CircleEvaluations<F, M> {
    pub(crate) fn from_cfft_order(domain: CircleDomain<F>, values: M) -> Self {
        assert_eq!(1 << domain.log_n, values.height());
        Self { domain, values }
    }
    pub fn from_natural_order(
        domain: CircleDomain<F>,
        values: M,
    ) -> CircleEvaluations<F, CfftView<M>> {
        CircleEvaluations::from_cfft_order(domain, values.cfft_perm_rows())
    }
    pub fn to_cfft_order(self) -> M {
        self.values
    }
    pub fn to_natural_order(self) -> CfftView<M> {
        self.values.cfft_perm_rows()
    }
}

impl<F: ComplexExtendable, M: Matrix<F>> CircleEvaluations<F, M> {
    #[instrument(skip_all, fields(dims = %self.values.dimensions()))]
    pub fn interpolate(self) -> RowMajorMatrix<F> {
        let Self { domain, values } = self;
        let mut values = debug_span!("to_rmm").in_scope(|| values.to_row_major_matrix());

        let twiddles = debug_span!("twiddles").in_scope(|| {
            compute_twiddles(domain)
                .into_iter()
                .map(|ts| {
                    batch_multiplicative_inverse(&ts)
                        .into_iter()
                        .map(|t| DifButterfly(t))
                        .collect_vec()
                })
                .collect_vec()
        });

        assert_eq!(twiddles.len(), domain.log_n);

        // The interpolation must divide every element by the domain size. Folding the
        // scaling into the last butterfly pass touches the data while it is cache-resident,
        // instead of paying a separate full sweep over the matrix.
        let h_inv = F::ONE.div_2exp_u64(domain.log_n as u64);
        cfft_layers(&mut values.values, values.width, &twiddles, Some(h_inv));

        values
    }

    #[instrument(skip_all, fields(dims = %self.values.dimensions()))]
    pub fn extrapolate(
        self,
        target_domain: CircleDomain<F>,
    ) -> CircleEvaluations<F, RowMajorMatrix<F>> {
        assert!(target_domain.log_n >= self.domain.log_n);
        let Self { domain, values } = self;

        // Materialize the evaluations into a buffer sized for the full LDE up front.
        // `interpolate` keeps the buffer (and its spare capacity) through
        // `to_row_major_matrix`, so the blow-up in `evaluate` fills the spare
        // capacity instead of reallocating: the whole extrapolation performs a
        // single allocation, and every page is first touched by a parallel write.
        let w = values.width();
        let initial_len = values.height() * w;
        let target_len = target_domain.size() * w;
        let values = debug_span!("to_rmm").in_scope(|| {
            let mut buf = Vec::with_capacity(target_len);
            // Each source row is copied into its slot by exactly one task,
            // covering `[0, initial_len)` of the spare capacity.
            buf.spare_capacity_mut()[..initial_len]
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(r, dst_row)| {
                    // SAFETY: `r < values.height()`, since the chunks cover
                    // exactly `values.height()` rows.
                    let src_row = unsafe { values.row_slice_unchecked(r) };
                    // `MaybeUninit::write` stores without reading or dropping
                    // the uninitialised bytes.
                    for (dst, &src) in dst_row.iter_mut().zip(src_row.iter()) {
                        dst.write(src);
                    }
                });
            // SAFETY: the loop above initialised the first `initial_len`
            // elements, and the capacity reserved covers them.
            unsafe {
                buf.set_len(initial_len);
            }
            RowMajorMatrix::new(buf, w)
        });

        let coeffs = CircleEvaluations::from_cfft_order(domain, values).interpolate();
        CircleEvaluations::evaluate(target_domain, coeffs)
    }

    pub fn evaluate_at_point<EF: ExtensionField<F>>(&self, point: Point<EF>) -> Vec<EF> {
        // Permute the domain to get it into the right format.
        let permuted_points = cfft_permute_slice(&self.domain.points().collect_vec());

        // Compute the lagrange denominators. This is batched as it lets us make use of batched_multiplicative_inverse.
        let lagrange_den = compute_lagrange_den_batched(&permuted_points, point, self.domain.log_n);

        self.evaluate_at_point_with_den(point, &lagrange_den)
    }

    /// Evaluate at `point` given precomputed Lagrange denominators for `(self.domain, point)`,
    /// as produced by [`compute_lagrange_den_batched`] on the CFFT-ordered domain points.
    pub(crate) fn evaluate_at_point_with_den<EF: ExtensionField<F>>(
        &self,
        point: Point<EF>,
        lagrange_den: &[EF],
    ) -> Vec<EF> {
        // Compute z_H
        let lagrange_num = self.domain.vanishing_poly(point);

        // The columnwise_dot_product here consumes about 5% of the runtime for example prove_poseidon2_m31_keccak.
        // Definitely something worth optimising further.
        self.values
            .columnwise_dot_product(lagrange_den)
            .into_iter()
            .map(|x| x * lagrange_num)
            .collect_vec()
    }

    /// Evaluate at two points in a single pass over the matrix, given precomputed Lagrange
    /// denominators for each point.
    ///
    /// Equivalent to two [`Self::evaluate_at_point_with_den`] calls, but each matrix row is
    /// only loaded once.
    pub(crate) fn evaluate_at_two_points_with_dens<EF: ExtensionField<F>>(
        &self,
        points: [Point<EF>; 2],
        dens: [&[EF]; 2],
    ) -> [Vec<EF>; 2] {
        let lagrange_nums = points.map(|point| self.domain.vanishing_poly(point));

        let interleaved_dens = izip!(dens[0], dens[1])
            .map(|(&den_0, &den_1)| FieldArray([den_0, den_1]))
            .collect_vec();

        let (ps_at_point_0, ps_at_point_1) = self
            .values
            .columnwise_dot_product_batched::<EF, 2>(&interleaved_dens)
            .into_iter()
            .map(|FieldArray([dot_0, dot_1])| (dot_0 * lagrange_nums[0], dot_1 * lagrange_nums[1]))
            .unzip();
        [ps_at_point_0, ps_at_point_1]
    }

    #[cfg(test)]
    pub(crate) fn dim(&self) -> usize
    where
        M: Clone,
    {
        let coeffs = self.clone().interpolate();
        for (i, mut row) in coeffs.rows().enumerate() {
            if row.all(|x| x.is_zero()) {
                return i;
            }
        }
        coeffs.height()
    }
}

impl<F: ComplexExtendable> CircleEvaluations<F, RowMajorMatrix<F>> {
    #[instrument(skip_all, fields(dims = %coeffs.dimensions()))]
    pub fn evaluate(domain: CircleDomain<F>, mut coeffs: RowMajorMatrix<F>) -> Self {
        let log_n = log2_strict_usize(coeffs.height());
        assert!(log_n <= domain.log_n);

        if log_n < domain.log_n {
            // We could simply pad coeffs like this:
            // coeffs.pad_to_height(target_domain.size(), F::ZERO);
            // But the first `added_bits` layers will simply fill out the zeros
            // with the lower order values. (In `DitButterfly`, `x_2` is 0, so
            // both `x_1` and `x_2` are set to `x_1`).
            // So instead we directly repeat the coeffs and skip the initial layers.
            //
            // After any number of doublings the buffer is the original `h` rows tiled end to end.
            // Output row `j` is therefore a copy of original row `j mod h`.
            //
            //     originals : r_0 r_1 ... r_{h-1}
            //     filled    : r_0 ... r_{h-1} | r_0 ... r_{h-1} | r_0 ... r_{h-1}
            //                 \_h originals_/   \___ identical tiled copies ___/
            //
            // Reserve the full target once, then fill the new tail in parallel.
            // Each tail row is written by exactly one task.
            // The original rows are only read.
            // So the concurrent writes never race.
            debug_span!("extend coeffs").in_scope(|| {
                let w = coeffs.width();
                // Rows present before the blow-up.
                let initial_h = coeffs.height();
                // Rows required after the blow-up.
                let target_h = domain.size();
                let initial_len = initial_h * w;
                let target_len = target_h * w;

                // Grow to the final size in one allocation; the tail stays uninitialised.
                coeffs.values.reserve_exact(target_len - initial_len);

                // SAFETY:
                // - The reservation above guarantees capacity for `target_len` elements.
                // - The read view covers `[0, initial_len)`.
                // - The write view covers `[initial_len, target_len)`.
                // - The two ranges are disjoint parts of one allocation.
                // - Each tail row is handed to exactly one task, so the writes never alias.
                // - `MaybeUninit::write` stores without reading or dropping the prior bytes.
                // - The length is updated only after every tail element is written.
                unsafe {
                    // Base pointer of the reserved allocation.
                    let ptr = coeffs.values.as_mut_ptr();

                    // Read-only view of the rows already present.
                    let initial_region: &[F] = core::slice::from_raw_parts(ptr, initial_len);

                    // Write view of the uninitialised tail, just past the existing rows.
                    let new_region: &mut [MaybeUninit<F>] = core::slice::from_raw_parts_mut(
                        ptr.add(initial_len).cast::<MaybeUninit<F>>(),
                        target_len - initial_len,
                    );

                    // One task per destination row.
                    new_region
                        .par_chunks_mut(w)
                        .enumerate()
                        .for_each(|(i, dst_row)| {
                            // `i` indexes the tail, so this is global output row `initial_h + i`.
                            // The tail starts on a multiple of `initial_h`.
                            // So the source row reduces to `i mod initial_h`.
                            let src_row_start = (i % initial_h) * w;
                            let src_row = &initial_region[src_row_start..src_row_start + w];
                            // Copy the chosen original row into this tail slot.
                            for (dst, &src) in dst_row.iter_mut().zip(src_row) {
                                dst.write(src);
                            }
                        });

                    // Every tail element is initialised; publish them as live `Vec` entries.
                    coeffs.values.set_len(target_len);
                }
            });
        }
        assert_eq!(coeffs.height(), 1 << domain.log_n);

        let twiddles = debug_span!("twiddles").in_scope(|| {
            compute_twiddles(domain)
                .into_iter()
                .map(|ts| ts.into_iter().map(|t| DitButterfly(t)).collect_vec())
                .rev()
                .skip(domain.log_n - log_n)
                .collect_vec()
        });

        cfft_layers(&mut coeffs.values, coeffs.width, &twiddles, None);

        Self::from_cfft_order(domain, coeffs)
    }
}

/// The bit position by which a layer's butterfly partners differ.
///
/// A layer whose twiddle array has `blocks` entries acts on `blocks` equal blocks of rows,
/// pairing the two halves of each block: rows `j` and `j ^ (h / (2 * blocks))`.
const fn flipped_bit(log_h: usize, blocks: usize) -> usize {
    log_h - log2_strict_usize(blocks) - 1
}

/// The binary log of the number of rows each task keeps cache-resident in [`cfft_layers`].
fn log_group_rows<F>(log_h: usize, width: usize) -> usize {
    // Cap the per-task working set so that all layers of a pass run from cache.
    const TARGET_GROUP_BYTES: usize = 1 << 19;
    let log_cache = log2_ceil_usize(TARGET_GROUP_BYTES / (width * size_of::<F>()).max(1)).max(1);
    // Keep enough groups around for the thread pool to stay busy, but never fewer than 8 rows
    // per group so small transforms still fuse several layers per pass.
    let log_par = log_h
        .saturating_sub(log2_ceil_usize(4 * current_num_threads()))
        .max(3);
    log_cache.min(log_par).min(log_h)
}

/// Apply a full sequence of butterfly layers, fusing as many layers as possible per pass.
///
/// Layers are batched into maximal consecutive runs whose [`flipped_bit`]s fit in a window of
/// `log_group` bits, `[log_stride, log_stride + log_group)`. The layers of one run only ever
/// combine rows that agree on all index bits outside that window, so the matrix splits into
/// independent groups of `2^log_group` rows sitting `2^log_stride` rows apart. Each parallel task
/// applies every layer of the run to one cache-resident group, costing one pass over memory per
/// run instead of one per layer.
///
/// When `scale` is set, every element is additionally multiplied by it during the final pass,
/// while its group is still cache-resident.
fn cfft_layers<F: Field, B: Butterfly<F>>(
    values: &mut [F],
    width: usize,
    layers: &[Vec<B>],
    scale: Option<F>,
) {
    let h = values.len() / width;
    let log_h = log2_strict_usize(h);
    let log_group = log_group_rows::<F>(log_h, width);

    let mut start = 0;
    while start < layers.len() {
        let first_bit = flipped_bit(log_h, layers[start].len());
        let (mut lo_bit, mut hi_bit) = (first_bit, first_bit);
        let mut end = start + 1;
        while let Some(ts) = layers.get(end) {
            let bit = flipped_bit(log_h, ts.len());
            if bit.max(hi_bit) - bit.min(lo_bit) >= log_group {
                break;
            }
            (lo_bit, hi_bit) = (bit.min(lo_bit), bit.max(hi_bit));
            end += 1;
        }
        let log_stride = lo_bit.min(log_h - log_group);
        let pass_scale = scale.filter(|_| end == layers.len());
        debug_span!("fused_layers", layers = end - start, log_group, log_stride).in_scope(|| {
            par_group_pass(
                values,
                width,
                &layers[start..end],
                log_group,
                log_stride,
                pass_scale,
            );
        });
        start = end;
    }
}

/// Apply consecutive butterfly layers whose [`flipped_bit`]s all lie in
/// `[log_stride, log_stride + log_group)`, parallelizing over independent row groups.
///
/// Group `g = (hi, lo)` consists of the rows `j = hi << (log_group + log_stride) | t << log_stride
/// | lo` for `t` in `[0, 2^log_group)`. A layer with `b` blocks pairs rows differing in bit
/// `e = flipped_bit - log_stride` of `t` and applies the twiddle `b * j / h`, which reduces to
/// index `t >> (e + 1)` into the contiguous twiddle slice for `hi`.
fn par_group_pass<F: Field, B: Butterfly<F>>(
    values: &mut [F],
    width: usize,
    layers: &[Vec<B>],
    log_group: usize,
    log_stride: usize,
    scale: Option<F>,
) {
    let h = values.len() / width;
    let log_h = log2_strict_usize(h);
    let num_groups = h >> log_group;
    let base_addr = values.as_mut_ptr() as usize;
    let packed_scale = scale.map(F::Packing::from);
    (0..num_groups).into_par_iter().for_each(|g| {
        let base = base_addr as *mut F;
        let hi = g >> log_stride;
        let lo = g & ((1 << log_stride) - 1);
        let first_row = (hi << (log_group + log_stride)) | lo;
        for ts in layers {
            let e = flipped_bit(log_h, ts.len()) - log_stride;
            let slice_len = 1 << (log_group - e - 1);
            let slice = &ts[hi * slice_len..][..slice_len];
            for (s, &t) in slice.iter().enumerate() {
                for u in 0..1usize << e {
                    let row_lo = first_row + (((s << (e + 1)) | u) << log_stride);
                    let row_hi = row_lo + (1 << (e + log_stride));
                    // SAFETY: every row index decomposes uniquely as
                    // `hi << (log_group + log_stride) | t << log_stride | lo`, so the task for
                    // group `g = (hi, lo)` is the only one touching its rows, and within a layer
                    // each row appears in exactly one butterfly, so `row_lo` and `row_hi` never
                    // alias. All indices stay below `h` since `t < 2^log_group`.
                    let (row_lo, row_hi) = unsafe {
                        (
                            core::slice::from_raw_parts_mut(base.add(row_lo * width), width),
                            core::slice::from_raw_parts_mut(base.add(row_hi * width), width),
                        )
                    };
                    t.apply_to_rows(row_lo, row_hi);
                }
            }
        }
        if let (Some(scale), Some(packed_scale)) = (scale, packed_scale) {
            for t in 0..1usize << log_group {
                let row = first_row + (t << log_stride);
                // SAFETY: the group decomposition above guarantees this task is the only
                // one touching its rows, and `row < h` since `t < 2^log_group`.
                let row = unsafe { core::slice::from_raw_parts_mut(base.add(row * width), width) };
                let (packed, suffix) = F::Packing::pack_slice_with_suffix_mut(row);
                for x in packed {
                    *x *= packed_scale;
                }
                for x in suffix {
                    *x *= scale;
                }
            }
        }
    });
}

impl<F: ComplexExtendable> CircleDomain<F> {
    pub(crate) fn y_twiddles(&self) -> Vec<F> {
        let mut ys = self.coset0().map(|p| p.y).collect_vec();
        reverse_slice_index_bits(&mut ys);
        ys
    }
    pub(crate) fn nth_y_twiddle(&self, index: usize) -> F {
        self.nth_point(cfft_permute_index(index << 1, self.log_n)).y
    }
    pub(crate) fn x_twiddles(&self, layer: usize) -> Vec<F> {
        let generator = self.subgroup_generator() * (1 << layer);
        let shift = self.shift * (1 << layer);
        let mut xs = iterate(shift, move |&p| p + generator)
            .map(|p| p.x)
            .take(1 << (self.log_n - layer - 2))
            .collect_vec();
        reverse_slice_index_bits(&mut xs);
        xs
    }
    pub(crate) fn nth_x_twiddle(&self, index: usize) -> F {
        (self.shift + self.subgroup_generator() * index).x
    }
}

fn compute_twiddles<F: ComplexExtendable>(domain: CircleDomain<F>) -> Vec<Vec<F>> {
    assert!(domain.log_n >= 1);
    let mut pts = domain.coset0().collect_vec();
    reverse_slice_index_bits(&mut pts);
    let mut twiddles = vec![pts.iter().map(|p| p.y).collect_vec()];
    if domain.log_n >= 2 {
        twiddles.push(pts.iter().step_by(2).map(|p| p.x).collect_vec());
        for i in 0..(domain.log_n - 2) {
            let prev = twiddles.last().unwrap();
            assert_eq!(prev.len(), 1 << (domain.log_n - 2 - i));
            let cur = prev
                .iter()
                .step_by(2)
                .map(|x| x.square().double() - F::ONE)
                .collect_vec();
            twiddles.push(cur);
        }
    }
    twiddles
}

pub fn circle_basis<F: Field>(p: Point<F>, log_n: usize) -> Vec<F> {
    let mut b = vec![F::ONE, p.y];
    let mut x = p.x;
    for _ in 0..(log_n - 1) {
        for i in 0..b.len() {
            b.push(b[i] * x);
        }
        x = x.square().double() - F::ONE;
    }
    assert_eq!(b.len(), 1 << log_n);
    b
}

#[cfg(test)]
mod tests {
    use itertools::iproduct;
    use p3_field::extension::BinomialExtensionField;
    use p3_mersenne_31::Mersenne31;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type F = Mersenne31;
    type EF = BinomialExtensionField<F, 3>;

    #[test]
    fn test_cfft_icfft() {
        let mut rng = SmallRng::seed_from_u64(1);
        for (log_n, width) in iproduct!(2..5, [1, 4, 11]) {
            let shift = Point::generator(F::CIRCLE_TWO_ADICITY) * (rng.random::<u16>() as usize);
            let domain = CircleDomain::new(log_n, shift);
            let trace = RowMajorMatrix::<F>::rand(&mut rng, 1 << log_n, width);
            let coeffs = CircleEvaluations::from_natural_order(domain, trace.clone()).interpolate();
            assert_eq!(
                CircleEvaluations::evaluate(domain, coeffs.clone())
                    .to_natural_order()
                    .to_row_major_matrix(),
                trace,
                "icfft(cfft(evals)) is identity",
            );
            for (i, pt) in domain.points().enumerate() {
                assert_eq!(
                    &*trace.row_slice(i).unwrap(),
                    coeffs.columnwise_dot_product(&circle_basis(pt, log_n)),
                    "coeffs can be evaluated with circle_basis",
                );
            }
        }
    }

    #[test]
    fn test_extrapolation() {
        let mut rng = SmallRng::seed_from_u64(1);
        for (log_n, log_blowup) in iproduct!(2..5, [1, 2, 3]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut rng, 1 << log_n, 11),
            );
            let lde = evals
                .clone()
                .extrapolate(CircleDomain::standard(log_n + log_blowup));

            let coeffs = evals.interpolate();
            let lde_coeffs = lde.interpolate();

            for r in 0..coeffs.height() {
                assert_eq!(
                    &*coeffs.row_slice(r).unwrap(),
                    &*lde_coeffs.row_slice(r).unwrap()
                );
            }
            for r in coeffs.height()..lde_coeffs.height() {
                assert!(lde_coeffs.row(r).unwrap().into_iter().all(|x| x.is_zero()));
            }
        }
    }

    #[test]
    fn eval_at_point_matches_cfft() {
        let mut rng = SmallRng::seed_from_u64(1);
        for (log_n, width) in iproduct!(2..5, [1, 4, 11]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut rng, 1 << log_n, width),
            );

            let pt = Point::<EF>::from_projective_line(rng.random());

            assert_eq!(
                evals.clone().evaluate_at_point(pt),
                evals
                    .interpolate()
                    .columnwise_dot_product(&circle_basis(pt, log_n))
            );
        }
    }

    #[test]
    fn eval_at_point_matches_lde() {
        let mut rng = SmallRng::seed_from_u64(1);
        for (log_n, width, log_blowup) in iproduct!(2..8, [1, 4, 11], [1, 2]) {
            let evals = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut rng, 1 << log_n, width),
            );
            let lde = evals
                .clone()
                .extrapolate(CircleDomain::standard(log_n + log_blowup));
            let zeta = Point::<EF>::from_projective_line(rng.random());
            assert_eq!(evals.evaluate_at_point(zeta), lde.evaluate_at_point(zeta));
            assert_eq!(
                evals.evaluate_at_point(zeta),
                evals
                    .interpolate()
                    .columnwise_dot_product(&circle_basis(zeta, log_n))
            );
            assert_eq!(
                lde.evaluate_at_point(zeta),
                lde.interpolate()
                    .columnwise_dot_product(&circle_basis(zeta, log_n + log_blowup))
            );
        }
    }
}
