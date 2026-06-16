use alloc::vec;
use alloc::vec::Vec;
use core::mem::MaybeUninit;

use itertools::{Itertools, iterate, izip};
use p3_commit::PolynomialSpace;
use p3_dft::{Butterfly, DifButterfly, DitButterfly};
use p3_field::extension::ComplexExtendable;
use p3_field::{ExtensionField, Field, FieldArray, PackedValue, batch_multiplicative_inverse};
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
        let len = self.domain.size() * self.values.width();
        self.interpolate_with_capacity(len)
    }

    /// Interpolate into a freshly allocated buffer with at least `capacity` elements
    /// reserved, so a caller can extend the result in place without reallocating.
    ///
    /// The source matrix is read directly by the first butterfly pass: each parallel
    /// task copies its own rows into the output buffer and immediately applies every
    /// layer of the pass while they are cache-resident, so the input is never
    /// materialized separately.
    fn interpolate_with_capacity(self, capacity: usize) -> RowMajorMatrix<F> {
        let Self { domain, values } = self;
        let w = values.width();
        let len = domain.size() * w;

        let twiddles = debug_span!("twiddles").in_scope(|| {
            compute_twiddles(domain)
                .into_iter()
                .map(|ts| {
                    CfftLayer::Butterflies(
                        batch_multiplicative_inverse(&ts)
                            .into_iter()
                            .map(|t| DifButterfly(t))
                            .collect_vec(),
                    )
                })
                .collect_vec()
        });

        assert_eq!(twiddles.len(), domain.log_n);

        // The interpolation must divide every element by the domain size. Folding the
        // scaling into the last butterfly pass touches the data while it is cache-resident,
        // instead of paying a separate full sweep over the matrix.
        let h_inv = F::ONE.div_2exp_u64(domain.log_n as u64);

        let mut buf: Vec<F> = Vec::with_capacity(capacity.max(len));
        cfft_layers(
            buf.as_mut_ptr(),
            domain.size(),
            w,
            &twiddles,
            Some(&values),
            Some(h_inv),
        );
        // SAFETY: the first pass of `cfft_layers` copied every row of `values` into
        // `buf` before transforming it, so all `len` elements are initialised, and the
        // reservation above covers them.
        unsafe {
            buf.set_len(len);
        }
        RowMajorMatrix::new(buf, w)
    }

    #[instrument(skip_all, fields(dims = %self.values.dimensions()))]
    pub fn extrapolate(
        self,
        target_domain: CircleDomain<F>,
    ) -> CircleEvaluations<F, RowMajorMatrix<F>> {
        assert!(target_domain.log_n >= self.domain.log_n);

        // Reserve the buffer for the full LDE up front: the blow-up in `evaluate` then
        // fills the spare capacity instead of reallocating, so the whole extrapolation
        // performs a single allocation, and every page is first touched by a parallel
        // write inside the first interpolation pass.
        let target_len = target_domain.size() * self.values.width();
        let coeffs = self.interpolate_with_capacity(target_len);
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

        let added_bits = domain.log_n - log_n;
        let w = coeffs.width();
        let target_len = domain.size() * w;

        // A `DitButterfly` layer acting on coefficients whose upper half is zero sets both
        // outputs to the lower input, so the first `added_bits` layers of the transform are
        // pure row duplications. Instead of materializing the zero-padding (or the tiled
        // copies it collapses to) in a separate sweep, the duplications run as [`Dup`]
        // layers inside the first fused pass, while the rows are cache-resident. Only the
        // capacity is reserved here; the duplication layers initialise the tail.
        coeffs
            .values
            .reserve_exact(target_len - coeffs.values.len());

        let twiddles = debug_span!("twiddles").in_scope(|| {
            compute_twiddles(domain)
                .into_iter()
                .map(|ts| ts.into_iter().map(|t| DitButterfly(t)).collect_vec())
                .rev()
                .skip(added_bits)
                .map(CfftLayer::Butterflies)
                .collect_vec()
        });
        let layers = (0..added_bits)
            .map(|l| CfftLayer::Dup { blocks: 1 << l })
            .chain(twiddles)
            .collect_vec();

        cfft_layers(
            coeffs.values.as_mut_ptr(),
            domain.size(),
            w,
            &layers,
            None::<&RowMajorMatrix<F>>,
            None,
        );

        // SAFETY: every row with one of the top `added_bits` index bits set is written by
        // the duplication layer of its highest such bit (later duplication layers rewrite
        // it consistently), so all `target_len` elements behind the reservation above are
        // initialised once `cfft_layers` returns.
        unsafe {
            coeffs.values.set_len(target_len);
        }

        Self::from_cfft_order(domain, coeffs)
    }
}

/// One layer of a fused CFFT pass.
enum CfftLayer<B> {
    /// Copy the lower half of each of `blocks` row blocks onto its upper half.
    ///
    /// This realizes a `DitButterfly` layer acting on coefficients whose upper half is
    /// zero (for which both outputs equal the lower input), i.e. the zero-padding part
    /// of an LDE, without materializing the padding in a separate sweep.
    Dup { blocks: usize },
    /// A twiddle butterfly layer, one twiddle per block.
    Butterflies(Vec<B>),
}

impl<B> CfftLayer<B> {
    const fn blocks(&self) -> usize {
        match self {
            Self::Dup { blocks } => *blocks,
            Self::Butterflies(ts) => ts.len(),
        }
    }
}

/// The bit position by which a layer's butterfly partners differ.
///
/// A layer with `blocks` blocks acts on `blocks` equal blocks of rows,
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
/// When `ingest` is set, the buffer behind `base` may be entirely uninitialised: each task of
/// the first pass copies its own rows from the source matrix before transforming them, so the
/// input is pulled in during the first pass instead of a separate materialization sweep. After
/// the call, all `h * width` elements behind `base` are initialised.
///
/// When `scale` is set, every element is additionally multiplied by it during the final pass,
/// while its group is still cache-resident.
///
/// [`CfftLayer::Dup`] layers may act on uninitialised rows: a duplication layer writes every
/// row with its flipped bit set, so once all duplication layers have run, every row beyond the
/// original (lowest-index) block is initialised. Since duplications are copies rather than
/// arithmetic, each one widens the window budget of its run by one bit instead of consuming it,
/// keeping the number of passes unchanged.
///
/// # Safety-relevant contract (not `unsafe fn` to keep call sites readable)
///
/// `base` must be valid for reads and writes of `h * width` elements. Elements must be
/// initialised, except (without `ingest`) rows whose index has a [`CfftLayer::Dup`] flipped
/// bit set, and (with `ingest`) the whole buffer. `layers` must not be empty if `ingest` or
/// `scale` is set (otherwise no pass would perform the copy or the scaling).
fn cfft_layers<F: Field, B: Butterfly<F>, M: Matrix<F>>(
    base: *mut F,
    h: usize,
    width: usize,
    layers: &[CfftLayer<B>],
    ingest: Option<&M>,
    scale: Option<F>,
) {
    let log_h = log2_strict_usize(h);
    let log_group = log_group_rows::<F>(log_h, width);
    debug_assert!(ingest.is_none_or(|m| m.height() == h && m.width() == width));
    assert!(
        !layers.is_empty() || (ingest.is_none() && scale.is_none()),
        "ingest and scale require at least one layer pass"
    );

    // Duplication layers may widen a window beyond `log_group` (see above), but never so far
    // that the larger groups leave the thread pool idle.
    let budget_cap =
        log_group.max(log_h.saturating_sub(log2_ceil_usize(4 * current_num_threads())));

    let mut start = 0;
    while start < layers.len() {
        let is_dup = |l: &CfftLayer<B>| matches!(l, CfftLayer::Dup { .. });
        let first_bit = flipped_bit(log_h, layers[start].blocks());
        let (mut lo_bit, mut hi_bit) = (first_bit, first_bit);
        let mut budget = (log_group + usize::from(is_dup(&layers[start]))).min(budget_cap);
        let mut end = start + 1;
        while let Some(layer) = layers.get(end) {
            let bit = flipped_bit(log_h, layer.blocks());
            let new_budget = (budget + usize::from(is_dup(layer))).min(budget_cap);
            if bit.max(hi_bit) - bit.min(lo_bit) >= new_budget {
                break;
            }
            budget = new_budget;
            (lo_bit, hi_bit) = (bit.min(lo_bit), bit.max(hi_bit));
            end += 1;
        }
        let log_group_run = log_group.max(hi_bit - lo_bit + 1).min(log_h);
        let log_stride = lo_bit.min(log_h - log_group_run);
        let pass_ingest = if start == 0 { ingest } else { None };
        let pass_scale = scale.filter(|_| end == layers.len());
        debug_span!(
            "fused_layers",
            layers = end - start,
            log_group_run,
            log_stride
        )
        .in_scope(|| {
            par_group_pass(
                base,
                h,
                width,
                &layers[start..end],
                log_group_run,
                log_stride,
                pass_ingest,
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
///
/// See [`cfft_layers`] for the `ingest` and `scale` semantics and the safety contract.
#[allow(clippy::too_many_arguments)]
fn par_group_pass<F: Field, B: Butterfly<F>, M: Matrix<F>>(
    base: *mut F,
    h: usize,
    width: usize,
    layers: &[CfftLayer<B>],
    log_group: usize,
    log_stride: usize,
    ingest: Option<&M>,
    scale: Option<F>,
) {
    let log_h = log2_strict_usize(h);
    let num_groups = h >> log_group;
    let base_addr = base as usize;
    let packed_scale = scale.map(F::Packing::from);
    (0..num_groups).into_par_iter().for_each(|g| {
        let base = base_addr as *mut F;
        let hi = g >> log_stride;
        let lo = g & ((1 << log_stride) - 1);
        let first_row = (hi << (log_group + log_stride)) | lo;
        if let Some(src) = ingest {
            for t in 0..1usize << log_group {
                let row = first_row + (t << log_stride);
                // SAFETY: `row < h` since `t < 2^log_group`, and this task owns the
                // destination row (the groups partition the rows), so the raw copy
                // neither races nor reads uninitialised destination memory.
                unsafe {
                    let src_row = src.row_slice_unchecked(row);
                    core::ptr::copy_nonoverlapping(src_row.as_ptr(), base.add(row * width), width);
                }
            }
        }
        for layer in layers {
            let e = flipped_bit(log_h, layer.blocks()) - log_stride;
            match layer {
                CfftLayer::Dup { .. } => {
                    for s in 0..1usize << (log_group - e - 1) {
                        for u in 0..1usize << e {
                            let row_lo = first_row + (((s << (e + 1)) | u) << log_stride);
                            let row_hi = row_lo + (1 << (e + log_stride));
                            // SAFETY: row ownership and non-aliasing as for the butterfly
                            // case below. The copy goes through `MaybeUninit`, so it is
                            // sound even while either row is still uninitialised (a
                            // garbage copy is later overwritten by the duplication layer
                            // of the destination row's highest flipped bit).
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    base.add(row_lo * width).cast::<MaybeUninit<F>>(),
                                    base.add(row_hi * width).cast::<MaybeUninit<F>>(),
                                    width,
                                );
                            }
                        }
                    }
                }
                CfftLayer::Butterflies(ts) => {
                    let slice_len = 1 << (log_group - e - 1);
                    let slice = &ts[hi * slice_len..][..slice_len];
                    if log_stride == 0 {
                        // The `2^e` row pairs sharing the twiddle `t` span two contiguous
                        // row blocks, so they merge into a single call: one twiddle
                        // broadcast and one long inner loop instead of one per row pair.
                        for (s, &t) in slice.iter().enumerate() {
                            let row_lo = first_row + (s << (e + 1));
                            let row_hi = row_lo + (1 << e);
                            let len = width << e;
                            // SAFETY: every row index decomposes uniquely as
                            // `hi << (log_group + log_stride) | t << log_stride | lo`, so the
                            // task for group `g = (hi, lo)` is the only one touching its rows,
                            // and within a layer each row appears in exactly one butterfly, so
                            // the two blocks never alias. All indices stay below `h` since
                            // `t < 2^log_group`.
                            let (block_lo, block_hi) = unsafe {
                                (
                                    core::slice::from_raw_parts_mut(base.add(row_lo * width), len),
                                    core::slice::from_raw_parts_mut(base.add(row_hi * width), len),
                                )
                            };
                            t.apply_to_rows(block_lo, block_hi);
                        }
                    } else {
                        for (s, &t) in slice.iter().enumerate() {
                            for u in 0..1usize << e {
                                let row_lo = first_row + (((s << (e + 1)) | u) << log_stride);
                                let row_hi = row_lo + (1 << (e + log_stride));
                                // SAFETY: as in the contiguous case above; `row_lo` and
                                // `row_hi` never alias and stay below `h`.
                                let (row_lo, row_hi) = unsafe {
                                    (
                                        core::slice::from_raw_parts_mut(
                                            base.add(row_lo * width),
                                            width,
                                        ),
                                        core::slice::from_raw_parts_mut(
                                            base.add(row_hi * width),
                                            width,
                                        ),
                                    )
                                };
                                t.apply_to_rows(row_lo, row_hi);
                            }
                        }
                    }
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

    /// The first `2^(log_n - b)` rows of a CFFT-ordered matrix over a domain `D` are the
    /// CFFT-ordered matrix over the twin-coset `CircleDomain::new(log_n - b, D.shift)`, and a
    /// polynomial of degree below that size is determined by its values there. Out-of-domain
    /// evaluation can therefore work on the prefix alone.
    #[test]
    fn eval_at_point_on_subdomain_prefix_matches_full() {
        let mut rng = SmallRng::seed_from_u64(1);
        for (log_n, width, log_blowup) in iproduct!(2..8, [1, 4, 11], [1, 2]) {
            let lde_domain = CircleDomain::standard(log_n + log_blowup);
            let lde = CircleEvaluations::<F>::from_natural_order(
                CircleDomain::standard(log_n),
                RowMajorMatrix::rand(&mut rng, 1 << log_n, width),
            )
            .extrapolate(lde_domain);

            let sub_domain = CircleDomain::new(log_n, lde_domain.shift);
            // The prefix rows are the subdomain's CFFT order: the same selection applies to
            // the domain points.
            assert_eq!(
                cfft_permute_slice(&sub_domain.points().collect_vec()),
                cfft_permute_slice(&lde_domain.points().collect_vec())[..1 << log_n],
            );

            let zeta = Point::<EF>::from_projective_line(rng.random());
            let full = lde.evaluate_at_point(zeta);
            let prefix = lde.values.split_rows(1 << log_n).0;
            let sub_evals = CircleEvaluations::from_cfft_order(sub_domain, prefix);
            assert_eq!(sub_evals.evaluate_at_point(zeta), full);
        }
    }
}
