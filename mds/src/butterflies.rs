use p3_field::{Algebra, Field};

/// DIT butterfly.
#[inline]
pub(crate) fn dit_butterfly<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    idx_1: usize,
    idx_2: usize,
    twiddle: F,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone() * twiddle;
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = val_1 - val_2;
}

/// DIF butterfly.
#[inline]
pub(crate) fn dif_butterfly<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    idx_1: usize,
    idx_2: usize,
    twiddle: F,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone();
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = (val_1 - val_2) * twiddle;
}

/// Butterfly with twiddle factor 1 (works in either DIT or DIF).
#[inline]
pub(crate) fn twiddle_free_butterfly<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    idx_1: usize,
    idx_2: usize,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone();
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = val_1 - val_2;
}

/// One layer of a Bowers G network.
///
/// This is used in both `CosetMds` and `IntegratedCosetMds` implementations.
/// Equivalent to `bowers_g_t_layer` except for the butterfly type.
#[inline]
pub(crate) fn bowers_g_layer<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = N >> log_block_size;

    // Unroll first iteration with a twiddle factor of 1.
    for hi in 0..half_block_size {
        let lo = hi + half_block_size;
        twiddle_free_butterfly(values, hi, lo);
    }

    for (block, &twiddle) in (1..num_blocks).zip(&twiddles[1..]) {
        let block_start = block << log_block_size;
        for hi in block_start..block_start + half_block_size {
            let lo = hi + half_block_size;
            dif_butterfly(values, hi, lo, twiddle);
        }
    }
}

/// One layer of a Bowers G^T network.
///
/// This is used in the `CosetMds` implementation.
/// Equivalent to `bowers_g_layer` except for the butterfly type.
///
/// This version skips the first block (which has twiddle factor 1) in the twiddle iteration.
#[inline]
pub(crate) fn bowers_g_t_layer<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = N >> log_block_size;

    // Unroll first iteration with a twiddle factor of 1.
    for hi in 0..half_block_size {
        let lo = hi + half_block_size;
        twiddle_free_butterfly(values, hi, lo);
    }

    for (block, &twiddle) in (1..num_blocks).zip(&twiddles[1..]) {
        let block_start = block << log_block_size;
        for hi in block_start..block_start + half_block_size {
            let lo = hi + half_block_size;
            dit_butterfly(values, hi, lo, twiddle);
        }
    }
}

/// One layer of a Bowers G^T network with integrated twiddles.
///
/// This is used in the `IntegratedCosetMds` implementation where twiddle factors
/// are pre-computed for each layer and include the coset shifts.
///
/// Unlike `bowers_g_t_layer`, this version applies twiddles to all blocks starting from block 0.
#[inline]
pub(crate) fn bowers_g_t_layer_integrated<F: Field, A: Algebra<F>, const N: usize>(
    values: &mut [A; N],
    log_half_block_size: usize,
    twiddles: &[F],
) {
    let log_block_size = log_half_block_size + 1;
    let half_block_size = 1 << log_half_block_size;
    let num_blocks = N >> log_block_size;

    for (block, &twiddle) in (0..num_blocks).zip(twiddles) {
        let block_start = block << log_block_size;
        for hi in block_start..block_start + half_block_size {
            let lo = hi + half_block_size;
            dit_butterfly(values, hi, lo, twiddle);
        }
    }
}
