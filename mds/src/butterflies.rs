use p3_field::FieldAlgebra;

/// DIT butterfly.
#[inline]
pub(crate) fn dit_butterfly<FA: FieldAlgebra, const N: usize>(
    values: &mut [FA; N],
    idx_1: usize,
    idx_2: usize,
    twiddle: FA::F,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone() * twiddle;
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = val_1 - val_2;
}

/// DIF butterfly.
#[inline]
pub(crate) fn dif_butterfly<FA: FieldAlgebra, const N: usize>(
    values: &mut [FA; N],
    idx_1: usize,
    idx_2: usize,
    twiddle: FA::F,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone();
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = (val_1 - val_2) * twiddle;
}

/// Butterfly with twiddle factor 1 (works in either DIT or DIF).
#[inline]
pub(crate) fn twiddle_free_butterfly<FA: FieldAlgebra, const N: usize>(
    values: &mut [FA; N],
    idx_1: usize,
    idx_2: usize,
) {
    let val_1 = values[idx_1].clone();
    let val_2 = values[idx_2].clone();
    values[idx_1] = val_1.clone() + val_2.clone();
    values[idx_2] = val_1 - val_2;
}
