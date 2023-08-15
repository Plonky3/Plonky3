use p3_field::AbstractField;

// NB: These four are MDS for M31, BabyBear and Goldilocks
//const MATRIX_CIRC_MDS_8_2EXP: [u64; 8] = [1, 1, 2, 1, 8, 32, 4, 256];
pub(crate) const MATRIX_CIRC_MDS_8_SML: [u64; 8] = [4, 1, 2, 9, 10, 5, 1, 1];

//const MATRIX_CIRC_MDS_12_2EXP: [u64; 12] = [1, 1, 2, 1, 8, 32, 2, 256, 4096, 8, 65536, 1024];
pub(crate) const MATRIX_CIRC_MDS_12_SML: [u64; 12] = [9, 7, 4, 1, 16, 2, 256, 128, 3, 32, 1, 1];

fn dot_vec<F: AbstractField, const N: usize>(u: &[F; N], v: &[F; N]) -> F {
    u.iter().zip(v).map(|(x, y)| x.clone() * y.clone()).sum()
}

// F: Field ==> need dot_vec for each field
pub(crate) fn apply_circulant<F: AbstractField, const N: usize>(
    circ_matrix: &[u64; N],
    input: [F; N],
) -> [F; N] {
    let mut matrix: [F; N] = circ_matrix.map(F::from_canonical_u64);

    let mut output = [F::ZERO; N];
    for i in 0..N - 1 {
        output[i] = dot_vec(&input, &matrix);
        matrix.rotate_right(1);
    }
    output[N - 1] = dot_vec(&input, &matrix);
    output
}
