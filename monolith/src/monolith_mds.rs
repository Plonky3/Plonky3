use p3_field::PrimeField32;
use p3_symmetric::mds::{MDSPermutation, NaiveMDSMatrix};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128, Shake128Reader,
};

pub fn monolith_mds<F: PrimeField32, const WIDTH: usize>(init_string: &str, num_rounds: usize) -> Box<dyn MDSPermutation<F, WIDTH>> {
    let matrix = if WIDTH == 16 {
        let row = [
            61402, 17845, 26798, 59689, 12021, 40901, 41351, 27521, 56951, 12034, 53865, 43244,
            7454, 33823, 28750, 1108,
        ];
        circulant_matrix(row.as_ref().try_into().unwrap())
    } else {
        let mut shake = Shake128::default();
        shake.update(init_string.as_bytes());
        shake.update(&[WIDTH as u8, num_rounds as u8]);
        shake.update(&F::ORDER_U32.to_le_bytes());
        shake.update(&[16, 15]);
        shake.update("MDS".as_bytes());
        let mut shake_finalized = shake.finalize_xof();
        cauchy_mds_matrix(&mut shake_finalized)
    };
    Box::new(NaiveMDSMatrix::new(matrix))
}

fn circulant_matrix<F: PrimeField32, const WIDTH: usize>(row: &[u64; WIDTH]) -> [[F; WIDTH]; WIDTH] {
    let mut mat = [[F::ZERO; WIDTH]; WIDTH];
    let mut rot: Vec<F> = row.iter().map(|i| F::from_canonical_u64(*i)).collect();
    mat[0].copy_from_slice(&rot);
    for row in mat.iter_mut().skip(1) {
        rot.rotate_right(1);
        row.copy_from_slice(&rot);
    }
    mat
}

fn cauchy_mds_matrix<F: PrimeField32, const WIDTH: usize>(shake: &mut Shake128Reader) -> [[F; WIDTH]; WIDTH] {
    let mut p = F::ORDER_U32;
    let mut tmp = 0;
    while p != 0 {
        tmp += 1;
        p >>= 1;
    }
    let x_mask = (1 << (tmp - 7 - 2)) - 1;
    let y_mask = ((1 << tmp) - 1) >> 2;

    let mut res = [[F::ZERO; WIDTH]; WIDTH];

    let y = get_random_y_i::<F, WIDTH>(shake, x_mask, y_mask);
    let mut x = y.to_owned();
    x.iter_mut().for_each(|x_i| *x_i &= x_mask);

    for (i, x_i) in x.iter().enumerate() {
        for (j, yj) in y.iter().enumerate() {
            res[i][j] = F::from_canonical_u32(x_i + yj).inverse();
        }
    }

    res
}

fn get_random_y_i<F: PrimeField32, const WIDTH: usize>(shake: &mut Shake128Reader, x_mask: u32, y_mask: u32) -> [u32; WIDTH] {
    let mut res = [0; WIDTH];
    for i in 0..WIDTH {
        let mut valid = false;
        while !valid {
            let mut rand = [0u8; 4];
            shake.read(&mut rand);

            let y_i = u32::from_le_bytes(rand) & y_mask;
            
            // check distinct x_i
            let x_i = y_i & x_mask;
            valid = true;
            for r in res.iter().take(i) {
                if r & x_mask == x_i {
                    valid = false;
                    break;
                }
            }
            if valid {
                res[i] = y_i;
            }
        }
    }

    res
}