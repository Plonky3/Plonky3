//! The convert byte-sliced against GFNI matrices, the eq weight baked into them per window.

use p3_binary_field::Poly192;

use crate::byteslice::{Native, OUT_BYTES, SliceRegister, plane_matrices, store_f192};
use crate::skip::{SKIP_SIZE, phi8};
use crate::zerocheck::fixed::{MEDIUM_VALUES, gamma_powers};
use crate::zerocheck::planar::F192x8;

/// `g^b` per medium position in coefficient planes, and `phi8` of the unit bytes.
pub(crate) struct ConvertTables {
    /// `g^b`, eight positions a set of planes.
    gammas: [F192x8; 2],
    /// `phi8(2^s)` in lane `s`.
    units: Native,
}

impl ConvertTables {
    /// The constants, built once per sweep.
    pub(crate) fn new() -> Self {
        let g = gamma_powers();
        let units: [u64; 8] = core::array::from_fn(|s| phi8(1 << s).to_bits());
        let (halves, _) = g.as_chunks::<8>();
        Self {
            gammas: [F192x8::load(&halves[0]), F192x8::load(&halves[1])],
            // SAFETY: the array is eight quadwords, one register.
            units: unsafe { Native::load(units.as_ptr().cast()) },
        }
    }

    /// Medium position `b`'s matrices: the weights `(g^b eq_lo) phi8(2^s)`, for every `b`.
    #[inline(always)]
    fn matrices(&self, eq_lo: Poly192) -> [[u64; OUT_BYTES]; MEDIUM_VALUES] {
        let eq = F192x8::splat(eq_lo);
        let mut out = [[0u64; OUT_BYTES]; MEDIUM_VALUES];
        for (h, gammas) in self.gammas.iter().enumerate() {
            let t = gammas.mul(eq);
            for l in 0..8 {
                let weights = t.broadcast(l).mul_base(self.units);
                for (k, reg) in plane_matrices(&weights.0).into_iter().enumerate() {
                    // SAFETY: each store fills eight of the 24 matrices of one position.
                    unsafe { Native::store(out[8 * h + l][8 * k..].as_mut_ptr().cast(), reg) };
                }
            }
        }
        out
    }
}

/// One worker's byte-sliced sums for `a b` and for `c`.
pub(crate) struct Convert {
    /// Register `o` holds byte `o` of every lane's `a b` sum.
    ab: [Native; OUT_BYTES],
    /// Register `o` holds byte `o` of every lane's `c` sum.
    c: [Native; OUT_BYTES],
}

impl Convert {
    /// Empty sums.
    pub(crate) fn new() -> Self {
        Self {
            ab: [Native::zero(); OUT_BYTES],
            c: [Native::zero(); OUT_BYTES],
        }
    }

    /// Add one window's medium rows at weight `eq_lo`.
    #[inline(always)]
    pub(crate) fn accumulate(
        &mut self,
        tables: &ConvertTables,
        ab: &[[u8; 64]],
        c: &[[u8; 64]],
        eq_lo: Poly192,
    ) {
        // Medium position b's matrices, the eq weight baked in.
        let matrices = tables.matrices(eq_lo);

        // Eight output bytes at a time keep sixteen accumulators in registers.
        for g in 0..3 {
            let mut acc_ab: [Native; 8] = core::array::from_fn(|l| self.ab[8 * g + l]);
            let mut acc_c: [Native; 8] = core::array::from_fn(|l| self.c[8 * g + l]);
            for ((ab, c), m) in ab.iter().zip(c).zip(&matrices) {
                // SAFETY: each row is 64 bytes.
                let (xa, xc) = unsafe { (Native::load(ab.as_ptr()), Native::load(c.as_ptr())) };
                for l in 0..8 {
                    let a = Native::splat(m[8 * g + l]);
                    acc_ab[l] = acc_ab[l].xor(xa.affine(a));
                    acc_c[l] = acc_c[l].xor(xc.affine(a));
                }
            }
            self.ab[8 * g..8 * g + 8].copy_from_slice(&acc_ab);
            self.c[8 * g..8 * g + 8].copy_from_slice(&acc_c);
        }
    }

    /// The `a b` and `c` sums.
    pub(crate) fn values(&self) -> ([Poly192; SKIP_SIZE], [Poly192; SKIP_SIZE]) {
        let (mut ab, mut c) = (
            [Poly192::default(); SKIP_SIZE],
            [Poly192::default(); SKIP_SIZE],
        );
        store_f192(&self.ab, &mut ab);
        store_f192(&self.c, &mut c);
        (ab, c)
    }
}
