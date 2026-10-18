use core::borrow::{Borrow, BorrowMut};
use core::mem::size_of;
#[cfg(test)]
use core::mem::transmute;

#[cfg(test)]
use p3_util::indices_arr;

use crate::constants::R;
use crate::{NUM_ROUNDS, U64_LIMBS};

/// Note: The ordering of each array is based on the input mapping. As the spec says,
///
/// > The mapping between the bits of s and those of a is `s[w(5y + x) + z] = a[x][y][z]`.
///
/// Thus, for example, `a_prime` is stored in `y, x, z` order. This departs from the more common
/// convention of `x, y, z` order, but it has the benefit that input lists map to AIR columns in a
/// nicer way.
#[derive(Debug)]
#[repr(C)]
pub struct KeccakCols<T> {
    /// The `i`th value is set to 1 if we are in the `i`th round, otherwise 0.
    pub step_flags: [T; NUM_ROUNDS],

    /// A register which indicates if a row should be exported, i.e. included in a multiset equality
    /// argument. Should be 1 only for certain rows which are final steps, i.e. with
    /// `step_flags[23] = 1`.
    pub export: T,

    /// Permutation inputs, stored in y-major order.
    pub preimage: [[[T; U64_LIMBS]; 5]; 5],

    pub a: [[[T; U64_LIMBS]; 5]; 5],

    /// ```ignore
    /// C[x] = xor(A[x, 0], A[x, 1], A[x, 2], A[x, 3], A[x, 4])
    /// ```
    pub c: [[T; 64]; 5],

    /// ```ignore
    /// C'[x, z] = xor(C[x, z], C[x - 1, z], C[x + 1, z - 1])
    /// ```
    pub c_prime: [[T; 64]; 5],

    // Note: D is inlined, not stored in the witness.
    /// ```ignore
    /// A'[x, y] = xor(A[x, y], D[x])
    ///          = xor(A[x, y], C[x - 1], ROT(C[x + 1], 1))
    /// ```
    pub a_prime: [[[T; 64]; 5]; 5],

    /// ```ignore
    /// A''[x, y] = xor(B[x, y], andn(B[x + 1, y], B[x + 2, y])).
    /// ```
    pub a_prime_prime: [[[T; U64_LIMBS]; 5]; 5],

    /// The bits of `A''[0, 0]`.
    pub a_prime_prime_0_0_bits: [T; 64],

    /// ```ignore
    /// A'''[0, 0, z] = A''[0, 0, z] ^ RC[k, z]
    /// ```
    pub a_prime_prime_prime_0_0_limbs: [T; U64_LIMBS],
}

impl<T: Copy> KeccakCols<T> {
    pub fn b(&self, x: usize, y: usize, z: usize) -> T {
        debug_assert!(x < 5);
        debug_assert!(y < 5);
        debug_assert!(z < 64);

        // B is just a rotation of A', so these are aliases for A' registers.
        // From the spec,
        //     B[y, (2x + 3y) % 5] = ROT(A'[x, y], r[x, y])
        // So,
        //     B[x, y] = f((x + 3y) % 5, x)
        // where f(a, b) = ROT(A'[a, b], r[a, b])
        let a = (x + 3 * y) % 5;
        let b = x;
        let rot = R[a][b] as usize;
        self.a_prime[b][a][(z + 64 - rot) % 64]
    }

    pub fn a_prime_prime_prime(&self, y: usize, x: usize, limb: usize) -> T {
        debug_assert!(y < 5);
        debug_assert!(x < 5);
        debug_assert!(limb < U64_LIMBS);

        if y == 0 && x == 0 {
            self.a_prime_prime_prime_0_0_limbs[limb]
        } else {
            self.a_prime_prime[y][x][limb]
        }
    }
}

pub const NUM_KECCAK_COLS: usize = size_of::<KeccakCols<u8>>();

#[cfg(test)]
pub(crate) const KECCAK_COL_MAP: KeccakCols<usize> = make_col_map();

#[cfg(test)]
const fn make_col_map() -> KeccakCols<usize> {
    unsafe { transmute(indices_arr::<NUM_KECCAK_COLS>()) }
}

impl<T> Borrow<KeccakCols<T>> for [T] {
    fn borrow(&self) -> &KeccakCols<T> {
        debug_assert_eq!(self.len(), NUM_KECCAK_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<KeccakCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T> BorrowMut<KeccakCols<T>> for [T] {
    fn borrow_mut(&mut self) -> &mut KeccakCols<T> {
        debug_assert_eq!(self.len(), NUM_KECCAK_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<KeccakCols<T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
