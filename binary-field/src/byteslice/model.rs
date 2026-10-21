//! A scalar stand-in for the sliced register, so every host tests the sliced kernels.

use super::lanes::SliceRegister;

/// One register of bytes, held in a plain array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Model([u8; 64]);

impl Model {
    /// Quadword `i`, little-endian.
    const fn qword(&self, i: usize) -> u64 {
        u64::from_le_bytes(self.0.as_chunks::<8>().0[i])
    }

    /// The register of eight quadwords.
    fn from_qwords(qwords: [u64; 8]) -> Self {
        Self(core::array::from_fn(|i| qwords[i / 8].to_le_bytes()[i % 8]))
    }
}

/// The image of one byte under one block, output bit `k` reading block byte `7 - k`.
fn apply(matrix: u64, input: u8) -> u8 {
    let rows = matrix.to_le_bytes();
    (0..8).fold(0, |out, k| {
        out | (u8::from((rows[7 - k] & input).count_ones() % 2 == 1) << k)
    })
}

impl SliceRegister for Model {
    fn zero() -> Self {
        Self([0; 64])
    }

    fn splat(qword: u64) -> Self {
        Self::from_qwords([qword; 8])
    }

    unsafe fn load(from: *const u8) -> Self {
        // SAFETY: readability for one whole register is the caller's obligation.
        Self(core::array::from_fn(|i| unsafe { *from.add(i) }))
    }

    unsafe fn store(to: *mut u8, value: Self) {
        for (i, byte) in value.0.into_iter().enumerate() {
            // SAFETY: writability for one whole register is the caller's obligation.
            unsafe { *to.add(i) = byte };
        }
    }

    fn prefetch(_at: *const u8) {}

    fn xor(self, other: Self) -> Self {
        Self(core::array::from_fn(|i| self.0[i] ^ other.0[i]))
    }

    fn xor3(self, a: Self, b: Self) -> Self {
        self.xor(a).xor(b)
    }

    fn permute(self, index: Self) -> Self {
        Self(index.0.map(|j| self.0[usize::from(j % 64)]))
    }

    fn permute2(self, index: Self, high: Self) -> Self {
        Self(index.0.map(|j| {
            let source = if j & 64 == 0 { &self } else { &high };
            source.0[usize::from(j % 64)]
        }))
    }

    fn permute_merge(self, mask: u64, index: Self, source: Self) -> Self {
        Self(core::array::from_fn(|i| {
            if mask >> i & 1 == 1 {
                source.0[usize::from(index.0[i] % 64)]
            } else {
                self.0[i]
            }
        }))
    }

    fn permute2_qwords(self, index: Self, high: Self) -> Self {
        Self::from_qwords(core::array::from_fn(|i| {
            let j = index.qword(i) as usize;
            let source = if j & 8 == 0 { &self } else { &high };
            source.qword(j % 8)
        }))
    }

    fn permute_merge_qwords(self, mask: u8, index: Self, source: Self) -> Self {
        Self::from_qwords(core::array::from_fn(|i| {
            if mask >> i & 1 == 1 {
                source.qword(index.qword(i) as usize % 8)
            } else {
                self.qword(i)
            }
        }))
    }

    fn affine(self, matrices: Self) -> Self {
        Self(core::array::from_fn(|i| {
            apply(matrices.qword(i / 8), self.0[i])
        }))
    }

    fn transpose_qwords(rows: [Self; 8]) -> [Self; 8] {
        core::array::from_fn(|g| Self::from_qwords(core::array::from_fn(|r| rows[r].qword(g))))
    }
}
