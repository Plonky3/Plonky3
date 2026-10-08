// CREDIT: https://github.com/succinctlabs/flock (flock-core), MIT OR Apache-2.0.
//! §2.1 single-table collapse of the LDE matrix `M = fwd_NTT_Λ ∘ inv_NTT_S`.
//!
//! Background: the URM round-1 needs to map each `ell`-bit row of the boolean
//! witness (packed as `n_chunks = ell/8` bytes) to `ell` evaluations on the
//! NTT domain `Λ`. The naive way computes inv_NTT on S then fwd_NTT on Λ for
//! every row, which is too slow.
//!
//! The optimization (§2.1 of the paper): `M = α · M̃` with `M̃` Cauchy and `α`
//! a scalar. The columns of `M` satisfy a XOR-shift relation, so the `n_chunks`
//! per-byte sub-tables collapse to a single 256-row base table `T_0`:
//!
//!   M[i', 8b + t]  =  T_0[bit-t-mask(8b+t)][i' ⊕ 8b]
//!
//! Per-byte-chunk b contributes `π_b(T_0[byte_b])` to the output, where
//! `π_b(i') = i' ⊕ 8b`.
//!
//! Storage: 256 × ell bytes (16 KB at k=6, 32 KB at k=7), which fits in L1.
//! Lookups per row: n_chunks (= ell/8), each load is `ell` contiguous bytes.

use alloc::vec;
use alloc::vec::Vec;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

use p3_binary_field::Rijndael8b as F8;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};

use crate::BasisNtt;

/// Byte-table evaluation extension over the AES field.
#[derive(Clone, Debug)]
pub struct RijndaelLde {
    /// The base-two logarithm of the row length.
    k: usize,
    /// The number of evaluations per row.
    ell: usize,
    /// The number of packed Boolean bytes per input row.
    n_chunks: usize,
    /// `data[w * ell .. (w+1) * ell]` = `T_0[w]`, the XOR-sum of columns of `M`
    /// indexed by the set bits of `w`.
    data: Vec<F8>,
    /// The interpolation plan on the source coset.
    source: BasisNtt<F8>,
    /// The evaluation plan on the destination coset.
    target: BasisNtt<F8>,
}

impl RijndaelLde {
    /// Build the byte-packed extension table between two polynomial-coordinate cosets.
    ///
    /// # Panics
    /// Panics unless the dimension is between three and seven and both cosets have the same subspace.
    pub fn new(k: usize, source_shift: F8, target_shift: F8) -> Self {
        assert!(
            (3..=7).contains(&k),
            "byte table dimension must be between three and seven"
        );
        let ntt_s = BasisNtt::polynomial(k, source_shift);
        let ntt_l = BasisNtt::polynomial(k, target_shift);
        let ell = 1usize << k;
        assert!(ell >= 8, "ell must be ≥ 8 so n_chunks ≥ 1");
        let n_chunks = ell / 8;
        assert!(
            n_chunks <= 16,
            "n_chunks must fit the i'/chunk XOR encoding"
        );

        let mut data = vec![F8::ZERO; 256 * ell];

        // Compute the 8 unit-column images cols[t] = fwd_NTT_Λ ∘ inv_NTT_S (e_t)
        // for t ∈ 0..8. The remaining columns of M are XOR-shifted versions.
        let mut tmp = vec![F8::ZERO; ell];
        let mut cols: Vec<Vec<F8>> = Vec::with_capacity(8);
        for t in 0..8 {
            tmp.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp[t] = F8::ONE;
            ntt_s.inverse(&mut tmp);
            ntt_l.forward(&mut tmp);
            cols.push(tmp.clone());
        }

        // T_0[0] already zero. T_0[2^t] = cols[t]. Then for non-power-of-two w,
        // T_0[w] = T_0[w ^ lo_bit] ⊕ T_0[lo_bit]; this builds all 256 entries
        // with one XOR per entry.
        for (t, col) in cols.iter().enumerate() {
            let entry_start = (1usize << t) * ell;
            data[entry_start..entry_start + ell].copy_from_slice(col);
        }
        for w in 3usize..256 {
            if (w & (w - 1)) == 0 {
                continue; // skip powers of 2 (already written)
            }
            let lo_bit = 1usize << w.trailing_zeros();
            let parent = w ^ lo_bit;
            // Borrow-checker friendly: read parent + bit_v slices, then write entry.
            let (parent_off, bit_off, entry_off) = (parent * ell, lo_bit * ell, w * ell);
            for i in 0..ell {
                let v = data[parent_off + i] + data[bit_off + i];
                data[entry_off + i] = v;
            }
        }

        Self {
            k,
            ell,
            n_chunks,
            data,
            source: ntt_s,
            target: ntt_l,
        }
    }

    /// The base-two logarithm of the evaluation count.
    pub const fn log_domain_size(&self) -> usize {
        self.k
    }

    /// The number of evaluations produced by one Boolean input row.
    pub const fn row_len(&self) -> usize {
        self.ell
    }

    /// The number of packed Boolean bytes consumed by one row.
    pub const fn input_bytes(&self) -> usize {
        self.n_chunks
    }

    /// The interpolation plan for the source evaluations.
    pub const fn source(&self) -> &BasisNtt<F8> {
        &self.source
    }

    /// The evaluation plan for the destination domain.
    pub const fn target(&self) -> &BasisNtt<F8> {
        &self.target
    }

    /// Apply M to a single byte-packed row, in place.
    /// `bytes` is `n_chunks` bytes (the LCH-coefficient bits of the row);
    /// `out` will be filled with the `ell` evaluations on Λ.
    ///
    /// Dispatches: NEON on aarch64 / SSE2 on x86_64 when `ell ≥ 16`, which
    /// covers every supported arch at the protocol size (k_skip=6 ⇒ ell=64).
    /// The scalar arm is reachable only at `ell < 16`, i.e. k=3, which occurs
    /// only in tests.
    #[inline]
    pub fn apply(&self, bytes: &[u8], out: &mut [F8]) {
        #[cfg(target_arch = "aarch64")]
        if self.ell == 64 {
            self.apply_neon64(bytes, out);
            return;
        }
        #[cfg(target_arch = "aarch64")]
        if self.ell >= 16 {
            // SAFETY: aarch64 statically guarantees NEON; ell ≥ 16 ⇒ at least
            // one 128-bit chunk; method validates slice lengths.
            unsafe { self.apply_v128::<Neon>(bytes, out) };
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        if self.ell == 64 {
            // SAFETY: avx512f is enabled at compile time; the method validates
            // slice lengths and requires exactly this `ell`.
            unsafe { self.apply_avx512(bytes, out) };
            return;
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx2",
            not(target_feature = "avx512f")
        ))]
        if self.ell == 64 {
            // SAFETY: avx2 is enabled at compile time; the method validates
            // slice lengths and requires exactly this `ell`.
            unsafe { self.apply_avx2(bytes, out) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.ell >= 16 {
            // SAFETY: x86_64 statically guarantees SSE2; ell ≥ 16 ⇒ at least
            // one 128-bit chunk; method validates slice lengths.
            unsafe { self.apply_v128::<Sse2>(bytes, out) };
            return;
        }
        self.apply_scalar(bytes, out);
    }

    /// Extend matching Boolean rows, then sum their weighted pointwise products.
    ///
    /// Inputs are row-major packed bits, with `input_bytes()` bytes per row.
    /// For output point `j`, the result is `sum_r weights[r] * LDE(a[r])[j] * LDE(b[r])[j]`.
    /// No intermediate columns or heap allocations are exposed to the caller.
    ///
    /// # Panics
    /// Panics unless both inputs contain one complete row per weight and the
    /// output has `row_len()` elements.
    #[inline]
    pub fn weighted_product_sum(&self, a: &[u8], b: &[u8], weights: &[F8], out: &mut [F8]) {
        assert_eq!(a.len(), b.len(), "input matrices differ in size");
        assert_eq!(a.len() % self.n_chunks, 0, "incomplete Boolean row");
        assert_eq!(
            a.len() / self.n_chunks,
            weights.len(),
            "one weight per row required"
        );
        assert_eq!(out.len(), self.ell, "output length differs from domain");
        #[cfg(all(
            target_arch = "aarch64",
            target_endian = "little",
            target_feature = "aes"
        ))]
        if self.ell == 64 {
            if weights.len() == 8
                && weights
                    .iter()
                    .enumerate()
                    .all(|(i, weight)| weight.to_byte() == 1 << i)
            {
                self.power_product_sum_neon64(a, b, out);
                return;
            }
            self.weighted_product_sum_neon64(a, b, weights, out);
            return;
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "gfni",
            target_feature = "avx512f",
            target_feature = "avx512bw"
        ))]
        if self.ell == 64 {
            self.weighted_product_sum_zmm(a, b, weights, out);
            return;
        }
        type Packing = <F8 as Field>::Packing;
        out.fill(F8::ZERO);
        let mut a_col = [F8::ZERO; 128];
        let mut b_col = [F8::ZERO; 128];
        for ((a, b), &weight) in a
            .chunks_exact(self.n_chunks)
            .zip(b.chunks_exact(self.n_chunks))
            .zip(weights)
        {
            self.apply(a, &mut a_col[..self.ell]);
            self.apply(b, &mut b_col[..self.ell]);
            let done = self.ell / Packing::WIDTH * Packing::WIDTH;
            for start in (0..done).step_by(Packing::WIDTH) {
                let a = *Packing::from_slice(&a_col[start..start + Packing::WIDTH]);
                let b = *Packing::from_slice(&b_col[start..start + Packing::WIDTH]);
                *Packing::from_slice_mut(&mut out[start..start + Packing::WIDTH]) +=
                    a * b * Packing::from(weight);
            }
            for j in done..self.ell {
                out[j] += a_col[j] * b_col[j] * weight;
            }
        }
    }

    /// Sum eight generator-weighted row products with only one final weight reduction.
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    ))]
    #[inline]
    fn power_product_sum_neon64(&self, a: &[u8], b: &[u8], out: &mut [F8]) {
        use p3_binary_field::{PackedRijndael8b, RijndaelPowerAccumulator};
        let mut sums = [RijndaelPowerAccumulator::<16>::new(); 4];
        macro_rules! add {
            ($power:literal) => {{
                let offset = 8 * $power;
                let a = self.lookup_neon64(&a[offset..offset + 8]);
                let b = self.lookup_neon64(&b[offset..offset + 8]);
                // SAFETY: both representations are sixteen unrestricted bytes per block.
                // PackedValue and the scalar's transparent representation guarantee this layout.
                let a: [PackedRijndael8b<16>; 4] = unsafe { core::mem::transmute(a) };
                let b: [PackedRijndael8b<16>; 4] = unsafe { core::mem::transmute(b) };
                for i in 0..4 {
                    sums[i].add::<$power>(a[i] * b[i]);
                }
            }};
        }
        add!(0);
        add!(1);
        add!(2);
        add!(3);
        add!(4);
        add!(5);
        add!(6);
        add!(7);
        for (chunk, sum) in out.as_chunks_mut::<16>().0.iter_mut().zip(sums) {
            chunk.copy_from_slice(sum.finish().as_slice());
        }
    }

    /// Keep the two extended rows and their weighted sum in ZMM registers.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[inline]
    fn weighted_product_sum_zmm(&self, a: &[u8], b: &[u8], weights: &[F8], out: &mut [F8]) {
        type Packing = <F8 as Field>::Packing;
        let mut sum = Packing::ZERO;
        for ((a, b), &weight) in a
            .as_chunks::<8>()
            .0
            .iter()
            .zip(b.as_chunks::<8>().0)
            .zip(weights)
        {
            // SAFETY: the compile-time feature gate enables AVX-512. PackedValue
            // represents sixty-four unrestricted scalar bytes, just as each ZMM does.
            let (a, b): (Packing, Packing) = unsafe {
                (
                    core::mem::transmute::<__m512i, Packing>(self.apply_zmm(a)),
                    core::mem::transmute::<__m512i, Packing>(self.apply_zmm(b)),
                )
            };
            sum += a * b * Packing::from(weight);
        }
        out.copy_from_slice(sum.as_slice());
    }

    /// Fuse extension lookup, packed multiplication, and the weighted sum on ARM.
    #[cfg(all(
        target_arch = "aarch64",
        target_endian = "little",
        target_feature = "aes"
    ))]
    #[inline]
    fn weighted_product_sum_neon64(&self, a: &[u8], b: &[u8], weights: &[F8], out: &mut [F8]) {
        type Packing = <F8 as Field>::Packing;
        let mut sums = [Packing::ZERO; 4];
        for ((a, b), &weight) in a
            .as_chunks::<8>()
            .0
            .iter()
            .zip(b.as_chunks::<8>().0)
            .zip(weights)
        {
            let a = self.lookup_neon64(a);
            let b = self.lookup_neon64(b);
            // SAFETY: PackedValue guarantees the packing represents sixteen scalar bytes.
            // F8 is transparent over u8 and every byte is a valid element. Both types
            // are sixteen bytes under this module's ARM AES feature gate.
            let a: [Packing; 4] = unsafe {
                core::mem::transmute::<[core::arch::aarch64::uint8x16_t; 4], [Packing; 4]>(a)
            };
            let b: [Packing; 4] = unsafe {
                core::mem::transmute::<[core::arch::aarch64::uint8x16_t; 4], [Packing; 4]>(b)
            };
            let weight = Packing::from(weight);
            sums = core::array::from_fn(|i| sums[i] + a[i] * b[i] * weight);
        }
        for (chunk, sum) in out.as_chunks_mut::<16>().0.iter_mut().zip(sums) {
            chunk.copy_from_slice(sum.as_slice());
        }
    }

    /// Extend a 64-bit Boolean row while keeping its four output vectors in registers.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    fn apply_neon64(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::aarch64::*;
        assert_eq!(out.len(), 64);
        let sums = self.lookup_neon64(bytes);
        // SAFETY: the checked output holds four unaligned vector stores; F8 is a byte.
        unsafe {
            for (i, sum) in sums.into_iter().enumerate() {
                vst1q_u8(out.as_mut_ptr().cast::<u8>().add(16 * i), sum);
            }
        }
    }

    /// The four vectors of one Boolean extension, before any output store.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn lookup_neon64(&self, bytes: &[u8]) -> [core::arch::aarch64::uint8x16_t; 4] {
        use core::arch::aarch64::*;
        assert_eq!(bytes.len(), 8);
        assert_eq!(self.ell, 64);
        // SAFETY: NEON is part of the aarch64 baseline. Each table lookup covers
        // a complete 64-byte row inside the constructor's 256-row table.
        unsafe {
            let base = self.data.as_ptr().cast::<u8>();
            let row = base.add(bytes[0] as usize * 64);
            let mut sums: [uint8x16_t; 4] = core::array::from_fn(|i| vld1q_u8(row.add(16 * i)));
            for (b, &byte) in bytes.iter().enumerate().skip(1) {
                let row = base.add(byte as usize * 64);
                sums = core::array::from_fn(|i| {
                    let value = vld1q_u8(row.add(16 * (i ^ (b >> 1))));
                    let value = if b & 1 == 0 {
                        value
                    } else {
                        vextq_u8::<8>(value, value)
                    };
                    veorq_u8(sums[i], value)
                });
            }
            sums
        }
    }

    /// [`apply`](Self::apply) at the protocol's `ell = 64`, one register wide.
    ///
    /// # Safety
    /// Requires AVX-512F, and `self.ell` must be 64. The method validates slice
    /// lengths.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn apply_avx512(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(out.len(), self.ell);
        let bytes: &[u8; 8] = bytes.try_into().expect("8 bytes at ell = 64");
        // SAFETY: the single store covers exactly `out`.
        unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), self.apply_zmm(bytes)) };
    }

    /// The 64 evaluations of one 8-byte row as one ZMM, at `ell = 64`.
    ///
    /// Byte `b`'s row enters permuted by `i' ⊕ 8b`, an XOR of `b` on the qword index.
    /// The rows are summed as a tree, so each bit of `b` is one fixed shuffle:
    /// bit 0 swaps the qwords of each 128-bit lane, bits 1 and 2 swap lanes, and only those two cross a lane.
    ///
    /// # Panics
    /// Panics unless `self.ell` is 64.
    ///
    /// # Safety
    /// Requires AVX-512F, which the target enables wherever this is compiled.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    #[inline]
    #[target_feature(enable = "avx512f")]
    fn apply_zmm(&self, bytes: &[u8; 8]) -> __m512i {
        assert_eq!(self.ell, 64);
        let base = self.data.as_ptr().cast::<u8>();
        // SAFETY: every row offset is `byte * 64` into a `256 * 64` table.
        let row = |b: usize| unsafe { _mm512_loadu_si512(base.add(bytes[b] as usize * 64).cast()) };
        let swap_qwords = |v| _mm512_shuffle_epi32::<0x4E>(v);
        let swap_lanes = |v| _mm512_shuffle_i64x2::<0xB1>(v, v);
        let swap_halves = |v| _mm512_shuffle_i64x2::<0x4E>(v, v);
        let pair = |b: usize| _mm512_xor_si512(row(b), swap_qwords(row(b + 1)));
        let lo = _mm512_xor_si512(pair(0), swap_lanes(pair(2)));
        let hi = _mm512_xor_si512(pair(4), swap_lanes(pair(6)));
        _mm512_xor_si512(lo, swap_halves(hi))
    }

    /// [`apply`](Self::apply) at the protocol's `ell = 64`, two registers wide.
    ///
    /// The `i' ⊕ 8b` permutation is a qword-index XOR by `b`: bit 0 swaps the qwords of each lane, bit 1 the lanes, bit 2
    /// the registers.
    ///
    /// # Safety
    /// Requires AVX2, and `self.ell` must be 64. The method validates slice
    /// lengths.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[cfg_attr(target_feature = "avx512f", allow(dead_code))]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn apply_avx2(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(self.ell, 64);
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        // SAFETY: every row offset is `byte * 64` into a `256 * 64` table, and
        // the two stores cover exactly `out`.
        unsafe {
            let base = self.data.as_ptr().cast::<u8>();
            let row = |b: usize| {
                let p = base.add(bytes[b] as usize * 64);
                (
                    _mm256_loadu_si256(p.cast()),
                    _mm256_loadu_si256(p.add(32).cast()),
                )
            };
            let (mut lo, mut hi) = row(0);
            for b in 1..8 {
                let (mut l, mut h) = row(b);
                if b & 1 != 0 {
                    (l, h) = (
                        _mm256_shuffle_epi32::<0b01_00_11_10>(l),
                        _mm256_shuffle_epi32::<0b01_00_11_10>(h),
                    );
                }
                if b & 2 != 0 {
                    (l, h) = (
                        _mm256_permute4x64_epi64::<0b01_00_11_10>(l),
                        _mm256_permute4x64_epi64::<0b01_00_11_10>(h),
                    );
                }
                if b & 4 != 0 {
                    (l, h) = (h, l);
                }
                lo = _mm256_xor_si256(lo, l);
                hi = _mm256_xor_si256(hi, h);
            }
            let dst = out.as_mut_ptr().cast::<u8>();
            _mm256_storeu_si256(dst.cast(), lo);
            _mm256_storeu_si256(dst.add(32).cast(), hi);
        }
    }

    /// Extend one row using scalar byte lookups.
    fn apply_scalar(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        out.iter_mut().for_each(|x| *x = F8::ZERO);
        for (b, &byte_b) in bytes.iter().enumerate() {
            let row_off = byte_b as usize * self.ell;
            let row = &self.data[row_off..row_off + self.ell];
            let shift = 8 * b;
            for i in 0..self.ell {
                out[i] += row[i ^ shift];
            }
        }
    }

    /// SIMD variant of `apply`, operating in 16-byte chunks.
    ///
    /// For each output chunk `c ∈ 0..ell/16`:
    ///   * `b = 0`: straight 16-byte copy from `row0[c]`
    ///   * `b ≥ 1`: load `row_b[c ⊕ (b>>1)]`, half-swap if `b` is odd, XOR
    ///
    /// The `b>>1` chunk-XOR and the `8 · b` within-chunk shift together
    /// implement the `π_b(i') = i' ⊕ 8b` permutation that the §2.1 collapse
    /// requires.
    ///
    /// This is the URM round-1 inner loop and it must inline into flock's
    /// `shift_reduce_inner_ab_gfni`, hence `#[inline(always)]` here and on
    /// every [`Vec128`] method.
    ///
    /// # Safety
    /// `V`'s target features must be available (statically true at the
    /// dispatch site for both NEON on aarch64 and SSE2 on x86_64). The method
    /// validates slice lengths.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    #[inline(always)]
    unsafe fn apply_v128<V: Vec128>(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        let n128 = self.ell / 16; // 4 for ell = 64
        let base = self.data.as_ptr() as *const u8;
        let out_ptr = out.as_mut_ptr() as *mut u8;

        // SAFETY: the caller guarantees `V`'s features. `data` is `256 * ell` bytes and each row index is a byte, so
        // row `bytes[b] * ell` has `ell` bytes; `ell` is a power of two, `n128 = ell / 16` and
        // `b >> 1 <= (n_chunks - 1) / 2 < n128`, so every chunk index `c ^ (b >> 1)` stays below `n128`, and each
        // 16-byte access lies inside its row or inside `out`, whose length is asserted to be `ell`.
        unsafe {
            // b = 0: identity permutation, a straight copy from row 0.
            let row0 = base.add(bytes[0] as usize * self.ell);
            for c in 0..n128 {
                V::store(out_ptr.add(c * 16), V::load(row0.add(c * 16)));
            }

            // b ≥ 1: XOR with table row[bytes[b]], permuted.
            for (b, &byte) in bytes.iter().enumerate().take(self.n_chunks).skip(1) {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(byte as usize * self.ell);
                if b_odd {
                    for c in 0..n128 {
                        let v = V::load(row_b.add((c ^ b_high) * 16)).swap64();
                        let dst = out_ptr.add(c * 16);
                        V::store(dst, V::load(dst).xor(v));
                    }
                } else {
                    for c in 0..n128 {
                        let v = V::load(row_b.add((c ^ b_high) * 16));
                        let dst = out_ptr.add(c * 16);
                        V::store(dst, V::load(dst).xor(v));
                    }
                }
            }
        }
    }
}

/// The four inlined 128-bit primitives used by `apply_v128`'s inner loop.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
trait Vec128: Copy {
    /// # Safety
    /// `p` must be readable for 16 bytes (alignment not required).
    unsafe fn load(p: *const u8) -> Self;
    /// # Safety
    /// `p` must be writable for 16 bytes (alignment not required).
    unsafe fn store(p: *mut u8, v: Self);
    fn xor(self, other: Self) -> Self;
    /// Swap the two 64-bit halves.
    fn swap64(self) -> Self;
}

#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy)]
struct Neon(core::arch::aarch64::uint8x16_t);

#[cfg(target_arch = "aarch64")]
impl Vec128 for Neon {
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: NEON is part of the aarch64 baseline, and the caller guarantees 16 readable bytes at `p`.
        Self(unsafe { core::arch::aarch64::vld1q_u8(p) })
    }
    #[inline(always)]
    unsafe fn store(p: *mut u8, v: Self) {
        // SAFETY: NEON is part of the aarch64 baseline, and the caller guarantees 16 writable bytes at `p`.
        unsafe { core::arch::aarch64::vst1q_u8(p, v.0) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        // SAFETY: NEON is part of the aarch64 baseline; registers only.
        Self(unsafe { core::arch::aarch64::veorq_u8(self.0, other.0) })
    }
    #[inline(always)]
    fn swap64(self) -> Self {
        // SAFETY: NEON is part of the aarch64 baseline; registers only.
        Self(unsafe { core::arch::aarch64::vextq_u8::<8>(self.0, self.0) })
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct Sse2(core::arch::x86_64::__m128i);

#[cfg(target_arch = "x86_64")]
impl Vec128 for Sse2 {
    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: SSE2 is part of the x86-64 baseline, and the caller guarantees 16 readable bytes at `p`;
        // the load is unaligned.
        Self(unsafe {
            core::arch::x86_64::_mm_loadu_si128(p as *const core::arch::x86_64::__m128i)
        })
    }
    #[inline(always)]
    unsafe fn store(p: *mut u8, v: Self) {
        // SAFETY: SSE2 is part of the x86-64 baseline, and the caller guarantees 16 writable bytes at `p`;
        // the store is unaligned.
        unsafe { core::arch::x86_64::_mm_storeu_si128(p as *mut core::arch::x86_64::__m128i, v.0) }
    }
    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        // SAFETY: SSE2 is part of the x86-64 baseline; registers only.
        unsafe { Self(core::arch::x86_64::_mm_xor_si128(self.0, other.0)) }
    }
    #[inline(always)]
    fn swap64(self) -> Self {
        // SAFETY: SSE2 is part of the x86-64 baseline; registers only.
        unsafe {
            Self(core::arch::x86_64::_mm_shuffle_epi32::<0b01_00_11_10>(
                self.0,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn check(k: usize, source: u8, target: u8, bytes: &[u8]) {
        let table = RijndaelLde::new(k, F8::from_byte(source), F8::from_byte(target));
        // The oracle interpolates every unpacked Boolean coordinate independently of the byte table.
        let mut expected: Vec<_> = bytes
            .iter()
            .flat_map(|&b| (0..8).map(move |bit| F8::from_byte((b >> bit) & 1)))
            .collect();
        table.source().inverse(&mut expected);
        table.target().forward(&mut expected);
        let mut scalar = vec![F8::ZERO; table.ell];
        let mut packed = scalar.clone();
        table.apply_scalar(bytes, &mut scalar);
        table.apply(bytes, &mut packed);
        assert_eq!(scalar, expected);
        assert_eq!(packed, expected);
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        if k == 6 {
            // SAFETY: AVX2 is statically enabled and the rows contain exactly 64 bytes.
            unsafe { table.apply_avx2(bytes, &mut packed) };
            assert_eq!(packed, expected);
        }
    }

    proptest! {
        #[test]
        fn weighted_sum_matches_independent_extensions(
            k in 3usize..=7, count in 0usize..=8,
            source in any::<u8>(), target in any::<u8>(),
            a in any::<[u8;128]>(), b in any::<[u8;128]>(), w in any::<[u8;8]>()
        ) {
            let table=RijndaelLde::new(k,F8::from_byte(source),F8::from_byte(target));
            let bytes=count*table.input_bytes();
            let weights=w.map(F8::from_byte);
            let guard=F8::from_byte(0xa5);
            let mut got=vec![guard;table.row_len()+2];
            table.weighted_product_sum(&a[..bytes],&b[..bytes],&weights[..count],&mut got[1..=table.row_len()]);
            let mut expected=vec![F8::ZERO;table.row_len()];
            for (row, &weight) in weights.iter().enumerate().take(count) {
                let extend=|data:&[u8]| {
                    let mut values:Vec<_>=data.iter().flat_map(|&byte|(0..8).map(move|bit|F8::from_byte((byte>>bit)&1))).collect();
                    table.source().inverse(&mut values);
                    table.target().forward(&mut values);
                    values
                };
                let start=row*table.input_bytes();
                let end=start+table.input_bytes();
                let (x,y)=(extend(&a[start..end]),extend(&b[start..end]));
                for j in 0..table.row_len(){expected[j]+=x[j]*y[j]*weight;}
            }
            prop_assert_eq!(&got[1..=table.row_len()],&expected);
            prop_assert_eq!(got[0],guard);
            prop_assert_eq!(got[table.row_len()+1],guard);
        }

        #[test]
        fn generator_weighted_rows_match_independent_extensions(
            source in any::<u8>(), target in any::<u8>(),
            a in any::<[u8;64]>(), b in any::<[u8;64]>()
        ) {
            let table = RijndaelLde::new(6, F8::from_byte(source), F8::from_byte(target));
            let weights = core::array::from_fn::<_,8,_>(|i| F8::from_byte(1 << i));
            let guard = F8::from_byte(0xa5);
            let mut got = [guard; 66];
            table.weighted_product_sum(&a, &b, &weights, &mut got[1..65]);
            let mut expected = [F8::ZERO;64];
            for row in 0..8 {
                let extend = |bytes:&[u8]| {
                    let mut values:Vec<_> = bytes.iter().flat_map(|&byte|(0..8).map(move|bit|F8::from_byte((byte>>bit)&1))).collect();
                    table.source().inverse(&mut values);
                    table.target().forward(&mut values);
                    values
                };
                let (x,y) = (extend(&a[8*row..8*row+8]), extend(&b[8*row..8*row+8]));
                for j in 0..64 { expected[j] += x[j]*y[j]*weights[row]; }
            }
            prop_assert_eq!(&got[1..65], &expected);
            prop_assert_eq!(got[0], guard);
            prop_assert_eq!(got[65], guard);
        }

        #[test]
        fn packed_extension_matches_interpolation(k in 3usize..=7, source in any::<u8>(), target in any::<u8>(), bytes in prop::array::uniform16(any::<u8>())) {
            check(k, source, target, &bytes[..(1 << k)/8]);
        }
    }

    #[test]
    fn zero_and_every_unit_byte_match_interpolation() {
        // These vectors pin the XOR-column construction, including the zero lookup.
        for k in 3..=7 {
            let mut bytes = vec![0; (1 << k) / 8];
            check(k, 0, 1 << k, &bytes);
            for position in 0..1usize << k {
                bytes[position / 8] = 1 << (position % 8);
                check(k, 0, 1 << k, &bytes);
                bytes[position / 8] = 0;
            }
        }
    }
}
