//! Various simple utilities.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::any::type_name;
use core::hint::unreachable_unchecked;
use core::mem::MaybeUninit;

pub mod array_serialization;
pub mod linear_map;

/// Computes `ceil(log_2(n))`.
#[must_use]
pub const fn log2_ceil_usize(n: usize) -> usize {
    (usize::BITS - n.saturating_sub(1).leading_zeros()) as usize
}

#[must_use]
pub fn log2_ceil_u64(n: u64) -> u64 {
    (u64::BITS - n.saturating_sub(1).leading_zeros()).into()
}

/// Computes `log_2(n)`
///
/// # Panics
/// Panics if `n` is not a power of two.
#[must_use]
#[inline]
pub fn log2_strict_usize(n: usize) -> usize {
    let res = n.trailing_zeros();
    assert_eq!(n.wrapping_shr(res), 1, "Not a power of two: {n}");
    res as usize
}

/// Returns `[0, ..., N - 1]`.
#[must_use]
pub const fn indices_arr<const N: usize>() -> [usize; N] {
    let mut indices_arr = [0; N];
    let mut i = 0;
    while i < N {
        indices_arr[i] = i;
        i += 1;
    }
    indices_arr
}

#[inline]
pub const fn reverse_bits(x: usize, n: usize) -> usize {
    reverse_bits_len(x, n.trailing_zeros() as usize)
}

#[inline]
pub const fn reverse_bits_len(x: usize, bit_len: usize) -> usize {
    // NB: The only reason we need overflowing_shr() here as opposed
    // to plain '>>' is to accommodate the case n == num_bits == 0,
    // which would become `0 >> 64`. Rust thinks that any shift of 64
    // bits causes overflow, even when the argument is zero.
    x.reverse_bits()
        .overflowing_shr(usize::BITS - bit_len as u32)
        .0
}

/// Permutes `arr` such that each index is mapped to its reverse in binary.
pub fn reverse_slice_index_bits<F>(vals: &mut [F]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let log_n = log2_strict_usize(n);

    for i in 0..n {
        let j = reverse_bits_len(i, log_n);
        if i < j {
            vals.swap(i, j);
        }
    }
}

#[inline(always)]
pub fn assume(p: bool) {
    debug_assert!(p);
    if !p {
        unsafe {
            unreachable_unchecked();
        }
    }
}

/// Try to force Rust to emit a branch. Example:
///
/// ```no_run
/// let x = 100;
/// if x > 20 {
///     println!("x is big!");
///     p3_util::branch_hint();
/// } else {
///     println!("x is small!");
/// }
/// ```
///
/// This function has no semantics. It is a hint only.
#[inline(always)]
pub fn branch_hint() {
    // NOTE: These are the currently supported assembly architectures. See the
    // [nightly reference](https://doc.rust-lang.org/nightly/reference/inline-assembly.html) for
    // the most up-to-date list.
    #[cfg(any(
        target_arch = "aarch64",
        target_arch = "arm",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "x86",
        target_arch = "x86_64",
    ))]
    unsafe {
        core::arch::asm!("", options(nomem, nostack, preserves_flags));
    }
}

/// Convenience methods for Vec.
pub trait VecExt<T> {
    /// Push `elem` and return a reference to it.
    fn pushed_ref(&mut self, elem: T) -> &T;
    /// Push `elem` and return a mutable reference to it.
    fn pushed_mut(&mut self, elem: T) -> &mut T;
}

impl<T> VecExt<T> for alloc::vec::Vec<T> {
    fn pushed_ref(&mut self, elem: T) -> &T {
        self.push(elem);
        self.last().unwrap()
    }
    fn pushed_mut(&mut self, elem: T) -> &mut T {
        self.push(elem);
        self.last_mut().unwrap()
    }
}

pub fn transpose_vec<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

/// Return a String containing the name of T but with all the crate
/// and module prefixes removed.
pub fn pretty_name<T>() -> String {
    let name = type_name::<T>();
    let mut result = String::new();
    for qual in name.split_inclusive(&['<', '>', ',']) {
        result.push_str(qual.split("::").last().unwrap());
    }
    result
}

/// A C-style buffered input reader, similar to
/// `std::iter::Iterator::next_chunk()` from nightly.
///
/// Unsafe because the returned array may contain uninitialised
/// elements.
#[inline]
unsafe fn iter_next_chunk<const BUFLEN: usize, I: Iterator>(
    iter: &mut I,
) -> ([I::Item; BUFLEN], usize)
where
    I::Item: Copy,
{
    let mut buf = unsafe {
        let t = [const { MaybeUninit::<I::Item>::uninit() }; BUFLEN];
        // We are forced to use `transmute_copy` here instead of
        // `transmute` because `BUFLEN` is a const generic parameter.
        // The compiler *should* be smart enough not to emit a copy though.
        core::mem::transmute_copy::<_, [I::Item; BUFLEN]>(&t)
    };
    let mut i = 0;

    // Read BUFLEN values from `iter` into `buf` at a time.
    for c in iter {
        // Copy the next Item into `buf`.
        unsafe {
            *buf.get_unchecked_mut(i) = c;
            i = i.unchecked_add(1);
        }
        // If `buf` is full
        if i == BUFLEN {
            break;
        }
    }
    (buf, i)
}

/// Repeatedly read `BUFLEN` elements from `input` into an array and
/// pass the array to `func` as a slice. If less than `BUFLEN`
/// elements are remaining, that smaller slice is passed to `func` (if
/// it is non-empty) and the function returns.
#[inline]
pub fn apply_to_chunks<const BUFLEN: usize, I, H>(input: I, mut func: H)
where
    I: IntoIterator<Item = u8>,
    H: FnMut(&[I::Item]),
{
    let mut iter = input.into_iter();
    loop {
        let (buf, n) = unsafe { iter_next_chunk::<BUFLEN, _>(&mut iter) };
        if n == 0 {
            break;
        }
        func(unsafe { buf.get_unchecked(..n) });
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_reverse_bits_len() {
        assert_eq!(reverse_bits_len(0b0000000000, 10), 0b0000000000);
        assert_eq!(reverse_bits_len(0b0000000001, 10), 0b1000000000);
        assert_eq!(reverse_bits_len(0b1000000000, 10), 0b0000000001);
        assert_eq!(reverse_bits_len(0b00000, 5), 0b00000);
        assert_eq!(reverse_bits_len(0b01011, 5), 0b11010);
    }

    #[test]
    fn test_reverse_index_bits() {
        let mut arg = vec![10, 20, 30, 40];
        reverse_slice_index_bits(&mut arg);
        assert_eq!(arg, vec![10, 30, 20, 40]);

        let mut input256: Vec<u64> = (0..256).collect();
        #[rustfmt::skip]
        let output256: Vec<u64> = vec![
            0x00, 0x80, 0x40, 0xc0, 0x20, 0xa0, 0x60, 0xe0, 0x10, 0x90, 0x50, 0xd0, 0x30, 0xb0, 0x70, 0xf0,
            0x08, 0x88, 0x48, 0xc8, 0x28, 0xa8, 0x68, 0xe8, 0x18, 0x98, 0x58, 0xd8, 0x38, 0xb8, 0x78, 0xf8,
            0x04, 0x84, 0x44, 0xc4, 0x24, 0xa4, 0x64, 0xe4, 0x14, 0x94, 0x54, 0xd4, 0x34, 0xb4, 0x74, 0xf4,
            0x0c, 0x8c, 0x4c, 0xcc, 0x2c, 0xac, 0x6c, 0xec, 0x1c, 0x9c, 0x5c, 0xdc, 0x3c, 0xbc, 0x7c, 0xfc,
            0x02, 0x82, 0x42, 0xc2, 0x22, 0xa2, 0x62, 0xe2, 0x12, 0x92, 0x52, 0xd2, 0x32, 0xb2, 0x72, 0xf2,
            0x0a, 0x8a, 0x4a, 0xca, 0x2a, 0xaa, 0x6a, 0xea, 0x1a, 0x9a, 0x5a, 0xda, 0x3a, 0xba, 0x7a, 0xfa,
            0x06, 0x86, 0x46, 0xc6, 0x26, 0xa6, 0x66, 0xe6, 0x16, 0x96, 0x56, 0xd6, 0x36, 0xb6, 0x76, 0xf6,
            0x0e, 0x8e, 0x4e, 0xce, 0x2e, 0xae, 0x6e, 0xee, 0x1e, 0x9e, 0x5e, 0xde, 0x3e, 0xbe, 0x7e, 0xfe,
            0x01, 0x81, 0x41, 0xc1, 0x21, 0xa1, 0x61, 0xe1, 0x11, 0x91, 0x51, 0xd1, 0x31, 0xb1, 0x71, 0xf1,
            0x09, 0x89, 0x49, 0xc9, 0x29, 0xa9, 0x69, 0xe9, 0x19, 0x99, 0x59, 0xd9, 0x39, 0xb9, 0x79, 0xf9,
            0x05, 0x85, 0x45, 0xc5, 0x25, 0xa5, 0x65, 0xe5, 0x15, 0x95, 0x55, 0xd5, 0x35, 0xb5, 0x75, 0xf5,
            0x0d, 0x8d, 0x4d, 0xcd, 0x2d, 0xad, 0x6d, 0xed, 0x1d, 0x9d, 0x5d, 0xdd, 0x3d, 0xbd, 0x7d, 0xfd,
            0x03, 0x83, 0x43, 0xc3, 0x23, 0xa3, 0x63, 0xe3, 0x13, 0x93, 0x53, 0xd3, 0x33, 0xb3, 0x73, 0xf3,
            0x0b, 0x8b, 0x4b, 0xcb, 0x2b, 0xab, 0x6b, 0xeb, 0x1b, 0x9b, 0x5b, 0xdb, 0x3b, 0xbb, 0x7b, 0xfb,
            0x07, 0x87, 0x47, 0xc7, 0x27, 0xa7, 0x67, 0xe7, 0x17, 0x97, 0x57, 0xd7, 0x37, 0xb7, 0x77, 0xf7,
            0x0f, 0x8f, 0x4f, 0xcf, 0x2f, 0xaf, 0x6f, 0xef, 0x1f, 0x9f, 0x5f, 0xdf, 0x3f, 0xbf, 0x7f, 0xff,
        ];
        reverse_slice_index_bits(&mut input256[..]);
        assert_eq!(input256, output256);
    }
}
