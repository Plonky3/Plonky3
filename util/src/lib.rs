//! Various simple utilities.

#![no_std]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::any::type_name;
use core::hint::unreachable_unchecked;

pub mod array_serialization;
pub mod linear_map;

/// Computes `ceil(a / b)`. Assumes `a + b` does not overflow.
#[must_use]
pub const fn ceil_div_usize(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

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
