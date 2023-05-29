//! Various simple utilities.

#![no_std]

use core::hint::unreachable_unchecked;

/// Computes `ceil(a / b)`. Assumes `a + b` does not overflow.
#[must_use]
pub const fn ceil_div_usize(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

/// Computes `ceil(log_2(n))`.
#[must_use]
pub fn log2_ceil_usize(n: usize) -> usize {
    (usize::BITS - n.saturating_sub(1).leading_zeros()) as usize
}

/// Computes `log_2(n)`
///
/// # Panics
/// Panics if `n` is not a power of two.
#[must_use]
pub fn log2_strict_usize(n: usize) -> usize {
    let res = n.trailing_zeros();
    assert_eq!(n.wrapping_shr(res), 1, "Not a power of two: {n}");
    res as usize
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
///     if x > 2 {
///         y = foo();
///         branch_hint();
///     } else {
///         y = bar();
///     }
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
