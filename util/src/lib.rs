//! Various simple utilities.

#![no_std]

extern crate alloc;

use alloc::slice;
use alloc::string::String;
use alloc::vec::Vec;
use core::any::type_name;
use core::hint::unreachable_unchecked;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::{iter, mem};

use crate::transpose::transpose_in_place_square;

pub mod array_serialization;
pub mod linear_map;
pub mod transpose;
pub mod zip_eq;

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
    // Tell the optimizer about the semantics of `log2_strict`. i.e. it can replace `n` with
    // `1 << res` and vice versa.
    assume(n == 1 << res);
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
    // Assert that n is a power of 2
    debug_assert!(n.is_power_of_two());
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

// Lookup table of 6-bit reverses.
// NB: 2^6=64 bytes is a cache line. A smaller table wastes cache space.
#[cfg(not(target_arch = "aarch64"))]
#[rustfmt::skip]
const BIT_REVERSE_6BIT: &[u8] = &[
    0o00, 0o40, 0o20, 0o60, 0o10, 0o50, 0o30, 0o70,
    0o04, 0o44, 0o24, 0o64, 0o14, 0o54, 0o34, 0o74,
    0o02, 0o42, 0o22, 0o62, 0o12, 0o52, 0o32, 0o72,
    0o06, 0o46, 0o26, 0o66, 0o16, 0o56, 0o36, 0o76,
    0o01, 0o41, 0o21, 0o61, 0o11, 0o51, 0o31, 0o71,
    0o05, 0o45, 0o25, 0o65, 0o15, 0o55, 0o35, 0o75,
    0o03, 0o43, 0o23, 0o63, 0o13, 0o53, 0o33, 0o73,
    0o07, 0o47, 0o27, 0o67, 0o17, 0o57, 0o37, 0o77,
];

// Ensure that SMALL_ARR_SIZE >= 4 * BIG_T_SIZE.
const BIG_T_SIZE: usize = 1 << 14;
const SMALL_ARR_SIZE: usize = 1 << 16;

/// Permutes `arr` such that each index is mapped to its reverse in binary.
///
/// If the whole array fits in fast cache, then the trivial algorithm is cache friendly. Also, if
/// `T` is really big, then the trivial algorithm is cache-friendly, no matter the size of the array.
pub fn reverse_slice_index_bits<F>(vals: &mut [F]) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let log_n = log2_strict_usize(n);

    // If the whole array fits in fast cache, then the trivial algorithm is cache friendly. Also, if
    // `T` is really big, then the trivial algorithm is cache-friendly, no matter the size of the array.
    if core::mem::size_of::<F>() << log_n <= SMALL_ARR_SIZE
        || core::mem::size_of::<F>() >= BIG_T_SIZE
    {
        reverse_slice_index_bits_small(vals, log_n);
    } else {
        debug_assert!(n >= 4); // By our choice of `BIG_T_SIZE` and `SMALL_ARR_SIZE`.

        // Algorithm:
        //
        // Treat `arr` as a `sqrt(n)` by `sqrt(n)` row-major matrix. (Assume for now that `lb_n` is
        // even, i.e., `n` is a square number.) To perform bit-order reversal we:
        //  1. Bit-reverse the order of the rows. (They are contiguous in memory, so this is
        //     basically a series of large `memcpy`s.)
        //  2. Transpose the matrix.
        //  3. Bit-reverse the order of the rows.
        //
        // This is equivalent to, for every index `0 <= i < n`:
        //  1. bit-reversing `i[lb_n / 2..lb_n]`,
        //  2. swapping `i[0..lb_n / 2]` and `i[lb_n / 2..lb_n]`,
        //  3. bit-reversing `i[lb_n / 2..lb_n]`.
        //
        // If `lb_n` is odd, i.e., `n` is not a square number, then the above procedure requires
        // slight modification. At steps 1 and 3 we bit-reverse bits `ceil(lb_n / 2)..lb_n`, of the
        // index (shuffling `floor(lb_n / 2)` chunks of length `ceil(lb_n / 2)`). At step 2, we
        // perform _two_ transposes. We treat `arr` as two matrices, one where the middle bit of the
        // index is `0` and another, where the middle bit is `1`; we transpose each individually.

        let lb_num_chunks = log_n >> 1;
        let lb_chunk_size = log_n - lb_num_chunks;
        unsafe {
            reverse_slice_index_bits_chunks(vals, lb_num_chunks, lb_chunk_size);
            transpose_in_place_square(vals, lb_chunk_size, lb_num_chunks, 0);
            if lb_num_chunks != lb_chunk_size {
                // `arr` cannot be interpreted as a square matrix. We instead interpret it as a
                // `1 << lb_num_chunks` by `2` by `1 << lb_num_chunks` tensor, in row-major order.
                // The above transpose acted on `tensor[..., 0, ...]` (all indices with middle bit
                // `0`). We still need to transpose `tensor[..., 1, ...]`. To do so, we advance
                // arr by `1 << lb_num_chunks` effectively, adding that to every index.
                let vals_with_offset = &mut vals[1 << lb_num_chunks..];
                transpose_in_place_square(vals_with_offset, lb_chunk_size, lb_num_chunks, 0);
            }
            reverse_slice_index_bits_chunks(vals, lb_num_chunks, lb_chunk_size);
        }
    }
}

// Both functions below are semantically equivalent to:
//     for i in 0..n {
//         result.push(arr[reverse_bits(i, n_power)]);
//     }
// where reverse_bits(i, n_power) computes the n_power-bit reverse. The complications are there
// to guide the compiler to generate optimal assembly.

#[cfg(not(target_arch = "aarch64"))]
fn reverse_slice_index_bits_small<F>(vals: &mut [F], lb_n: usize) {
    if lb_n <= 6 {
        // BIT_REVERSE_6BIT holds 6-bit reverses. This shift makes them lb_n-bit reverses.
        let dst_shr_amt = 6 - lb_n as u32;
        #[allow(clippy::needless_range_loop)]
        for src in 0..vals.len() {
            let dst = (BIT_REVERSE_6BIT[src] as usize).wrapping_shr(dst_shr_amt);
            if src < dst {
                vals.swap(src, dst);
            }
        }
    } else {
        // LLVM does not know that it does not need to reverse src at each iteration (which is
        // expensive on x86). We take advantage of the fact that the low bits of dst change rarely and the high
        // bits of dst are dependent only on the low bits of src.
        let dst_lo_shr_amt = usize::BITS - (lb_n - 6) as u32;
        let dst_hi_shl_amt = lb_n - 6;
        for src_chunk in 0..(vals.len() >> 6) {
            let src_hi = src_chunk << 6;
            let dst_lo = src_chunk.reverse_bits().wrapping_shr(dst_lo_shr_amt);
            #[allow(clippy::needless_range_loop)]
            for src_lo in 0..(1 << 6) {
                let dst_hi = (BIT_REVERSE_6BIT[src_lo] as usize) << dst_hi_shl_amt;
                let src = src_hi + src_lo;
                let dst = dst_hi + dst_lo;
                if src < dst {
                    vals.swap(src, dst);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn reverse_slice_index_bits_small<F>(vals: &mut [F], lb_n: usize) {
    // Aarch64 can reverse bits in one instruction, so the trivial version works best.
    for src in 0..vals.len() {
        let dst = src.reverse_bits().wrapping_shr(usize::BITS - lb_n as u32);
        if src < dst {
            vals.swap(src, dst);
        }
    }
}

/// Split `arr` chunks and bit-reverse the order of the chunks. There are `1 << lb_num_chunks`
/// chunks, each of length `1 << lb_chunk_size`.
/// SAFETY: ensure that `arr.len() == 1 << lb_num_chunks + lb_chunk_size`.
unsafe fn reverse_slice_index_bits_chunks<F>(
    vals: &mut [F],
    lb_num_chunks: usize,
    lb_chunk_size: usize,
) {
    for i in 0..1usize << lb_num_chunks {
        // `wrapping_shr` handles the silly case when `lb_num_chunks == 0`.
        let j = i
            .reverse_bits()
            .wrapping_shr(usize::BITS - lb_num_chunks as u32);
        if i < j {
            unsafe {
                core::ptr::swap_nonoverlapping(
                    vals.get_unchecked_mut(i << lb_chunk_size),
                    vals.get_unchecked_mut(j << lb_chunk_size),
                    1 << lb_chunk_size,
                );
            }
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
/// `core::iter::Iterator::next_chunk()` from nightly.
///
/// Returns an array of `MaybeUninit<T>` and the number of items in the
/// array which have been correctly initialized.
#[inline]
fn iter_next_chunk_erased<const BUFLEN: usize, I: Iterator>(
    iter: &mut I,
) -> ([MaybeUninit<I::Item>; BUFLEN], usize)
where
    I::Item: Copy,
{
    let mut buf = [const { MaybeUninit::<I::Item>::uninit() }; BUFLEN];
    let mut i = 0;

    while i < BUFLEN {
        if let Some(c) = iter.next() {
            // Copy the next Item into `buf`.
            unsafe {
                buf.get_unchecked_mut(i).write(c);
                i = i.unchecked_add(1);
            }
        } else {
            // No more items in the iterator.
            break;
        }
    }
    (buf, i)
}

/// Gets a shared reference to the contained value.
///
/// # Safety
///
/// Calling this when the content is not yet fully initialized causes undefined
/// behavior: it is up to the caller to guarantee that every `MaybeUninit<T>` in
/// the slice really is in an initialized state.
///
/// Copied from:
/// https://doc.rust-lang.org/std/primitive.slice.html#method.assume_init_ref
/// Once that is stabilized, this should be removed.
#[inline(always)]
pub const unsafe fn assume_init_ref<T>(slice: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: casting `slice` to a `*const [T]` is safe since the caller guarantees that
    // `slice` is initialized, and `MaybeUninit` is guaranteed to have the same layout as `T`.
    // The pointer obtained is valid since it refers to memory owned by `slice` which is a
    // reference and thus guaranteed to be valid for reads.
    unsafe { &*(slice as *const [MaybeUninit<T>] as *const [T]) }
}

/// Split an iterator into small arrays and apply `func` to each.
///
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
        let (buf, n) = iter_next_chunk_erased::<BUFLEN, _>(&mut iter);
        if n == 0 {
            break;
        }
        func(unsafe { assume_init_ref(buf.get_unchecked(..n)) });
    }
}

/// Pulls `N` items from `iter` and returns them as an array. If the iterator
/// yields fewer than `N` items (but more than `0`), pads by the given default value.
///
/// Since the iterator is passed as a mutable reference and this function calls
/// `next` at most `N` times, the iterator can still be used afterwards to
/// retrieve the remaining items.
///
/// If `iter.next()` panics, all items already yielded by the iterator are
/// dropped.
#[inline]
fn iter_next_chunk_padded<T: Copy, const N: usize>(
    iter: &mut impl Iterator<Item = T>,
    default: T, // Needed due to [T; M] not always implementing Default. Can probably be dropped if const generics stabilize.
) -> Option<[T; N]> {
    let (mut arr, n) = iter_next_chunk_erased::<N, _>(iter);
    (n != 0).then(|| {
        // Fill the rest of the array with default values.
        arr[n..].fill(MaybeUninit::new(default));
        unsafe { mem::transmute_copy::<_, [T; N]>(&arr) }
    })
}

/// Returns an iterator over `N` elements of the iterator at a time.
///
/// The chunks do not overlap. If `N` does not divide the length of the
/// iterator, then the last `N-1` elements will be padded with the given default value.
///
/// This is essentially a copy pasted version of the nightly `array_chunks` function.
/// https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.array_chunks
/// Once that is stabilized this and the functions above it should be removed.
#[inline]
pub fn iter_array_chunks_padded<T: Copy, const N: usize>(
    iter: impl IntoIterator<Item = T>,
    default: T, // Needed due to [T; M] not always implementing Default. Can probably be dropped if const generics stabilize.
) -> impl Iterator<Item = [T; N]> {
    let mut iter = iter.into_iter();
    iter::from_fn(move || iter_next_chunk_padded(&mut iter, default))
}

/// Convert a vector of `BaseArray` elements to a vector of `Base` elements without any
/// reallocations.
///
/// This is useful to convert `Vec<[F; N]>` to `Vec<F>` or `Vec<A>` to `Vec<F>` where
/// `A` has the same size, alignment and memory layout as `[F; N]` for some `N`. It can also,
/// be used to safely convert `Vec<u32>` to `Vec<F>` if `F` is a `32` bit field
/// or `Vec<u64>` to `Vec<F>` if `F` is a `64` bit field.
///
/// # Safety
///
/// This is assumes that `BaseArray` has the same alignment and memory layout as `[Base; N]`.
/// As Rust guarantees that arrays elements are contiguous in memory and the alignment of
/// the array is the same as the alignment of its elements, this means that `BaseArray`
/// must have the same alignment as `Base`.
///
/// # Panics
///
/// This panics if the size of `BaseArray` is not a multiple of the size of `Base`.
#[inline]
pub unsafe fn flatten_to_base<Base, BaseArray>(vec: Vec<BaseArray>) -> Vec<Base> {
    debug_assert_eq!(align_of::<Base>(), align_of::<BaseArray>());

    assert!(
        size_of::<BaseArray>() % size_of::<Base>() == 0,
        "Size of BaseArray (got {}) must be a multiple of the size of Base ({}).",
        size_of::<BaseArray>(),
        size_of::<Base>()
    );

    let d = size_of::<BaseArray>() / size_of::<Base>();
    // Prevent running `vec`'s destructor so we are in complete control
    // of the allocation.
    let mut values = ManuallyDrop::new(vec);

    // Each `Self` is an array of `d` elements, so the length and capacity of
    // the new vector will be multiplied by `d`.
    let new_len = values.len() * d;
    let new_cap = values.capacity() * d;

    // Safe as BaseArray and Base have the same alignment.
    let ptr = values.as_mut_ptr() as *mut Base;

    unsafe {
        // Safety:
        // - BaseArray and Base have the same alignment.
        // - As size_of::<BaseArray>() == size_of::<Base>() * d:
        //      -- The capacity of the new vector is equal to the capacity of the old vector.
        //      -- The first new_len elements of the new vector correspond to the first
        //         len elements of the old vector and so are properly initialized.
        Vec::from_raw_parts(ptr, new_len, new_cap)
    }
}

/// Convert a vector of `Base` elements to a vector of `BaseArray` elements ideally without any
/// reallocations.
///
/// This is an inverse of `flatten_to_base`. Unfortunately, unlike `flatten_to_base`, it may not be
/// possible to avoid allocations. This issue is that there is not way to guarantee that the capacity
/// of the vector is a multiple of `d`.
///
/// # Safety
///
/// This is assumes that `BaseArray` has the same alignment and memory layout as `[Base; N]`.
/// As Rust guarantees that arrays elements are contiguous in memory and the alignment of
/// the array is the same as the alignment of its elements, this means that `BaseArray`
/// must have the same alignment as `Base`.
///
/// # Panics
///
/// This panics if the size of `BaseArray` is not a multiple of the size of `Base`.
/// This panics if the length of the vector is not a multiple of the ratio of the sizes.
#[inline]
pub unsafe fn reconstitute_from_base<Base, BaseArray: Clone>(mut vec: Vec<Base>) -> Vec<BaseArray> {
    assert!(
        size_of::<BaseArray>() % size_of::<Base>() == 0,
        "Size of BaseArray (got {}) must be a multiple of the size of Base ({}).",
        size_of::<BaseArray>(),
        size_of::<Base>()
    );

    let d = size_of::<BaseArray>() / size_of::<Base>();

    assert!(
        vec.len() % d == 0,
        "Vector length (got {}) must be a multiple of the extension field dimension ({}).",
        vec.len(),
        d
    );

    let new_len = vec.len() / d;

    // We could call vec.shrink_to_fit() here to try and increase the probability that
    // the capacity is a multiple of d. That might cause a reallocation though which
    // would defeat the whole purpose.
    let cap = vec.capacity();

    // The assumption is that basically all callers of `reconstitute_from_base_vec` will be calling it
    // with a vector constructed from `flatten_to_base` and so the capacity should be a multiple of `d`.
    // But capacities can do strange things so we need to support both possibilities.
    // Note that the `else` branch would also work if the capacity is a multiple of `d` but it is slower.
    if cap % d == 0 {
        // Prevent running `vec`'s destructor so we are in complete control
        // of the allocation.
        let mut values = ManuallyDrop::new(vec);

        // If we are on this branch then the capacity is a multiple of `d`.
        let new_cap = cap / d;

        // Safe as BaseArray and Base have the same alignment.
        let ptr = values.as_mut_ptr() as *mut BaseArray;

        unsafe {
            // Safety:
            // - BaseArray and Base have the same alignment.
            // - As size_of::<Base>() == size_of::<BaseArray>() / d:
            //      -- If we have reached this point, the length and capacity are both divisible by `d`.
            //      -- The capacity of the new vector is equal to the capacity of the old vector.
            //      -- The first new_len elements of the new vector correspond to the first
            //         len elements of the old vector and so are properly initialized.
            Vec::from_raw_parts(ptr, new_len, new_cap)
        }
    } else {
        // If the capacity is not a multiple of `D`, we go via slices.

        let buf_ptr = vec.as_mut_ptr().cast::<BaseArray>();
        let slice = unsafe {
            // Safety:
            // - BaseArray and Base have the same alignment.
            // - As size_of::<Base>() == size_of::<BaseArray>() / D:
            //      -- If we have reached this point, the length is divisible by `D`.
            //      -- The first new_len elements of the slice correspond to the first
            //         len elements of the old slice and so are properly initialized.
            slice::from_raw_parts(buf_ptr, new_len)
        };

        // Ideally the compiler could optimize this away to avoid the copy but it appears not to.
        slice.to_vec()
    }
}

#[inline(always)]
pub const fn relatively_prime_u64(mut u: u64, mut v: u64) -> bool {
    // Check that neither input is 0.
    if u == 0 || v == 0 {
        return false;
    }

    // Check divisibility by 2.
    if (u | v) & 1 == 0 {
        return false;
    }

    // Remove factors of 2 from `u` and `v`
    u >>= u.trailing_zeros();
    if u == 1 {
        return true;
    }

    while v != 0 {
        v >>= v.trailing_zeros();
        if v == 1 {
            return true;
        }

        // Ensure u <= v
        if u > v {
            core::mem::swap(&mut u, &mut v);
        }

        // This looks inefficient for v >> u but thanks to the fact that we remove
        // trailing_zeros of v in every iteration, it ends up much more performative
        // than first glance implies.
        v -= u
    }
    // If we made it through the loop, at no point is u or v equal to 1 and so the gcd
    // must be greater than 1.
    false
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

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

    #[test]
    fn test_apply_to_chunks_exact_fit() {
        const CHUNK_SIZE: usize = 4;
        let input: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut results: Vec<Vec<u8>> = Vec::new();

        apply_to_chunks::<CHUNK_SIZE, _, _>(input, |chunk| {
            results.push(chunk.to_vec());
        });

        assert_eq!(results, vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]]);
    }

    #[test]
    fn test_apply_to_chunks_with_remainder() {
        const CHUNK_SIZE: usize = 3;
        let input: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7];
        let mut results: Vec<Vec<u8>> = Vec::new();

        apply_to_chunks::<CHUNK_SIZE, _, _>(input, |chunk| {
            results.push(chunk.to_vec());
        });

        assert_eq!(results, vec![vec![1, 2, 3], vec![4, 5, 6], vec![7]]);
    }

    #[test]
    fn test_apply_to_chunks_empty_input() {
        const CHUNK_SIZE: usize = 4;
        let input: Vec<u8> = vec![];
        let mut results: Vec<Vec<u8>> = Vec::new();

        apply_to_chunks::<CHUNK_SIZE, _, _>(input, |chunk| {
            results.push(chunk.to_vec());
        });

        assert!(results.is_empty());
    }

    #[test]
    fn test_apply_to_chunks_single_chunk() {
        const CHUNK_SIZE: usize = 10;
        let input: Vec<u8> = vec![1, 2, 3, 4, 5];
        let mut results: Vec<Vec<u8>> = Vec::new();

        apply_to_chunks::<CHUNK_SIZE, _, _>(input, |chunk| {
            results.push(chunk.to_vec());
        });

        assert_eq!(results, vec![vec![1, 2, 3, 4, 5]]);
    }

    #[test]
    fn test_apply_to_chunks_large_chunk_size() {
        const CHUNK_SIZE: usize = 100;
        let input: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let mut results: Vec<Vec<u8>> = Vec::new();

        apply_to_chunks::<CHUNK_SIZE, _, _>(input, |chunk| {
            results.push(chunk.to_vec());
        });

        assert_eq!(results, vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]);
    }

    #[test]
    fn test_apply_to_chunks_large_input() {
        const CHUNK_SIZE: usize = 5;
        let input: Vec<u8> = (1..=20).collect();
        let mut results: Vec<Vec<u8>> = Vec::new();

        apply_to_chunks::<CHUNK_SIZE, _, _>(input, |chunk| {
            results.push(chunk.to_vec());
        });

        assert_eq!(
            results,
            vec![
                vec![1, 2, 3, 4, 5],
                vec![6, 7, 8, 9, 10],
                vec![11, 12, 13, 14, 15],
                vec![16, 17, 18, 19, 20]
            ]
        );
    }

    #[test]
    fn test_reverse_slice_index_bits_random() {
        let lengths = [32, 128, 1 << 16];
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..32 {
            for &length in &lengths {
                let mut rand_list: Vec<u32> = Vec::with_capacity(length);
                rand_list.resize_with(length, || rng.random());
                let expect = reverse_index_bits_naive(&rand_list);

                let mut actual = rand_list.clone();
                reverse_slice_index_bits(&mut actual);

                assert_eq!(actual, expect);
            }
        }
    }

    #[test]
    fn test_log2_strict_usize_edge_cases() {
        assert_eq!(log2_strict_usize(1), 0);
        assert_eq!(log2_strict_usize(2), 1);
        assert_eq!(log2_strict_usize(1 << 18), 18);
        assert_eq!(log2_strict_usize(1 << 31), 31);
        assert_eq!(
            log2_strict_usize(1 << (usize::BITS - 1)),
            usize::BITS as usize - 1
        );
    }

    #[test]
    #[should_panic]
    fn test_log2_strict_usize_zero() {
        let _ = log2_strict_usize(0);
    }

    #[test]
    #[should_panic]
    fn test_log2_strict_usize_nonpower_2() {
        let _ = log2_strict_usize(0x78c341c65ae6d262);
    }

    #[test]
    #[should_panic]
    fn test_log2_strict_usize_max() {
        let _ = log2_strict_usize(usize::MAX);
    }

    #[test]
    fn test_log2_ceil_usize_comprehensive() {
        // Powers of 2
        assert_eq!(log2_ceil_usize(0), 0);
        assert_eq!(log2_ceil_usize(1), 0);
        assert_eq!(log2_ceil_usize(2), 1);
        assert_eq!(log2_ceil_usize(1 << 18), 18);
        assert_eq!(log2_ceil_usize(1 << 31), 31);
        assert_eq!(
            log2_ceil_usize(1 << (usize::BITS - 1)),
            usize::BITS as usize - 1
        );

        // Nonpowers; want to round up
        assert_eq!(log2_ceil_usize(3), 2);
        assert_eq!(log2_ceil_usize(0x14fe901b), 29);
        assert_eq!(
            log2_ceil_usize((1 << (usize::BITS - 1)) + 1),
            usize::BITS as usize
        );
        assert_eq!(log2_ceil_usize(usize::MAX - 1), usize::BITS as usize);
        assert_eq!(log2_ceil_usize(usize::MAX), usize::BITS as usize);
    }

    fn reverse_index_bits_naive<T: Copy>(arr: &[T]) -> Vec<T> {
        let n = arr.len();
        let n_power = log2_strict_usize(n);

        let mut out = vec![None; n];
        for (i, v) in arr.iter().enumerate() {
            let dst = i.reverse_bits() >> (usize::BITS - n_power as u32);
            out[dst] = Some(*v);
        }

        out.into_iter().map(|x| x.unwrap()).collect()
    }

    #[test]
    fn test_relatively_prime_u64() {
        // Zero cases (should always return false)
        assert!(!relatively_prime_u64(0, 0));
        assert!(!relatively_prime_u64(10, 0));
        assert!(!relatively_prime_u64(0, 10));
        assert!(!relatively_prime_u64(0, 123456789));

        // Number with itself (if greater than 1, not relatively prime)
        assert!(relatively_prime_u64(1, 1));
        assert!(!relatively_prime_u64(10, 10));
        assert!(!relatively_prime_u64(99999, 99999));

        // Powers of 2 (always false since they share factor 2)
        assert!(!relatively_prime_u64(2, 4));
        assert!(!relatively_prime_u64(16, 32));
        assert!(!relatively_prime_u64(64, 128));
        assert!(!relatively_prime_u64(1024, 4096));
        assert!(!relatively_prime_u64(u64::MAX, u64::MAX));

        // One number is a multiple of the other (always false)
        assert!(!relatively_prime_u64(5, 10));
        assert!(!relatively_prime_u64(12, 36));
        assert!(!relatively_prime_u64(15, 45));
        assert!(!relatively_prime_u64(100, 500));

        // Co-prime numbers (should be true)
        assert!(relatively_prime_u64(17, 31));
        assert!(relatively_prime_u64(97, 43));
        assert!(relatively_prime_u64(7919, 65537));
        assert!(relatively_prime_u64(15485863, 32452843));

        // Small prime numbers (should be true)
        assert!(relatively_prime_u64(13, 17));
        assert!(relatively_prime_u64(101, 103));
        assert!(relatively_prime_u64(1009, 1013));

        // Large numbers (some cases where they are relatively prime or not)
        assert!(!relatively_prime_u64(
            190266297176832000,
            10430732356495263744
        ));
        assert!(!relatively_prime_u64(
            2040134905096275968,
            5701159354248194048
        ));
        assert!(!relatively_prime_u64(
            16611311494648745984,
            7514969329383038976
        ));
        assert!(!relatively_prime_u64(
            14863931409971066880,
            7911906750992527360
        ));

        // Max values
        assert!(relatively_prime_u64(u64::MAX, 1));
        assert!(relatively_prime_u64(u64::MAX, u64::MAX - 1));
        assert!(!relatively_prime_u64(u64::MAX, u64::MAX));
    }
}
