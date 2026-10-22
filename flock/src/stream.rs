//! Streaming stores for tables written once and read in a later pass.
//!
//! A streaming store skips the read a cache fill would make, so a written table costs its bytes once.

use core::marker::PhantomData;
use core::mem::{MaybeUninit, size_of_val};

/// A region of streaming stores, fenced when dropped.
///
/// Streaming stores are weakly ordered, so the region belongs to one thread and its fence lands before the task returns.
pub(crate) struct Stream(PhantomData<*const ()>);

impl Stream {
    /// Open a region.
    pub(crate) const fn new() -> Self {
        Self(PhantomData)
    }

    /// Copy `src` into unwritten slots, streaming whole cache lines.
    #[inline]
    pub(crate) fn write<T: Copy>(&self, dst: &mut [MaybeUninit<T>], src: &[T]) {
        assert_eq!(dst.len(), src.len(), "a streaming copy is elementwise");
        // SAFETY: distinct slices are valid for `size_of_val(src)` bytes each and cannot overlap.
        unsafe {
            copy_raw(
                dst.as_mut_ptr().cast(),
                src.as_ptr().cast(),
                size_of_val(src),
            );
        }
    }

    /// Write `a & b` into unwritten words, streaming whole cache lines.
    #[inline]
    pub(crate) fn write_and(&self, dst: &mut [MaybeUninit<u64>], a: &[u64], b: &[u64]) {
        assert!(
            dst.len() == a.len() && dst.len() == b.len(),
            "a streaming AND is elementwise"
        );
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        {
            use core::arch::x86_64::{_mm512_and_si512, _mm512_loadu_si512, _mm512_stream_si512};

            /// Words of a cache line.
            const LINE: usize = 8;
            let head = dst.as_ptr().align_offset(64).min(dst.len());
            let body = (dst.len() - head) & !(LINE - 1);
            for i in (0..head).chain(head + body..dst.len()) {
                dst[i].write(a[i] & b[i]);
            }
            for i in (head..head + body).step_by(LINE) {
                // SAFETY: the line lies in all three slices and `dst + i` is line-aligned.
                unsafe {
                    let v = _mm512_and_si512(
                        _mm512_loadu_si512(a.as_ptr().add(i).cast()),
                        _mm512_loadu_si512(b.as_ptr().add(i).cast()),
                    );
                    _mm512_stream_si512(dst.as_mut_ptr().add(i).cast(), v);
                }
            }
        }
        #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
        for ((d, &a), &b) in dst.iter_mut().zip(a).zip(b) {
            d.write(a & b);
        }
    }
}

impl Drop for Stream {
    #[inline]
    fn drop(&mut self) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `sfence` is unconditionally available on x86-64.
        unsafe {
            core::arch::x86_64::_mm_sfence();
        }
    }
}

/// Copy `bytes` from `src` to `dst`, streaming the whole cache lines.
///
/// # Safety
///
/// `dst` and `src` must be valid for `bytes` and must not overlap.
#[cfg_attr(
    not(all(target_arch = "x86_64", target_feature = "avx512f")),
    expect(
        clippy::missing_const_for_fn,
        reason = "the x86 path streams at run time"
    )
)]
#[inline]
unsafe fn copy_raw(dst: *mut u8, src: *const u8, bytes: usize) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
    {
        use core::arch::x86_64::{_mm512_loadu_si512, _mm512_stream_si512};

        /// Bytes of a cache line.
        const LINE: usize = 64;
        let head = dst.align_offset(LINE).min(bytes);
        let body = (bytes - head) & !(LINE - 1);
        // SAFETY: the three runs partition `bytes`, which both pointers cover, and `dst + head` is line-aligned.
        unsafe {
            core::ptr::copy_nonoverlapping(src, dst, head);
            let mut off = head;
            while off < head + body {
                _mm512_stream_si512(dst.add(off).cast(), _mm512_loadu_si512(src.add(off).cast()));
                off += LINE;
            }
            core::ptr::copy_nonoverlapping(src.add(off), dst.add(off), bytes - off);
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
    // SAFETY: forwarded to the caller's obligation.
    unsafe {
        core::ptr::copy_nonoverlapping(src, dst, bytes);
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn a_streamed_copy_is_a_plain_one() {
        let src: Vec<u32> = (0..600u32).collect();
        for head in 0..20 {
            for len in [0, 1, 15, 16, 17, 64, 129, 500] {
                let mut got = vec![MaybeUninit::new(0u32); head + len];
                {
                    let stream = Stream::new();
                    stream.write(&mut got[head..], &src[..len]);
                }
                // SAFETY: every slot was initialized above.
                let got: Vec<u32> = got.iter().map(|x| unsafe { x.assume_init() }).collect();
                let mut want = vec![0u32; head + len];
                want[head..].copy_from_slice(&src[..len]);
                assert_eq!(got, want, "head {head}, len {len}");
            }
        }
    }

    #[test]
    fn a_streamed_and_is_a_plain_one() {
        let a: Vec<u64> = (0..300u64)
            .map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15))
            .collect();
        let b: Vec<u64> = (0..300u64).map(|i| !i.rotate_left(17)).collect();
        for head in 0..12 {
            for len in [0, 1, 7, 8, 9, 64, 131] {
                let mut got = vec![MaybeUninit::new(0u64); head + len];
                {
                    let stream = Stream::new();
                    stream.write_and(&mut got[head..], &a[..len], &b[..len]);
                }
                // SAFETY: every slot was initialized above.
                let got: Vec<u64> = got.iter().map(|x| unsafe { x.assume_init() }).collect();
                let mut want = vec![0u64; head + len];
                for (w, (&a, &b)) in want[head..].iter_mut().zip(a.iter().zip(&b)) {
                    *w = a & b;
                }
                assert_eq!(got, want, "head {head}, len {len}");
            }
        }
    }
}
