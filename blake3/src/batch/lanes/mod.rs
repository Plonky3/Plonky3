//! Vector backends for the batched compression, one lane per message.
//!
//! x86-64 compiles AVX-512 and AVX2 even when the build does not enable them.
//!
//! It then picks the widest backend the running CPU has, falling back to SSE2.
//!
//! Other targets pick their backend at build time: NEON on AArch64, SIMD128 on wasm32, and one lane elsewhere.
//!
//! Soft-float x86-64 targets lack SSE2, so they take the one-lane backend.
//!
//! Their ABI forbids enabling vector features on a function, so no x86 backend compiles there.

/// Implement the out-of-line steps of [`Backend`] for `W` lanes, with an optional target feature.
///
/// Each step forwards to its generic body in the driver.
///
/// The feature compiles that body for the backend, so its word arithmetic inlines to single instructions.
macro_rules! out_of_line_steps {
    ($w:ident $(, $feature:literal)?) => {
        $(#[target_feature(enable = $feature)])?
        unsafe fn hash_group<const G: usize>(
            mode: $crate::batch::Mode,
            lanes: &$crate::batch::Lanes<'_, $w, G>,
            len: usize,
            out: &mut [[[u8; blake3::OUT_LEN]; $w]; G],
        ) {
            // SAFETY: the caller runs this on a CPU with the backend's features.
            unsafe { $crate::batch::hash_group::<Self, $w, G>(mode, lanes, len, out) }
        }

        $(#[target_feature(enable = $feature)])?
        unsafe fn subtree<const G: usize>(
            mode: $crate::batch::Mode,
            lanes: &$crate::batch::Lanes<'_, $w, G>,
            len: usize,
            first: usize,
            chunks: usize,
            root: u32,
        ) -> $crate::batch::State<Self, G> {
            // SAFETY: the caller runs this on a CPU with the backend's features.
            unsafe { $crate::batch::subtree::<Self, $w, G>(mode, lanes, len, first, chunks, root) }
        }

        #[inline(never)]
        $(#[target_feature(enable = $feature)])?
        unsafe fn leaf<const G: usize>(
            mode: $crate::batch::Mode,
            lanes: &$crate::batch::Lanes<'_, $w, G>,
            len: usize,
            index: usize,
            root: u32,
        ) -> $crate::batch::State<Self, G> {
            $crate::batch::chunk::<Self, $w, G>(mode, lanes, len, index, root)
        }

        $(#[target_feature(enable = $feature)])?
        unsafe fn spread<const G: usize>(
            mode: $crate::batch::Mode,
            input: &[u8],
            len: usize,
            out: &mut [[u8; blake3::OUT_LEN]],
        ) {
            // SAFETY: the caller runs this on a CPU with the backend's features.
            unsafe { $crate::batch::spread::hash::<Self, $w, G>(mode, input, len, out) }
        }

        #[inline(never)]
        $(#[target_feature(enable = $feature)])?
        unsafe fn pass<P: $crate::batch::spread::Pass<Self, $w>, const G: usize>(
            pass: &P,
            lanes: &$crate::batch::Lanes<'_, $w, G>,
            counters: &[[u64; $w]; G],
            out: &mut [[[u8; blake3::OUT_LEN]; $w]; G],
        ) {
            // The chaining value of every lane, written out as its digest bytes.
            let state = pass.run(lanes, counters);
            for (state, out) in state.iter().zip(out) {
                <Self as Backend<$w>>::store_digests(state, out);
            }
        }

        #[inline(never)]
        $(#[target_feature(enable = $feature)])?
        unsafe fn parent<const G: usize>(
            mode: $crate::batch::Mode,
            left: &$crate::batch::State<Self, G>,
            right: &$crate::batch::State<Self, G>,
            root: u32,
        ) -> $crate::batch::State<Self, G> {
            $crate::batch::parent::<Self, G>(mode, left, right, root)
        }
    };
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod x86_64_avx512;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx512f")
))]
mod x86_64_avx2;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "sse2",
    not(target_feature = "avx2")
))]
mod x86_64_sse2;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_endian = "little"
))]
mod aarch64_neon;

#[cfg(all(
    target_arch = "wasm32",
    any(target_feature = "simd128", feature = "wasm32-simd")
))]
mod wasm32_simd128;

#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "sse2"),
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ),
    all(
        target_arch = "wasm32",
        any(target_feature = "simd128", feature = "wasm32-simd")
    )
)))]
mod portable;

use core::fmt;

use blake3::{BLOCK_LEN, OUT_LEN};

use super::compress::{BLOCK_WORDS, STATE_WORDS};
use super::{Lanes, Mode, State};

/// Every backend this build compiles, widest first.
///
/// The last one runs on every CPU of the target.
const KERNELS: &[Kernel] = &[
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    x86_64_avx512::KERNEL,
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx512f")
    ))]
    x86_64_avx2::KERNEL,
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "sse2",
        not(target_feature = "avx2")
    ))]
    x86_64_sse2::KERNEL,
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_endian = "little"
    ))]
    aarch64_neon::KERNEL,
    #[cfg(all(
        target_arch = "wasm32",
        any(target_feature = "simd128", feature = "wasm32-simd")
    ))]
    wasm32_simd128::KERNEL,
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "sse2"),
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_endian = "little"
        ),
        all(
            target_arch = "wasm32",
            any(target_feature = "simd128", feature = "wasm32-simd")
        )
    )))]
    portable::KERNEL,
];

/// Messages the widest compiled backend advances at once.
pub(crate) const LANES: usize = KERNELS[0].lanes;

/// The batched driver of one backend, picked at run time.
#[derive(Clone, Copy)]
pub(crate) struct Kernel {
    /// The backend's name, for diagnostics.
    name: &'static str,
    /// Lanes in one register.
    pub(crate) width: usize,
    /// Messages one batched compression advances at once.
    pub(crate) lanes: usize,
    /// Whether the running CPU has the backend's target features.
    supported: fn() -> bool,
    /// The driver compiled for the backend, sound to call only when `supported` holds.
    run: unsafe fn(Mode, &[u8], usize, &mut [[u8; OUT_LEN]]),
    /// The spreading driver alone, so tests reach it whatever the cost model picks.
    #[cfg(test)]
    spread: unsafe fn(Mode, &[u8], usize, &mut [[u8; OUT_LEN]]),
}

impl Kernel {
    /// The kernel of backend `V`, with `W` lanes per register and `G` register groups.
    const fn new<V: Backend<W>, const W: usize, const G: usize>(name: &'static str) -> Self {
        Self {
            name,
            width: W,
            lanes: W * G,
            supported: V::supported,
            run: super::hash_many_with::<V, W, G>,
            #[cfg(test)]
            spread: V::spread::<G>,
        }
    }

    /// Hash equal-length messages of `len` bytes laid end to end in `input`.
    ///
    /// The caller guarantees `input.len() == len * out.len()`.
    #[inline]
    pub(crate) fn hash_many(self, mode: Mode, input: &[u8], len: usize, out: &mut [[u8; OUT_LEN]]) {
        // SAFETY: kernels only leave this module through `supported`, which checks the CPU.
        unsafe { (self.run)(mode, input, len, out) }
    }

    /// Hash through the spreading driver, whatever the cost model would pick.
    ///
    /// The caller guarantees:
    ///
    /// - `input.len() == len * out.len()`;
    /// - every message spans at least two chunks;
    /// - there are at most 32 messages.
    #[cfg(test)]
    pub(crate) fn spread(self, mode: Mode, input: &[u8], len: usize, out: &mut [[u8; OUT_LEN]]) {
        // SAFETY: kernels only leave this module through `supported`, which checks the CPU.
        unsafe { (self.spread)(mode, input, len, out) }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name)
    }
}

/// Every backend the running CPU supports, widest first.
pub(crate) fn supported() -> impl Iterator<Item = Kernel> {
    KERNELS
        .iter()
        .copied()
        .filter(|kernel| (kernel.supported)())
}

/// The widest backend the running CPU supports.
#[inline]
pub(crate) fn detect() -> Kernel {
    supported()
        .next()
        .expect("the last backend runs on every CPU of the target")
}

/// The arithmetic the compression function needs, on every lane at once.
pub(super) trait Word: Copy {
    /// The same word in every lane.
    fn splat(value: u32) -> Self;

    /// Lane-wise addition modulo 2^32.
    fn add(self, rhs: Self) -> Self;

    /// Lane-wise bitwise and.
    fn and(self, rhs: Self) -> Self;

    /// Lane-wise exclusive or.
    fn xor(self, rhs: Self) -> Self;

    /// Lane-wise rotation right by 16 bits.
    fn rotr_16(self) -> Self;

    /// Lane-wise rotation right by 12 bits.
    fn rotr_12(self) -> Self;

    /// Lane-wise rotation right by 8 bits.
    fn rotr_8(self) -> Self;

    /// Lane-wise rotation right by 7 bits.
    fn rotr_7(self) -> Self;

    /// Advance `G` groups by one block with a hand-scheduled kernel, if this backend has one.
    ///
    /// `params` is the second half of the working vector: IV[0..4], counter, block length, flags.
    ///
    /// Returns false when there is none, and the generic rounds run instead.
    #[inline(always)]
    fn compress_scheduled<const G: usize>(
        _h: &mut [[Self; STATE_WORDS]; G],
        _m: &[[Self; BLOCK_WORDS]; G],
        _params: &[u32; STATE_WORDS],
    ) -> bool {
        false
    }

    /// The hand-scheduled kernel of this backend with a counter per lane, if it has one.
    ///
    /// `counters[g]` holds the low and then the high counter word of every lane of group `g`.
    ///
    /// The counter slots of `params` are ignored.
    ///
    /// Returns false when there is none, and the generic rounds run instead.
    #[inline(always)]
    fn compress_scheduled_counters<const G: usize>(
        _h: &mut [[Self; STATE_WORDS]; G],
        _m: &[[Self; BLOCK_WORDS]; G],
        _params: &[u32; STATE_WORDS],
        _counters: &[[Self; 2]; G],
    ) -> bool {
        false
    }
}

/// A register of `W` lanes, with the out-of-line steps of the driver compiled for it.
///
/// Its word arithmetic runs the backend's instructions, which the running CPU may lack.
///
/// The driver only reaches a backend through a [`Kernel`], and only once [`Backend::supported`] holds.
pub(super) trait Backend<const W: usize>: Word {
    /// Whether the running CPU has this backend's target features.
    fn supported() -> bool;

    /// Transpose one block from each lane into sixteen message words.
    ///
    /// - `rows[l]` is one block of lane `l`.
    /// - Word `w` of the result holds word `w` of every lane.
    fn load_block(rows: &[&[u8; BLOCK_LEN]; W]) -> [Self; BLOCK_WORDS];

    /// Write the chaining value of every lane as a digest.
    fn store_digests(state: &[Self; STATE_WORDS], out: &mut [[u8; OUT_LEN]; W]);

    /// [`super::hash_group`], compiled with this backend's target features.
    ///
    /// # Safety
    ///
    /// The running CPU has this backend's target features.
    unsafe fn hash_group<const G: usize>(
        mode: Mode,
        lanes: &Lanes<'_, W, G>,
        len: usize,
        out: &mut [[[u8; OUT_LEN]; W]; G],
    );

    /// [`super::subtree`], compiled with this backend's target features.
    ///
    /// # Safety
    ///
    /// The running CPU has this backend's target features.
    unsafe fn subtree<const G: usize>(
        mode: Mode,
        lanes: &Lanes<'_, W, G>,
        len: usize,
        first: usize,
        chunks: usize,
        root: u32,
    ) -> State<Self, G>;

    /// A chunk of the tree, kept out of line so the recursion in [`super::subtree`] carries small frames.
    ///
    /// # Safety
    ///
    /// The running CPU has this backend's target features.
    unsafe fn leaf<const G: usize>(
        mode: Mode,
        lanes: &Lanes<'_, W, G>,
        len: usize,
        index: usize,
        root: u32,
    ) -> State<Self, G>;

    /// [`super::spread::hash`], compiled with this backend's target features.
    ///
    /// # Safety
    ///
    /// The running CPU has this backend's target features.
    unsafe fn spread<const G: usize>(
        mode: Mode,
        input: &[u8],
        len: usize,
        out: &mut [[u8; OUT_LEN]],
    );

    /// One pass of the spreading driver, kept out of line like [`Backend::leaf`].
    ///
    /// Writes the digest of every lane of every group to `out`.
    ///
    /// # Safety
    ///
    /// The running CPU has this backend's target features.
    unsafe fn pass<P: super::spread::Pass<Self, W>, const G: usize>(
        pass: &P,
        lanes: &Lanes<'_, W, G>,
        counters: &[[u64; W]; G],
        out: &mut [[[u8; OUT_LEN]; W]; G],
    );

    /// A parent node, kept out of line like [`Backend::leaf`].
    ///
    /// # Safety
    ///
    /// The running CPU has this backend's target features.
    unsafe fn parent<const G: usize>(
        mode: Mode,
        left: &State<Self, G>,
        right: &State<Self, G>,
        root: u32,
    ) -> State<Self, G>;
}

/// Every lane of one vector, in lane order.
#[cfg(test)]
pub(super) const fn to_lanes<V: Backend<W>, const W: usize>(vector: V) -> [u32; W] {
    const { assert!(size_of::<V>() == size_of::<[u32; W]>()) };
    // SAFETY: a vector is exactly its lanes, packed from lane 0 at the lowest address.
    unsafe { core::mem::transmute_copy(&vector) }
}

/// One vector holding the given lanes, in lane order.
pub(super) const fn from_lanes<V: Backend<W>, const W: usize>(lanes: [u32; W]) -> V {
    const { assert!(size_of::<V>() == size_of::<[u32; W]>()) };
    // SAFETY: a vector is exactly its lanes, packed from lane 0 at the lowest address.
    unsafe { core::mem::transmute_copy(&lanes) }
}
