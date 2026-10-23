//! Independent compression oracles and lane-boundary checks.

use proptest::prelude::*;

use crate::{compress_raw, compress_raw_packed, compress_raw_xof};

// This reference follows the specification with a dynamically permuted message,
// independently of the production schedule and vector arithmetic.
fn reference(cv: [u32; 8], mut m: [u32; 16], params: [u32; 4]) -> [u32; 16] {
    fn g(v: &mut [u32; 16], [a, b, c, d]: [usize; 4], x: u32, y: u32) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    }
    let mut v = [0; 16];
    v[..8].copy_from_slice(&cv);
    v[8..12].copy_from_slice(&[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a]);
    v[12..].copy_from_slice(&params);
    for _ in 0..7 {
        for (indices, pair) in [
            [0, 4, 8, 12],
            [1, 5, 9, 13],
            [2, 6, 10, 14],
            [3, 7, 11, 15],
            [0, 5, 10, 15],
            [1, 6, 11, 12],
            [2, 7, 8, 13],
            [3, 4, 9, 14],
        ]
        .into_iter()
        .zip(m.as_chunks::<2>().0)
        {
            g(&mut v, indices, pair[0], pair[1]);
        }
        m = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8].map(|i| m[i]);
    }
    core::array::from_fn(|i| {
        if i < 8 {
            v[i] ^ v[i + 8]
        } else {
            v[i] ^ cv[i - 8]
        }
    })
}

proptest! {
    #[test]
    fn scalar_matches_official_portable_compression(
        cv in any::<[u32; 8]>(), block in any::<[u32; 16]>(),
        counter in any::<u64>(), len in 0u8..=64, flags in 0u8..=127,
    ) {
        let bytes: [u8; 64] = core::array::from_fn(|i| block[i / 4].to_le_bytes()[i % 4]);
        let official = blake3::platform::Platform::Portable.compress_xof(&cv, &bytes, len, counter, flags);
        let expected: [u32; 16] = core::array::from_fn(|i| u32::from_le_bytes(official[4*i..4*i+4].try_into().unwrap()));
        let params = [counter as u32, (counter >> 32) as u32, u32::from(len), u32::from(flags)];
        prop_assert_eq!(reference(cv, block, params), expected);
        prop_assert_eq!(compress_raw_xof(&cv, &block, &params), expected);
        prop_assert_eq!(&compress_raw(&cv, &block, &params)[..], &expected[..8]);
    }

    #[test]
    fn arbitrary_parameter_words_are_preserved(
        cv in any::<[u32; 8]>(), block in any::<[u32; 16]>(), params in any::<[u32; 4]>(),
    ) {
        let expected = reference(cv, block, params);
        prop_assert_eq!(compress_raw_xof(&cv, &block, &params), expected);
        prop_assert_eq!(&compress_raw(&cv, &block, &params)[..], &expected[..8]);
    }
}

fn check_packed<const L: usize>() {
    // Every CV word and block word differs across lanes, including the high bit.
    let cv: [[u32; L]; 8] = core::array::from_fn(|w| {
        core::array::from_fn(|l| {
            (w as u32).wrapping_mul(0x12345679) ^ (l as u32).wrapping_mul(0x87654321)
        })
    });
    let block: [[u32; L]; 16] = core::array::from_fn(|w| {
        core::array::from_fn(|l| {
            (w as u32).wrapping_mul(0xfedcba97) ^ (l as u32).wrapping_mul(0x13579bdf)
        })
    });
    for params in [
        [0, 0, 64, 11],
        [u32::MAX, 0x87654321, 0x80000040, 0xffffff80],
    ] {
        let expected_lanes: [[u32; 16]; L] = core::array::from_fn(|l| {
            reference(cv.map(|word| word[l]), block.map(|word| word[l]), params)
        });
        let expected: [[u32; L]; 8] =
            core::array::from_fn(|w| core::array::from_fn(|l| expected_lanes[l][w]));
        assert_eq!(compress_raw_packed(&cv, &block, &params), expected);
        // Exercise every available backend, including those runtime detection skips.
        for kernel in crate::batch::supported() {
            assert_eq!(
                kernel.compress_raw_packed(&cv, &block, &params),
                expected,
                "{kernel:?}, L={L}"
            );
        }
    }
}

#[test]
fn packed_matches_independent_compressions_at_lane_boundaries() {
    check_packed::<0>();
    check_packed::<1>();
    check_packed::<3>();
    check_packed::<4>();
    check_packed::<5>();
    check_packed::<7>();
    check_packed::<8>();
    check_packed::<9>();
    check_packed::<15>();
    check_packed::<16>();
    check_packed::<17>();
    check_packed::<31>();
    check_packed::<32>();
    check_packed::<33>();
    check_packed::<47>();
    check_packed::<48>();
    check_packed::<63>();
    check_packed::<64>();
    check_packed::<65>();
}
