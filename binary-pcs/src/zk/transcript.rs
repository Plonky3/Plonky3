//! Protocol v2 hash domains and complete-prefix counter-mode public coins.
//!
//! Transcript records: field = 1 || repr128LE; cap = 2 || count64LE || roots;
//! bytes = 3 || length64LE || bytes; usize = 4 || value64LE; draw =
//! 5 || label_length64LE || label || output_length64LE || complete_raw_output.
//! All lengths and ordinals are checked u64; block counters are u32LE. A draw
//! hashes the unchanged complete prefix for every block, then records its output.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_challenger::{CanObserve, CanSample, CanSampleBits};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CryptographicHasher, MerkleCap};

use super::{Val, ZkError};

macro_rules! role_hasher {
    ($name:ident, $prefix:literal) => {
        #[derive(Clone, Copy, Debug)]
        pub(super) struct $name;
        impl CryptographicHasher<u8, [u8; 32]> for $name {
            fn hash_iter<I: IntoIterator<Item = u8>>(&self, input: I) -> [u8; 32] {
                Keccak256Hash.hash_iter($prefix.iter().copied().chain(input))
            }
        }
    };
}
role_hasher!(LeafHash, b"p3-binary-zk-v2/leaf\0");
role_hasher!(NodeHash, b"p3-binary-zk-v2/node\0");
role_hasher!(FiatShamirHash, b"p3-binary-zk-v2/fiat-shamir\0");

/// Protocol-owned complete public transcript. Cloning replays public coins only;
/// private proving randomness must never be derived from this state.
#[derive(Clone, Debug)]
pub struct Challenger {
    transcript: Vec<u8>,
    draw_ordinal: u64,
    #[cfg(test)]
    pub(super) draws: Vec<RecordedDraw>,
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(super) struct RecordedDraw {
    pub(super) prefix: Vec<u8>,
    pub(super) label: Vec<u8>,
    pub(super) ordinal: u64,
    pub(super) raw: Vec<u8>,
}

/// Start a fresh protocol v2 transcript.
pub const fn challenger() -> Challenger {
    Challenger {
        transcript: Vec::new(),
        draw_ordinal: 0,
        #[cfg(test)]
        draws: Vec::new(),
    }
}

fn fixed(value: usize) -> [u8; 8] {
    u64::try_from(value)
        .expect("protocol length exceeds u64")
        .to_le_bytes()
}

impl Challenger {
    pub(super) fn observe_bytes(&mut self, bytes: &[u8]) {
        self.transcript.push(3);
        self.transcript.extend_from_slice(&fixed(bytes.len()));
        self.transcript.extend_from_slice(bytes);
    }

    pub(super) fn observe_usize(&mut self, value: usize) {
        self.transcript.push(4);
        self.transcript.extend_from_slice(&fixed(value));
    }

    fn draw(&mut self, label: &[u8], output_len: usize) -> Vec<u8> {
        let mut input = Vec::new();
        input.extend_from_slice(&fixed(label.len()));
        input.extend_from_slice(label);
        input.extend_from_slice(&fixed(self.transcript.len()));
        input.extend_from_slice(&self.transcript);
        input.extend_from_slice(&self.draw_ordinal.to_le_bytes());
        input.extend_from_slice(&fixed(output_len));
        let mut output = Vec::with_capacity(output_len);
        for block in 0..output_len.div_ceil(32) {
            let counter = u32::try_from(block).expect("protocol draw exceeds u32 blocks");
            let digest =
                FiatShamirHash.hash_iter(input.iter().copied().chain(counter.to_le_bytes()));
            output.extend_from_slice(&digest[..(output_len - output.len()).min(32)]);
        }
        #[cfg(test)]
        self.draws.push(RecordedDraw {
            prefix: self.transcript.clone(),
            label: label.to_vec(),
            ordinal: self.draw_ordinal,
            raw: output.clone(),
        });
        self.transcript.push(5);
        self.transcript.extend_from_slice(&fixed(label.len()));
        self.transcript.extend_from_slice(label);
        self.transcript.extend_from_slice(&fixed(output_len));
        self.transcript.extend_from_slice(&output);
        self.draw_ordinal = self
            .draw_ordinal
            .checked_add(1)
            .expect("protocol draw ordinal overflow");
        output
    }

    pub(super) fn sample_accepted(
        &mut self,
        label: &'static [u8],
        error: &'static str,
        accepts: impl Fn(Val) -> bool,
    ) -> Result<Val, ZkError> {
        let raw = self.draw(label, 16 * 16);
        select_candidate(&raw, error, accepts)
    }

    pub(super) fn sample_nonzero(&mut self) -> Result<Val, ZkError> {
        self.sample_accepted(b"blend-scalar", "blend-scalar", |v| v != Val::ZERO)
    }

    pub(super) fn sample_indices(&mut self, q: usize, log_n: usize) -> Vec<usize> {
        assert!(log_n < 64 && log_n < usize::BITS as usize);
        let mut label = b"query-indices".to_vec();
        label.extend_from_slice(&fixed(q));
        label.extend_from_slice(&fixed(log_n));
        self.draw(&label, q.checked_mul(8).expect("query byte count overflow"))
            .as_chunks::<8>()
            .0
            .iter()
            .map(|chunk| {
                let raw = u64::from_le_bytes(*chunk);
                usize::try_from(raw & ((1u64 << log_n) - 1)).expect("bounded row index")
            })
            .collect()
    }
}

fn select_candidate(
    raw: &[u8],
    error: &'static str,
    accepts: impl Fn(Val) -> bool,
) -> Result<Val, ZkError> {
    raw.as_chunks::<16>()
        .0
        .iter()
        .map(|bytes| Val::from_repr(u128::from_le_bytes(*bytes)))
        .find(|&v| accepts(v))
        .ok_or(ZkError::ChallengeExhausted(error))
}

impl CanObserve<Val> for Challenger {
    fn observe(&mut self, value: Val) {
        self.transcript.push(1);
        self.transcript
            .extend_from_slice(&value.to_repr().to_le_bytes());
    }
}
impl CanObserve<&MerkleCap<Val, [u8; 32]>> for Challenger {
    fn observe(&mut self, cap: &MerkleCap<Val, [u8; 32]>) {
        self.transcript.push(2);
        self.transcript.extend_from_slice(&fixed(cap.num_roots()));
        for root in cap.roots() {
            self.transcript.extend_from_slice(root);
        }
    }
}
impl CanSample<Val> for Challenger {
    fn sample(&mut self) -> Val {
        Val::from_repr(u128::from_le_bytes(
            self.draw(b"generic-field", 16)
                .try_into()
                .expect("field draw"),
        ))
    }
}
impl CanSampleBits<usize> for Challenger {
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits <= usize::BITS as usize);
        let mut label = b"generic-bits".to_vec();
        label.extend_from_slice(&fixed(bits));
        let raw = u64::from_le_bytes(self.draw(&label, 8).try_into().expect("bits draw"));
        let mask = u64::MAX >> (64 - bits).min(63);
        if bits == 0 {
            0
        } else {
            usize::try_from(raw & mask).expect("bounded bits")
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn hash_roles_tag_scalar_and_batched_paths() {
        fn check<H: CryptographicHasher<u8, [u8; 32]>>(hash: &H, tag: &[u8]) {
            let input = [1, 2, 3, 4, 5, 6];
            let mut actual = [[0; 32]; 2];
            hash.hash_many(&input, &mut actual);
            for (i, bytes) in input.as_chunks::<3>().0.iter().enumerate() {
                let mut tagged = tag.to_vec();
                tagged.extend_from_slice(bytes);
                assert_eq!(actual[i], Keccak256Hash.hash_slice(&tagged));
                assert_eq!(actual[i], hash.hash_slice(bytes));
            }
            let mut empty = [[0; 32]; 2];
            hash.hash_many(&[], &mut empty);
            assert_eq!(empty, [Keccak256Hash.hash_slice(tag); 2]);
            assert_eq!(H::LANES, 1);
        }
        check(&LeafHash, b"p3-binary-zk-v2/leaf\0");
        check(&NodeHash, b"p3-binary-zk-v2/node\0");
        check(&FiatShamirHash, b"p3-binary-zk-v2/fiat-shamir\0");
        let hashes = [
            LeafHash.hash_slice(b"x"),
            NodeHash.hash_slice(b"x"),
            FiatShamirHash.hash_slice(b"x"),
        ];
        assert!(hashes[0] != hashes[1] && hashes[0] != hashes[2] && hashes[1] != hashes[2]);
    }

    fn independent_block(
        prefix: &[u8],
        label: &[u8],
        ordinal: u64,
        len: u64,
        block: u32,
    ) -> [u8; 32] {
        let mut input = b"p3-binary-zk-v2/fiat-shamir\0".to_vec();
        input.extend_from_slice(&(label.len() as u64).to_le_bytes());
        input.extend_from_slice(label);
        input.extend_from_slice(&(prefix.len() as u64).to_le_bytes());
        input.extend_from_slice(prefix);
        input.extend_from_slice(&ordinal.to_le_bytes());
        input.extend_from_slice(&len.to_le_bytes());
        input.extend_from_slice(&block.to_le_bytes());
        Keccak256Hash.hash_slice(&input)
    }

    #[test]
    fn full_prefix_multiblock_known_answer_and_draw_records() {
        let mut c = challenger();
        c.observe(Val::from_repr(7));
        let mut prefix = vec![1];
        prefix.extend_from_slice(&7u128.to_le_bytes());
        // Compute blocks out of order; every block uses exactly the same prefix.
        let third = independent_block(&prefix, b"vector", 0, 70, 2);
        let first = independent_block(&prefix, b"vector", 0, 70, 0);
        let second = independent_block(&prefix, b"vector", 0, 70, 1);
        let mut expected = first.to_vec();
        expected.extend_from_slice(&second);
        expected.extend_from_slice(&third[..6]);
        let raw = c.draw(b"vector", 70);
        assert_eq!(raw, expected);
        // Pinned portable byte vector: fixed prefix, label, length and counters.
        assert_eq!(
            raw,
            [
                94, 104, 40, 4, 55, 146, 42, 123, 92, 185, 124, 121, 152, 73, 8, 217, 119, 222,
                149, 18, 5, 250, 182, 72, 72, 41, 155, 75, 219, 75, 78, 182, 116, 91, 200, 15, 190,
                64, 74, 144, 128, 206, 198, 97, 36, 211, 76, 50, 247, 53, 221, 223, 77, 44, 94,
                218, 154, 87, 4, 119, 92, 55, 10, 170, 57, 218, 189, 43, 141, 170
            ]
        );
        prefix.push(5);
        prefix.extend_from_slice(&6u64.to_le_bytes());
        prefix.extend_from_slice(b"vector");
        prefix.extend_from_slice(&70u64.to_le_bytes());
        prefix.extend_from_slice(&raw);
        c.observe_bytes(b"after");
        prefix.push(3);
        prefix.extend_from_slice(&5u64.to_le_bytes());
        prefix.extend_from_slice(b"after");
        assert_eq!(c.transcript, prefix);
        assert_eq!(
            c.draw(b"next", 16),
            independent_block(&prefix, b"next", 1, 16, 0)[..16]
        );
    }

    #[test]
    fn observation_boundaries_roles_lengths_and_labels_are_distinct() {
        let mut a = challenger();
        a.observe_bytes(b"a");
        a.observe_bytes(b"bc");
        let mut b = challenger();
        b.observe_bytes(b"ab");
        b.observe_bytes(b"c");
        assert_ne!(a.draw(b"x", 64), b.draw(b"x", 64));
        let mut c = challenger();
        c.observe(Val::from_repr(1));
        let mut d = challenger();
        d.observe_usize(1);
        let mut e = challenger();
        e.observe_bytes(&1u128.to_le_bytes());
        let mut f = challenger();
        f.observe(&MerkleCap::new(vec![[0; 32]]));
        let mut g = challenger();
        g.observe_bytes(&[0; 32]);
        assert_ne!(c.clone().draw(b"x", 32), d.draw(b"x", 32));
        assert_ne!(c.draw(b"x", 32), e.draw(b"x", 32));
        assert_ne!(f.draw(b"x", 32), g.draw(b"x", 32));
        assert_ne!(challenger().draw(b"a", 32), challenger().draw(b"b", 32));
        assert_ne!(
            challenger().draw(b"a", 32),
            challenger().draw(b"a", 64)[..32]
        );
        let mut early_a = challenger();
        early_a.observe_bytes(b"a");
        early_a.draw(b"one", 256);
        let mut early_b = challenger();
        early_b.observe_bytes(b"b");
        early_b.draw(b"one", 256);
        assert_ne!(early_a.draw(b"later", 32), early_b.draw(b"later", 32));
    }

    #[test]
    fn query_batch_is_one_draw_with_bound_geometry_and_replays() {
        let mut c = challenger();
        c.observe_bytes(b"queries");
        let prefix = c.transcript.clone();
        let mut label = b"query-indices".to_vec();
        label.extend_from_slice(&9u64.to_le_bytes());
        label.extend_from_slice(&6u64.to_le_bytes());
        let raw: Vec<_> = (0..3)
            .flat_map(|i| independent_block(&prefix, &label, 0, 72, i))
            .take(72)
            .collect();
        let expected: Vec<_> = raw
            .as_chunks::<8>()
            .0
            .iter()
            .map(|v| (u64::from_le_bytes(*v) & 63) as usize)
            .collect();
        let mut replay = c.clone();
        assert_eq!(c.sample_indices(9, 6), expected);
        assert_eq!(c.draw_ordinal, 1);
        assert_eq!(replay.sample_indices(9, 6), expected);
        assert_eq!(c.draw(b"after", 80), replay.draw(b"after", 80));
        let mut changed_q = challenger();
        changed_q.observe_bytes(b"queries");
        assert_ne!(changed_q.sample_indices(8, 6), expected[..8]);
    }

    #[test]
    fn fixed_rejection_batch_selects_first_valid_or_returns_typed_error() {
        let mut bytes = vec![0; 256];
        bytes[32..48].copy_from_slice(&7u128.to_le_bytes());
        bytes[48..64].copy_from_slice(&9u128.to_le_bytes());
        assert_eq!(
            select_candidate(&bytes, "blend-scalar", |v| v != Val::ZERO),
            Ok(Val::from_repr(7))
        );
        assert_eq!(
            select_candidate(&[0; 256], "blend-scalar", |v| v != Val::ZERO),
            Err(ZkError::ChallengeExhausted("blend-scalar"))
        );
        // Trace H={0,1}; reject both members, then accept the first outside H.
        bytes[16..32].copy_from_slice(&1u128.to_le_bytes());
        let outside_h = |v: Val| v * (v + Val::ONE) != Val::ZERO;
        assert_eq!(
            select_candidate(&bytes, "air-point", outside_h),
            Ok(Val::from_repr(7))
        );
        assert_eq!(
            select_candidate(&[0; 256], "air-point", outside_h),
            Err(ZkError::ChallengeExhausted("air-point"))
        );
        let mut c = challenger();
        let prefix = c.transcript.clone();
        let mut expected = Vec::new();
        for block in 0..8 {
            expected.extend_from_slice(&independent_block(&prefix, b"air-point", 0, 256, block));
        }
        assert_eq!(
            c.sample_accepted(b"air-point", "air-point", outside_h),
            select_candidate(&expected, "air-point", outside_h)
        );
        assert_eq!(c.draw_ordinal, 1);
        assert!(c.transcript.ends_with(&expected));
    }

    #[test]
    fn generic_bits_use_separate_framed_draws_and_cover_edge_widths() {
        for bits in [0, 1, 16, usize::BITS as usize] {
            let mut c = challenger();
            let value = c.sample_bits(bits);
            if bits < usize::BITS as usize {
                assert!(value < (1usize << bits));
            }
            assert_eq!(c.draw_ordinal, 1);
            assert_eq!(value, challenger().sample_bits(bits));
        }
    }
}
