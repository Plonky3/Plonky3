//! The SHA-256 compression function (FIPS 180-4, Section 6.2.2): the message schedule, 64 rounds and the feed-forward.

pub use super::arx::IV;
use super::{Gadget, pack_u32s, unpack_u32s};
use crate::circuit::{Builder, Circuit, Word};

/// The round constants (FIPS 180-4, Section 4.2.2).
pub(crate) const K: [u32; 64] = [
    0x428A_2F98,
    0x7137_4491,
    0xB5C0_FBCF,
    0xE9B5_DBA5,
    0x3956_C25B,
    0x59F1_11F1,
    0x923F_82A4,
    0xAB1C_5ED5,
    0xD807_AA98,
    0x1283_5B01,
    0x2431_85BE,
    0x550C_7DC3,
    0x72BE_5D74,
    0x80DE_B1FE,
    0x9BDC_06A7,
    0xC19B_F174,
    0xE49B_69C1,
    0xEFBE_4786,
    0x0FC1_9DC6,
    0x240C_A1CC,
    0x2DE9_2C6F,
    0x4A74_84AA,
    0x5CB0_A9DC,
    0x76F9_88DA,
    0x983E_5152,
    0xA831_C66D,
    0xB003_27C8,
    0xBF59_7FC7,
    0xC6E0_0BF3,
    0xD5A7_9147,
    0x06CA_6351,
    0x1429_2967,
    0x27B7_0A85,
    0x2E1B_2138,
    0x4D2C_6DFC,
    0x5338_0D13,
    0x650A_7354,
    0x766A_0ABB,
    0x81C2_C92E,
    0x9272_2C85,
    0xA2BF_E8A1,
    0xA81A_664B,
    0xC24B_8B70,
    0xC76C_51A3,
    0xD192_E819,
    0xD699_0624,
    0xF40E_3585,
    0x106A_A070,
    0x19A4_C116,
    0x1E37_6C08,
    0x2748_774C,
    0x34B0_BCB5,
    0x391C_0CB3,
    0x4ED8_AA4A,
    0x5B9C_CA4F,
    0x682E_6FF3,
    0x748F_82EE,
    0x78A5_636F,
    0x84C8_7814,
    0x8CC7_0208,
    0x90BE_FFFA,
    0xA450_6CEB,
    0xBEF9_A3F7,
    0xC671_78F2,
];

/// One compression: the new chaining value from `state` and one block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sha256Compress {
    /// The input chaining value.
    pub state: [u32; 8],
    /// The message block, as big-endian words.
    pub block: [u32; 16],
}

/// Three rotations of `x` XORed, free (FIPS 180-4, Section 4.1.2).
fn sigma3(c: &mut Builder, x: &Word, r: [usize; 3], shift: bool) -> Word {
    // The small sigmas shift instead of rotating the third time.
    let third = if shift {
        x >> r[2]
    } else {
        x.rotate_right(r[2])
    };
    let ab = c.xor_word(&x.rotate_right(r[0]), &x.rotate_right(r[1]));
    c.xor_word(&ab, &third)
}

impl Gadget for Sha256Compress {
    /// The chaining value, then the block.
    const INPUT_WORDS: usize = 12;

    /// The new chaining value.
    type Output = [u32; 8];

    fn circuit() -> Circuit {
        let mut c = Builder::new(&[256, 512], &[256]);
        let state: [Word; 8] = Word::split_u32(c.input(0)).try_into().expect("eight words");
        let block: [Word; 16] = Word::split_u32(c.input(1))
            .try_into()
            .expect("sixteen words");
        let out = compress(&mut c, &state, &block);
        for (i, word) in out.iter().enumerate() {
            for (bit, &wire) in word.iter().enumerate() {
                c.output(0, 32 * i + bit, wire);
            }
        }
        c.finish()
    }

    fn write_input(&self, words: &mut [u64]) {
        pack_u32s(&self.state, &mut words[..4]);
        pack_u32s(&self.block, &mut words[4..12]);
    }

    fn decode_output(words: &[u64]) -> [u32; 8] {
        unpack_u32s(words)
    }

    fn write_output(output: &[u32; 8], words: &mut [u64]) {
        pack_u32s(output, words);
    }
}

/// The compression on circuit words: the new chaining value from `state` and a block of big-endian words.
pub fn compress(c: &mut Builder, state: &[Word; 8], block: &[Word; 16]) -> [Word; 8] {
    // The schedule: one four-operand sum per word.
    let mut w = block.to_vec();
    for t in 16..64 {
        let s0 = sigma3(c, &w[t - 15], [7, 18, 3], true);
        let s1 = sigma3(c, &w[t - 2], [17, 19, 10], true);
        let next = c.wrapping_sum(&[&s1, &w[t - 7], &s0, &w[t - 16]]);
        w.push(next);
    }

    // Rounds: `T1` feeds both new words.
    let mut v = state.clone();
    for (t, &k) in K.iter().enumerate() {
        let [a, b, cc, d, e, f, g, h] = &v;
        let s1 = sigma3(c, e, [6, 11, 25], false);
        let ch = c.ch(e, f, g);
        let k = c.literal(u64::from(k), 32);
        let t1 = c.wrapping_sum(&[h, &s1, &ch, &k, &w[t]]);
        let s0 = sigma3(c, a, [2, 13, 22], false);
        let maj = c.maj(a, b, cc);
        let new_e = c.wrapping_sum(&[d, &t1]);
        let new_a = c.wrapping_sum(&[&t1, &s0, &maj]);
        // The words shift down a place; `a` and `e` take the new values.
        v.rotate_right(1);
        (v[0], v[4]) = (new_a, new_e);
    }

    // The feed-forward.
    core::array::from_fn(|i| c.wrapping_sum(&[&state[i], &v[i]]))
}
