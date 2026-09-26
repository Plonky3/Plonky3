//! The BLAKE2s compression function, run on any number of messages at once.
//!
//! Every operation is applied to a lane array rather than to one word, so the same code
//! compresses one message or sixteen. The compiler fills those arrays with whatever vector
//! unit the target has; nothing here names an instruction set.

/// Bytes in one compression block.
pub(crate) const BLOCK_BYTES: usize = 64;

/// Words in one compression block.
const BLOCK_WORDS: usize = 16;

/// Rounds in one compression.
const ROUNDS: usize = 10;

/// The initialization vector, the same eight words SHA-256 starts from.
const IV: [u32; 8] = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// The parameter block of an unkeyed BLAKE2s-256, XORed into the first state word.
///
/// It reads as digest length, key length, fanout, depth: 32-byte digests, no key, and the
/// sequential tree shape.
const PARAM_BLOCK_0: u32 = 0x0101_0020;

/// Which message word each of the sixteen G inputs reads, for each round.
const SIGMA: [[usize; BLOCK_WORDS]; ROUNDS] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
];

/// Which state words each G step of a round mixes: four columns, then four diagonals.
const G_SCHEDULE: [[usize; 4]; 8] = [
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [1, 6, 11, 12],
    [2, 7, 8, 13],
    [3, 4, 9, 14],
];

/// A chaining value held for `N` messages at once, one lane per message.
pub(crate) type State<const N: usize> = [[u32; N]; 8];

/// One message block held for `N` messages at once.
pub(crate) type Block<const N: usize> = [[u32; N]; BLOCK_WORDS];

/// The chaining value every unkeyed BLAKE2s-256 message starts from.
#[inline]
pub(crate) const fn initial_state<const N: usize>() -> State<N> {
    let mut state = [[0u32; N]; 8];
    let mut word = 0;
    while word < 8 {
        let value = if word == 0 {
            IV[0] ^ PARAM_BLOCK_0
        } else {
            IV[word]
        };
        state[word] = [value; N];
        word += 1;
    }
    state
}

/// Advance `N` states by one block each.
///
/// The lanes share `counter` and `last`, which is what makes the batch legal: they are the
/// same block index of messages of the same length.
#[inline]
pub(crate) fn compress_lanes<const N: usize>(
    state: &mut State<N>,
    block: &Block<N>,
    counter: u64,
    last: bool,
) {
    // The working state: the chaining value, then the initialization vector with the
    // counter halves and the finalization flag folded in. RFC 7693 section 3.2 inverts
    // v[14] on the last block and leaves v[15] alone.
    let mut v = [[0u32; N]; BLOCK_WORDS];
    v[..8].copy_from_slice(state);
    for (word, iv) in v[8..].iter_mut().zip(IV) {
        *word = [iv; N];
    }
    xor_constant(&mut v[12], counter as u32);
    xor_constant(&mut v[13], (counter >> 32) as u32);
    if last {
        xor_constant(&mut v[14], u32::MAX);
    }

    for schedule in SIGMA {
        for (step, [a, b, c, d]) in G_SCHEDULE.into_iter().enumerate() {
            g(
                &mut v,
                [a, b, c, d],
                block[schedule[2 * step]],
                block[schedule[2 * step + 1]],
            );
        }
    }

    // h'[i] = h[i] ^ v[i] ^ v[i + 8].
    for (word, (low, high)) in state.iter_mut().zip(v[..8].iter().zip(&v[8..])) {
        for lane in 0..N {
            word[lane] ^= low[lane] ^ high[lane];
        }
    }
}

/// XOR one constant into every lane of a word.
#[inline(always)]
fn xor_constant<const N: usize>(word: &mut [u32; N], value: u32) {
    for lane in word {
        *lane ^= value;
    }
}

/// The BLAKE2s mixing function, on every lane at once.
#[inline(always)]
fn g<const N: usize>(
    v: &mut [[u32; N]; BLOCK_WORDS],
    [a, b, c, d]: [usize; 4],
    mx: [u32; N],
    my: [u32; N],
) {
    for lane in 0..N {
        let (mut va, mut vb, mut vc, mut vd) = (v[a][lane], v[b][lane], v[c][lane], v[d][lane]);

        va = va.wrapping_add(vb).wrapping_add(mx[lane]);
        vd = (vd ^ va).rotate_right(16);
        vc = vc.wrapping_add(vd);
        vb = (vb ^ vc).rotate_right(12);

        va = va.wrapping_add(vb).wrapping_add(my[lane]);
        vd = (vd ^ va).rotate_right(8);
        vc = vc.wrapping_add(vd);
        vb = (vb ^ vc).rotate_right(7);

        (v[a][lane], v[b][lane], v[c][lane], v[d][lane]) = (va, vb, vc, vd);
    }
}
