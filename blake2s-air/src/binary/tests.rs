use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use core::array;
use core::borrow::{Borrow, BorrowMut};

use blake2::digest::consts::U32;
use blake2::{Blake2s, Digest};
use hex_literal::hex;
use p3_air::{
    AirLayout, BaseAir, BaseEntry, BaseLeaf, ConstraintFailure, SymbolicExpr,
    check_all_constraints, check_constraints, get_max_constraint_degree, get_symbolic_constraints,
};
use p3_binary_field::BinaryField128;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use super::air::NUM_INPUT_BITS;
use super::{
    Blake2sBinaryAir, Blake2sBinaryCols, Blake2sCompressionInput, NUM_BLAKE2S_BINARY_COLS,
    generate_binary_trace_packed, generate_binary_trace_rows,
};
use crate::constants::{IV, SIGMA};

type F = BinaryField128;

/// Bytes hashed by one compression, from RFC 7693 section 2.1.
const BLOCK_BYTES: usize = 64;

/// BLAKE2s-256 of "abc", from RFC 7693 appendix B.
const ABC_DIGEST: [u8; 32] =
    hex!("508C5E8C327C14E2E1A72BA34EEB452F37458B209ED63A294D999B4C86675982");

/// The working state while hashing "abc", from RFC 7693 appendix B.
///
/// Entry 0 is the state before the first round.
///
/// Entry `r` is the state after round `r - 1`.
const ABC_ROUND_STATES: [[u32; 16]; 11] = [
    [
        0x6B08E647, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19, 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527C, 0x9B05688C,
        0xE07C2654, 0x5BE0CD19,
    ],
    [
        0x16A3242E, 0xD7B5E238, 0xCE8CE24B, 0x927AEDE1, 0xA7B430D9, 0x93A4A14E, 0xA44E7C31,
        0x41D4759B, 0x95BF33D3, 0x9A99C181, 0x608A3A6B, 0xB666383E, 0x7A8DD50F, 0xBE378ED7,
        0x353D1EE6, 0x3BB44C6B,
    ],
    [
        0x3AE30FE3, 0x0982A96B, 0xE88185B4, 0x3E339B16, 0xF24338CD, 0x0E66D326, 0xE005ED0C,
        0xD591A277, 0x180B1F3A, 0xFCF43914, 0x30DB62D6, 0x4847831C, 0x7F00C58E, 0xFB847886,
        0xC544E836, 0x524AB0E2,
    ],
    [
        0x7A3BE783, 0x997546C1, 0xD45246DF, 0xEDB5F821, 0x7F98A742, 0x10E864E2, 0xD4AB70D0,
        0xC63CB1AB, 0x6038DA9E, 0x414594B0, 0xF2C218B5, 0x8DA0DCB7, 0xD7CD7AF5, 0xAB4909DF,
        0x85031A52, 0xC4EDFC98,
    ],
    [
        0x2A8B8CB7, 0x1ACA82B2, 0x14045D7F, 0xCC7258ED, 0x383CF67C, 0xE090E7F9, 0x3025D276,
        0x57D04DE4, 0x994BACF0, 0xF0982759, 0xF17EE300, 0xD48FC2D5, 0xDC854C10, 0x523898A9,
        0xC03A0F89, 0x47D6CD88,
    ],
    [
        0xC4AA2DDB, 0x111343A3, 0xD54A700A, 0x574A00A9, 0x857D5A48, 0xB1E11989, 0x6F5C52DF,
        0xDD2C53A3, 0x678E5F8E, 0x9718D4E9, 0x622CB684, 0x92976076, 0x0E41A517, 0x359DC2BE,
        0x87A87DDD, 0x643F9CEC,
    ],
    [
        0x3453921C, 0xD7595EE1, 0x592E776D, 0x3ED6A974, 0x4D997CB3, 0xDE9212C3, 0x35ADF5C9,
        0x9916FD65, 0x96562E89, 0x4EAD0792, 0xEBFC2712, 0x2385F5B2, 0xF34600FB, 0xD7BC20FB,
        0xEB452A7B, 0xECE1AA40,
    ],
    [
        0xBE851B2D, 0xA85F6358, 0x81E6FC3B, 0x0BB28000, 0xFA55A33A, 0x87BE1FAD, 0x4119370F,
        0x1E2261AA, 0xA1318FD3, 0xF4329816, 0x071783C2, 0x6E536A8D, 0x9A81A601, 0xE7EC80F1,
        0xACC09948, 0xF849A584,
    ],
    [
        0x07E5B85A, 0x069CC164, 0xF9DE3141, 0xA56F4680, 0x9E440AD2, 0x9AB659EA, 0x3C84B971,
        0x21DBD9CF, 0x46699F8C, 0x765257EC, 0xAF1D998C, 0x75E4C3B6, 0x523878DC, 0x30715015,
        0x397FEE81, 0x4F1FA799,
    ],
    [
        0x435148C4, 0xA5AA2D11, 0x4B354173, 0xD543BC9E, 0xBDA2591C, 0xBF1D2569, 0x4FCB3120,
        0x707ADA48, 0x565B3FDE, 0x32C9C916, 0xEAF4A1AB, 0xB1018F28, 0x8078D978, 0x68ADE4B5,
        0x9778FDA3, 0x2863B92E,
    ],
    [
        0xD9C994AA, 0xCFEC3AA6, 0x700D0AB2, 0x2C38670E, 0xAF6A1F66, 0x1D023EF3, 0x1D9EC27D,
        0x945357A5, 0x3E9FFEBD, 0x969FE811, 0xEF485E21, 0xA632797A, 0xDEEF082E, 0xAF3D80E1,
        0x4E86829B, 0x4DEAFD3A,
    ],
];

/// The chaining value after hashing "abc", from RFC 7693 appendix B.
const ABC_OUTPUT: [u32; 8] = [
    0x8C5E8C50, 0xE2147C32, 0xA32BA7E1, 0x2F45EB4E, 0x208B4537, 0x293AD69E, 0x4C9B994D, 0x82596786,
];

/// The self-test result, from RFC 7693 appendix E.
///
/// It is the BLAKE2s-256 digest of 48 digests, keyed and unkeyed, concatenated.
const SELF_TEST_DIGEST: [u8; 32] =
    hex!("6A411F08CE25ADCDFB02ABA641451CEC53C598B24F4FC787FBDC88797F4C1DFE");

/// The BLAKE2s G function on the state words at `a`, `b`, `c`, `d`, from RFC 7693 section 3.1.
fn reference_g(v: &mut [u32; 16], [a, b, c, d]: [usize; 4], mx: u32, my: u32) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(mx);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(12);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(my);
    v[d] = (v[d] ^ v[a]).rotate_right(8);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(7);
}

/// The working state before the first round, from RFC 7693 section 3.2.
fn reference_initial_state(input: &Blake2sCompressionInput) -> [u32; 16] {
    let cv = input.chaining_value;
    // A set flag inverts every bit of its word.
    let last_block = if input.last_block { u32::MAX } else { 0 };

    [
        cv[0],
        cv[1],
        cv[2],
        cv[3],
        cv[4],
        cv[5],
        cv[6],
        cv[7],
        IV[0],
        IV[1],
        IV[2],
        IV[3],
        IV[4] ^ (input.counter as u32),
        IV[5] ^ ((input.counter >> 32) as u32),
        IV[6] ^ last_block,
        IV[7],
    ]
}

/// The BLAKE2s compression function on plain words, from RFC 7693 section 3.2.
fn reference_compress(input: &Blake2sCompressionInput) -> [u32; 8] {
    let mut v = reference_initial_state(input);
    let m = input.block;
    // Four column steps, then four diagonal steps, in each of the ten rounds.
    for schedule in SIGMA {
        reference_g(&mut v, [0, 4, 8, 12], m[schedule[0]], m[schedule[1]]);
        reference_g(&mut v, [1, 5, 9, 13], m[schedule[2]], m[schedule[3]]);
        reference_g(&mut v, [2, 6, 10, 14], m[schedule[4]], m[schedule[5]]);
        reference_g(&mut v, [3, 7, 11, 15], m[schedule[6]], m[schedule[7]]);
        reference_g(&mut v, [0, 5, 10, 15], m[schedule[8]], m[schedule[9]]);
        reference_g(&mut v, [1, 6, 11, 12], m[schedule[10]], m[schedule[11]]);
        reference_g(&mut v, [2, 7, 8, 13], m[schedule[12]], m[schedule[13]]);
        reference_g(&mut v, [3, 4, 9, 14], m[schedule[14]], m[schedule[15]]);
    }
    // Fold both halves of the state into the chaining value.
    let cv = input.chaining_value;
    array::from_fn(|i| cv[i] ^ v[i] ^ v[i + 8])
}

/// A 64-byte block read as sixteen little-endian words, from RFC 7693 section 2.4.
fn block_words(bytes: &[u8; BLOCK_BYTES]) -> [u32; 16] {
    array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

/// The compression inputs of a BLAKE2s hash, from RFC 7693 section 3.3.
///
/// Every input carries the IV as its chaining value.
///
/// The caller replaces it with the previous output while running the chain.
fn hash_inputs(key: &[u8], message: &[u8]) -> Vec<Blake2sCompressionInput> {
    // The key, when present, fills a block of its own, padded with zeros.
    let mut padded = Vec::new();
    if !key.is_empty() {
        padded.extend_from_slice(key);
        padded.resize(BLOCK_BYTES, 0);
    }

    // The message follows, padded with zeros to whole blocks.
    //
    // An unkeyed empty message still compresses one all-zero block.
    padded.extend_from_slice(message);
    let num_blocks = padded.len().div_ceil(BLOCK_BYTES).max(1);
    padded.resize(num_blocks * BLOCK_BYTES, 0);

    // The counter of the last block counts only real bytes: the whole key block, then the message.
    let key_block_len = if key.is_empty() { 0 } else { BLOCK_BYTES };
    let total_len = (key_block_len + message.len()) as u64;

    padded
        .as_chunks::<BLOCK_BYTES>()
        .0
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            let last = i + 1 == num_blocks;
            Blake2sCompressionInput {
                chaining_value: IV,
                block: block_words(bytes),
                // Earlier blocks count every byte through their own end.
                counter: if last {
                    total_len
                } else {
                    (BLOCK_BYTES * (i + 1)) as u64
                },
                last_block: last,
            }
        })
        .collect()
}

/// Hash a message with BLAKE2s by proving every compression, from RFC 7693 section 3.3.
///
/// Each compression reads the previous output as its chaining value.
///
/// So each one is proved on its own trace, once the previous output is known.
fn proved_blake2s(key: &[u8], message: &[u8], digest_len: usize) -> Vec<u8> {
    // The parameter block p[0] = 0x0101kknn: digest length, key length, fanout 1 and depth 1.
    let mut chaining_value = IV;
    chaining_value[0] ^= 0x0101_0000 ^ ((key.len() as u32) << 8) ^ digest_len as u32;

    // Chain the compressions, checking every constraint of each.
    for mut input in hash_inputs(key, message) {
        input.chaining_value = chaining_value;
        chaining_value = trace_outputs(&[input])[0];
    }

    // The digest is the first bytes of the chaining value, read little-endian.
    chaining_value
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .take(digest_len)
        .collect()
}

/// Unkeyed BLAKE2s-256, proved.
fn proved_hash(message: &[u8]) -> [u8; 32] {
    proved_blake2s(&[], message, 32).try_into().unwrap()
}

/// Generate a trace for the inputs, check every constraint, and return each row's output.
fn trace_outputs(inputs: &[Blake2sCompressionInput]) -> Vec<[u32; 8]> {
    let air = Blake2sBinaryAir::default();

    // Pad to a power-of-two height with all-zero inputs, which are valid compressions too.
    let mut padded_inputs = inputs.to_vec();
    padded_inputs.resize(
        inputs.len().next_power_of_two(),
        Blake2sCompressionInput::default(),
    );

    let trace = generate_binary_trace_rows::<F>(padded_inputs, 0);
    check_constraints(&air, &trace, &[]);

    // Read each output back from the witness of its row.
    (0..inputs.len())
        .map(|r| {
            let row = trace.row_slice(r).unwrap();
            let row: &Blake2sBinaryCols<F> = (*row).borrow();
            row.compression_output()
        })
        .collect()
}

/// Constraint failures of a random four-row trace after an edit to one of its rows.
///
/// Every failure must sit on the edited row, since the AIR reads no next row.
fn failures_after_edit(
    row_index: usize,
    mutate: impl FnOnce(&mut Blake2sBinaryCols<F>),
) -> Vec<ConstraintFailure> {
    let air = Blake2sBinaryAir::default();
    let mut trace = air.generate_random_trace_rows::<F>(4, 0);
    mutate(trace.row_mut(row_index).borrow_mut());
    let failures = check_all_constraints(&air, &trace, &[], None).failures;
    assert!(failures.iter().all(|failure| failure.row == row_index));
    failures
}

/// Random compression inputs, with every field drawn at random.
fn random_compression_inputs(rng: &mut SmallRng, count: usize) -> Vec<Blake2sCompressionInput> {
    (0..count)
        .map(|_| Blake2sCompressionInput {
            chaining_value: rng.random(),
            block: rng.random(),
            counter: rng.random(),
            last_block: rng.random(),
        })
        .collect()
}

/// The deterministic byte sequence of the RFC 7693 appendix E self-test.
///
/// It is a Fibonacci sequence over 32-bit words, keeping the top byte of each term.
fn self_test_sequence(len: usize, seed: u32) -> Vec<u8> {
    let mut a = 0xDEAD_4BAD_u32.wrapping_mul(seed);
    let mut b = 1_u32;
    (0..len)
        .map(|_| {
            let t = a.wrapping_add(b);
            a = b;
            b = t;
            (t >> 24) as u8
        })
        .collect()
}

#[test]
fn width_and_constraint_hints_match_symbolic_evaluation() {
    // Fixture state:
    //
    //     inputs     26 words * 32 bits + 1 flag cell      =    833 columns
    //     witness    10 rounds * 8 steps * 190 columns     = 15,200 columns
    //     total                                            = 16,033 columns
    //
    // Constraints: one booleanity per input cell, plus 190 per G step.
    let air = Blake2sBinaryAir::default();
    assert_eq!(NUM_INPUT_BITS, 833);
    assert_eq!(NUM_BLAKE2S_BINARY_COLS, 16_033);
    assert_eq!(<Blake2sBinaryAir as BaseAir<F>>::width(&air), 16_033);
    assert!(<Blake2sBinaryAir as BaseAir<F>>::main_next_row_columns(&air).is_empty());

    let layout = AirLayout {
        main_width: NUM_BLAKE2S_BINARY_COLS,
        ..Default::default()
    };
    for (air, assumes_boolean_trace, num_constraints) in [
        (Blake2sBinaryAir::default(), false, 16_033),
        (Blake2sBinaryAir::assuming_boolean_trace(), true, 15_200),
    ] {
        // The declared counts must match what the constraints actually are.
        let constraints = get_symbolic_constraints::<F, _>(&air, layout);
        assert_eq!(constraints.len(), num_constraints);
        assert_eq!(
            <Blake2sBinaryAir as BaseAir<F>>::num_constraints(&air),
            Some(constraints.len())
        );
        assert_eq!(
            <Blake2sBinaryAir as BaseAir<F>>::assumes_boolean_trace(&air),
            assumes_boolean_trace
        );

        // The majority function is the only product, so the degree is exactly 2.
        let degree = get_max_constraint_degree::<F, _>(&air, layout, 1 << 4);
        assert_eq!(degree, 2);
        assert_eq!(
            <Blake2sBinaryAir as BaseAir<F>>::max_constraint_degree(&air),
            Some(degree)
        );
    }
}

#[test]
fn the_rfc_7693_digest_of_abc_is_proved() {
    assert_eq!(proved_hash(b"abc"), ABC_DIGEST);
}

#[test]
fn rfc_7693_appendix_b_matches_after_every_round() {
    // RFC 7693 appendix B prints the whole state of the "abc" compression after every round.
    //
    // Comparing each round separately pins a wrong schedule row, rotation or step wiring
    // to the round where it happens, not only to the digest.
    //
    // Fixture state: one block, 3 bytes, the final block, an unkeyed 32-byte digest.
    // The chaining value is the IV with the parameter block 0x01010020 folded into its first word.
    let mut input = hash_inputs(&[], b"abc")[0];
    input.chaining_value[0] ^= 0x0101_0020;

    // The message block holds "abc" little-endian in its first word, zero elsewhere.
    assert_eq!(input.block[0], 0x0063_6261);
    assert!(input.block[1..].iter().all(|&word| word == 0));

    // Round 0 of the table is the state before mixing: parameter block, counter 3, inverted v[14].
    assert_eq!(reference_initial_state(&input), ABC_ROUND_STATES[0]);

    // Prove the compression and read the state after each round from the witness.
    let trace = generate_binary_trace_rows::<F>(vec![input], 0);
    check_constraints(&Blake2sBinaryAir::default(), &trace, &[]);
    let row = trace.row_slice(0).unwrap();
    let row: &Blake2sBinaryCols<F> = (*row).borrow();
    for round in 0..10 {
        assert_eq!(
            row.state_after_round(round),
            ABC_ROUND_STATES[round + 1],
            "state after round {round}"
        );
    }

    // The folded chaining value, then its little-endian bytes as the digest.
    assert_eq!(row.compression_output(), ABC_OUTPUT);
}

#[test]
fn rfc_7693_appendix_e_self_test() {
    // RFC 7693 appendix E hashes a grid of cases and hashes all their digests together.
    //
    //     digest lengths   16, 20, 28, 32 bytes
    //     message lengths  0, 3, 64, 65, 255, 1024 bytes
    //     each case        once unkeyed, once keyed with a key as long as the digest
    //
    // This covers what short unkeyed messages leave untested:
    // - the key length and digest length bytes of the parameter block,
    // - the key block and the counter that includes it,
    // - chains of up to 17 compressions on pseudorandom data.
    //
    // Every compression, the final grand hash included, goes through a checked trace.
    let mut digests = Vec::new();
    for digest_len in [16, 20, 28, 32] {
        for message_len in [0, 3, 64, 65, 255, 1024] {
            let message = self_test_sequence(message_len, message_len as u32);
            digests.extend(proved_blake2s(&[], &message, digest_len));

            let key = self_test_sequence(digest_len, digest_len as u32);
            digests.extend(proved_blake2s(&key, &message, digest_len));
        }
    }
    assert_eq!(proved_hash(&digests), SELF_TEST_DIGEST);
}

#[test]
fn proved_hashes_match_the_blake2_crate() {
    let mut rng = SmallRng::seed_from_u64(2);
    // Lengths around the block boundaries:
    //
    //     0            one all-zero block
    //     1, 55, 63    one partial block
    //     64, 128      messages ending exactly on a block
    //     65, 100      two compressions
    //     129, 200     three compressions
    for length in [0, 1, 55, 63, 64, 65, 100, 128, 129, 200] {
        let message: Vec<u8> = (0..length).map(|_| rng.random()).collect();
        let expected: [u8; 32] = Blake2s::<U32>::digest(&message).into();
        assert_eq!(proved_hash(&message), expected, "message of {length} bytes");
    }
}

#[test]
fn random_inputs_match_the_reference_compression() {
    // Random counters reach the high counter word, and random flags reach both flag values.
    let mut rng = SmallRng::seed_from_u64(3);
    let inputs = random_compression_inputs(&mut rng, 16);

    for (input, output) in inputs.iter().zip(trace_outputs(&inputs)) {
        assert_eq!(output, reference_compress(input));
    }
}

#[test]
fn random_traces_satisfy_constraints() {
    let air = Blake2sBinaryAir::default();
    for height in [1, 2, 4] {
        let trace = air.generate_random_trace_rows::<F>(height, 0);
        assert_eq!(trace.height(), height);
        check_constraints(&air, &trace, &[]);
    }
}

#[test]
fn packed_trace_matches_dense_trace_at_word_boundaries() {
    let mut rng = SmallRng::seed_from_u64(17);
    // All-ones inputs carry out of every bit of the first additions.
    //
    // The full carry chains are then exercised whatever the random draws.
    let all_ones = Blake2sCompressionInput {
        chaining_value: [u32::MAX; 8],
        block: [u32::MAX; 16],
        counter: u64::MAX,
        last_block: true,
    };
    // Heights below, at and above one packed word of 64 rows.
    for height in [1usize, 2, 32, 64, 128] {
        let random = random_compression_inputs(&mut rng, height);
        for inputs in [random, vec![all_ones; height]] {
            let packed = generate_binary_trace_packed::<F>(&inputs);
            let dense = generate_binary_trace_rows::<F>(inputs, 0);
            assert_eq!(packed.width, NUM_BLAKE2S_BINARY_COLS);
            assert_eq!(packed.height(), height.div_ceil(64));

            // Bit k of a packed word is cell (64 * w + k) of the dense trace.
            for row in 0..height {
                for column in 0..NUM_BLAKE2S_BINARY_COLS {
                    let expected = dense.values[row * NUM_BLAKE2S_BINARY_COLS + column];
                    let word = packed.values[(row / 64) * packed.width + column];
                    assert_eq!(expected, F::from_bool((word >> (row % 64)) & 1 == 1));
                }
            }

            // Lanes past the last input stay zero.
            if !height.is_multiple_of(64) {
                for word in &packed.values[packed.width * (packed.height() - 1)..] {
                    assert_eq!(*word >> (height % 64), 0);
                }
            }
        }
    }
}

#[test]
fn packed_random_trace_uses_the_dense_generator_sequence() {
    let air = Blake2sBinaryAir::default();
    for height in [1usize, 64, 128] {
        let dense = air.generate_random_trace_rows::<F>(height, 0);
        let packed = air.generate_random_trace_packed::<F>(height);
        // Bit k of a packed word is cell (64 * w + k) of the dense trace.
        for row in 0..height {
            for column in 0..NUM_BLAKE2S_BINARY_COLS {
                let word = packed.values[(row / 64) * packed.width + column];
                assert_eq!(
                    dense.values[row * NUM_BLAKE2S_BINARY_COLS + column],
                    F::from_bool((word >> (row % 64)) & 1 == 1)
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "at least one input")]
fn packed_generator_rejects_empty_input() {
    let _ = generate_binary_trace_packed::<F>(&[]);
}

#[test]
#[should_panic(expected = "power of two")]
fn packed_generator_rejects_non_power_of_two_input() {
    let _ = generate_binary_trace_packed::<F>(&[Blake2sCompressionInput::default(); 3]);
}

#[test]
fn rejects_flipped_carry_bit() {
    let failures = failures_after_edit(1, |row| row.rounds[3][5].add2_carries[17] += F::ONE);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_flipped_d1_bit() {
    let failures = failures_after_edit(0, |row| row.rounds[0][2].d1[0] += F::ONE);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_flipped_output_bits_in_last_round() {
    // The output is never stored, so the last round's words are what bind it.
    for g in 0..8 {
        let failures = failures_after_edit(2, |row| row.rounds[9][g].b2[3 * g] += F::ONE);
        assert!(!failures.is_empty(), "b2 of step {g}");
        let failures = failures_after_edit(3, |row| row.rounds[9][g].d2[31 - g] += F::ONE);
        assert!(!failures.is_empty(), "d2 of step {g}");
    }
}

#[test]
fn rejects_a_flipped_counter_or_flag_bit() {
    // The counter and the flag reach the state only through the initial d words.
    //
    // A flip in one of them is caught only if it breaks the first round's additions.
    let failures = failures_after_edit(1, |row| row.counter_low[9] += F::ONE);
    assert!(!failures.is_empty(), "counter low");
    let failures = failures_after_edit(2, |row| row.counter_high[0] += F::ONE);
    assert!(!failures.is_empty(), "counter high");

    // Fixture state: every random row is a final block, so the flag is set.
    //
    // Mutation: clear it, which un-inverts v[14] and changes the whole compression.
    let failures = failures_after_edit(3, |row| row.last_block += F::ONE);
    assert!(!failures.is_empty(), "last block flag");
}

#[test]
fn rejects_non_boolean_input() {
    assert!(F::GENERATOR != F::ZERO && F::GENERATOR != F::ONE);
    let air = Blake2sBinaryAir::default();

    // The input cells are the first columns, and each gets its booleanity constraint in order:
    //
    //     cells 0..832    the 26 input words, 32 bits each
    //     cell  832       last-block flag
    //     cell  833       last-node flag
    //
    // So a non-bit in input cell i fails constraint i.
    let cells = (0..NUM_INPUT_BITS - 2).step_by(39).chain([832, 833]);
    for cell in cells {
        let mut trace = air.generate_random_trace_rows::<F>(4, 0);
        trace.row_mut(1)[cell] = F::GENERATOR;
        let failures = check_all_constraints(&air, &trace, &[], None).failures;
        assert!(
            failures.iter().any(|failure| failure.constraint == cell),
            "input cell {cell}"
        );
    }
}

#[test]
fn rejects_non_boolean_witness_at_the_top_bit() {
    // The carry into bit 31 of a 3-operand addition.
    let failures = failures_after_edit(2, |row| row.rounds[2][6].add1_carries[30] = F::GENERATOR);
    assert!(!failures.is_empty());
    // Bit 23 of d_2 feeds bit 31 of a_2 = d_1 ^ (d_2 <<< 8).
    let failures = failures_after_edit(3, |row| row.rounds[4][1].d2[23] = F::GENERATOR);
    assert!(!failures.is_empty());
}

#[test]
fn rejects_message_bit_flip_without_regeneration() {
    let failures = failures_after_edit(2, |row| row.block[5][9] += F::ONE);
    assert!(!failures.is_empty());
}

/// Every current-row main column the expression reads.
fn columns_of(expr: &SymbolicExpr<BaseLeaf<F>>, out: &mut BTreeSet<usize>) {
    match expr {
        SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => {
            if v.entry == (BaseEntry::Main { offset: 0 }) {
                out.insert(v.index);
            }
        }
        SymbolicExpr::Leaf(_) => {}
        SymbolicExpr::Add { x, y, .. }
        | SymbolicExpr::Sub { x, y, .. }
        | SymbolicExpr::Mul { x, y, .. } => {
            columns_of(x, out);
            columns_of(y, out);
        }
        SymbolicExpr::Neg { x, .. } => columns_of(x, out),
    }
}

/// Evaluate an expression on the cells of one row.
fn eval_on_row(expr: &SymbolicExpr<BaseLeaf<F>>, row: &[F]) -> F {
    match expr {
        SymbolicExpr::Leaf(BaseLeaf::Variable(v)) => {
            assert_eq!(v.entry, BaseEntry::Main { offset: 0 });
            row[v.index]
        }
        SymbolicExpr::Leaf(BaseLeaf::Constant(c)) => *c,
        SymbolicExpr::Leaf(_) => unreachable!("the AIR uses no selector"),
        SymbolicExpr::Add { x, y, .. } => eval_on_row(x, row) + eval_on_row(y, row),
        SymbolicExpr::Sub { x, y, .. } => eval_on_row(x, row) - eval_on_row(y, row),
        SymbolicExpr::Mul { x, y, .. } => eval_on_row(x, row) * eval_on_row(y, row),
        SymbolicExpr::Neg { x, .. } => -eval_on_row(x, row),
    }
}

/// Set every non-input cell of a row to the value the constraints give it from the inputs.
///
/// Each constraint after input booleanity reads one new column.
///
/// It is that column plus an expression in columns already set.
///
/// In characteristic 2 that column then equals the expression.
///
/// No cell has to be a bit for this.
fn derive_from_inputs(row: &mut [F]) {
    let layout = AirLayout {
        main_width: NUM_BLAKE2S_BINARY_COLS,
        ..Default::default()
    };
    let constraints = get_symbolic_constraints::<F, _>(&Blake2sBinaryAir::default(), layout);

    // Columns whose value is already settled.
    let mut fixed: BTreeSet<usize> = BTreeSet::new();
    for (i, constraint) in constraints.iter().enumerate() {
        let mut columns = BTreeSet::new();
        columns_of(constraint, &mut columns);

        // The booleanity constraints come first and only read input cells, which stay as given.
        if i < NUM_INPUT_BITS {
            fixed.extend(columns);
            continue;
        }

        // Every later constraint must read exactly one column not yet settled.
        let new: Vec<usize> = columns.difference(&fixed).copied().collect();
        let [column] = new[..] else {
            panic!("constraint {i} introduces columns {new:?}");
        };

        // The constraint reads column + rest = 0.
        // Evaluating it with the column at zero therefore yields the column.
        row[column] = F::ZERO;
        row[column] = eval_on_row(constraint, row);
        fixed.insert(column);
    }
}

#[test]
fn only_booleanity_rejects_a_non_boolean_input() {
    // Invariant: booleanity is the only constraint that holds the inputs to bits.
    //
    // Mutation: put a non-bit in one input cell, then derive every other cell from the inputs.
    //
    // Expected:
    // - the full AIR fails exactly one constraint, that cell's booleanity,
    // - the AIR without booleanity accepts the row.
    //
    // Input cell i is booleanity constraint i.
    assert!(F::GENERATOR != F::ZERO && F::GENERATOR != F::ONE);
    let air = Blake2sBinaryAir::default();
    // A chaining value bit, a message bit, a counter bit, and the flag.
    for cell in [0, 447, 800, 832] {
        let mut trace = air.generate_random_trace_rows::<F>(1, 0);
        let row = trace.row_mut(0);
        row[cell] = F::GENERATOR;
        derive_from_inputs(row);

        // The non-bit reached the derived cells.
        let non_bits = row.iter().filter(|&&c| c != F::ZERO && c != F::ONE).count();
        assert!(non_bits > 1, "input cell {cell}");

        let failures: Vec<usize> = check_all_constraints(&air, &trace, &[], None)
            .failures
            .iter()
            .map(|failure| failure.constraint)
            .collect();
        assert_eq!(failures, [cell], "input cell {cell}");

        let unconstrained = Blake2sBinaryAir::assuming_boolean_trace();
        assert!(
            check_all_constraints(&unconstrained, &trace, &[], None).is_ok(),
            "input cell {cell}"
        );
    }
}
