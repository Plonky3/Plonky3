use core::arch::x86_64::{_mm256_extract_epi64, _mm256_setr_epi64x, _mm256_setzero_si256};

use criterion::{criterion_group, criterion_main, Criterion};
use p3_keccak::avx2;
use tiny_keccak::keccakf;

const STATES: [[u64; 25]; 4] = [
    [
        0xc22c4c11dbedc46a,
        0x317f74268c4f5cd0,
        0x838719da5aa295b6,
        0x9e9b17211985a3ba,
        0x92927b963ce29d69,
        0xf9a7169e38cc7216,
        0x639a594d6fbfe341,
        0x2335ebd8d15777bd,
        0x44e1abc0d022823b,
        0xb3657f9d16b36c13,
        0x26d9217c32b3010a,
        0x6e73d6e9c7e5bcc8,
        0x400aa469d130a391,
        0x1aa7c8a2cb97188a,
        0xdc3084a09bd0a6e3,
        0xbcfe3b656841baea,
        0x325f41887c840166,
        0x844656e313674bfe,
        0xd63de8bad19d156c,
        0x49ef0ac0ab52e147,
        0x8b92ee811c654ca9,
        0x42a9310fedf09bda,
        0x182dbdac03a5358e,
        0x3b4692ce58af8cb5,
        0x534da610f01b8fb3,
    ],
    [
        0x1c322ff4aea07d26,
        0xbd67bde061c97612,
        0x517565bd02ab410a,
        0xb251273ddc12a725,
        0x24f0979fe4f4fedc,
        0xc32d063a64f0bf03,
        0xd33c6709a7b103d2,
        0xaf33a8224b5c8828,
        0x6544ca066e997f1c,
        0xd53ad41e39f06d68,
        0x67695f6fb71d77d9,
        0xd6378cf19ee510f2,
        0x49472ea57abcbd08,
        0xcf3739df1eefbbb4,
        0x0fac1bf30e8ef101,
        0x7ff04c9b90de0f27,
        0xf3d63ec0e64cb2ab,
        0x76388c05f377d4bd,
        0x7886dd8f5b14ef5b,
        0xb036d289ba24a513,
        0x011e8fd6be65a408,
        0x695e2d20848eec67,
        0x31f9e80c5f45f8ee,
        0xcdf873daf7a5fdeb,
        0xfe98ff5bf28d560a,
    ],
    [
        0xed7423c3e4cda469,
        0x8bbbe52577993e33,
        0x93182a78487f96db,
        0x3c5209456d78dc73,
        0x8b66bde37b967146,
        0x226ae6e2148314fc,
        0x302aed4f30cd2db9,
        0x621a7c7751302084,
        0x4054874254bc66cc,
        0xb513327739d3c109,
        0x0ae03189599dd81a,
        0x457e6f324e512ed9,
        0x3870ea63c7367728,
        0xb08c7c0e401d2623,
        0xa1316890c9bb3ac1,
        0x0a313e02f34f6c7e,
        0x7c1325754df4dbf5,
        0x287e3d88240bedd2,
        0xc7c0f3c5058290bb,
        0x39471c62d065a4d1,
        0x050d8ecb5c7911bf,
        0x7a6cd7ca757186a7,
        0xed14a51934a17895,
        0x8a75418d7ffb98dd,
        0x8096f8d803188d57,
    ],
    [
        0x118d693606b316cd,
        0xc2614f04c0dfca91,
        0x5eb3da95450a214f,
        0x193eb69a8198e724,
        0xc24dea1c58e5fa6d,
        0xcf8630adb771d47c,
        0xe612253b23ade1df,
        0x281b2b53f2d5fe61,
        0x9a3f8fb149d7c419,
        0x7ac1eeffbd426464,
        0xb1bdb03caa7b20a3,
        0x4e38a03b709d47d1,
        0x35cafd22e9a91879,
        0x26c9ae757c5b9f80,
        0x9113b092720d90db,
        0xeb504ed5104a0a09,
        0x1405c220c45ee2b1,
        0x8055d2c37b02472b,
        0x545eeff9734b4b99,
        0x79eb67721385dff8,
        0x4133f750b0446360,
        0x7167e6c1fd8ff59b,
        0x576436ac7d46936e,
        0x8db22fbb547d0826,
        0xa855d775d64f0110,
    ],
];

fn hash_tiny_keccak(states: &mut [[u64; 25]; 4]) {
    keccakf(&mut states[0]);
    keccakf(&mut states[1]);
    keccakf(&mut states[2]);
    keccakf(&mut states[3]);
}

fn hash_avx2(states: &mut [[u64; 25]; 4]) {
    let mut packed_states = [unsafe { _mm256_setzero_si256() }; 25];
    for i in 0..25 {
        packed_states[i] = unsafe {
            _mm256_setr_epi64x(
                states[0][i] as i64,
                states[1][i] as i64,
                states[2][i] as i64,
                states[3][i] as i64,
            )
        };
    }

    avx2::keccak_perm(&mut packed_states);

    for i in 0..25 {
        unsafe {
            states[0][i] = _mm256_extract_epi64::<0>(packed_states[i]) as u64;
            states[1][i] = _mm256_extract_epi64::<1>(packed_states[i]) as u64;
            states[2][i] = _mm256_extract_epi64::<2>(packed_states[i]) as u64;
            states[3][i] = _mm256_extract_epi64::<3>(packed_states[i]) as u64;
        }
    }
}
fn bench_keccak_avx2(c: &mut Criterion) {
    c.bench_function("keccak_avx2_baseline", |b| {
        b.iter(|| {
            let mut states = STATES;
            for _ in 0..1000 {
                hash_tiny_keccak(&mut states);
            }
            states
        });
    });

    c.bench_function("keccak_avx2_impl", |b| {
        b.iter(|| {
            let mut states = STATES;
            for _ in 0..1000 {
                hash_avx2(&mut states);
            }
            states
        });
    });
}

criterion_group!(keccak_avx2, bench_keccak_avx2);
criterion_main!(keccak_avx2);
