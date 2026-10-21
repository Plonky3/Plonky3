use alloc::vec;
use alloc::vec::Vec;

use num_bigint::BigUint;
use p3_baby_bear::BabyBear;
use p3_bn254::Bn254;
use p3_field::{Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::Mersenne31;
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::prime::{is_bpsw, miller_rabin, strong_lucas_selfridge, trial_division};
use crate::{
    CanonicalResidue, ModField, ModRing, ModulusError, MontField, MontModulus, PowerOfTwo,
    PrimeError, PrimePolicy, Residue, StaticBigPrime, StaticPrime, Uint, sample_prime,
};

fn big<const L: usize>(x: Uint<L>) -> BigUint {
    x.into()
}

fn uint<const L: usize>(x: &BigUint) -> Uint<L> {
    Uint::try_from(x).expect("fits")
}

/// A byte stream from a seeded generator.
fn stream(seed: u64) -> impl FnMut(&mut [u8]) {
    let mut rng = StdRng::seed_from_u64(seed);
    move |out: &mut [u8]| rng.fill_bytes(out)
}

/// An odd modulus of exactly `bits` bits, from random limbs.
fn modulus_of<const L: usize>(limbs: [u64; L], bits: u32) -> Uint<L> {
    let mask = Uint::<L>::MAX.shr(Uint::<L>::BITS - bits);
    let mut m = Uint(core::array::from_fn(|i| limbs[i] & mask.0[i]));
    m = Uint(core::array::from_fn(|i| {
        m.0[i] | Uint::<L>::ONE.shl(bits - 1).0[i]
    }));
    m.0[0] |= 1;
    m
}

/// Values below `m`: the boundary values, then random ones.
fn values_below(m: &BigUint, rng: &mut StdRng, n: usize) -> Vec<BigUint> {
    let mut out = vec![
        BigUint::ZERO,
        BigUint::from(1u8),
        m - 1u8,
        m - 2u8.min(m.bits() as u8 - 1),
    ];
    out.extend((0..n).map(|_| {
        let bytes: Vec<u8> = (0..m.bits().div_ceil(8) + 8)
            .map(|_| rng.random())
            .collect();
        BigUint::from_bytes_le(&bytes) % m
    }));
    out
}

/// Every ring operation against num-bigint, with lazy sums at the largest stored values.
fn check_ring<const L: usize>(m: Uint<L>, seed: u64) {
    let ring = MontModulus::new(m).expect("valid modulus");
    let mb = big(m);
    let r = BigUint::from(1u8) << (64 * L);
    let mut rng = StdRng::seed_from_u64(seed);
    let vals = values_below(&mb, &mut rng, 6);
    let res: Vec<Residue<L>> = vals.iter().map(|v| ring.from_uint(&uint(v))).collect();
    let back = |x: Residue<L>| big(ring.to_uint(x));

    assert_eq!(ring.modulus(), mb);
    assert_eq!(ring.modulus_bits() as u64, mb.bits());
    assert_eq!(back(ring.one()), BigUint::from(1u8) % &mb);
    assert_eq!(back(ring.zero()), BigUint::ZERO);
    for (a, &x) in vals.iter().zip(&res) {
        // The stored form is the Montgomery form and is canonical.
        assert_eq!(big(x.0), (a * &r) % &mb);
        assert_eq!(back(x), *a);
        assert_eq!(back(ring.neg(x)), (&mb - a) % &mb);
        assert_eq!(back(ring.square(x)), (a * a) % &mb);
        assert_eq!(back(ring.half(x)), (a * ((&mb + 1u8) >> 1)) % &mb);
        assert_eq!(ring.from_canonical(&ring.to_canonical(x)), Some(x));
        let mut lift = [0u64; L];
        ring.lift_limbs(x, &mut lift);
        assert_eq!(Uint(lift), uint(a));
        let mut bytes = Vec::new();
        ring.encode(x, &mut bytes);
        assert_eq!(bytes.len(), 8 * L);
        assert_eq!(ring.decode(&bytes), Some(x));
        for (b, &y) in vals.iter().zip(&res) {
            assert_eq!(back(ring.add(x, y)), (a + b) % &mb);
            assert_eq!(back(ring.sub(x, y)), (a + &mb - b) % &mb);
            assert_eq!(back(ring.mul(x, y)), (a * b) % &mb);
        }
        let e: [u64; 3] = [rng.random(), rng.random(), rng.random()];
        let eb = big(Uint(e));
        assert_eq!(back(ring.pow(x, &e)), a.modpow(&eb, &mb));
        assert_eq!(back(ring.pow(x, &[])), BigUint::from(1u8) % &mb);
    }

    // Integers of any size and sign.
    let x: u64 = rng.random();
    assert_eq!(back(ring.from_u64(x)), BigUint::from(x) % &mb);
    let xi = -(x as i64).abs().max(1);
    let neg = (&mb - (BigUint::from(xi.unsigned_abs()) % &mb)) % &mb;
    assert_eq!(back(ring.from_i64(xi)), neg);
    for len in [0, 1, L - 1, L, L + 1, 3 * L + 2] {
        let limbs: Vec<u64> = (0..len).map(|_| rng.random()).collect();
        let n = limbs
            .iter()
            .rev()
            .fold(BigUint::ZERO, |acc, &w| (acc << 64) + w);
        assert_eq!(back(ring.reduce_limbs(&limbs)), n % &mb);
    }
    let any: [u64; L] = core::array::from_fn(|_| rng.random());
    assert_eq!(back(ring.from_uint(&Uint(any))), big(Uint(any)) % &mb);

    // Decoding refuses m and above, and wrong lengths.
    let mut bytes = vec![0u8; 8 * L];
    m.write_le_bytes(&mut bytes);
    assert_eq!(ring.decode(&bytes), None);
    Uint::<L>::MAX.write_le_bytes(&mut bytes);
    assert_eq!(ring.decode(&bytes), None);
    assert_eq!(ring.decode(&bytes[1..]), None);
    assert_eq!(ring.from_canonical(&CanonicalResidue(m.0)), None);

    // Slice kernels, including stored values at m - 1 that load every column.
    let top = Residue(m.sbb(&Uint::ONE).0);
    for len in [0, 1, 7, 8, 9, 33, 200] {
        let mut a: Vec<Residue<L>> = (0..len).map(|i| res[i % res.len()]).collect();
        let b: Vec<Residue<L>> = (0..len)
            .map(|i| {
                if i % 3 == 0 {
                    top
                } else {
                    res[(i * 5 + 1) % res.len()]
                }
            })
            .collect();
        let s = res[len % res.len()];
        let want = a
            .iter()
            .zip(&b)
            .fold(BigUint::ZERO, |acc, (&x, &y)| acc + back(x) * back(y))
            % &mb;
        assert_eq!(back(ring.dot(&a, &b)), want);
        let tops = vec![top; len];
        let want = (back(top) * back(top) * BigUint::from(len)) % &mb;
        assert_eq!(back(ring.dot(&tops, &tops)), want);

        let mut acc = a.clone();
        ring.scale_add(&mut acc, &b, s);
        for ((&o, &x), &y) in acc.iter().zip(&a).zip(&b) {
            assert_eq!(back(o), (back(x) + back(s) * back(y)) % &mb);
        }
        let orig = a.clone();
        ring.mul_slices(&mut a, &b);
        for ((&o, &x), &y) in a.iter().zip(&orig).zip(&b) {
            assert_eq!(back(o), (back(x) * back(y)) % &mb);
        }
    }

    // eq(point, i) = prod_k (x_k if bit k of i else 1 - x_k), point[0] on the top bit.
    let point = &res[..6];
    let table = ring.eq_table(point);
    for (i, &t) in table.iter().enumerate() {
        let want = point
            .iter()
            .enumerate()
            .fold(BigUint::from(1u8), |acc, (k, &x)| {
                let bit = (i >> (point.len() - 1 - k)) & 1 == 1;
                let f = if bit {
                    back(x)
                } else {
                    (&mb + 1u8 - back(x)) % &mb
                };
                acc * f % &mb
            });
        assert_eq!(back(t), want);
    }
}

/// The modulus `2^(64 L - 1) - 1 - 2k`, at the top of the allowed range.
fn edge_modulus<const L: usize>(k: u64) -> Uint<L> {
    Uint::<L>::MAX.shr(1).sbb(&Uint::from_u64(2 * k)).0
}

proptest! {
    #[test]
    fn ring_ops_match_bigint_1(limbs: [u64; 1], bits in 2u32..64, seed: u64) {
        check_ring(modulus_of(limbs, bits), seed);
    }

    #[test]
    fn ring_ops_match_bigint_2(limbs: [u64; 2], bits in 2u32..128, seed: u64) {
        check_ring(modulus_of(limbs, bits), seed);
    }

    #[test]
    fn ring_ops_match_bigint_3(limbs: [u64; 3], bits in 2u32..192, seed: u64) {
        check_ring(modulus_of(limbs, bits), seed);
    }

    #[test]
    fn ring_ops_match_bigint_4(limbs: [u64; 4], bits in 2u32..256, seed: u64) {
        check_ring(modulus_of(limbs, bits), seed);
    }

    #[test]
    fn ring_ops_at_the_spare_bit(k in 0u64..64, seed: u64) {
        check_ring(edge_modulus::<1>(k), seed);
        check_ring(edge_modulus::<2>(k), seed);
        check_ring(edge_modulus::<3>(k), seed);
        check_ring(edge_modulus::<4>(k), seed);
    }
}

#[test]
fn moduli_without_a_montgomery_form_are_refused() {
    assert_eq!(MontModulus::new(Uint([1u64])), Err(ModulusError::TooSmall));
    assert_eq!(MontModulus::new(Uint([0u64])), Err(ModulusError::Even));
    assert_eq!(MontModulus::new(Uint([10u64])), Err(ModulusError::Even));
    assert_eq!(
        MontModulus::new(Uint([1u64 << 63 | 1])),
        Err(ModulusError::TooWide)
    );
    assert_eq!(
        MontModulus::new(Uint([1, 1u64 << 63])),
        Err(ModulusError::TooWide)
    );
    assert!(MontModulus::new(Uint([3u64])).is_ok());
    assert!(MontModulus::new(Uint([u64::MAX, u64::MAX >> 1])).is_ok());
}

/// Known primes across the limb counts.
fn known_primes() -> Vec<BigUint> {
    let one = BigUint::from(1u8);
    let hex = |s: &str| BigUint::parse_bytes(s.as_bytes(), 16).expect("hex");
    vec![
        BigUint::from(3u8),
        BigUint::from(1621u32),
        BigUint::from(65537u32),
        (&one << 31) - 1u8,
        BigUint::from(0x7800_0001u32),
        BigUint::from(0x7f00_0001u32),
        (&one << 61) - 1u8,
        BigUint::from(0xffff_ffff_0000_0001u64),
        (&one << 89) - 1u8,
        (&one << 107) - 1u8,
        (&one << 127) - 1u8,
        (&one << 255) - 19u8,
        hex("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"),
    ]
}

#[test]
fn known_primes_pass() {
    for p in known_primes() {
        let limbs = (p.bits() + 1).div_ceil(64);
        match limbs {
            1 => assert!(
                is_bpsw(&MontModulus::<1>::new(uint(&p)).expect("odd")),
                "{p}"
            ),
            2 => assert!(
                is_bpsw(&MontModulus::<2>::new(uint(&p)).expect("odd")),
                "{p}"
            ),
            _ => assert!(
                is_bpsw(&MontModulus::<4>::new(uint(&p)).expect("odd")),
                "{p}"
            ),
        }
    }
    assert!(MontField::new(Uint::<4>::try_from(&known_primes()[11]).expect("fits")).is_ok());
    assert!(MontField::new(Uint([3u64])).is_ok());
}

#[test]
fn bpsw_matches_a_sieve_below_two_hundred_thousand() {
    const N: usize = 200_000;
    let mut sieve = vec![true; N];
    sieve[0] = false;
    sieve[1] = false;
    for i in 2..N {
        if sieve[i] {
            for j in (i * i..N).step_by(i) {
                sieve[j] = false;
            }
        }
    }
    for (n, &prime) in sieve.iter().enumerate().skip(3).step_by(2) {
        let ring = MontModulus::new(Uint([n as u64])).expect("odd");
        assert_eq!(is_bpsw(&ring), prime, "{n}");
        assert_eq!(MontField::new(Uint([n as u64])).is_ok(), prime, "{n}");
    }
    for n in 0..1700u64 {
        let small = n <= 1619 || n % 2 == 0 || !sieve[n as usize];
        let want = small.then(|| sieve[n as usize]);
        assert_eq!(trial_division(&Uint([n])), want, "{n}");
    }
}

#[test]
fn carmichael_numbers_fail() {
    let carmichael: [u64; 14] = [
        561,
        1105,
        1729,
        2465,
        2821,
        6601,
        8911,
        41041,
        825_265,
        321_197_185,
        3_215_031_751,
        5_394_826_801,
        232_250_619_601,
        9_746_347_772_161,
    ];
    for n in carmichael {
        let ring = MontModulus::new(Uint([n])).expect("odd");
        assert!(!is_bpsw(&ring), "{n}");
        assert_eq!(MontField::new(Uint([n])), Err(ModulusError::Composite));
    }
}

#[test]
fn strong_pseudoprimes_pass_their_bases_and_fail_bpsw() {
    // Strong pseudoprimes to base 2.
    for n in [
        2047u64, 3277, 4033, 4681, 8321, 15841, 29341, 42799, 49141, 52633, 65281, 74665, 80581,
        85489, 88357, 90751,
    ] {
        let ring = MontModulus::new(Uint([n])).expect("odd");
        assert!(miller_rabin(&ring, &Uint([2])), "{n}");
        assert!(!is_bpsw(&ring), "{n}");
    }
    // The least strong pseudoprimes to all prime bases up to 37 and 41.
    let primes = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];
    for (n, bases) in [
        (BigUint::from(3_215_031_751u64), 4),
        (BigUint::from(3_825_123_056_546_413_051u64), 9),
        (
            BigUint::parse_bytes(b"318665857834031151167461", 10).expect("dec"),
            12,
        ),
        (
            BigUint::parse_bytes(b"3317044064679887385961981", 10).expect("dec"),
            13,
        ),
    ] {
        let ring = MontModulus::<2>::new(uint(&n)).expect("odd");
        for &b in &primes[..bases] {
            assert!(miller_rabin(&ring, &Uint::from_u64(b)), "{n} base {b}");
        }
        assert!(!is_bpsw(&ring), "{n}");
        assert!(MontField::<2>::new(uint(&n)).is_err());
    }
}

#[test]
fn strong_lucas_pseudoprimes_pass_lucas_and_fail_bpsw() {
    for n in [
        5459u64, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519, 75077, 97439,
    ] {
        let ring = MontModulus::new(Uint([n])).expect("odd");
        assert!(strong_lucas_selfridge(&ring), "{n}");
        assert!(!miller_rabin(&ring, &Uint([2])), "{n}");
        assert!(!is_bpsw(&ring), "{n}");
    }
}

#[test]
fn squares_fail_the_lucas_test() {
    for p in [1627u64, 65537, 4_294_967_291] {
        let ring = MontModulus::<2>::new(uint(&(BigUint::from(p) * p))).expect("odd");
        assert!(!strong_lucas_selfridge(&ring), "{p}^2");
        assert!(!is_bpsw(&ring), "{p}^2");
    }
}

proptest! {
    #[test]
    fn bpsw_agrees_with_fermat_on_random_odd_numbers(limbs: [u64; 2], bits in 12u32..128) {
        let n = modulus_of(limbs, bits);
        let ring = MontModulus::new(n).expect("odd");
        let nb = big(n);
        let fermat = [2u32, 3, 5, 7, 11].iter().all(|&a| BigUint::from(a).modpow(&(&nb - 1u8), &nb) == BigUint::from(1u8));
        // Fermat to several bases misses only Carmichael-like numbers, which random draws never hit.
        prop_assert_eq!(is_bpsw(&ring), fermat);
    }

    #[test]
    fn sampled_primes_have_the_bit_length_and_are_prime(bits in 2u32..128, seed: u64) {
        let policy = PrimePolicy { mr_rounds: 8, ..PrimePolicy::new(bits) };
        let s = sample_prime::<2>(stream(seed), &policy).expect("a prime");
        let p = big(s.prime());
        prop_assert_eq!(p.bits(), u64::from(bits));
        prop_assert!(s.candidates() >= 1);
        if bits > 11 {
            prop_assert_eq!(BigUint::from(3u8).modpow(&(&p - 1u8), &p), BigUint::from(1u8));
        }
        let again = sample_prime::<2>(stream(seed), &policy).expect("a prime");
        prop_assert_eq!(&again, &s);
        let field = MontField::from_sample(s);
        let x = field.from_u64(seed | 1);
        if x != field.zero() {
            let inv = field.inverse(x).expect("nonzero");
            prop_assert_eq!(field.mul(x, inv), field.one());
        }
    }
}

#[test]
fn sampling_at_full_width_for_each_limb_count() {
    let s1 = sample_prime::<1>(stream(1), &PrimePolicy::new(63)).expect("prime");
    let s3 = sample_prime::<3>(stream(3), &PrimePolicy::new(191)).expect("prime");
    let s4 = sample_prime::<4>(stream(4), &PrimePolicy::new(255)).expect("prime");
    assert_eq!(s1.prime().bits(), 63);
    assert_eq!(s3.prime().bits(), 191);
    assert_eq!(s4.prime().bits(), 255);
}

#[test]
fn invalid_policies_and_exhaustion_are_reported() {
    let zeros = |out: &mut [u8]| out.fill(0);
    assert_eq!(
        sample_prime::<2>(zeros, &PrimePolicy::new(1)),
        Err(PrimeError::InvalidPolicy)
    );
    assert_eq!(
        sample_prime::<2>(zeros, &PrimePolicy::new(128)),
        Err(PrimeError::InvalidPolicy)
    );
    let none = PrimePolicy {
        max_candidates: 0,
        ..PrimePolicy::new(64)
    };
    assert_eq!(
        sample_prime::<2>(zeros, &none),
        Err(PrimeError::InvalidPolicy)
    );
    // The all-zero stream proposes only 2^63 + 1, a multiple of 3.
    let few = PrimePolicy {
        max_candidates: 3,
        ..PrimePolicy::new(64)
    };
    assert_eq!(sample_prime::<2>(zeros, &few), Err(PrimeError::Exhausted));
}

#[test]
fn field_inverses() {
    let p = (BigUint::from(1u8) << 127) - 1u8;
    let field = MontField::<2>::new(uint(&p)).expect("prime");
    assert_eq!(field.inverse(field.zero()), None);
    let mut rng = StdRng::seed_from_u64(9);
    let mut xs: Vec<Residue<2>> = (0..20).map(|_| field.from_u64(rng.random())).collect();
    xs[3] = field.zero();
    xs[11] = field.zero();
    let orig = xs.clone();
    field.batch_inverse(&mut xs);
    for (&x, &y) in orig.iter().zip(&xs) {
        if x == field.zero() {
            assert_eq!(y, field.zero());
        } else {
            assert_eq!(field.mul(x, y), field.one());
            assert_eq!(field.inverse(x), Some(y));
        }
    }
    assert_eq!(
        MontField::<2>::new(uint(&(&p - 2u8))),
        Err(ModulusError::Composite)
    );
}

/// Every ring operation of a static field against num-bigint.
fn check_static<R: ModRing>(ring: &R, seed: u64) {
    let mb = ring.modulus();
    let mut rng = StdRng::seed_from_u64(seed);
    let limbs = ring.limbs();
    let lift = |x: R::Elem| {
        let mut out = vec![0u64; limbs];
        ring.lift_limbs(x, &mut out);
        out.iter()
            .rev()
            .fold(BigUint::ZERO, |acc, &w| (acc << 64) + w)
    };
    let to_limbs = |v: &BigUint| {
        let mut out = v.to_u64_digits();
        out.resize(limbs, 0);
        out
    };
    let vals = values_below(&mb, &mut rng, 4);
    let elems: Vec<R::Elem> = vals
        .iter()
        .map(|v| ring.reduce_limbs(&to_limbs(v)))
        .collect();
    for (a, &x) in vals.iter().zip(&elems) {
        assert_eq!(lift(x), *a);
        assert_eq!(lift(ring.neg(x)), (&mb - a) % &mb);
        assert_eq!(lift(ring.square(x)), (a * a) % &mb);
        let mut bytes = Vec::new();
        ring.encode(x, &mut bytes);
        assert_eq!(ring.decode(&bytes), Some(x));
        for (b, &y) in vals.iter().zip(&elems) {
            assert_eq!(lift(ring.add(x, y)), (a + b) % &mb);
            assert_eq!(lift(ring.sub(x, y)), (a + &mb - b) % &mb);
            assert_eq!(lift(ring.mul(x, y)), (a * b) % &mb);
        }
    }
    let wide: Vec<u64> = (0..5).map(|_| rng.random()).collect();
    let n = wide
        .iter()
        .rev()
        .fold(BigUint::ZERO, |acc, &w| (acc << 64) + w);
    assert_eq!(lift(ring.reduce_limbs(&wide)), &n % &mb);
    assert_eq!(
        lift(ring.from_i64(-5)),
        (&mb - BigUint::from(5u8) % &mb) % &mb
    );
    // M itself, when the lift width can express it, and any wrong length are refused.
    if mb.bits() <= 64 * limbs as u64 {
        let mut bytes = Vec::new();
        for limb in to_limbs(&mb) {
            bytes.extend_from_slice(&limb.to_le_bytes());
        }
        assert_eq!(ring.decode(&bytes), None);
    }
    assert_eq!(ring.decode(&[0u8; 3]), None);
    let a = elems.clone();
    let b: Vec<R::Elem> = elems.iter().rev().copied().collect();
    let want = vals
        .iter()
        .zip(vals.iter().rev())
        .fold(BigUint::ZERO, |acc, (x, y)| acc + x * y)
        % &mb;
    assert_eq!(lift(ring.dot(&a, &b)), want);
}

proptest! {
    #[test]
    fn static_fields_match_bigint(seed: u64) {
        check_static(&StaticPrime::<BabyBear>::new(), seed);
        check_static(&StaticPrime::<KoalaBear>::new(), seed);
        check_static(&StaticPrime::<Goldilocks>::new(), seed);
        check_static(&StaticPrime::<Mersenne31>::new(), seed);
        check_static(&StaticBigPrime::<Bn254>::new(), seed);
        check_static(&StaticBigPrime::<Goldilocks>::new(), seed);
    }

    #[test]
    fn powers_of_two_match_bigint(bits in 1u32..=192, seed: u64) {
        check_static(&PowerOfTwo::new(bits).expect("in range"), seed);
    }
}

#[test]
fn static_rings_report_their_modulus() {
    assert_eq!(StaticPrime::<KoalaBear>::new().modulus_bits(), 31);
    assert_eq!(StaticBigPrime::<Bn254>::new().limbs(), 4);
    assert_eq!(StaticBigPrime::<Bn254>::new().modulus(), Bn254::order());
    assert_eq!(StaticPrime::<BabyBear>::new().inverse(BabyBear::ZERO), None);
    assert_eq!(
        StaticBigPrime::<Bn254>::new().inverse(Bn254::TWO),
        Some(Bn254::TWO.inverse())
    );
    assert_eq!(PowerOfTwo::new(0), Err(ModulusError::TooSmall));
    assert_eq!(PowerOfTwo::new(193), Err(ModulusError::TooWide));
    assert_eq!(PowerOfTwo::new(64).expect("in range").limbs(), 1);
}

#[test]
fn ring_ids_separate_kind_and_modulus() {
    let p = Uint([0x7f00_0001u64]);
    let field = MontField::new(p).expect("prime");
    let ring = MontModulus::new(p).expect("odd");
    assert_eq!(field.id(), StaticPrime::<KoalaBear>::new().id());
    assert_ne!(field.id(), ring.id());
    assert_ne!(ring.id(), PowerOfTwo::new(31).expect("in range").id());
    assert_eq!(field.id().to_bytes(), [1, 4, 0, 0, 0, 1, 0, 0, 0x7f]);
}

#[test]
fn canonical_residues_round_trip_postcard() {
    let x = CanonicalResidue([1u64, 2, 3]);
    let bytes = postcard::to_allocvec(&x).expect("serialize");
    assert_eq!(
        postcard::from_bytes::<CanonicalResidue<3>>(&bytes).expect("deserialize"),
        x
    );
}
