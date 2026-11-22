use core::array;
use std::hint::black_box;
use std::time::Instant;

use p3_field::extension::{
    BinomialExtensionField, Complex, HasComplexBinomialExtension, qm31_mul_min_rank,
};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_mersenne_31::Mersenne31;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

type CM31 = Complex<Mersenne31>;
type EF2 = BinomialExtensionField<CM31, 2>;
type EF3 = BinomialExtensionField<CM31, 3>;

/// Exact pre-optimization `0d80373` `Complex<Mersenne31>::mul` (`quadratic_mul` with `W = -1` mul).
#[inline(always)]
fn legacy_cm31_mul(a: CM31, b: CM31) -> CM31 {
    let [a0, a1] = a.to_array();
    let [b0, b1] = b.to_array();
    let b1_w = b1 * Mersenne31::NEG_ONE;
    CM31::new_complex(
        Mersenne31::dot_product(&[a0, a1], &[b0, b1_w]),
        Mersenne31::dot_product(&[a0, a1], &[b1, b0]),
    )
}

/// Exact pre-optimization `0d80373` `Complex<Mersenne31>::square` (`quadratic_extension_square`).
#[inline(always)]
fn legacy_cm31_square(a: CM31) -> CM31 {
    let [a0, a1] = a.to_array();
    CM31::new_complex(
        a0.square() + a1.square() * Mersenne31::NEG_ONE,
        (a0 * a1).double(),
    )
}

/// Exact pre-optimization `0d80373` `QM31::mul` (`binomial_mul::<CM31, ..., 2>`).
#[inline(always)]
fn legacy_qm31_mul(a: &[CM31; 2], b: &[CM31; 2], res: &mut [CM31; 2]) {
    let w = CM31::new_complex(Mersenne31::TWO, Mersenne31::ONE);
    let b1_w = legacy_cm31_mul(b[1], w);
    res[0] = legacy_cm31_mul(a[0], b[0]) + legacy_cm31_mul(a[1], b1_w);
    res[1] = legacy_cm31_mul(a[0], b[1]) + legacy_cm31_mul(a[1], b[0]);
}

/// Karatsuba-3 `CM31` + add-only `mul_by_w` `QM31::mul` (generic `HasComplexBinomialExtension<2>`).
#[inline(always)]
fn karatsuba3_qm31_mul(a: &[CM31; 2], b: &[CM31; 2], res: &mut [CM31; 2]) {
    let m0 = a[0] * b[0];
    let m1 = a[1] * b[1];
    let m2 = (a[0] + a[1]) * (b[0] + b[1]);
    res[0] = m0 + <Mersenne31 as HasComplexBinomialExtension<2>>::mul_by_w(m1);
    res[1] = m2 - m0 - m1;
}

/// Minimum-rank 7-multiply `QM31::mul` (`qm31_mul_min_rank` over `Mersenne31`).
#[inline(always)]
fn rank7_qm31_mul(a: &[CM31; 2], b: &[CM31; 2], res: &mut [CM31; 2]) {
    let fa = [a[0].real(), a[0].imag(), a[1].real(), a[1].imag()];
    let fb = [b[0].real(), b[0].imag(), b[1].real(), b[1].imag()];
    let mut out = [Mersenne31::ZERO; 4];
    qm31_mul_min_rank::<Mersenne31, Mersenne31>(&fa, &fb, &mut out);
    res[0] = CM31::new_complex(out[0], out[1]);
    res[1] = CM31::new_complex(out[2], out[3]);
}

/// Exact pre-optimization `0d80373` `QM31::square`.
#[inline(always)]
fn legacy_qm31_square(a: &[CM31; 2], res: &mut [CM31; 2]) {
    let s1 = legacy_cm31_square(a[1]);
    let w_s1 = CM31::new_complex(
        s1.real() + s1.real() - s1.imag(),
        s1.real() + s1.imag() + s1.imag(),
    );
    res[0] = legacy_cm31_square(a[0]) + w_s1;
    res[1] = legacy_cm31_mul(a[0], a[1]).double();
}

/// Exact pre-optimization `0d80373` `Sextic::mul` (`binomial_mul::<CM31, ..., 3>`).
#[inline(always)]
fn legacy_sextic_mul(a: &[CM31; 3], b: &[CM31; 3], res: &mut [CM31; 3]) {
    let w = CM31::new_imag(Mersenne31::new(5));
    let m0 = legacy_cm31_mul(a[0], b[0]);
    let m1 = legacy_cm31_mul(a[1], b[1]);
    let m2 = legacy_cm31_mul(a[2], b[2]);
    res[0] = m0 + legacy_cm31_mul(legacy_cm31_mul(a[1] + a[2], b[1] + b[2]) - m1 - m2, w);
    res[1] = legacy_cm31_mul(a[0] + a[1], b[0] + b[1]) - m0 - m1 + legacy_cm31_mul(m2, w);
    res[2] = legacy_cm31_mul(a[0] + a[2], b[0] + b[2]) - m0 - m2 + m1;
}

/// Karatsuba-6 `CM31` + `mul_by_w` `Sextic::mul` (generic `HasComplexBinomialExtension<3>`).
#[inline(always)]
fn karatsuba6_sextic_mul(a: &[CM31; 3], b: &[CM31; 3], res: &mut [CM31; 3]) {
    let m0 = a[0] * b[0];
    let m1 = a[1] * b[1];
    let m2 = a[2] * b[2];
    let t0 = (a[1] + a[2]) * (b[1] + b[2]) - m1 - m2;
    let t1 = (a[0] + a[1]) * (b[0] + b[1]) - m0 - m1;
    let t2 = (a[0] + a[2]) * (b[0] + b[2]) - m0 - m2;
    res[0] = m0 + <Mersenne31 as HasComplexBinomialExtension<3>>::mul_by_w(t0);
    res[1] = t1 + <Mersenne31 as HasComplexBinomialExtension<3>>::mul_by_w(m2);
    res[2] = t2 + m1;
}

/// Exact pre-optimization `0d80373` `Sextic::square` (`cubic_square` with full `W` mul).
#[inline(always)]
fn legacy_sextic_square(a: &[CM31; 3], res: &mut [CM31; 3]) {
    let w = CM31::new_imag(Mersenne31::new(5));
    let w_a2 = legacy_cm31_mul(a[2], w);
    res[0] = legacy_cm31_square(a[0]) + legacy_cm31_mul(a[1], w_a2).double();
    res[1] = legacy_cm31_mul(w_a2, a[2]) + legacy_cm31_mul(a[0], a[1]).double();
    res[2] = legacy_cm31_square(a[1]) + legacy_cm31_mul(a[0], a[2]).double();
}

#[inline(always)]
fn ef_to_array<const D: usize>(x: BinomialExtensionField<CM31, D>) -> [CM31; D]
where
    CM31: p3_field::extension::BinomiallyExtendable<D>,
{
    x.as_basis_coefficients_slice().try_into().unwrap()
}

fn bench_latency<T: Copy>(
    domain: &str,
    label: &str,
    inputs: &[T],
    iters: usize,
    mut f: impl FnMut(T, T) -> T,
) {
    let mut acc = inputs[0];
    for &item in inputs.iter().take(64) {
        acc = f(acc, item);
    }
    let start = Instant::now();
    for step in 0..iters {
        let rhs = inputs[step & (inputs.len() - 1)];
        acc = f(black_box(acc), black_box(rhs));
    }
    let elapsed = start.elapsed();
    let _ = black_box(acc);
    let ns_per_op = elapsed.as_secs_f64() * 1e9 / (iters as f64);
    println!("{domain:<16} | {label:<34} : {ns_per_op:>7.3} ns/op");
}

fn bench_throughput<T: Copy>(
    domain: &str,
    label: &str,
    inputs: &[T],
    rounds: usize,
    mut f: impl FnMut(T, T) -> T,
) {
    let mut lanes: [T; 8] = array::from_fn(|i| inputs[i]);
    let start = Instant::now();
    for r in 0..rounds {
        let rhs = inputs[(r + 8) & (inputs.len() - 1)];
        for lane in &mut lanes {
            *lane = f(black_box(*lane), black_box(rhs));
        }
    }
    let elapsed = start.elapsed();
    let _ = black_box(lanes);
    let total_ops = rounds * 8;
    let ns_per_op = elapsed.as_secs_f64() * 1e9 / (total_ops as f64);
    println!("{domain:<16} | {label:<34} : {ns_per_op:>7.3} ns/op");
}

#[test]
fn benchmark_mersenne31_complex_extensions_before_vs_after() {
    let mut rng = SmallRng::seed_from_u64(42);
    let cm_inputs: Vec<CM31> = (0..1024).map(|_| rng.random()).collect();
    let qm_inputs: Vec<EF2> = (0..1024).map(|_| rng.random()).collect();
    let sx_inputs: Vec<EF3> = (0..1024).map(|_| rng.random()).collect();

    let iters = if cfg!(debug_assertions) {
        1_000
    } else {
        2_000_000
    };
    let rounds = iters / 8;
    let inv_iters = iters / 16;

    println!("\n=== CM31 (Complex<Mersenne31>) ===");
    bench_latency(
        "CM31 Latency",
        "Before (0d80373 mul)",
        &cm_inputs,
        iters,
        legacy_cm31_mul,
    );
    bench_latency(
        "CM31 Latency",
        "After  (CM31::mul)",
        &cm_inputs,
        iters,
        |a, b| a * b,
    );
    bench_throughput(
        "CM31 Throughput",
        "Before (0d80373 mul)",
        &cm_inputs,
        rounds,
        legacy_cm31_mul,
    );
    bench_throughput(
        "CM31 Throughput",
        "After  (CM31::mul)",
        &cm_inputs,
        rounds,
        |a, b| a * b,
    );
    bench_latency(
        "CM31 Square",
        "Before (0d80373 square)",
        &cm_inputs,
        iters,
        |a, _| legacy_cm31_square(a),
    );
    bench_latency(
        "CM31 Square",
        "After  (CM31::square)",
        &cm_inputs,
        iters,
        |a, _| a.square(),
    );

    println!("\n=== QM31 (BinomialExtensionField<Complex<Mersenne31>, 2>) ===");
    bench_latency(
        "QM31 Latency",
        "Before (0d80373 binomial_mul)",
        &qm_inputs,
        iters,
        |a, b| {
            let mut out = [CM31::ZERO; 2];
            legacy_qm31_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF2::new(out)
        },
    );
    bench_latency(
        "QM31 Latency",
        "Generic Karatsuba-3 + mul_by_w",
        &qm_inputs,
        iters,
        |a, b| {
            let mut out = [CM31::ZERO; 2];
            karatsuba3_qm31_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF2::new(out)
        },
    );
    bench_latency(
        "QM31 Latency",
        "Min-Rank-7 (qm31_mul_min_rank)",
        &qm_inputs,
        iters,
        |a, b| {
            let mut out = [CM31::ZERO; 2];
            rank7_qm31_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF2::new(out)
        },
    );
    bench_latency(
        "QM31 Latency",
        "After  (QM31::mul wired dot4)",
        &qm_inputs,
        iters,
        |a, b| a * b,
    );

    bench_throughput(
        "QM31 Throughput",
        "Before (0d80373 binomial_mul)",
        &qm_inputs,
        rounds,
        |a, b| {
            let mut out = [CM31::ZERO; 2];
            legacy_qm31_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF2::new(out)
        },
    );
    bench_throughput(
        "QM31 Throughput",
        "Generic Karatsuba-3 + mul_by_w",
        &qm_inputs,
        rounds,
        |a, b| {
            let mut out = [CM31::ZERO; 2];
            karatsuba3_qm31_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF2::new(out)
        },
    );
    bench_throughput(
        "QM31 Throughput",
        "After  (QM31::mul wired dot4)",
        &qm_inputs,
        rounds,
        |a, b| a * b,
    );

    bench_latency(
        "QM31 Square",
        "Before (0d80373 ext_square)",
        &qm_inputs,
        iters,
        |a, _| {
            let mut out = [CM31::ZERO; 2];
            legacy_qm31_square(&ef_to_array(a), &mut out);
            EF2::new(out)
        },
    );
    bench_latency(
        "QM31 Square",
        "After  (QM31::square wired)",
        &qm_inputs,
        iters,
        |a, _| a.square(),
    );
    bench_latency(
        "QM31 Inverse",
        "After  (QM31::inverse)",
        &qm_inputs,
        inv_iters,
        |a, b| (if a == EF2::ZERO { EF2::ONE } else { a }).inverse() + b,
    );

    println!("\n=== Sextic M31^6 (BinomialExtensionField<Complex<Mersenne31>, 3>) ===");
    bench_latency(
        "M31^6 Latency",
        "Before (0d80373 cubic_mul)",
        &sx_inputs,
        iters,
        |a, b| {
            let mut out = [CM31::ZERO; 3];
            legacy_sextic_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF3::new(out)
        },
    );
    bench_latency(
        "M31^6 Latency",
        "Generic Karatsuba-6 + mul_by_w",
        &sx_inputs,
        iters,
        |a, b| {
            let mut out = [CM31::ZERO; 3];
            karatsuba6_sextic_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF3::new(out)
        },
    );
    bench_latency(
        "M31^6 Latency",
        "After  (EF3::mul wired dot6)",
        &sx_inputs,
        iters,
        |a, b| a * b,
    );

    bench_throughput(
        "M31^6 Throughput",
        "Before (0d80373 cubic_mul)",
        &sx_inputs,
        rounds,
        |a, b| {
            let mut out = [CM31::ZERO; 3];
            legacy_sextic_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF3::new(out)
        },
    );
    bench_throughput(
        "M31^6 Throughput",
        "Generic Karatsuba-6 + mul_by_w",
        &sx_inputs,
        rounds,
        |a, b| {
            let mut out = [CM31::ZERO; 3];
            karatsuba6_sextic_mul(&ef_to_array(a), &ef_to_array(b), &mut out);
            EF3::new(out)
        },
    );
    bench_throughput(
        "M31^6 Throughput",
        "After  (EF3::mul wired dot6)",
        &sx_inputs,
        rounds,
        |a, b| a * b,
    );

    bench_latency(
        "M31^6 Square",
        "Before (0d80373 cubic_square)",
        &sx_inputs,
        iters,
        |a, _| {
            let mut out = [CM31::ZERO; 3];
            legacy_sextic_square(&ef_to_array(a), &mut out);
            EF3::new(out)
        },
    );
    bench_latency(
        "M31^6 Square",
        "After  (EF3::square wired)",
        &sx_inputs,
        iters,
        |a, _| a.square(),
    );
    bench_latency(
        "M31^6 Inverse",
        "After  (EF3::inverse)",
        &sx_inputs,
        inv_iters,
        |a, b| (if a == EF3::ZERO { EF3::ONE } else { a }).inverse() + b,
    );
}
