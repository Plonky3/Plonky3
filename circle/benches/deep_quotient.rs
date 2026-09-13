use std::hint::black_box;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};
use p3_circle::Point;
use p3_field::{PrimeCharacteristicRing, batch_multiplicative_inverse};
use p3_mersenne_31::{Mersenne31, QM31};

type F = Mersenne31;
type EF = QM31;

fn domain_points(log_n: usize) -> Vec<Point<F>> {
    let generator = Point::generator(log_n - 1);
    let mut coset_0 = Point::generator(log_n + 1);
    let mut coset_1 = generator - coset_0;
    let mut points = Vec::with_capacity(1 << log_n);
    for _ in 0..1 << (log_n - 1) {
        points.push(coset_0);
        points.push(coset_1);
        coset_0 += generator;
        coset_1 += generator;
    }
    points
}

fn current_geometry(points: &[Point<F>], zeta: Point<EF>) -> (Vec<EF>, Vec<EF>, Vec<EF>) {
    let (re, im): (Vec<_>, Vec<_>) = points.iter().map(|&point| point.v_p(zeta)).unzip();
    let denoms = re
        .iter()
        .zip(&im)
        .map(|(&re, &im)| re.square() + im.square())
        .collect::<Vec<_>>();
    (re, im, batch_multiplicative_inverse(&denoms))
}

fn factored_geometry(points: &[Point<F>], zeta: Point<EF>) -> Vec<EF> {
    let (re, im): (Vec<_>, Vec<_>) = points.iter().map(|&point| point.v_p(zeta)).unzip();
    let denoms = re.iter().map(|re| re.double()).collect::<Vec<_>>();
    let denom_inv = batch_multiplicative_inverse(&denoms);
    im.into_iter()
        .zip(denom_inv)
        .map(|(im, denom_inv)| im * denom_inv)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn current_accumulate(
    ro: &mut [EF],
    alpha_offset: EF,
    alpha_pow_width: EF,
    reduced_rows: &[EF],
    re: &[EF],
    im: &[EF],
    denom_inv: &[EF],
    reduced_at_zeta: EF,
) {
    for ((((ro, &row), &re), &im), &denom_inv) in ro
        .iter_mut()
        .zip(reduced_rows)
        .zip(re)
        .zip(im)
        .zip(denom_inv)
    {
        *ro += alpha_offset * (re - alpha_pow_width * im) * denom_inv * (row - reduced_at_zeta);
    }
}

fn factored_accumulate(
    ro: &mut [EF],
    alpha_offset: EF,
    alpha_pow_width: EF,
    reduced_rows: &[EF],
    im_over_twice_re: &[EF],
    reduced_at_zeta: EF,
) {
    let alpha_offset_over_two = alpha_offset.halve();
    let alpha_offset_alpha_pow_width = alpha_offset * alpha_pow_width;
    for ((ro, &row), &im_over_twice_re) in ro.iter_mut().zip(reduced_rows).zip(im_over_twice_re) {
        *ro += (alpha_offset_over_two - alpha_offset_alpha_pow_width * im_over_twice_re)
            * (row - reduced_at_zeta);
    }
}

fn bench_deep_quotient(c: &mut Criterion) {
    let points = domain_points(18);
    let zeta = Point::<EF>::from_projective_line(EF::from_u8(9));
    let alpha_offset = EF::from_u8(7);
    let alpha_pow_width = EF::from_u8(11);
    let reduced_at_zeta = EF::from_u8(13);
    let reduced_rows = (0..points.len())
        .map(|i| EF::from_u64(i as u64 + 1))
        .collect::<Vec<_>>();
    let (re, im, denom_inv) = current_geometry(&points, zeta);
    let im_over_twice_re = factored_geometry(&points, zeta);

    let mut group = c.benchmark_group("circle/deep_quotient/log_n=18");
    group.bench_function("geometry/current", |b| {
        b.iter(|| black_box(current_geometry(black_box(&points), black_box(zeta))));
    });
    group.bench_function("geometry/factored", |b| {
        b.iter(|| black_box(factored_geometry(black_box(&points), black_box(zeta))));
    });

    let mut current_ro = EF::zero_vec(points.len());
    group.bench_function("accumulate/current", |b| {
        b.iter(|| {
            current_accumulate(
                black_box(&mut current_ro),
                alpha_offset,
                alpha_pow_width,
                black_box(&reduced_rows),
                black_box(&re),
                black_box(&im),
                black_box(&denom_inv),
                reduced_at_zeta,
            );
            black_box(&current_ro);
        });
    });

    let mut factored_ro = EF::zero_vec(points.len());
    group.bench_function("accumulate/factored", |b| {
        b.iter(|| {
            factored_accumulate(
                black_box(&mut factored_ro),
                alpha_offset,
                alpha_pow_width,
                black_box(&reduced_rows),
                black_box(&im_over_twice_re),
                reduced_at_zeta,
            );
            black_box(&factored_ro);
        });
    });
    group.finish();
}

criterion_group! {
    name = deep_quotient;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_deep_quotient
}
criterion_main!(deep_quotient);
