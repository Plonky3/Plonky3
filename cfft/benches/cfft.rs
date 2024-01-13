#![allow(unused_imports)]
#![allow(unused_variables)]
#![allow(dead_code)]

use std::any::type_name;

use p3_cfft::{cfft, cfft_inv, cfft_inv_twiddles, cfft_twiddles};
use p3_mersenne_31::{Mersenne31, Mersenne31Complex};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

fn bench_cfft(c: &mut Criterion) {
    // log_sizes correspond to the sizes of DFT we want to benchmark;
    let log_sizes = &[3, 6, 10];

    cfft_timing(c, log_sizes);
    cfft_inv_timing(c, log_sizes);
}

fn cfft_timing(c: &mut Criterion, log_sizes: &[usize]) {
    let mut group = c.benchmark_group(&format!("cfft::<{}>", type_name::<Mersenne31>(),));
    group.sample_size(10);

    let mut rng = rand::thread_rng();
    for log_n in log_sizes {
        let n = 1 << log_n;

        // let mut message = Vec::new();
        // for _ in 0..n {
        //     message.push(rng.gen::<Mersenne31>())
        // }

        let mut message: Vec<_> = vec![(); n]
            .iter()
            .map(|_| rng.gen::<Mersenne31>())
            .collect();

        let twiddles = cfft_twiddles(*log_n);

        group.bench_function(&format!("Benching Size {}", n), |b| {
            b.iter(|| cfft(&mut message, &twiddles))
        });
    }
}

fn cfft_inv_timing(c: &mut Criterion, log_sizes: &[usize]) {
    let mut group = c.benchmark_group(&format!("cfft_inv::<{}>", type_name::<Mersenne31>(),));
    group.sample_size(10);

    let mut rng = rand::thread_rng();
    for log_n in log_sizes {
        let n = 1 << log_n;

        // let mut message = Vec::new();
        // for _ in 0..n {
        //     message.push(rng.gen::<Mersenne31>())
        // }

        let mut message: Vec<_> = vec![(); n]
            .iter()
            .map(|_| rng.gen::<Mersenne31>())
            .collect();

        let twiddles = cfft_inv_twiddles(*log_n);

        group.bench_function(&format!("Benching Size {}", n), |b| {
            b.iter(|| cfft_inv(&mut message, &twiddles))
        });
    }
}

criterion_group!(benches, bench_cfft);
criterion_main!(benches);
