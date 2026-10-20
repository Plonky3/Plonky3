//! Lincheck and opening of one circuit batch at `k_log = 14`.

use core::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use p3_binary_field::Poly64;
use p3_challenger::CanSample;
use p3_flock::opening::WitnessPcs;
use p3_flock::zerocheck::ZerocheckClaims;
use p3_flock::{
    Batch, Builder, Challenge, Circuit, FlockChallenger, PcsParams, QuirkyClaim, QuirkyPoint, Wire,
    Witness, lincheck,
};
use p3_keccak::Keccak256Hash;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Products of the benchmark circuit: with its ports, just under `2^14` slots.
const ANDS: usize = 14_000;

/// A random XOR/AND circuit of `ANDS` products over 512 input and 512 output bits.
fn circuit() -> Circuit {
    let mut rng = SmallRng::seed_from_u64(0xF10C);
    let mut c = Builder::new(&[512], &[512]);
    let mut pool: Vec<Wire> = c.input(0).to_vec();
    let mut ands = 0;
    while ands < ANDS {
        let x = pool[rng.random_range(0..pool.len())];
        let y = pool[rng.random_range(0..pool.len())];
        let w = if rng.random_bool(0.5) {
            ands += 1;
            c.and(x, y)
        } else {
            c.xor(x, y)
        };
        pool.push(w);
    }
    for bit in 0..512 {
        let w = pool[pool.len() - 1 - bit];
        c.output(0, bit, w);
    }
    let circuit = c.finish();
    assert_eq!(circuit.k_log(), 14);
    circuit
}

/// A fresh sponge.
const fn challenger() -> FlockChallenger {
    FlockChallenger::from_hasher(Vec::new(), Keccak256Hash)
}

/// A quirky point over `m` variables drawn from the sponge.
fn draw_point(sponge: &mut FlockChallenger, m: usize) -> QuirkyPoint {
    let mut draw = || {
        Challenge::new(core::array::from_fn(|_| {
            CanSample::<Poly64>::sample(sponge)
        }))
    };
    QuirkyPoint {
        skip: draw(),
        rest: (0..m - 6).map(|_| draw()).collect(),
    }
}

/// The batch witness and the zerocheck's claims at a drawn point.
fn batch(circuit: &Circuit, log_instances: usize) -> (Witness, ZerocheckClaims) {
    let mut rng = SmallRng::seed_from_u64(log_instances as u64);
    let rows: Vec<[u64; 8]> = (0..1 << log_instances)
        .map(|_| core::array::from_fn(|_| rng.random()))
        .collect();
    let w = Witness::generate(circuit, log_instances, &rows, &[0; 8], |row, words| {
        words.copy_from_slice(row);
    });
    let k_log = circuit.k_log();
    let point = draw_point(&mut challenger(), k_log + log_instances);
    let claims = ZerocheckClaims {
        a: point.evaluate_packed(w.az(), k_log),
        b: point.evaluate_packed(w.bz(), k_log),
        c: point.evaluate_packed(w.z(), k_log),
        point,
    };
    (w, claims)
}

fn lincheck(c: &mut Criterion) {
    let circuit = circuit();
    let mut group = c.benchmark_group("lincheck");
    group.sample_size(10);
    for log_instances in [16, 17, 18] {
        let (w, claims) = batch(&circuit, log_instances);
        let statement = Batch::new(&circuit, log_instances).expect("a zerocheck batch");
        group.bench_function(BenchmarkId::new("prove", log_instances), |b| {
            b.iter(|| lincheck::prove(&statement, &w, &claims, &mut challenger()));
        });
        let (proof, _) = lincheck::prove(&statement, &w, &claims, &mut challenger());
        group.bench_function(BenchmarkId::new("verify", log_instances), |b| {
            b.iter(|| lincheck::verify(&statement, &claims, &proof, &mut challenger()).unwrap());
        });
        group.bench_function(BenchmarkId::new("evaluate_packed", log_instances), |b| {
            b.iter(|| claims.point.evaluate_packed(black_box(w.z()), 14));
        });
    }
    group.finish();
}

fn opening(c: &mut Criterion) {
    let circuit = circuit();
    let mut group = c.benchmark_group("opening");
    group.sample_size(10);
    for log_instances in [16, 17] {
        let (w, claims) = batch(&circuit, log_instances);
        let m = circuit.k_log() + log_instances;
        let pcs = WitnessPcs::new(m, &PcsParams::default()).unwrap();
        let claim = QuirkyClaim {
            point: claims.point.clone(),
            value: claims.c,
        };
        group.bench_function(BenchmarkId::new("commit", log_instances), |b| {
            b.iter(|| pcs.commit(&w, &mut challenger()).unwrap());
        });
        group.bench_function(BenchmarkId::new("open", log_instances), |b| {
            b.iter_batched(
                || pcs.commit(&w, &mut challenger()).unwrap(),
                |(_, data)| {
                    pcs.open(data, core::slice::from_ref(&claim), &mut challenger())
                        .unwrap()
                },
                BatchSize::PerIteration,
            );
        });
        let mut sponge = challenger();
        let (commitment, data) = pcs.commit(&w, &mut sponge).unwrap();
        let proof = pcs
            .open(data, core::slice::from_ref(&claim), &mut sponge)
            .unwrap();
        group.bench_function(BenchmarkId::new("verify", log_instances), |b| {
            b.iter(|| {
                let mut sponge = challenger();
                pcs.observe(&commitment, &mut sponge);
                pcs.verify(
                    &commitment,
                    core::slice::from_ref(&claim),
                    &proof,
                    &mut sponge,
                )
                .unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, lincheck, opening);
criterion_main!(benches);
