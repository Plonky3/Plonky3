//! Multi-claim bit ring-switching benchmarks using the production arithmetic representations.

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use p3_binary_field::{
    BinaryChallenger, BinaryField128, BitCoordinates, Ghash128, Poly64, Poly192, TowerLevel,
};
use p3_challenger::HashChallenger;
use p3_challenger::fs::TranscriptField;
use p3_field::ExtensionField;
use p3_keccak::Keccak256Hash;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::ring_switch::bits::{BitPacking, BitRingSwitch, BitRingSwitchClaims};
use p3_sumcheck::strategy::{FromTable, IntoTranscriptField};
use rand::SeedableRng;
use rand::rngs::SmallRng;

/// Multi-claim bit reduction, including the coordinate-weight construction and accumulation.
fn bench_bit_claims(c: &mut Criterion) {
    bench_bit_claims_for::<BinaryField128, BinaryField128, Ghash128>(c, "ghash128");
    bench_bit_claims_for::<Poly64, Poly192, Poly192>(c, "poly192");
}

fn bench_bit_claims_for<F, EF, R>(c: &mut Criterion, field: &str)
where
    F: TranscriptField + TowerLevel + Send + Sync,
    EF: BitCoordinates + ExtensionField<F> + Send + Sync,
    R: IntoTranscriptField<EF> + FromTable<F> + Sync,
    rand::distr::StandardUniform: rand::distr::Distribution<F> + rand::distr::Distribution<EF>,
{
    let mut group = c.benchmark_group(format!("ring_switch/bit_claims/{field}"));
    for packed_variables in [12, 16] {
        let mut rng = SmallRng::seed_from_u64(0xC1A1);
        let packed = Poly::<F>::rand(&mut rng, packed_variables);
        let packing = BitPacking::from_packed(packed).unwrap();
        let variables = packed_variables + BitRingSwitch::<F, EF>::ABSORBED;
        for successors in [false, true] {
            for count in [2, 4, 8] {
                let claims = BitRingSwitchClaims::new(
                    (0..count)
                        .map(|_| {
                            let point = Point::<EF>::rand(&mut rng, variables);
                            if successors {
                                BitRingSwitch::<F, EF>::with_successor(&point, variables - 1)
                            } else {
                                BitRingSwitch::<F, EF>::new(&point)
                            }
                            .unwrap()
                        })
                        .collect(),
                )
                .unwrap();
                let kind = if successors { "successor" } else { "plain" };
                group.throughput(Throughput::Elements((count as u64) << packed_variables));
                group.bench_function(
                    BenchmarkId::new(kind, format!("n{packed_variables}_k{count}")),
                    |b| {
                        b.iter(|| {
                            let mut challenger = BinaryChallenger::<
                                F,
                                HashChallenger<u8, Keccak256Hash, 32>,
                            >::from_hasher(
                                Vec::new(), Keccak256Hash
                            );
                            black_box(claims.prove::<R, _, _>(black_box(&packing), &mut challenger))
                        });
                    },
                );
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_bit_claims);
criterion_main!(benches);
