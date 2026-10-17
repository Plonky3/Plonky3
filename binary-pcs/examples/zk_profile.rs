//! Finite native profiling of the current-row full-blend reference protocol.
//!
//! Build with `cargo build -p p3-binary-pcs --features zk --example zk_profile
//! --release`, then run the binary directly as `zk_profile product-4 3`.
//! Accepted cases: product-4, product-64, product-256, four-products-64.
//! Measure process peak RSS separately, e.g. `/usr/bin/time -l <binary> ...`;
//! peak RSS includes process/runtime memory and is not a live-heap measurement.
//! Every sample uses fresh OS-seeded private randomness and consumed prover
//! state. Timings exclude witness generation and include only the named phases.

use std::time::Instant;

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::TowerLevel;
use p3_binary_pcs::zk::{ClassicalRomBudget, ReferenceZkAir, Val, ZkAirConfig};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::{StdRng, SysRng};
use rand::{RngExt, SeedableRng};

struct Products(usize);
impl BaseAir<Val> for Products {
    fn width(&self) -> usize {
        2 * self.0
    }
    fn num_public_values(&self) -> usize {
        self.0
    }
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}
impl<AB: AirBuilder<F = Val>> Air<AB> for Products {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        for pair in 0..self.0 {
            let product: AB::Expr = builder.public_values()[pair].into();
            builder.assert_zero(row[2 * pair] * row[2 * pair + 1] - product);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let case = args.next().unwrap_or_else(|| "product-4".into());
    let (log_height, pairs) = match case.as_str() {
        "product-4" => (2, 1),
        "product-64" => (6, 1),
        "product-256" => (8, 1),
        "four-products-64" => (6, 4),
        _ => {
            return Err(
                "unsupported case; use product-4/product-64/product-256/four-products-64".into(),
            );
        }
    };
    let iterations: usize = args.next().map_or(Ok(3), |s| s.parse())?;
    if !(1..=10).contains(&iterations) || args.next().is_some() {
        return Err("expected one case and iteration count 1..=10".into());
    }
    for iteration in 0..iterations {
        let mut rng = StdRng::try_from_rng(&mut SysRng)?;
        let public: Vec<_> = (0..pairs).map(|j| Val::from_repr(7 + j as u128)).collect();
        let mut values = Vec::with_capacity((1 << log_height) * 2 * pairs);
        for _ in 0..1 << log_height {
            for &product in &public {
                let factor = loop {
                    let v: Val = rng.random();
                    if v != Val::ZERO {
                        break v;
                    }
                };
                values.extend([factor, product * factor.inverse()]);
            }
        }
        let start = Instant::now();
        let air = ReferenceZkAir::compile(
            &Products(pairs),
            ZkAirConfig::classical_rom(log_height, ClassicalRomBudget::new(80, 32, 49)?)?,
        )?;
        let compile = start.elapsed();
        let start = Instant::now();
        let proof = air.prove(RowMajorMatrix::new(values, 2 * pairs), &public, &mut rng)?;
        let prove = start.elapsed();
        let start = Instant::now();
        let bytes = air.encode_proof(&proof)?;
        let encode = start.elapsed();
        let start = Instant::now();
        let decoded = air.decode_proof(&bytes)?;
        air.verify(&decoded, &public)?;
        let decode_verify = start.elapsed();
        println!(
            "case={case} sample={} bytes={} compile_ms={:.3} prove_ms={:.3} encode_ms={:.3} decode_verify_ms={:.3}",
            iteration + 1,
            bytes.len(),
            compile.as_secs_f64() * 1000.0,
            prove.as_secs_f64() * 1000.0,
            encode.as_secs_f64() * 1000.0,
            decode_verify.as_secs_f64() * 1000.0
        );
        if iteration == 0 {
            println!("public_security_report={:?}", air.security_report());
        }
    }
    Ok(())
}
