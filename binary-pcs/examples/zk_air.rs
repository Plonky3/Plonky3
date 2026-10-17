//! Native end-to-end private current-row AIR reference example.
//!
//! Run with `cargo run -p p3-binary-pcs --features experimental-zk --example zk_air --release`.
//! This uses the polynomial-size full-blend protocol, not BaseFold. The target
//! below budgets interactive algebra/query error only; hash and Fiat–Shamir
//! compilation losses remain unproved. See the crate README for the exact scope.

use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_binary_field::TowerLevel;
use p3_binary_pcs::zk::{ReferenceZkAir, Val, ZkAirConfig, ZkAirProof};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use rand::rngs::{StdRng, SysRng};
use rand::{RngExt, SeedableRng};

const LOG_TRACE_HEIGHT: usize = 2;
const INTERACTIVE_TARGET_BITS: usize = 32;

struct PrivateProduct;

impl BaseAir<Val> for PrivateProduct {
    fn width(&self) -> usize {
        2
    }

    fn num_public_values(&self) -> usize {
        1
    }

    // BaseAir defaults to allowing successor access; this protocol requires
    // an explicit declaration that every constraint uses only the current row.
    fn main_next_row_columns(&self) -> Vec<usize> {
        vec![]
    }
}

impl<AB: AirBuilder<F = Val>> Air<AB> for PrivateProduct {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.current_slice();
        let product: AB::Expr = builder.public_values()[0].into();
        builder.assert_zero(row[0] * row[1] - product);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fresh OS entropy seeds the private CSPRNG. The public Keccak transcript
    // is separate and never supplies witness randomness or hiding masks.
    let mut rng = StdRng::try_from_rng(&mut SysRng)?;
    let public_product = Val::from_repr(7);
    let height = 1 << LOG_TRACE_HEIGHT;
    let mut values = Vec::with_capacity(2 * height);
    for _ in 0..height {
        let factor = loop {
            let candidate: Val = rng.random();
            if candidate != Val::ZERO {
                break candidate;
            }
        };
        values.extend([factor, public_product * factor.inverse()]);
    }
    let trace = RowMajorMatrix::new(values, 2);
    let air = ReferenceZkAir::compile(
        &PrivateProduct,
        ZkAirConfig::new(LOG_TRACE_HEIGHT, INTERACTIVE_TARGET_BITS)?,
    )?;
    let proof = air.prove(trace, &[public_product], &mut rng)?;
    let bytes = postcard::to_allocvec(&proof)?;
    let decoded: ZkAirProof = postcard::from_bytes(&bytes)?;
    air.verify(&decoded, &[public_product])?;

    // Report only public dimensions and serialized size, never private inputs,
    // seeds, trace values, or unblinded polynomial coefficients.
    println!(
        "Verified serialized proof: {} bytes; {height} rows, 2 private columns, 1 public value; \
         {INTERACTIVE_TARGET_BITS}-bit interactive algebra/query target",
        bytes.len()
    );
    Ok(())
}
