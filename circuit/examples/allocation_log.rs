//! An example to show how the allocation log works.
//!
//! *NOTE*: The example must be run in debug mode or with a custom profile
//! that enables debug_assertions, and with `RUST_LOG=DEBUG` for tracing.

use p3_baby_bear::BabyBear;
use p3_circuit::CircuitBuilder;
use p3_field::PrimeCharacteristicRing;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

/// Initializes a global logger with default parameters.
fn init_logger() {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();
}

fn main() {
    init_logger();

    println!("Building a simple circuit to demonstrate the allocation log...\n");

    let mut builder = CircuitBuilder::<BabyBear>::new();

    // Build a simple circuit: (a + b * c) * (a - b) / 2
    println!("Creating circuit: result = (a + b * c) * (a - b) / 2\n");

    let a = builder.alloc_public_input("input_a");
    let b = builder.alloc_public_input("input_b");
    let c = builder.alloc_public_input("input_c");

    let two = builder.alloc_const(BabyBear::TWO, "2");

    let _two_bis = builder.alloc_const(BabyBear::TWO, "2 bis"); // should be ignored

    let bc = builder.alloc_mul(b, c, "b_times_c");
    let sum = builder.alloc_add(a, bc, "a_plus_bc");
    let diff = builder.alloc_sub(a, bc, "a_minus_bc");
    let product = builder.alloc_mul(sum, diff, "sum_times_diff");
    let _result = builder.alloc_div(product, two, "final_result");

    builder.dump_allocation_log();
}
