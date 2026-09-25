use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use p3_binary_field::Gf2;
use p3_blake3_air::Blake3BinaryAir;
use p3_examples::binary::{BinaryProofOptions, prove_boolean_air_cubic};
use p3_sumcheck::layout::Table;
use tracing::{Subscriber, span::Attributes};
use tracing_subscriber::{Layer, layer::Context, layer::SubscriberExt};

struct SlicedRound(Arc<AtomicBool>);

impl<S: Subscriber> Layer<S> for SlicedRound {
    fn on_new_span(&self, attrs: &Attributes<'_>, _: &tracing::span::Id, _: Context<'_, S>) {
        if attrs.metadata().name() == "fold_sliced" {
            self.0.store(true, Ordering::Relaxed);
        }
    }
}

#[test]
fn cubic_folding_keeps_early_zerocheck_rounds_sliced() {
    let seen = Arc::new(AtomicBool::new(false));
    tracing::subscriber::set_global_default(
        tracing_subscriber::registry().with(SlicedRound(seen.clone())),
    )
    .unwrap();

    let air = Blake3BinaryAir::default();
    let trace = Table::from_packed_bits(air.generate_random_trace_packed::<Gf2>(128), 7);
    prove_boolean_air_cubic(&air, trace, BinaryProofOptions::default()).unwrap();
    assert!(
        seen.load(Ordering::Relaxed),
        "first fold used the dense extension-field path"
    );
}
