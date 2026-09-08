//! Small AIR proofs with explicit, inexpensive demonstration parameters.
//! Run with `cargo run -p p3-configs --features baby-bear,koala-bear --example prove_prime_fields`
//! or select just `--features goldilocks`.
//! These parameters are for demonstration only; none of the configurations is hiding.

#[cfg(any(feature = "baby-bear", feature = "koala-bear", feature = "goldilocks"))]
mod prime;

fn main() {
    #[cfg(any(feature = "baby-bear", feature = "koala-bear", feature = "goldilocks"))]
    prime::run();
    #[cfg(not(any(feature = "baby-bear", feature = "koala-bear", feature = "goldilocks")))]
    eprintln!("Enable baby-bear, koala-bear or goldilocks to run this example.");
}
