use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger, GrindingChallenger,
};
use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Hash, MerkleCap};
use spin::Mutex;

use super::Shared;
use crate::{Base, PermutationChip};

/// A [`DuplexChallenger`] that mirrors every absorb and squeeze onto circuit wires.
///
/// The native challenger still runs, so the native transcript sees exactly the values it expects.
///
/// The mirror sponge runs the same duplex construction on wires, through the circuit's chip.
pub struct DuplexMirror<F, EF, P, Perm, const WIDTH: usize, const RATE: usize>
where
    F: Clone,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    /// The native challenger.
    native: DuplexChallenger<F, Perm, WIDTH, RATE>,
    /// The circuit under construction.
    shared: Arc<Mutex<Shared<F, EF, P>>>,
    /// The sponge state, as wires.
    state: Vec<Base<F>>,
    /// Absorbed values not yet permuted.
    input: Vec<Base<F>>,
    /// Squeezed values not yet sampled.
    output: Vec<Base<F>>,
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize>
    DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    /// A mirror of `native`, whose current state enters the circuit as constants.
    pub(crate) fn new(
        native: DuplexChallenger<F, Perm, WIDTH, RATE>,
        shared: Arc<Mutex<Shared<F, EF, P>>>,
    ) -> Self {
        assert_eq!(P::WIDTH, WIDTH, "the chip permutes the sponge state");
        let constants = |values: &[F]| values.iter().map(|&v| Base::from(v)).collect();
        Self {
            state: constants(&native.sponge_state),
            input: constants(&native.input_buffer),
            output: constants(&native.output_buffer),
            native,
            shared,
        }
    }

    /// The duplex step: overwrite the rate with the input, tag its length, permute.
    fn duplexing(&mut self, shared: &mut Shared<F, EF, P>) {
        let absorbed = self.input.len();
        for (i, value) in self.input.drain(..).enumerate() {
            self.state[i] = value;
        }
        if absorbed > 0 {
            self.state[absorbed..RATE].fill(Base::ZERO);
            self.state[RATE] = self.state[RATE].clone() + F::from_u8(absorbed as u8);
        }
        self.state = shared.builder.permute(&self.state);
        self.output.clear();
        self.output.extend_from_slice(&self.state[..RATE]);
    }

    /// Sample one base-field value and hand its wire to the verifier.
    fn sample_base(&mut self) -> F {
        let value: F = self.native.sample();
        let shared = Arc::clone(&self.shared);
        let mut shared = shared.lock();
        let sample = self.squeeze(&mut shared, value);
        shared.push_sample(sample);
        value
    }

    /// Squeeze one wire, as the native challenger squeezes its value.
    fn squeeze(&mut self, shared: &mut Shared<F, EF, P>, value: F) -> Base<F> {
        if !self.input.is_empty() || self.output.is_empty() {
            self.duplexing(shared);
        }
        let sample = self.output.pop().expect("a duplex step fills the output");
        assert_eq!(
            shared.builder.value(&sample),
            value,
            "the mirror squeezes what the sponge squeezes"
        );
        sample
    }
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize> CanObserve<F>
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, value: F) {
        self.native.observe(value);
        let shared = Arc::clone(&self.shared);
        let mut shared = shared.lock();
        let unit = shared.absorbed(value);
        self.output.clear();
        self.input.push(unit);
        if self.input.len() == RATE {
            self.duplexing(&mut shared);
        }
    }
}

impl<F, EF, P, Perm, const N: usize, const WIDTH: usize, const RATE: usize> CanObserve<[F; N]>
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: [F; N]) {
        values.into_iter().for_each(|value| self.observe(value));
    }
}

impl<F, EF, P, Perm, const N: usize, const WIDTH: usize, const RATE: usize>
    CanObserve<Hash<F, F, N>> for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, values: Hash<F, F, N>) {
        values.into_iter().for_each(|value| self.observe(value));
    }
}

impl<F, EF, P, Perm, const N: usize, const WIDTH: usize, const RATE: usize>
    CanObserve<MerkleCap<F, [F; N]>> for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    fn observe(&mut self, cap: MerkleCap<F, [F; N]>) {
        for digest in cap.roots() {
            digest.iter().for_each(|&value| self.observe(value));
        }
    }
}

impl<F, EF, P, Perm, T, const WIDTH: usize, const RATE: usize> CanSample<T>
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
    T: BasedVectorSpace<F>,
{
    fn sample(&mut self) -> T {
        // One base sample per basis coefficient, in the order the native sponge draws them.
        T::from_basis_coefficients_fn(|_| self.sample_base())
    }
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize> CanSampleBits<usize>
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    fn sample_bits(&mut self, bits: usize) -> usize {
        assert!(bits < (usize::BITS as usize));
        assert!((1 << bits) < F::ORDER_U64);
        let value = self.sample_base();
        value.as_canonical_u64() as usize & ((1 << bits) - 1)
    }
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize> FieldChallenger<F>
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F> + Send,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize> GrindingChallenger
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    type Witness = F;

    /// # Panics
    ///
    /// Always: a verifier checks witnesses and never grinds for one.
    fn grind(&mut self, _bits: usize) -> F {
        unreachable!("a verifier checks witnesses and never grinds for one")
    }

    fn check_witness(&mut self, bits: usize, witness: F) -> bool {
        if bits == 0 {
            return true;
        }
        self.observe(witness);

        // The witness passes when the low `bits` bits of the next sample are zero.
        let value: F = self.native.sample();
        let shared = Arc::clone(&self.shared);
        let mut shared = shared.lock();
        let sample = self.squeeze(&mut shared, value);
        let sample_bits = shared.builder.canonical_bits(&sample);
        for bit in &sample_bits[..bits] {
            // A failing witness leaves the circuit unsatisfiable.
            //
            // The native check rejects it too.
            let _ = shared.builder.assert_zero(bit);
        }
        value.as_canonical_u64() & ((1 << bits) - 1) == 0
    }
}
