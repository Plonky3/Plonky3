use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_challenger::{
    CanObserve, CanSample, CanSampleBits, CanSampleUniformBits, DuplexChallenger, FieldChallenger,
    GrindingChallenger, ResamplingError, UniformSamplingField,
};
use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::{CryptographicPermutation, Hash, MerkleCap};
use spin::Mutex;

use super::{Sample, Shared};
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
        shared.push_sample(Sample::Value(sample));
        value
    }

    /// Draw base samples until one lies below `bound`, as the native resample loop does.
    ///
    /// Each draw is constrained to the side of the bound it fell on, so the accepted draw is the native one.
    ///
    /// # Errors
    ///
    /// When a draw is rejected and `resample` is false.
    fn accepted_bits(
        &mut self,
        bound: u64,
        bits: usize,
        resample: bool,
    ) -> Result<Vec<Base<F>>, ResamplingError> {
        let shared = Arc::clone(&self.shared);
        let mut shared = shared.lock();
        loop {
            let value: F = self.native.sample();
            let sample = self.squeeze(&mut shared, value);
            let sample_bits = shared.builder.canonical_bits(&sample);
            let below = below(&sample_bits, bound);
            let accepted = value.as_canonical_u64() < bound;
            // A draw on the wrong side of the bound leaves the circuit unsatisfiable.
            let side = if accepted { Base::ONE - below } else { below };
            shared
                .builder
                .assert_zero(&side)
                .expect("a draw lies on the side of the bound the native loop saw");
            if accepted {
                return Ok(sample_bits[..bits].to_vec());
            }
            if !resample {
                return Err(ResamplingError::new(value.as_canonical_u64(), bound));
            }
        }
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

/// Whether the integer of `bits`, least significant first, lies below the constant `bound`, as a bit.
fn below<F: Field>(bits: &[Base<F>], bound: u64) -> Base<F> {
    // From the top bit down, the first bit where the integer and the bound differ decides.
    let (mut below, mut equal) = (Base::ZERO, Base::ONE);
    for (i, bit) in bits.iter().enumerate().rev() {
        if (bound >> i) & 1 == 1 {
            below += equal.clone() * (Base::ONE - bit.clone());
            equal *= bit.clone();
        } else {
            equal *= Base::ONE - bit.clone();
        }
    }
    below
}

impl<F, EF, P, Perm, const WIDTH: usize, const RATE: usize> CanSampleUniformBits<F>
    for DuplexMirror<F, EF, P, Perm, WIDTH, RATE>
where
    F: UniformSamplingField + PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
    Perm: CryptographicPermutation<[F; WIDTH]>,
{
    /// Mirrors the native rejection sampling squeeze for squeeze.
    ///
    /// The circuit's shape depends on how many draws were rejected, each draw being rejected with probability below `2^-24`.
    fn sample_uniform_bits<const RESAMPLE: bool>(
        &mut self,
        bits: usize,
    ) -> Result<usize, ResamplingError> {
        let index = if bits == 0 {
            Vec::new()
        } else {
            assert!(bits < usize::BITS as usize, "bit count must be valid");
            assert!(
                (1u64 << bits) < F::ORDER_U64,
                "bit count exceeds field order"
            );
            if bits <= F::MAX_SINGLE_SAMPLE_BITS {
                self.accepted_bits(F::SAMPLING_BITS_M[bits], bits, RESAMPLE)?
            } else {
                // Two draws of half the bits each, the first one holding the low half.
                let low = bits / 2;
                let mut index = self.accepted_bits(F::SAMPLING_BITS_M[low], low, RESAMPLE)?;
                let high = bits - low;
                index.extend(self.accepted_bits(F::SAMPLING_BITS_M[high], high, RESAMPLE)?);
                index
            }
        };
        let mut shared = self.shared.lock();
        let value = index
            .iter()
            .enumerate()
            .filter(|(_, bit)| shared.builder.value(bit).is_one())
            .map(|(i, _)| 1usize << i)
            .sum();
        shared.push_sample(Sample::Index(index));
        Ok(value)
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
