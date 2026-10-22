//! Verifying proofs inside a circuit.
//!
//! [`CircuitArith`] runs any verifier written over [`Arith`] and records it as a circuit.
//!
//! The verifier's native transcript drives a mirror challenger.
//!
//! The mirror absorbs and samples wires in lockstep with it.

mod challenger;
mod constraints;
mod mmcs;
mod stark;

use alloc::collections::VecDeque;
use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_field::{ExtensionField, PrimeCharacteristicRing, PrimeField64};
use p3_verifier_arith::{Arith, Lift};
use spin::Mutex;

pub use self::challenger::DuplexMirror;
pub use self::stark::{CircuitConfig, CircuitPcs, MirrorChallenger};
use crate::{Base, Circuit, CircuitBuilder, Ext, PermutationChip};

/// A draw of the native transcript, as the wires the verifier takes it as.
#[derive(Debug)]
pub(crate) enum Sample<F> {
    /// One base-field sample.
    Value(Base<F>),
    /// A query index, as its bits, least significant first.
    Index(Vec<Base<F>>),
}

/// The circuit being built, shared with the challenger that mirrors the native transcript.
#[derive(Debug)]
pub(crate) struct Shared<F, EF, P> {
    /// The circuit and its witness.
    pub(crate) builder: CircuitBuilder<F, EF, P>,
    /// Values the native transcript absorbs next, in absorb order.
    pending: VecDeque<Base<F>>,
    /// Samples the native transcript drew and the verifier has not yet taken.
    sampled: VecDeque<Sample<F>>,
    /// Whether the transcript under construction is absorbing its seed.
    seeding: bool,
}

impl<F, EF, P> Shared<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    /// The wire carrying the next value the native transcript absorbs.
    ///
    /// A seed is a constant of the transcript's shape, so it enters as a constant.
    ///
    /// # Panics
    ///
    /// When the verifier did not announce the value, or announced a different one.
    pub(crate) fn absorbed(&mut self, value: F) -> Base<F> {
        if self.seeding {
            return Base::from(value);
        }
        let unit = self
            .pending
            .pop_front()
            .expect("every absorbed proof value is announced before the transcript absorbs it");
        assert_eq!(
            self.builder.value(&unit),
            value,
            "the announced value is the one the transcript absorbs"
        );
        unit
    }

    /// Hand a sample the native transcript drew to the verifier.
    pub(crate) fn push_sample(&mut self, sample: Sample<F>) {
        self.sampled.push_back(sample);
    }
}

/// How a circuit takes the verifying keys of the proofs it checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KeyBinding {
    /// Each key is a constant, so the circuit accepts proofs of one statement only.
    #[default]
    Constant,
    /// Each key is a public input, so whoever checks this circuit's proof checks the key.
    Public,
}

/// An [`Arith`] whose values are circuit wires.
///
/// Running a verifier on it builds the circuit that checks the same proof.
#[derive(Debug)]
pub struct CircuitArith<F, EF, P> {
    /// The circuit under construction.
    pub(crate) shared: Arc<Mutex<Shared<F, EF, P>>>,
    /// How verifying keys enter the circuit.
    keys: KeyBinding,
}

impl<F, EF, P> CircuitArith<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    /// An empty circuit proving permutations with `chip`.
    pub fn new(chip: P) -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                builder: CircuitBuilder::new(chip),
                pending: VecDeque::new(),
                sampled: VecDeque::new(),
                seeding: false,
            })),
            keys: KeyBinding::default(),
        }
    }

    /// The same circuit, taking verifying keys as `keys` says.
    #[must_use]
    pub const fn with_key_binding(mut self, keys: KeyBinding) -> Self {
        self.keys = keys;
        self
    }

    /// Run `f` on the circuit under construction.
    pub fn with_builder<R>(&mut self, f: impl FnOnce(&mut CircuitBuilder<F, EF, P>) -> R) -> R {
        f(&mut self.shared.lock().builder)
    }

    /// A value of a verifying key, bound as the key binding says.
    pub(crate) fn load_key(&self, value: F) -> Base<F> {
        match self.keys {
            KeyBinding::Constant => Base::from(value),
            KeyBinding::Public => self.shared.lock().builder.public(value),
        }
    }

    /// Every inner public value, as a public input of this circuit.
    pub fn public_inputs(&mut self, values: &[Vec<F>]) -> Vec<Vec<Base<F>>> {
        let builder = &mut self.shared.lock().builder;
        values
            .iter()
            .map(|row| row.iter().map(|&value| builder.public(value)).collect())
            .collect()
    }

    /// The finished circuit.
    ///
    /// # Panics
    ///
    /// When a mirror challenger is still alive, or a transcript value was never consumed.
    pub fn build(self) -> Circuit<F, EF, P> {
        let shared = Arc::try_unwrap(self.shared)
            .unwrap_or_else(|_| {
                panic!("every mirror challenger is dropped before the circuit is built")
            })
            .into_inner();
        assert!(
            shared.pending.is_empty(),
            "every announced value is absorbed"
        );
        assert!(shared.sampled.is_empty(), "every sample is taken");
        shared.builder.build()
    }
}

impl<F, EF, P> Arith for CircuitArith<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    type Val = F;
    type Challenge = EF;
    type F = Base<F>;
    type EF = Ext<F, EF>;
    type Bit = Base<F>;
    type Index = Vec<Base<F>>;

    fn bit(&self, bit: &Base<F>) -> Base<F> {
        bit.clone()
    }

    fn index_bits(&mut self, index: &Vec<Base<F>>, count: usize) -> Vec<Base<F>> {
        // Bits above the sampled width are zero.
        let mut bits: Vec<Base<F>> = index.iter().take(count).cloned().collect();
        bits.resize(count, Base::ZERO);
        bits
    }

    fn index_shr(&self, index: &Vec<Base<F>>, shift: usize) -> Vec<Base<F>> {
        index.iter().skip(shift).cloned().collect()
    }

    fn constant_index(&mut self, value: usize) -> Vec<Base<F>> {
        (0..usize::BITS - value.leading_zeros())
            .map(|i| Base::from(F::from_bool((value >> i) & 1 == 1)))
            .collect()
    }

    fn index_with_high(
        &mut self,
        low: &Vec<Base<F>>,
        low_bits: usize,
        high: usize,
    ) -> Vec<Base<F>> {
        let mut bits = self.index_bits(low, low_bits);
        bits.extend(self.constant_index(high));
        bits
    }

    fn batch_inverse<E>(
        &mut self,
        values: &[Ext<F, EF>],
        zero: impl FnOnce(usize) -> E,
    ) -> Result<Vec<Ext<F, EF>>, E> {
        let mut shared = self.shared.lock();
        let mut inverses = Vec::with_capacity(values.len());
        for (i, value) in values.iter().enumerate() {
            match shared.builder.inverse_ext(value) {
                Ok(inverse) => inverses.push(inverse),
                Err(_) => return Err(zero(i)),
            }
        }
        Ok(inverses)
    }

    fn assert_zero<E>(&mut self, value: Ext<F, EF>, err: impl FnOnce() -> E) -> Result<(), E> {
        self.shared
            .lock()
            .builder
            .assert_zero_ext(&value)
            .map_err(|_| err())
    }

    fn coefficients(&mut self, value: &Ext<F, EF>) -> Vec<Base<F>> {
        self.shared.lock().builder.decompose(value)
    }
}

impl<F, EF, P> Lift for CircuitArith<F, EF, P>
where
    F: PrimeField64,
    EF: ExtensionField<F>,
    P: PermutationChip<F>,
{
    fn load(&mut self, value: F) -> Base<F> {
        self.shared.lock().builder.witness(value)
    }

    fn load_ext(&mut self, value: EF) -> Ext<F, EF> {
        self.shared.lock().builder.witness_ext(value)
    }

    fn value(&self, value: &Base<F>) -> F {
        self.shared.lock().builder.value(value)
    }

    fn value_ext(&self, value: &Ext<F, EF>) -> EF {
        self.shared.lock().builder.value_ext(value)
    }

    fn seeding<T>(&mut self, construct: impl FnOnce() -> T) -> T {
        self.shared.lock().seeding = true;
        let transcript = construct();
        self.shared.lock().seeding = false;
        transcript
    }

    fn will_observe(&mut self, units: Vec<Base<F>>) {
        self.shared.lock().pending.extend(units);
    }

    fn sampled_ext(&mut self, value: EF) -> Ext<F, EF> {
        let mut shared = self.shared.lock();
        let limbs: Vec<Base<F>> = (0..EF::DIMENSION)
            .map(|_| match shared.sampled.pop_front() {
                Some(Sample::Value(limb)) => limb,
                _ => panic!("a challenge-field sample was drawn"),
            })
            .collect();
        let sample = shared.builder.compose(&limbs);
        assert_eq!(
            shared.builder.value_ext(&sample),
            value,
            "the mirror samples what the transcript samples"
        );
        sample
    }

    fn sampled_index(&mut self, value: usize, bits: usize) -> Vec<Base<F>> {
        let mut shared = self.shared.lock();
        let index: Vec<Base<F>> = match shared.sampled.pop_front() {
            Some(Sample::Value(sample)) => shared.builder.canonical_bits(&sample),
            Some(Sample::Index(index)) => index,
            None => panic!("a query index sample was drawn"),
        }
        .into_iter()
        .take(bits)
        .collect();
        let recomposed = index
            .iter()
            .enumerate()
            .filter(|(_, bit)| shared.builder.value(bit).is_one())
            .map(|(i, _)| 1usize << i)
            .sum::<usize>();
        assert_eq!(
            recomposed, value,
            "the mirror samples what the transcript samples"
        );
        index
    }
}
