use p3_air::{Air, AirBuilder};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, extension::BinomialExtensionField};
use p3_koala_bear::{GenericPoseidon2LinearLayersKoalaBear, KoalaBear};
use p3_matrix::dense::RowMajorMatrixView;
use p3_poseidon2_air::{Poseidon2Air, RoundConstants};

pub type PrimeField<EF> = <EF as PrimeCharacteristicRing>::PrimeSubfield;
pub type ExtensionPacking<EF> = <EF as ExtensionField<PrimeField<EF>>>::ExtensionPacking;

#[derive(Debug)]
pub struct ConstraintFolderPackedExtension<'a, EF: Field + ExtensionField<PrimeField<EF>>> {
    pub main: RowMajorMatrixView<'a, ExtensionPacking<EF>>,
    pub alpha_powers: &'a [EF],
    pub accumulator: ExtensionPacking<EF>,
    pub constraint_index: usize,
}

impl<'a, EF: Field + ExtensionField<PrimeField<EF>>> AirBuilder
    for ConstraintFolderPackedExtension<'a, EF>
{
    type F = PrimeField<EF>;
    type I = <PrimeField<EF> as Field>::Packing;
    type Expr = ExtensionPacking<EF>;
    type Var = ExtensionPacking<EF>;
    type M = RowMajorMatrixView<'a, ExtensionPacking<EF>>;

    #[inline]
    fn main(&self) -> Self::M {
        self.main
    }

    #[inline]
    fn is_first_row(&self) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn is_last_row(&self) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn is_transition_window(&self, _: usize) -> Self::Expr {
        unreachable!()
    }

    #[inline]
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let alpha_power = self.alpha_powers[self.constraint_index];
        let x: ExtensionPacking<EF> = x.into();
        self.accumulator += x * alpha_power;
        self.constraint_index += 1;
    }

    #[inline]
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, _array: [I; N]) {
        unreachable!();
    }
}

fn main() {
    const WIDTH: usize = 16;
    const SBOX_DEGREE: u64 = 3;
    const SBOX_REGISTERS: usize = 0;
    const HALF_FULL_ROUNDS: usize = 4;
    const PARTIAL_ROUNDS: usize = 20;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    let constants =
        RoundConstants::<F, WIDTH, HALF_FULL_ROUNDS, PARTIAL_ROUNDS>::from_rng(&mut rand::rng());

    let poseidon_air = Poseidon2Air::<
        F,
        GenericPoseidon2LinearLayersKoalaBear,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    >::new(constants.clone());

    my_funct::<EF, _>(&poseidon_air);
}

fn my_funct<
    'a,
    EF: Field + ExtensionField<PrimeField<EF>>,
    A: Air<ConstraintFolderPackedExtension<'a, EF>>,
>(
    _air: &A,
) {
}
