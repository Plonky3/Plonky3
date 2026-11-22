use core::array;
use core::cell::Cell;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_baby_bear::BabyBear;
use p3_field::extension::{
    BinomialExtensionField, BinomiallyExtendable, CubicTrinomialExtensionField,
    QuinticTrinomialExtensionField, binomial_mul_min_rank, binomial_square_min_rank,
    cubic_trinomial_mul_min_rank, cubic_trinomial_square_min_rank, qm31_mul_min_rank,
    qm31_square_min_rank, quintic_trinomial_mul_min_rank, quintic_trinomial_square_min_rank,
};
use p3_field::{Algebra, BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks;
use p3_koala_bear::KoalaBear;
use p3_mersenne_31::{Mersenne31, QM31};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

thread_local! {
    static MUL_COUNT: Cell<usize> = const { Cell::new(0) };
    static SQUARE_COUNT: Cell<usize> = const { Cell::new(0) };
}

fn reset_counts() {
    MUL_COUNT.with(|c| c.set(0));
    SQUARE_COUNT.with(|c| c.set(0));
}

fn counts() -> (usize, usize) {
    (MUL_COUNT.with(Cell::get), SQUARE_COUNT.with(Cell::get))
}

/// Wrapper algebra over `F` that counts `A * A` multiplications and `A::square` calls,
/// while treating base-field scalar multiplications `A * F` as degree-0 linear ops.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct CountingAlgebra<F>(F);

impl<F: Field> From<F> for CountingAlgebra<F> {
    #[inline]
    fn from(value: F) -> Self {
        Self(value)
    }
}

impl<F: Field> Neg for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl<F: Field> Add for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl<F: Field> Add<F> for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: F) -> Self {
        Self(self.0 + rhs)
    }
}

impl<F: Field> AddAssign for CountingAlgebra<F> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl<F: Field> AddAssign<F> for CountingAlgebra<F> {
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.0 += rhs;
    }
}

impl<F: Field> Sub for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl<F: Field> Sub<F> for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: F) -> Self {
        Self(self.0 - rhs)
    }
}

impl<F: Field> SubAssign for CountingAlgebra<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl<F: Field> SubAssign<F> for CountingAlgebra<F> {
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        self.0 -= rhs;
    }
}

impl<F: Field> Mul for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        MUL_COUNT.with(|c| c.set(c.get().saturating_add(1)));
        Self(self.0 * rhs.0)
    }
}

impl<F: Field> Mul<F> for CountingAlgebra<F> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self(self.0 * rhs)
    }
}

impl<F: Field> MulAssign for CountingAlgebra<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F: Field> MulAssign<F> for CountingAlgebra<F> {
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        self.0 *= rhs;
    }
}

impl<F: Field> Sum for CountingAlgebra<F> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
    }
}

impl<F: Field> Product for CountingAlgebra<F> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |a, b| a * b)
    }
}

impl<F: Field> PrimeCharacteristicRing for CountingAlgebra<F> {
    type PrimeSubfield = F::PrimeSubfield;
    const ZERO: Self = Self(F::ZERO);
    const ONE: Self = Self(F::ONE);
    const TWO: Self = Self(F::TWO);
    const NEG_ONE: Self = Self(F::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self(F::from_prime_subfield(f))
    }

    #[inline]
    fn square(&self) -> Self {
        SQUARE_COUNT.with(|c| c.set(c.get() + 1));
        Self(self.0.square())
    }
}

impl<F: Field> Algebra<F> for CountingAlgebra<F> {}

fn check_binomial<F, const D: usize>(expected_muls: usize, expected_sq_ops: (usize, usize))
where
    F: BinomiallyExtendable<D>,
    rand::distr::StandardUniform: rand::distr::Distribution<F>,
{
    let w = <F as BinomiallyExtendable<D>>::W;

    // Exhaustive basis-pair verification.
    for i in 0..D {
        for j in 0..D {
            let mut a = [CountingAlgebra(F::ZERO); D];
            let mut b = [CountingAlgebra(F::ZERO); D];
            a[i] = CountingAlgebra(F::ONE);
            b[j] = CountingAlgebra(F::ONE);
            let mut res = [CountingAlgebra(F::ZERO); D];

            reset_counts();
            binomial_mul_min_rank::<F, CountingAlgebra<F>, D>(&a, &b, &mut res, w);
            assert_eq!(counts(), (expected_muls, 0));

            let ef_a = BinomialExtensionField::<F, D>::new(a.map(|x| x.0));
            let ef_b = BinomialExtensionField::<F, D>::new(b.map(|x| x.0));
            let expected = ef_a * ef_b;
            assert_eq!(&res.map(|x| x.0), expected.as_basis_coefficients_slice());
        }
    }

    // Random trials for both mul and square.
    let mut rng = SmallRng::seed_from_u64(0x517c_c1b7_2722_0a95 ^ (D as u64));
    for _ in 0..256 {
        let raw_a: [F; D] = array::from_fn(|_| rng.random());
        let raw_b: [F; D] = array::from_fn(|_| rng.random());
        let a = raw_a.map(CountingAlgebra);
        let b = raw_b.map(CountingAlgebra);

        let mut res_mul = [CountingAlgebra(F::ZERO); D];
        reset_counts();
        binomial_mul_min_rank::<F, CountingAlgebra<F>, D>(&a, &b, &mut res_mul, w);
        assert_eq!(counts(), (expected_muls, 0));

        let ef_a = BinomialExtensionField::<F, D>::new(raw_a);
        let ef_b = BinomialExtensionField::<F, D>::new(raw_b);
        assert_eq!(
            &res_mul.map(|x| x.0),
            (ef_a * ef_b).as_basis_coefficients_slice()
        );

        let mut res_sq = [CountingAlgebra(F::ZERO); D];
        reset_counts();
        binomial_square_min_rank::<F, CountingAlgebra<F>, D>(&a, &mut res_sq, w);
        assert_eq!(counts(), expected_sq_ops);
        assert_eq!(
            &res_sq.map(|x| x.0),
            ef_a.square().as_basis_coefficients_slice()
        );
    }
}

#[test]
fn binomial_min_rank_matches_all_plonky3_extensions() {
    check_binomial::<Goldilocks, 2>(3, (2, 0));
    check_binomial::<Mersenne31, 2>(3, (2, 0));
    check_binomial::<Mersenne31, 3>(5, (0, 5));
    check_binomial::<BabyBear, 4>(7, (0, 7));
    check_binomial::<KoalaBear, 4>(7, (0, 7));
    check_binomial::<BabyBear, 5>(9, (0, 9));
    check_binomial::<Goldilocks, 5>(9, (0, 9));
    check_binomial::<BabyBear, 8>(15, (0, 15));
    check_binomial::<KoalaBear, 8>(15, (0, 15));
}

#[test]
fn cubic_trinomial_min_rank_matches_goldilocks() {
    type F = Goldilocks;
    type EF = CubicTrinomialExtensionField<F>;

    for i in 0..3 {
        for j in 0..3 {
            let mut raw_a = [F::ZERO; 3];
            let mut raw_b = [F::ZERO; 3];
            raw_a[i] = F::ONE;
            raw_b[j] = F::ONE;
            let a = raw_a.map(CountingAlgebra);
            let b = raw_b.map(CountingAlgebra);
            let mut res = [CountingAlgebra(F::ZERO); 3];

            reset_counts();
            cubic_trinomial_mul_min_rank::<F, CountingAlgebra<F>>(&a, &b, &mut res);
            assert_eq!(counts(), (5, 0));
            assert_eq!(
                &res.map(|x| x.0),
                (EF::new(raw_a) * EF::new(raw_b)).as_basis_coefficients_slice()
            );
        }
    }

    let mut rng = SmallRng::seed_from_u64(0x3333_7777);
    for _ in 0..256 {
        let raw_a: [F; 3] = array::from_fn(|_| rng.random());
        let raw_b: [F; 3] = array::from_fn(|_| rng.random());
        let a = raw_a.map(CountingAlgebra);
        let b = raw_b.map(CountingAlgebra);

        let mut res_mul = [CountingAlgebra(F::ZERO); 3];
        reset_counts();
        cubic_trinomial_mul_min_rank::<F, CountingAlgebra<F>>(&a, &b, &mut res_mul);
        assert_eq!(counts(), (5, 0));
        assert_eq!(
            &res_mul.map(|x| x.0),
            (EF::new(raw_a) * EF::new(raw_b)).as_basis_coefficients_slice()
        );

        let mut res_sq = [CountingAlgebra(F::ZERO); 3];
        reset_counts();
        cubic_trinomial_square_min_rank::<F, CountingAlgebra<F>>(&a, &mut res_sq);
        assert_eq!(counts(), (0, 5));
        assert_eq!(
            &res_sq.map(|x| x.0),
            EF::new(raw_a).square().as_basis_coefficients_slice()
        );
    }
}

#[test]
fn quintic_trinomial_min_rank_matches_koalabear() {
    type F = KoalaBear;
    type EF = QuinticTrinomialExtensionField<F>;

    for i in 0..5 {
        for j in 0..5 {
            let mut raw_a = [F::ZERO; 5];
            let mut raw_b = [F::ZERO; 5];
            raw_a[i] = F::ONE;
            raw_b[j] = F::ONE;
            let a = raw_a.map(CountingAlgebra);
            let b = raw_b.map(CountingAlgebra);
            let mut res = [CountingAlgebra(F::ZERO); 5];

            reset_counts();
            quintic_trinomial_mul_min_rank::<F, CountingAlgebra<F>>(&a, &b, &mut res);
            assert_eq!(counts(), (9, 0));
            assert_eq!(
                &res.map(|x| x.0),
                (EF::new(raw_a) * EF::new(raw_b)).as_basis_coefficients_slice()
            );
        }
    }

    let mut rng = SmallRng::seed_from_u64(0x5555_9999);
    for _ in 0..256 {
        let raw_a: [F; 5] = array::from_fn(|_| rng.random());
        let raw_b: [F; 5] = array::from_fn(|_| rng.random());
        let a = raw_a.map(CountingAlgebra);
        let b = raw_b.map(CountingAlgebra);

        let mut res_mul = [CountingAlgebra(F::ZERO); 5];
        reset_counts();
        quintic_trinomial_mul_min_rank::<F, CountingAlgebra<F>>(&a, &b, &mut res_mul);
        assert_eq!(counts(), (9, 0));
        assert_eq!(
            &res_mul.map(|x| x.0),
            (EF::new(raw_a) * EF::new(raw_b)).as_basis_coefficients_slice()
        );

        let mut res_sq = [CountingAlgebra(F::ZERO); 5];
        reset_counts();
        quintic_trinomial_square_min_rank::<F, CountingAlgebra<F>>(&a, &mut res_sq);
        assert_eq!(counts(), (0, 9));
        assert_eq!(
            &res_sq.map(|x| x.0),
            EF::new(raw_a).square().as_basis_coefficients_slice()
        );
    }
}

#[test]
fn qm31_min_rank_matches_mersenne31_qm31() {
    type F = Mersenne31;

    for i in 0..4 {
        for j in 0..4 {
            let mut raw_a = [F::ZERO; 4];
            let mut raw_b = [F::ZERO; 4];
            raw_a[i] = F::ONE;
            raw_b[j] = F::ONE;
            let a = raw_a.map(CountingAlgebra);
            let b = raw_b.map(CountingAlgebra);
            let mut res = [CountingAlgebra(F::ZERO); 4];

            reset_counts();
            qm31_mul_min_rank::<F, CountingAlgebra<F>>(&a, &b, &mut res);
            assert_eq!(counts(), (7, 0));

            let qa = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(&raw_a).unwrap();
            let qb = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(&raw_b).unwrap();
            let expected = qa * qb;
            assert_eq!(
                &res.map(|x| x.0),
                <QM31 as BasedVectorSpace<F>>::as_basis_coefficients_slice(&expected)
            );
        }
    }

    let mut rng = SmallRng::seed_from_u64(0x2147_4836);
    for _ in 0..512 {
        let raw_a: [F; 4] = array::from_fn(|_| rng.random());
        let raw_b: [F; 4] = array::from_fn(|_| rng.random());
        let a = raw_a.map(CountingAlgebra);
        let b = raw_b.map(CountingAlgebra);

        let mut res_mul = [CountingAlgebra(F::ZERO); 4];
        reset_counts();
        qm31_mul_min_rank::<F, CountingAlgebra<F>>(&a, &b, &mut res_mul);
        assert_eq!(counts(), (7, 0));

        let qa = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(&raw_a).unwrap();
        let qb = <QM31 as BasedVectorSpace<F>>::from_basis_coefficients_slice(&raw_b).unwrap();
        assert_eq!(
            &res_mul.map(|x| x.0),
            <QM31 as BasedVectorSpace<F>>::as_basis_coefficients_slice(&(qa * qb))
        );

        let mut res_sq = [CountingAlgebra(F::ZERO); 4];
        reset_counts();
        qm31_square_min_rank::<F, CountingAlgebra<F>>(&a, &mut res_sq);
        assert_eq!(counts(), (6, 0));
        assert_eq!(
            &res_sq.map(|x| x.0),
            <QM31 as BasedVectorSpace<F>>::as_basis_coefficients_slice(&qa.square())
        );
    }
}
