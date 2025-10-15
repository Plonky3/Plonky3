use p3_field::PrimeCharacteristicRing;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

// Generic Algebra trait tests: R implements Algebra<F> if From<F>, +F, -F, *F and assign ops hold
pub fn test_algebra_over_base<R, F>()
where
    R: PrimeCharacteristicRing
        + From<F>
        + core::ops::Add<F, Output = R>
        + core::ops::AddAssign<F>
        + core::ops::Sub<F, Output = R>
        + core::ops::SubAssign<F>
        + core::ops::Mul<F, Output = R>
        + core::ops::MulAssign<F>
        + Eq
        + Clone,
    F: PrimeCharacteristicRing + Clone,
    StandardUniform: Distribution<R> + Distribution<F>,
{
    let mut rng = SmallRng::seed_from_u64(1);
    let r: R = rng.random();
    let f: F = rng.random();
    let g: F = rng.random();

    // from preserves identities
    assert_eq!(R::from(F::ONE), R::ONE);
    assert_eq!(R::from(F::ZERO), R::ZERO);

    // +F, -F, *F
    let add = r.clone() + f.clone();
    let sub = r.clone() - f.clone();
    let mul = r.clone() * f.clone();
    let mut add_asg = r.clone();
    add_asg += f.clone();
    let mut sub_asg = r.clone();
    sub_asg -= f.clone();
    let mut mul_asg = r.clone();
    mul_asg *= f.clone();

    assert_eq!(add_asg, add);
    assert_eq!(sub_asg, sub);
    assert_eq!(mul_asg, mul);

    // distributivity over F scalars on the right
    assert_eq!(
        r.clone() * (f.clone() + g.clone()),
        r.clone() * f.clone() + r * g
    );
}
