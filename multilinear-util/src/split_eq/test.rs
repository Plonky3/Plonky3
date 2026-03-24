use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_util::log2_strict_usize;
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

use crate::evals::Poly;
use crate::evals::test::eval_reference;
use crate::multilinear::Point;
use crate::split_eq::eq::SplitEq;

type F = BabyBear;
type PackedF = <F as Field>::Packing;
type EF = BinomialExtensionField<F, 4>;

fn accumulate_into_reference<F: Field>(out: &mut [F], point: &Point<F>, alpha: Option<F>) {
    let alpha = alpha.unwrap_or(F::ONE);
    let eq = Poly::new_from_point(point.as_slice(), alpha);
    out.iter_mut()
        .zip(eq.iter())
        .for_each(|(out, &eq)| *out += eq);
}

#[test]
fn test_accumulate() {
    let mut rng = SmallRng::seed_from_u64(0);

    for k in 0..=18 {
        let input = Poly::<EF>::rand(&mut rng, k);

        let mut out_ref = input.clone();
        let point = Point::rand(&mut rng, k);
        let alpha: EF = rng.random();
        accumulate_into_reference(out_ref.as_mut_slice(), &point, Some(alpha));

        let mut out = input.clone();
        SplitEq::<F, EF>::new_unpacked(&point, EF::ONE)
            .accumulate_into(out.as_mut_slice(), Some(alpha));
        assert_eq!(out_ref, out);

        let mut out = input.clone();
        SplitEq::<F, EF>::new_unpacked(&point, alpha).accumulate_into(out.as_mut_slice(), None);
        assert_eq!(out_ref, out);

        let mut out = input.clone();
        SplitEq::<F, EF>::new_packed(&point, EF::ONE)
            .accumulate_into(out.as_mut_slice(), Some(alpha));
        assert_eq!(out_ref, out);

        let mut out = input.clone();
        SplitEq::<F, EF>::new_packed(&point, alpha).accumulate_into(out.as_mut_slice(), None);
        assert_eq!(out_ref, out);

        if k > log2_strict_usize(PackedF::WIDTH) {
            let mut out = input.pack::<F, EF>();
            SplitEq::<F, EF>::new_packed(&point, EF::ONE)
                .accumulate_into_packed(out.as_mut_slice(), Some(alpha));
            assert_eq!(out_ref, out.unpack::<F, EF>());

            let mut out = input.pack::<F, EF>();
            SplitEq::<F, EF>::new_packed(&point, alpha)
                .accumulate_into_packed(out.as_mut_slice(), None);
            assert_eq!(out_ref, out.unpack::<F, EF>());
        }
    }
}

#[test]
fn test_eval() {
    let mut rng = SmallRng::seed_from_u64(0);

    for k in 0..=18 {
        // base field
        {
            let poly = Poly::<F>::rand(&mut rng, k);
            let point = Point::rand(&mut rng, k);
            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_base(&poly)
            );
            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_base(&poly)
            );
        }

        // extension field
        {
            let poly = Poly::<EF>::rand(&mut rng, k);
            let point = Point::rand(&mut rng, k);
            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_ext(&poly)
            );
            assert_eq!(
                eval_reference(poly.as_slice(), point.as_slice()),
                SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_ext(&poly)
            );
        }

        // packed extension field
        {
            if k >= log2_strict_usize(PackedF::WIDTH) {
                let poly = Poly::<EF>::rand(&mut rng, k);
                let point = Point::rand(&mut rng, k);
                let expected = eval_reference(poly.as_slice(), point.as_slice());
                let poly = poly.pack::<F, EF>();
                assert_eq!(
                    expected,
                    SplitEq::<F, EF>::new_packed(&point, EF::ONE).eval_packed(&poly)
                );
                assert_eq!(
                    expected,
                    SplitEq::<F, EF>::new_unpacked(&point, EF::ONE).eval_packed(&poly)
                );
            }
        }
    }
}

#[test]
fn test_compress() {
    let mut rng = SmallRng::seed_from_u64(0);

    for k in 0..=18 {
        for point_k in 0..=k {
            let poly = Poly::<F>::rand(&mut rng, k);
            let point: Point<EF> = Point::rand(&mut rng, k);
            let expected = eval_reference(poly.as_slice(), point.as_slice());

            let (point_lo, point_hi) = point.split_at(point_k);

            let split_lo = SplitEq::<F, EF>::new_unpacked(&point_lo, EF::ONE);
            let split_hi = SplitEq::<F, EF>::new_unpacked(&point_hi, EF::ONE);

            let compressed = split_lo.compress_lo(&poly);
            assert_eq!(expected, split_hi.eval_ext(&compressed));

            if k >= (point_k + log2_strict_usize(PackedF::WIDTH)) {
                let split_lo = SplitEq::<F, EF>::new_packed(&point_lo, EF::ONE);
                let split_hi = SplitEq::<F, EF>::new_packed(&point_hi, EF::ONE);
                let compressed = split_lo.compress_lo(&poly);
                assert_eq!(expected, split_hi.eval_ext(&compressed));

                let compressed = split_lo.compress_lo_to_packed(&poly).unpack::<F, EF>();
                assert_eq!(expected, split_hi.eval_ext(&compressed));

                let compressed = split_hi.compress_hi(&poly);
                assert_eq!(expected, split_lo.eval_ext(&compressed));
            }
        }
    }
}
