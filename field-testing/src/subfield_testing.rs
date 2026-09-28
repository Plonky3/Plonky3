use alloc::vec::Vec;

use num_bigint::BigUint;
use p3_field::{Field, HasSubfield};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// Subfields with at most this many elements are checked on every element.
const MAX_ENUMERATED_ORDER: u32 = 1 << 8;

/// The number of random elements drawn from each field, and of interpolation nodes checked.
const NUM_SAMPLES: usize = 64;

/// Slice lengths for `all_in_subfield`, straddling the widths a vectorized override chunks by.
const SLICE_LENS: [usize; 11] = [1, 2, 3, 4, 7, 8, 9, 16, 31, 32, 65];

/// Check the [`HasSubfield`] contract between `EF` and its subfield `S`.
///
/// Every element of `S` is checked when `S` is small, and random elements otherwise.
///
/// The elements known to lie outside the subfield are its elements translated by
/// `EF::GENERATOR`. A generator of the whole multiplicative group lies in no proper subfield.
/// A subfield is closed under addition, so none of those translates lies in it either.
///
/// The other elements of `EF` checked are its constants, its first interpolation nodes, and
/// random elements. The nodes are small bit patterns or small integers. Those sit just above a
/// subfield whose elements are the smallest patterns or integers, so they catch a membership test
/// that skips the lowest bits or digits above it. Nothing here targets a membership test that
/// skips high bits or digits: an implementation's own tests should set each of those in turn.
pub fn test_has_subfield<EF, S>()
where
    EF: HasSubfield<S>,
    S: Field,
    StandardUniform: Distribution<EF> + Distribution<S>,
{
    let mut rng = SmallRng::seed_from_u64(1);

    let (subfield, enumerated) = subfield_elements::<S>(&mut rng);
    let members: Vec<EF> = subfield.iter().map(|&s| EF::from(s)).collect();
    let outsiders: Vec<EF> = if S::order() < EF::order() {
        members.iter().map(|&m| m + EF::GENERATOR).collect()
    } else {
        Vec::new()
    };

    let num_nodes = usize::try_from(EF::order())
        .unwrap_or(usize::MAX)
        .min(NUM_SAMPLES);
    let samples: Vec<EF> = [EF::ZERO, EF::ONE, EF::GENERATOR]
        .into_iter()
        .chain((0..num_nodes).map(EF::interpolation_node))
        .chain((0..NUM_SAMPLES).map(|_| rng.random()))
        .collect();

    check_embedding::<EF, S>(&subfield);
    check_membership(&subfield, &members, &outsiders, &samples, enumerated);
    check_subfield_action(&subfield, &samples);
    check_all_in_subfield::<EF, S>(&members, &outsiders, &samples);
}

/// Elements of `S`, and whether they are all of them.
///
/// A small field is listed as zero followed by the powers of `S::GENERATOR`.
/// The walk asserts that the generator has full order, so that list has no repeats and
/// covers the field. A larger field gets its constants and random elements instead.
fn subfield_elements<S: Field>(rng: &mut SmallRng) -> (Vec<S>, bool)
where
    StandardUniform: Distribution<S>,
{
    if S::order() > BigUint::from(MAX_ENUMERATED_ORDER) {
        let elements = [S::ZERO, S::ONE, S::GENERATOR]
            .into_iter()
            .chain((0..NUM_SAMPLES).map(|_| rng.random()))
            .collect();
        return (elements, false);
    }

    let order = usize::try_from(S::order()).expect("the order was bounded above");
    let mut elements = Vec::with_capacity(order);
    elements.push(S::ZERO);
    let mut power = S::ONE;
    for exponent in 1..order {
        elements.push(power);
        power *= S::GENERATOR;
        assert_eq!(
            power == S::ONE,
            exponent == order - 1,
            "the subfield generator must have order |S| - 1, failing at exponent {exponent}"
        );
    }
    (elements, true)
}

/// The embedding must be a ring homomorphism on the given subfield elements.
fn check_embedding<EF: HasSubfield<S>, S: Field>(subfield: &[S]) {
    assert_eq!(EF::from(S::ZERO), EF::ZERO, "the embedding must fix zero");
    assert_eq!(EF::from(S::ONE), EF::ONE, "the embedding must fix one");

    for &a in subfield {
        for &b in subfield {
            assert_eq!(
                EF::from(a + b),
                EF::from(a) + EF::from(b),
                "the embedding must commute with {a} + {b}"
            );
            assert_eq!(
                EF::from(a * b),
                EF::from(a) * EF::from(b),
                "the embedding must commute with {a} * {b}"
            );
        }
    }
}

/// `as_subfield` must return `Some` exactly on the images of subfield elements.
fn check_membership<EF: HasSubfield<S>, S: Field>(
    subfield: &[S],
    members: &[EF],
    outsiders: &[EF],
    samples: &[EF],
    enumerated: bool,
) {
    // Narrowing undoes the embedding, which also makes the embedding injective.
    for (&s, &member) in subfield.iter().zip(members) {
        assert_eq!(
            member.as_subfield(),
            Some(s),
            "the image of {s} must narrow back to it"
        );
    }

    for &outsider in outsiders {
        assert_eq!(
            outsider.as_subfield(),
            None,
            "{outsider} lies outside the subfield"
        );
    }

    for &x in samples {
        match x.as_subfield() {
            Some(s) => assert_eq!(EF::from(s), x, "{x} narrowed to {s}, whose image differs"),
            // Only a full list of the subfield can rule out every preimage.
            None if enumerated => assert!(
                !members.contains(&x),
                "{x} is the image of a subfield element but did not narrow"
            ),
            None => {}
        }
    }
}

/// Each operation by a subfield element must agree with the same operation by its image.
fn check_subfield_action<EF: HasSubfield<S>, S: Field>(subfield: &[S], samples: &[EF]) {
    for &x in samples {
        for &s in subfield {
            let image = EF::from(s);
            assert_eq!(x + s, x + image, "the sum must agree at ({x}, {s})");
            assert_eq!(x - s, x - image, "the difference must agree at ({x}, {s})");
            assert_eq!(x * s, x * image, "the product must agree at ({x}, {s})");

            let mut assigned = [x; 3];
            assigned[0] += s;
            assigned[1] -= s;
            assigned[2] *= s;
            assert_eq!(
                assigned,
                [x + image, x - image, x * image],
                "the assigning operations must agree at ({x}, {s})"
            );
        }
    }
}

/// `all_in_subfield` must agree with narrowing each element of the slice in turn.
fn check_all_in_subfield<EF: HasSubfield<S>, S: Field>(
    members: &[EF],
    outsiders: &[EF],
    samples: &[EF],
) {
    let narrows = |v: &EF| v.as_subfield().is_some();
    let elementwise = |values: &[EF]| values.iter().all(narrows);

    assert!(
        EF::all_in_subfield(&[]),
        "an empty slice lies in the subfield"
    );
    assert!(
        EF::all_in_subfield(members),
        "every subfield element lies in the subfield"
    );

    // Every element that does not narrow, whether known to lie outside or merely sampled.
    let spoilers: Vec<EF> = outsiders
        .iter()
        .chain(samples.iter().filter(|v| !narrows(v)))
        .copied()
        .collect();

    for len in SLICE_LENS {
        let inside: Vec<EF> = members.iter().copied().cycle().take(len).collect();
        assert!(
            EF::all_in_subfield(&inside),
            "{len} subfield elements lie in the subfield"
        );

        // A single element that does not narrow must spoil the slice, whichever element it is
        // and wherever it sits. Each spoiler and each position is used at least once.
        if !spoilers.is_empty() {
            for i in 0..len.max(spoilers.len()) {
                let (position, spoiler) = (i % len, spoilers[i % spoilers.len()]);
                let mut spoiled = inside.clone();
                spoiled[position] = spoiler;
                assert!(
                    !EF::all_in_subfield(&spoiled),
                    "{spoiler} at position {position} of {len} went unnoticed"
                );
            }
        }

        let random: Vec<EF> = samples.iter().copied().cycle().take(len).collect();
        assert_eq!(
            EF::all_in_subfield(&random),
            elementwise(&random),
            "all_in_subfield must agree with narrowing each of {len} random elements"
        );
    }
}
