//! Vectors this crate publishes for its additive transform, and the check that it still meets them.
//!
//! The fixture restates the novel basis from its product definition.
//!
//! An outside implementation can reproduce the values without the identities used internally.

use p3_binary_dft::{AdditiveNtt, ButterflyField, LchNtt, NaiveAdditiveNtt};
use p3_binary_field::{BinaryField32, BinaryField128, TowerLevel};
use p3_matrix::dense::RowMajorMatrix;

const VECTORS: &str = include_str!("vectors/additive-transform.txt");

/// One published case: a coset, the input column and the expected output column.
struct Case {
    bits: usize,
    log_n: usize,
    shift: u128,
    coefficients: Vec<u128>,
    values: Vec<u128>,
}

/// Splits the fixture into its cases, in file order.
fn cases() -> Vec<Case> {
    let mut cases: Vec<Case> = Vec::new();
    for line in VECTORS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        let hex = |t: &str| u128::from_str_radix(t, 16).expect("hexadecimal element");
        match parts[0] {
            "transform" => cases.push(Case {
                bits: parts[1].parse().expect("level width"),
                log_n: parts[2].parse().expect("log length"),
                shift: hex(parts[3]),
                coefficients: Vec::new(),
                values: Vec::new(),
            }),
            "coeff" => {
                let case = cases.last_mut().expect("a case was opened");
                assert_eq!(case.coefficients.len(), parts[1].parse().unwrap());
                case.coefficients.push(hex(parts[2]));
            }
            "value" => {
                let case = cases.last_mut().expect("a case was opened");
                assert_eq!(case.values.len(), parts[1].parse().unwrap());
                case.values.push(hex(parts[2]));
            }
            other => panic!("unknown record {other}"),
        }
    }
    cases
}

/// Runs one case through both transforms of this crate.
fn run<F>(case: &Case)
where
    F: TowerLevel + ButterflyField,
    F::Repr: TryFrom<u128> + Copy,
{
    let read = |raw: u128| F::from_repr(F::Repr::try_from(raw).ok().expect("element fits"));
    let n = 1 << case.log_n;
    assert_eq!(case.coefficients.len(), n);
    assert_eq!(case.values.len(), n);

    let input: Vec<F> = case.coefficients.iter().copied().map(read).collect();
    let want: Vec<F> = case.values.iter().copied().map(read).collect();
    let shift = read(case.shift);

    let lch = LchNtt::<F>::default();
    let got = lch.shifted_ntt_batch(RowMajorMatrix::new_col(input.clone()), shift);
    assert_eq!(got.values, want, "level {} length {}", case.bits, n);

    // The reference transform must meet the same published values, or one of the two is drifting.
    let naive = NaiveAdditiveNtt::<F>::default();
    let got = naive.shifted_ntt_batch(RowMajorMatrix::new_col(input.clone()), shift);
    assert_eq!(got.values, want, "reference, level {}", case.bits);

    // The inverse returns the published coefficients.
    // That is what makes the vector usable in either direction.
    let back = lch.shifted_intt_batch(got, shift);
    assert_eq!(back.values, input);
}

#[test]
fn the_published_transform_vectors_still_hold() {
    let cases = cases();
    assert_eq!(cases.len(), 4, "the fixture lost a case");
    for case in &cases {
        match case.bits {
            32 => run::<BinaryField32>(case),
            128 => run::<BinaryField128>(case),
            other => panic!("no level of {other} bits"),
        }
    }
}

#[test]
fn a_transform_over_the_wrong_coset_misses_the_published_values() {
    // The coset shift enters every output, so a shifted case must not also hold at the origin.
    // Without this the fixture would not pin the shift at all.
    let case = cases()
        .into_iter()
        .find(|c| c.bits == 128 && c.shift != 0)
        .expect("a shifted case");
    let read = |raw: u128| BinaryField128::from_repr(raw);
    let input: Vec<BinaryField128> = case.coefficients.iter().copied().map(read).collect();
    let want: Vec<BinaryField128> = case.values.iter().copied().map(read).collect();

    let got = LchNtt::<BinaryField128>::default().ntt_batch(RowMajorMatrix::new_col(input));
    assert_ne!(got.values, want);
}
