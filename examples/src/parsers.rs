//! This file contains a collection of simple Enums which allow a nice command line interface.
//!
//! For each enum, we allow the user to specify the enum either using the whole string or any substring
//! which fully determines the choice. We additionally add a few extra aliases if other natural ones exist.
//!
//! For most of the enums, this allows the user to

use clap::builder::PossibleValue;
use clap::ValueEnum;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FieldOptions {
    BabyBear,
    KoalaBear,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProofOptions {
    Blake3Permutations,
    KeccakFPermutations,
    Poseidon2Permutations,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DftOptions {
    Radix2DitParallel,
    RecursiveDft,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MerkleHashOptions {
    KeccakF,
    Poseidon2,
}

/// Produce a collection of PossibleValue's for an Enum variant.
///
/// We allow any prefix of the full name which uniquely determines the variant.
/// We additionally allow the user to specify a collection of aliases which are
/// not prefixes. For each alias, we also allow any unique prefix of that alias.
///
/// For example, for the `KoalaBear` variant of `FieldOptions`, running
/// `get_aliases("koala-bear", 1, vec![("koalabear", 6), ("kb", 2)])` produces the following set of
/// allowed strings:
///
/// `koala-bear, k, ko, koa, koal, koala, koala-, koala-b, koala-be, koala-bea, koalab, koalabe, koalabea, koalabear, kb`
fn get_aliases(
    base: &'static str,
    min_unique_base_prefix: usize,
    alias: Option<Vec<(&'static str, usize)>>,
) -> PossibleValue {
    match alias {
        None => PossibleValue::new(base)
            .aliases((min_unique_base_prefix..base.len()).map(|i| &base[..i])),
        Some(vec) => PossibleValue::new(base).aliases(
            (min_unique_base_prefix..base.len())
                .map(|i| &base[..i])
                .chain(vec.into_iter().flat_map(|(alias, min_unique)| {
                    (min_unique..alias.len() + 1).map(|i| &alias[..i])
                })),
        ),
    }
}

impl ValueEnum for FieldOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[FieldOptions::BabyBear, FieldOptions::KoalaBear]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            FieldOptions::BabyBear => {
                get_aliases("baby-bear", 1, Some(vec![("babybear", 5), ("bb", 2)]))
            }
            FieldOptions::KoalaBear => {
                get_aliases("koala-bear", 1, Some(vec![("koalabear", 6), ("kb", 2)]))
            }
        })
    }
}

impl ValueEnum for ProofOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            ProofOptions::Blake3Permutations,
            ProofOptions::Poseidon2Permutations,
            ProofOptions::KeccakFPermutations,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            ProofOptions::Blake3Permutations => get_aliases(
                "blake-3-permutations",
                1,
                Some(vec![("blake3-permutations", 6), ("b3", 2)]),
            ),
            ProofOptions::KeccakFPermutations => get_aliases(
                "keccak-f-permutations",
                1,
                Some(vec![("keccakf-permutations", 7), ("kf", 2)]),
            ),
            ProofOptions::Poseidon2Permutations => get_aliases(
                "poseidon-2-permutations",
                1,
                Some(vec![("poseidon2-permutations", 9), ("p2", 2)]),
            ),
        })
    }
}

impl ValueEnum for DftOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[DftOptions::Radix2DitParallel, DftOptions::RecursiveDft]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            DftOptions::RecursiveDft => {
                get_aliases("recursive-dft", 2, Some(vec![("recursivedft", 10)]))
            }
            DftOptions::Radix2DitParallel => get_aliases(
                "radix-2-dit-parallel",
                2,
                Some(vec![("radix2ditparallel", 6), ("parallel", 1)]),
            ),
        })
    }
}

impl ValueEnum for MerkleHashOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[MerkleHashOptions::Poseidon2, MerkleHashOptions::KeccakF]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            MerkleHashOptions::KeccakF => {
                get_aliases("keccak-f", 1, Some(vec![("keccakf", 7), ("kf", 2)]))
            }
            MerkleHashOptions::Poseidon2 => {
                get_aliases("poseidon-2", 1, Some(vec![("poseidon2", 9), ("p2", 2)]))
            }
        })
    }
}
