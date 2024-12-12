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
pub enum ProofObjectives {
    Blake3Hashes,
    KeccakHashes,
    Poseidon2Hashes,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DFTOptions {
    Radix2DitParallel,
    RecursiveDft,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MerkleHashOptions {
    Keccak,
    Poseidon2,
}

/// Produce a collection of PossibleValue's for an Enum variant.
///
/// We allow any prefix of the full name which uniquely determines the variant.
/// We additionally allow the user to specify a collection of aliases which are
/// not prefixes. For each alias, we also allow any unique prefix of that alias.
///
/// For example, for the `KoalaBear` variant of `FieldOptions`, running
/// `get_aliases("koala-bear", 1, vec!["kb", 2])` produces the following set of
/// allowed strings:
///
/// ```
/// k, ko, koa, koal, koala, koala-, koala-b, koala-be, koala-bea, koala-bear, kb
/// ```
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
            FieldOptions::BabyBear => get_aliases("baby-bear", 1, Some(vec![("bb", 2)])),
            FieldOptions::KoalaBear => get_aliases("koala-bear", 1, Some(vec![("kb", 2)])),
        })
    }
}

impl ValueEnum for ProofObjectives {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            ProofObjectives::Blake3Hashes,
            ProofObjectives::Poseidon2Hashes,
            ProofObjectives::KeccakHashes,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            ProofObjectives::Blake3Hashes => {
                get_aliases("blake-3-hashes", 1, Some(vec![("b3", 2)]))
            }
            ProofObjectives::Poseidon2Hashes => {
                get_aliases("poseidon-2-hashes", 1, Some(vec![("p2", 2)]))
            }
            ProofObjectives::KeccakHashes => get_aliases("keccak-hashes", 1, None),
        })
    }
}

impl ValueEnum for DFTOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[DFTOptions::Radix2DitParallel, DFTOptions::RecursiveDft]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            DFTOptions::RecursiveDft => get_aliases("recursive-dft", 2, None),
            DFTOptions::Radix2DitParallel => {
                get_aliases("radix-2-dit-parallel", 2, Some(vec![("parallel", 1)]))
            }
        })
    }
}

impl ValueEnum for MerkleHashOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[MerkleHashOptions::Poseidon2, MerkleHashOptions::Keccak]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            MerkleHashOptions::Poseidon2 => get_aliases("poseidon-2", 1, Some(vec![("p2", 2)])),
            MerkleHashOptions::Keccak => get_aliases("keccak", 1, None),
        })
    }
}
