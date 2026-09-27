//! This file contains a collection of Enums which allow a nice command line interface.
//!
//! For each enum, we allow the user to specify the enum either using the whole string or any substring
//! which fully determines the choice. We additionally add a few extra aliases if other natural ones exist.
//!
//! For most of the enums, this allows the user to

use clap::ValueEnum;
use clap::builder::PossibleValue;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FieldOptions {
    BabyBear,
    KoalaBear,
    Mersenne31,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ProofOptions {
    Blake3Permutations,
    KeccakFPermutations,
    Poseidon1Permutations,
    Poseidon2Permutations,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DftOptions {
    None,
    Radix2DitParallel,
    RecursiveDft,
    SmallBatch,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MerkleHashOptions {
    KeccakF,
    Poseidon2,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PcsOptions {
    Fri,
    Stir,
}

/// How a prover example prints its measurements.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OutputFormat {
    /// Labelled lines for a person to read.
    Human,
    /// One JSON object per run, for collecting runs into a scoreboard.
    Json,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryHashOptions {
    Blake2sCompressions,
    Blake3Compressions,
    KeccakFPermutations,
    Sha256Compressions,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum RepresentationOptions {
    Auto,
    Subfield,
    PolyBasis,
    PolyBasisLate,
    Generic,
}

/// The byte hash a binary-field proof builds its Merkle trees and transcript from.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum BinaryCommitmentHashOptions {
    Keccak256,
    Blake3,
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
#[allow(clippy::option_if_let_else)]
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
        &[Self::BabyBear, Self::KoalaBear, Self::Mersenne31]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::BabyBear => get_aliases("baby-bear", 1, Some(vec![("babybear", 5), ("bb", 2)])),
            Self::KoalaBear => {
                get_aliases("koala-bear", 1, Some(vec![("koalabear", 6), ("kb", 2)]))
            }
            Self::Mersenne31 => {
                get_aliases("mersenne-31", 1, Some(vec![("mersenne31", 9), ("m31", 2)]))
            }
        })
    }
}

impl ValueEnum for ProofOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Blake3Permutations,
            Self::Poseidon1Permutations,
            Self::Poseidon2Permutations,
            Self::KeccakFPermutations,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Blake3Permutations => get_aliases(
                "blake-3-permutations",
                1,
                Some(vec![("blake3-permutations", 6), ("b3", 2)]),
            ),
            Self::KeccakFPermutations => get_aliases(
                "keccak-f-permutations",
                1,
                Some(vec![("keccakf-permutations", 7), ("kf", 2)]),
            ),
            Self::Poseidon1Permutations => get_aliases(
                "poseidon-1-permutations",
                10,
                Some(vec![("poseidon1-permutations", 9), ("p1", 2)]),
            ),
            Self::Poseidon2Permutations => get_aliases(
                "poseidon-2-permutations",
                10,
                Some(vec![("poseidon2-permutations", 9), ("p2", 2)]),
            ),
        })
    }
}

impl ValueEnum for DftOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Radix2DitParallel,
            Self::RecursiveDft,
            Self::SmallBatch,
            Self::None,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::RecursiveDft => get_aliases("recursive-dft", 2, Some(vec![("recursivedft", 10)])),
            Self::Radix2DitParallel => get_aliases(
                "radix-2-dit-parallel",
                2,
                Some(vec![("radix2ditparallel", 6), ("parallel", 1)]),
            ),
            Self::SmallBatch => get_aliases(
                "small-batch-dft",
                1,
                Some(vec![("smallbatchdft", 6), ("sb", 2)]),
            ),
            Self::None => PossibleValue::new(""),
        })
    }
}

impl ValueEnum for MerkleHashOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Poseidon2, Self::KeccakF]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::KeccakF => get_aliases("keccak-f", 1, Some(vec![("keccakf", 7), ("kf", 2)])),
            Self::Poseidon2 => {
                get_aliases("poseidon-2", 1, Some(vec![("poseidon2", 9), ("p2", 2)]))
            }
        })
    }
}

impl ValueEnum for PcsOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Fri, Self::Stir]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Fri => get_aliases("fri", 1, None),
            Self::Stir => get_aliases("stir", 1, None),
        })
    }
}

impl ValueEnum for OutputFormat {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Human, Self::Json]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Human => get_aliases("human", 1, None),
            Self::Json => get_aliases("json", 1, None),
        })
    }
}

impl ValueEnum for BinaryHashOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Blake2sCompressions,
            Self::Blake3Compressions,
            Self::KeccakFPermutations,
            Self::Sha256Compressions,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Blake2sCompressions => get_aliases(
                "blake-2s-compressions",
                7,
                Some(vec![("blake2s-compressions", 6), ("b2s", 3)]),
            ),
            Self::Blake3Compressions => get_aliases(
                "blake-3-compressions",
                1,
                Some(vec![("blake3-compressions", 6), ("b3", 2)]),
            ),
            Self::KeccakFPermutations => get_aliases(
                "keccak-f-permutations",
                1,
                Some(vec![("keccakf-permutations", 7), ("kf", 2)]),
            ),
            Self::Sha256Compressions => get_aliases(
                "sha-256-compressions",
                1,
                Some(vec![("sha256-compressions", 4)]),
            ),
        })
    }
}

impl ValueEnum for BinaryCommitmentHashOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::Keccak256, Self::Blake3]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Keccak256 => get_aliases("keccak-256", 1, Some(vec![("keccak256", 7)])),
            Self::Blake3 => get_aliases("blake-3", 1, Some(vec![("blake3", 6), ("b3", 2)])),
        })
    }
}

impl ValueEnum for RepresentationOptions {
    fn value_variants<'a>() -> &'a [Self] {
        &[
            Self::Auto,
            Self::Subfield,
            Self::PolyBasis,
            Self::PolyBasisLate,
            Self::Generic,
        ]
    }

    fn to_possible_value(&self) -> Option<PossibleValue> {
        Some(match self {
            Self::Auto => get_aliases("auto", 1, None),
            Self::Subfield => get_aliases("subfield", 1, None),
            Self::PolyBasis => {
                get_aliases("poly-basis", 1, Some(vec![("polybasis", 4), ("pb", 2)]))
            }
            Self::PolyBasisLate => get_aliases(
                "poly-basis-late",
                11,
                Some(vec![("polybasislate", 10), ("pbl", 3)]),
            ),
            Self::Generic => get_aliases("generic", 1, None),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poly_basis_late_has_an_explicit_cli_spelling() {
        assert_eq!(
            RepresentationOptions::from_str("poly-basis-late", true),
            Ok(RepresentationOptions::PolyBasisLate)
        );
        assert_eq!(
            RepresentationOptions::from_str("poly-basis", true),
            Ok(RepresentationOptions::PolyBasis)
        );
        assert_eq!(
            RepresentationOptions::from_str("poly", true),
            Ok(RepresentationOptions::PolyBasis)
        );
        assert_eq!(
            RepresentationOptions::from_str("polybasis", true),
            Ok(RepresentationOptions::PolyBasis)
        );
        assert_eq!(
            RepresentationOptions::from_str("p", true),
            Ok(RepresentationOptions::PolyBasis)
        );
        assert_eq!(
            RepresentationOptions::from_str("pbl", true),
            Ok(RepresentationOptions::PolyBasisLate)
        );
        assert_eq!(
            RepresentationOptions::from_str("polybasislate", true),
            Ok(RepresentationOptions::PolyBasisLate)
        );
    }
}
