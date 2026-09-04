//! A single typed step of an interactive protocol.
//!
//! # Overview
//!
//! Every transcript is a list of typed steps.
//!
//! Each step answers four questions about itself:
//!
//! - Where does it sit in the nesting hierarchy?
//! - What is its semantic role?
//! - How many concrete values does it carry?
//! - What kind of value are those, in compiler-independent terms?

use core::any::type_name;
use core::fmt::{Display, Formatter};

use p3_field::{BasedVectorSpace, PrimeField64};

/// Position of a step in the hierarchical structure of a transcript.
///
/// # Example
///
/// ```text
///     Begin   "outer"
///       Atomic Message    "commitment"
///       Atomic Challenge  "alpha"
///     End     "outer"
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Hierarchy {
    /// A leaf step. Carries one or more values.
    Atomic,
    /// Opens a nested sub-protocol.
    Begin,
    /// Closes the most recently opened sub-protocol.
    End,
}

/// Semantic role of a transcript step.
///
/// # Overview
///
/// The recording-side validator uses this to enforce that nested steps
/// are compatible with their surrounding sub-protocol.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Kind {
    /// Container that may hold steps of any kind.
    Protocol,
    /// Value the prover and verifier already share before the run.
    Public,
    /// Value the prover sends to the verifier; absorbed.
    Message,
    /// Value the prover sends to the verifier; not absorbed.
    Hint,
    /// Value the verifier derives from the sponge.
    Challenge,
    /// Proof-of-work step. The prover grinds, the verifier checks.
    Pow,
    /// Zero-knowledge salt absorbed before any sample.
    Salt,
}

/// How many values a single step carries.
///
/// Two patterns that absorb different counts of the same type cannot be
/// confused at the type level if they declare distinct length variants.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum Length {
    /// No carried value.
    None,
    /// Exactly one value.
    Scalar,
    /// A statically-known number of values.
    ///
    /// A `Kind::Pow` step reuses this variant to carry its difficulty in bits.
    Fixed(usize),
    /// At most `max` values.
    /// The actual count travels on the wire as a big-endian length prefix.
    ///
    /// The prefix width is the minimum number of bytes that can hold every value in `0..=max`.
    ///
    /// This is distinct from a fixed length because the bound is part of the pattern hash.
    ///
    /// Two protocols differing only in their capacity get distinct seeds and cannot be confused.
    Bounded(usize),
}

/// Compiler-independent identity of the values a step carries.
///
/// [`type_name`] cannot play this role.
/// Its output may change between compiler versions.
/// The pattern fingerprint is a cross-run protocol commitment, so it must not depend on that.
///
/// What the fingerprint has to pin down is the shape of a step.
/// Element width is shape.
///
/// Two protocols over two different 31-bit primes must not share a seed.
/// Neither must a step that is an `F` in one protocol and a degree-4 element in the other.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub enum TypeTag {
    /// Structural marker; carries no value.
    Marker,
    /// Raw bytes: hints, salts, digest-shaped messages.
    Bytes,
    /// `degree` coefficients over the prime field of order `modulus`.
    ///
    /// A base-field step is the `degree == 1` case.
    Algebra {
        /// Order of the prime field the coefficients live in.
        modulus: u64,
        /// Number of base-field coefficients per value.
        degree: usize,
    },
}

/// Compile-time identifier of a step.
///
/// Lives in the binary as a `&'static str`; never allocated at runtime.
pub type Label = &'static str;

/// A single typed step inside an interactive protocol.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Interaction {
    /// Position in the nesting hierarchy.
    hierarchy: Hierarchy,
    /// Semantic role of the step.
    kind: Kind,
    /// Compile-time identifier of the step.
    label: Label,
    /// Compiler-independent identity of the carried values.
    type_tag: TypeTag,
    /// Rust type name of the carried values, captured at construction.
    ///
    /// Local diagnostics only; excluded from the pattern fingerprint.
    type_name: &'static str,
    /// Number of carried values.
    length: Length,
}

impl Interaction {
    /// Build a step carrying `A` values, where `A` is an algebra over the prime field `F`.
    ///
    /// A base-field step is the `A = F` case, which has degree 1.
    ///
    /// # Arguments
    ///
    /// * `hierarchy` — position in the nesting hierarchy.
    /// * `kind` — semantic role.
    /// * `label` — compile-time identifier.
    /// * `length` — number of carried values.
    #[must_use]
    pub fn algebra<F, A>(hierarchy: Hierarchy, kind: Kind, label: Label, length: Length) -> Self
    where
        F: PrimeField64,
        A: BasedVectorSpace<F>,
    {
        Self {
            hierarchy,
            kind,
            label,
            // Modulus plus degree pins the element width across compilers and crates.
            type_tag: TypeTag::Algebra {
                modulus: F::ORDER_U64,
                degree: A::DIMENSION,
            },
            type_name: type_name::<A>(),
            length,
        }
    }

    /// Build a step carrying raw bytes: a hint, a salt, or a digest-shaped message.
    #[must_use]
    pub fn bytes(hierarchy: Hierarchy, kind: Kind, label: Label, length: Length) -> Self {
        Self {
            hierarchy,
            kind,
            label,
            type_tag: TypeTag::Bytes,
            type_name: type_name::<u8>(),
            length,
        }
    }

    /// Build a structural marker: the opener or closer of a sub-protocol.
    ///
    /// `T` names the sub-protocol at the type level and is compared locally,
    /// but does not reach the pattern fingerprint.
    #[must_use]
    pub fn marker<T: ?Sized>(hierarchy: Hierarchy, kind: Kind, label: Label) -> Self {
        Self {
            hierarchy,
            kind,
            label,
            type_tag: TypeTag::Marker,
            type_name: type_name::<T>(),
            length: Length::None,
        }
    }

    /// Hierarchy position of the step.
    #[must_use]
    pub const fn hierarchy(&self) -> Hierarchy {
        self.hierarchy
    }

    /// Semantic role of the step.
    #[must_use]
    pub const fn kind(&self) -> Kind {
        self.kind
    }

    /// Compile-time identifier of the step.
    #[must_use]
    pub const fn label(&self) -> Label {
        self.label
    }

    /// Compiler-independent identity of the carried values.
    #[must_use]
    pub const fn type_tag(&self) -> TypeTag {
        self.type_tag
    }

    /// Rust type captured at construction.
    #[must_use]
    pub const fn type_name(&self) -> &'static str {
        self.type_name
    }

    /// Length signature of the step.
    #[must_use]
    pub const fn length(&self) -> Length {
        self.length
    }

    /// Decide whether `self` (an `End` marker) closes the sub-protocol
    /// opened by `other` (a `Begin` marker).
    ///
    /// # Algorithm
    ///
    /// 1. Both records must be markers of opposite hierarchy.
    /// 2. Every other field must agree exactly: kind, label, type, length.
    #[must_use]
    pub fn closes(&self, other: &Self) -> bool {
        // Position check: only an End can close a Begin.
        self.hierarchy == Hierarchy::End
            && other.hierarchy == Hierarchy::Begin
            // Identity check: every non-positional field must match.
            && self.kind == other.kind
            && self.label == other.label
            && self.type_tag == other.type_tag
            && self.type_name == other.type_name
            && self.length == other.length
    }
}

impl Display for Interaction {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            // Alternate mode: hash-stable form feeding the pattern fingerprint.
            //
            // Type identity enters through `type_tag`, never through `type_name`.
            // Length-prefix the label so adjacent labels do not collapse.
            write!(f, "{} {}", self.hierarchy, self.kind)?;
            write!(f, " {} {}", self.label.len(), self.label)?;
            write!(f, " {} {}", self.length, self.type_tag)
        } else {
            // Default mode: human-readable form including the Rust type name.
            write!(
                f,
                "{} {} {} {} {}",
                self.hierarchy, self.kind, self.label, self.length, self.type_name,
            )
        }
    }
}

impl Display for Hierarchy {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Atomic => write!(f, "Atomic"),
            Self::Begin => write!(f, "Begin"),
            Self::End => write!(f, "End"),
        }
    }
}

impl Display for Kind {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Protocol => write!(f, "Protocol"),
            Self::Public => write!(f, "Public"),
            Self::Message => write!(f, "Message"),
            Self::Hint => write!(f, "Hint"),
            Self::Challenge => write!(f, "Challenge"),
            Self::Pow => write!(f, "Pow"),
            Self::Salt => write!(f, "Salt"),
        }
    }
}

impl Display for Length {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Scalar => write!(f, "Scalar"),
            Self::Fixed(n) => write!(f, "Fixed({n})"),
            Self::Bounded(n) => write!(f, "Bounded({n})"),
        }
    }
}

impl Display for TypeTag {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Marker => write!(f, "Marker"),
            Self::Bytes => write!(f, "Bytes"),
            Self::Algebra { modulus, degree } => write!(f, "Algebra({modulus}^{degree})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_goldilocks::Goldilocks;
    use p3_koala_bear::KoalaBear;

    use super::*;

    /// Degree-4 binomial extension used to exercise the degree field of the tag.
    type EF4 = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn alternate_display_carries_the_tag_and_length_prefixes_the_label() {
        // Hash-stable form: modulus and degree instead of a Rust type name.
        let i = Interaction::algebra::<BabyBear, BabyBear>(
            Hierarchy::Atomic,
            Kind::Message,
            "test-message",
            Length::Scalar,
        );
        assert_eq!(
            format!("{i:#}"),
            "Atomic Message 12 test-message Scalar Algebra(2013265921^1)"
        );
    }

    #[test]
    fn default_display_includes_type_name() {
        // Human form: carried Rust type appears at the end.
        //
        // The exact spelling is a compiler detail, which is why it is a
        // diagnostic aid here and never reaches the fingerprint.
        let i = Interaction::algebra::<BabyBear, BabyBear>(
            Hierarchy::Atomic,
            Kind::Challenge,
            "alpha",
            Length::Scalar,
        );
        let rendered = format!("{i}");
        assert!(rendered.starts_with("Atomic Challenge alpha Scalar "));
        assert!(rendered.contains("BabyBear"));
    }

    #[test]
    fn tag_separates_two_primes_of_the_same_width() {
        // Invariant: the fingerprint sees the modulus, not the bit width.
        //
        // BabyBear and KoalaBear are both 31-bit, so only the modulus tells them apart.
        let bb = Interaction::algebra::<BabyBear, BabyBear>(
            Hierarchy::Atomic,
            Kind::Message,
            "x",
            Length::Scalar,
        );
        let kb = Interaction::algebra::<KoalaBear, KoalaBear>(
            Hierarchy::Atomic,
            Kind::Message,
            "x",
            Length::Scalar,
        );
        assert_ne!(format!("{bb:#}"), format!("{kb:#}"));
    }

    #[test]
    fn tag_separates_a_base_element_from_an_extension_element() {
        // Invariant: element width is part of the shape.
        //
        // A base scalar and a degree-4 element occupy different wire widths.
        let base = Interaction::algebra::<BabyBear, BabyBear>(
            Hierarchy::Atomic,
            Kind::Message,
            "x",
            Length::Scalar,
        );
        let ext = Interaction::algebra::<BabyBear, EF4>(
            Hierarchy::Atomic,
            Kind::Message,
            "x",
            Length::Scalar,
        );
        assert_ne!(format!("{base:#}"), format!("{ext:#}"));
        assert_eq!(
            ext.type_tag(),
            TypeTag::Algebra {
                modulus: BabyBear::ORDER_U64,
                degree: 4,
            }
        );
    }

    #[test]
    fn tag_separates_bytes_from_field_elements() {
        // A 4-byte hint and a 4-byte BabyBear scalar are different shapes.
        let raw = Interaction::bytes(Hierarchy::Atomic, Kind::Hint, "x", Length::Fixed(4));
        let scalar = Interaction::algebra::<Goldilocks, Goldilocks>(
            Hierarchy::Atomic,
            Kind::Hint,
            "x",
            Length::Fixed(4),
        );
        assert_eq!(raw.type_tag(), TypeTag::Bytes);
        assert_ne!(format!("{raw:#}"), format!("{scalar:#}"));
    }

    #[test]
    fn bounded_length_renders_with_max() {
        // Invariant: the bound appears in both display modes.
        //
        // The alternate form feeds the pattern fingerprint, so the cap must be visible there too.

        // Fixture state: an atomic hint step with a cap of 64.
        let i = Interaction::bytes(
            Hierarchy::Atomic,
            Kind::Hint,
            "auth-path",
            Length::Bounded(64),
        );

        // Default form carries the type name and prints the bound at the end.
        assert_eq!(format!("{i}"), "Atomic Hint auth-path Bounded(64) u8");

        // Alternate form swaps the type name for the tag.
        assert_eq!(
            format!("{i:#}"),
            "Atomic Hint 9 auth-path Bounded(64) Bytes"
        );
    }

    #[test]
    fn closes_requires_every_field_to_match() {
        // Closure needs exact agreement on label, kind, type, and length.
        let begin = Interaction::marker::<()>(Hierarchy::Begin, Kind::Protocol, "x");
        let good_end = Interaction::marker::<()>(Hierarchy::End, Kind::Protocol, "x");
        assert!(good_end.closes(&begin));

        // Wrong label.
        let bad_label = Interaction::marker::<()>(Hierarchy::End, Kind::Protocol, "y");
        assert!(!bad_label.closes(&begin));

        // Wrong kind.
        let bad_kind = Interaction::marker::<()>(Hierarchy::End, Kind::Message, "x");
        assert!(!bad_kind.closes(&begin));

        // Wrong type.
        let bad_type = Interaction::marker::<u32>(Hierarchy::End, Kind::Protocol, "x");
        assert!(!bad_type.closes(&begin));

        // Wrong shape: an atomic byte step is not a closer at all.
        let bad_shape = Interaction::bytes(Hierarchy::End, Kind::Protocol, "x", Length::None);
        assert!(!bad_shape.closes(&begin));
    }

    #[test]
    fn closes_rejects_two_begins_or_two_ends() {
        // Closure requires opposite hierarchy positions.
        let begin_a = Interaction::marker::<()>(Hierarchy::Begin, Kind::Protocol, "x");
        let begin_b = Interaction::marker::<()>(Hierarchy::Begin, Kind::Protocol, "x");

        let end_a = Interaction::marker::<()>(Hierarchy::End, Kind::Protocol, "x");
        let end_b = Interaction::marker::<()>(Hierarchy::End, Kind::Protocol, "x");

        assert!(!begin_a.closes(&begin_b));
        assert!(!end_a.closes(&end_b));
    }
}
