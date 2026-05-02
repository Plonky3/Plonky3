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
//! - What is the Rust type of those values?

use core::any::type_name;
use core::fmt::{Display, Formatter};

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
    Fixed(usize),
    /// A dynamically-known number of values.
    Dynamic,
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
    /// Rust type name of the carried values, captured at construction.
    type_name: &'static str,
    /// Number of carried values.
    length: Length,
}

impl Interaction {
    /// Build a new step that carries values of type `T`.
    ///
    /// # Arguments
    ///
    /// * `hierarchy` — position in the nesting hierarchy.
    /// * `kind` — semantic role.
    /// * `label` — compile-time identifier.
    /// * `length` — number of carried values.
    #[must_use]
    pub fn new<T: ?Sized>(hierarchy: Hierarchy, kind: Kind, label: Label, length: Length) -> Self {
        // Capture the Rust type name once at construction time.
        //
        // Later equality checks compare two records as plain values
        // without re-resolving the generic parameter.
        let type_name = type_name::<T>();
        Self {
            hierarchy,
            kind,
            label,
            type_name,
            length,
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
            && self.type_name == other.type_name
            && self.length == other.length
    }
}

impl Display for Interaction {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        if f.alternate() {
            // Alternate mode: hash-stable form.
            //
            // Length-prefix the label so adjacent labels do not collapse.
            // The type name is intentionally omitted here.
            write!(f, "{} {}", self.hierarchy, self.kind)?;
            write!(f, " {} {}", self.label.len(), self.label)?;
            write!(f, " {}", self.length)
        } else {
            // Default mode: human-readable form including the type name.
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
            Self::Dynamic => write!(f, "Dynamic"),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn alternate_display_omits_type_name_and_length_prefixes_label() {
        // Hash-stable form: no type name, label is length-prefixed.
        let i = Interaction::new::<Vec<f64>>(
            Hierarchy::Atomic,
            Kind::Message,
            "test-message",
            Length::Scalar,
        );
        assert_eq!(format!("{i:#}"), "Atomic Message 12 test-message Scalar");
    }

    #[test]
    fn default_display_includes_type_name() {
        // Human form: carried type appears at the end.
        let i =
            Interaction::new::<u64>(Hierarchy::Atomic, Kind::Challenge, "alpha", Length::Scalar);
        assert_eq!(format!("{i}"), "Atomic Challenge alpha Scalar u64");
    }

    #[test]
    fn closes_requires_every_field_to_match() {
        // Closure needs exact agreement on label, kind, type, and length.
        let begin = Interaction::new::<()>(Hierarchy::Begin, Kind::Protocol, "x", Length::None);
        let good_end = Interaction::new::<()>(Hierarchy::End, Kind::Protocol, "x", Length::None);
        assert!(good_end.closes(&begin));

        // Wrong label.
        let bad_label = Interaction::new::<()>(Hierarchy::End, Kind::Protocol, "y", Length::None);
        assert!(!bad_label.closes(&begin));

        // Wrong kind.
        let bad_kind = Interaction::new::<()>(Hierarchy::End, Kind::Message, "x", Length::None);
        assert!(!bad_kind.closes(&begin));

        // Wrong type.
        let bad_type = Interaction::new::<u32>(Hierarchy::End, Kind::Protocol, "x", Length::None);
        assert!(!bad_type.closes(&begin));

        // Wrong length.
        let bad_length =
            Interaction::new::<()>(Hierarchy::End, Kind::Protocol, "x", Length::Scalar);
        assert!(!bad_length.closes(&begin));
    }

    #[test]
    fn closes_rejects_two_begins_or_two_ends() {
        // Closure requires opposite hierarchy positions.
        let begin_a = Interaction::new::<()>(Hierarchy::Begin, Kind::Protocol, "x", Length::None);
        let begin_b = Interaction::new::<()>(Hierarchy::Begin, Kind::Protocol, "x", Length::None);

        let end_a = Interaction::new::<()>(Hierarchy::End, Kind::Protocol, "x", Length::None);
        let end_b = Interaction::new::<()>(Hierarchy::End, Kind::Protocol, "x", Length::None);

        assert!(!begin_a.closes(&begin_b));
        assert!(!end_a.closes(&end_b));
    }
}
