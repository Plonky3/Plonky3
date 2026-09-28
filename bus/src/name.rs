//! Typed channel names for binary-native bus declarations.

use core::fmt;

use thiserror::Error;

/// A channel name, checked against the alphabet the bus separator binds.
///
/// A name decides which multiset a tuple joins, so it belongs to the statement rather than to the prose around it.
///
/// A machine names each channel once, in a const item the compiler checks, and every table imports that constant.
///
/// ```
/// use p3_bus::BusName;
///
/// const MEMORY: BusName<'static> = BusName::new("memory");
///
/// assert_eq!(MEMORY.as_str(), "memory");
/// ```
///
/// # Alphabet
///
/// A name is one to 64 bytes: an ASCII letter, then ASCII letters, digits, `_`, `.` or `-`.
///
/// The separator length-prefixes every name, so any bytes at all would encode injectively.
///
/// What arbitrary bytes would not do is print injectively.
///
/// Two names that render the same way are still two multisets, and a report shows them under one heading.
///
/// ASCII rules that pair out.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BusName<'a>(&'a str);

impl<'a> BusName<'a> {
    /// Longest name the alphabet admits.
    pub const MAX_LEN: usize = 64;

    /// Check one name against the alphabet.
    ///
    /// # Errors
    ///
    /// Returns an error naming the byte that left the alphabet, or the length that overran it.
    pub const fn try_new(name: &'a str) -> Result<Self, BusNameError> {
        let bytes = name.as_bytes();
        if bytes.is_empty() {
            return Err(BusNameError::Empty);
        }
        if bytes.len() > Self::MAX_LEN {
            return Err(BusNameError::TooLong {
                len: bytes.len(),
                max: Self::MAX_LEN,
            });
        }

        // A leading letter keeps a name from reading as a number or an option flag.
        if !bytes[0].is_ascii_alphabetic() {
            return Err(BusNameError::Leading { byte: bytes[0] });
        }

        // A const context rules out iterators, so the tail is walked by index.
        let mut index = 1;
        while index < bytes.len() {
            let byte = bytes[index];
            if !byte.is_ascii_alphanumeric() && byte != b'_' && byte != b'.' && byte != b'-' {
                return Err(BusNameError::Byte { index, byte });
            }
            index += 1;
        }
        Ok(Self(name))
    }

    /// Name one channel from a literal.
    ///
    /// # Panics
    ///
    /// Panics on a name outside the alphabet, which is a compile error in the const item a machine declares it from.
    #[must_use]
    pub const fn new(name: &'a str) -> Self {
        match Self::try_new(name) {
            Ok(name) => name,
            // A const panic cannot format, so the message states the whole rule.
            Err(_) => panic!(
                "a bus name is 1 to 64 bytes: an ASCII letter, then ASCII letters, digits, '_', '.' or '-'"
            ),
        }
    }

    /// Bytes this name contributes to the domain separator.
    #[must_use]
    pub const fn as_str(self) -> &'a str {
        self.0
    }
}

impl fmt::Debug for BusName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A name is a string in every report, so it reads as one inside a larger value.
        fmt::Debug::fmt(self.0, f)
    }
}

impl fmt::Display for BusName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl PartialEq<str> for BusName<'_> {
    fn eq(&self, other: &str) -> bool {
        self.0 == other
    }
}

impl PartialEq<BusName<'_>> for str {
    fn eq(&self, other: &BusName<'_>) -> bool {
        self == other.0
    }
}

/// Reason one channel name left the alphabet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum BusNameError {
    /// The name holds no bytes.
    #[error("a bus name is not empty")]
    Empty,
    /// The name is longer than the alphabet admits.
    #[error("a bus name holds at most {max} bytes, not {len}")]
    TooLong {
        /// Length the rejected name carried.
        len: usize,
        /// Longest admitted length.
        max: usize,
    },
    /// The first byte is not an ASCII letter.
    #[error("a bus name starts with an ASCII letter, not byte {byte:#04x}")]
    Leading {
        /// First byte of the rejected name.
        byte: u8,
    },
    /// A later byte is outside the alphabet.
    #[error(
        "a bus name continues with ASCII letters, digits, '_', '.' or '-', not byte {byte:#04x} at {index}"
    )]
    Byte {
        /// Position of the rejected byte.
        index: usize,
        /// Rejected byte.
        byte: u8,
    },
}

#[cfg(test)]
mod tests {
    use alloc::format;
    use alloc::string::String;

    use super::*;

    /// A machine names its channels once, in a const item the compiler checks.
    const MEMORY: BusName<'static> = BusName::new("memory");

    #[test]
    fn a_const_name_keeps_its_bytes() {
        assert_eq!(MEMORY.as_str(), "memory");
        assert_eq!(format!("{MEMORY}"), "memory");
        assert_eq!(format!("{MEMORY:?}"), "\"memory\"");
        assert!(MEMORY == *"memory");
    }

    #[test]
    fn the_alphabet_admits_every_shape_a_machine_needs() {
        // A leading letter, then every admitted class of continuation byte.
        for name in ["a", "memory", "state.init", "read_write", "bus-0", "Op2"] {
            assert_eq!(BusName::try_new(name).unwrap().as_str(), name);
        }

        // The bound is inclusive.
        let longest = String::from_iter(core::iter::repeat_n('a', BusName::MAX_LEN));
        assert!(BusName::try_new(&longest).is_ok());
    }

    #[test]
    fn the_alphabet_names_the_byte_it_rejects() {
        assert_eq!(BusName::try_new(""), Err(BusNameError::Empty));

        let overlong = String::from_iter(core::iter::repeat_n('a', BusName::MAX_LEN + 1));
        assert_eq!(
            BusName::try_new(&overlong),
            Err(BusNameError::TooLong {
                len: BusName::MAX_LEN + 1,
                max: BusName::MAX_LEN,
            })
        );

        // A digit, a separator and a space are all rejected in leading position.
        for (name, byte) in [("0bus", b'0'), ("_bus", b'_'), (" bus", b' ')] {
            assert_eq!(BusName::try_new(name), Err(BusNameError::Leading { byte }));
        }

        // A rejected continuation byte reports where it sits.
        assert_eq!(
            BusName::try_new("bus name"),
            Err(BusNameError::Byte {
                index: 3,
                byte: b' ',
            })
        );
        assert_eq!(
            BusName::try_new("bus\0name"),
            Err(BusNameError::Byte { index: 3, byte: 0 })
        );
    }

    #[test]
    fn two_names_that_print_alike_cannot_both_exist() {
        // U+03BF renders as the Latin letter it is not, so two multisets would share one heading.
        let greek = "mem\u{03bf}ry";
        assert_ne!(greek, "memory");
        assert!(BusName::try_new(greek).is_err());

        // Zero-width and non-breaking bytes are the same hazard without a visible carrier.
        for name in ["mem\u{200b}ory", "mem\u{00a0}ory", "memory\u{200e}"] {
            assert!(BusName::try_new(name).is_err());
        }
    }

    #[test]
    fn a_name_orders_and_hashes_by_its_bytes() {
        // Plan order is lexicographic over names, so the type has to agree with `str`.
        let mut names = [
            BusName::new("memory"),
            BusName::new("bytecode"),
            BusName::new("state"),
        ];
        names.sort_unstable();
        assert_eq!(names.map(BusName::as_str), ["bytecode", "memory", "state"]);
    }
}
