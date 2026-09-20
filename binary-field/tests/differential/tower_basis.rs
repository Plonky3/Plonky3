//! Vectors this crate publishes for its tower basis, and the check that it still meets them.
//!
//! Nobody else defines this basis for us, so these are a reference rather than a consumption.
//!
//! The fixture states the recurrence and the bit layout in full.
//!
//! An outside implementation can meet it without reading any code here.

use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, TowerLevel,
};
use p3_field::PrimeCharacteristicRing;

const TOWER: &str = include_str!("vectors/tower-basis.txt");

/// Applies one record to whichever level the fixture is currently reading.
fn check<F>(kind: &str, fields: &[&str], line: &str) -> bool
where
    F: TowerLevel,
    F::Repr: TryFrom<u128> + Copy,
{
    let read = |text: &str| {
        let raw = u128::from_str_radix(text, 16).expect("hexadecimal element");
        let repr = F::Repr::try_from(raw).ok().expect("element fits the level");
        F::from_repr(repr)
    };
    match kind {
        "mul" => {
            assert_eq!(read(fields[0]) * read(fields[1]), read(fields[2]), "{line}");
            true
        }
        "inv" => {
            let a = read(fields[0]);
            assert_eq!(a.inverse(), read(fields[1]), "{line}");
            assert_eq!(a * read(fields[1]), F::ONE, "{line}");
            true
        }
        "cantor" => {
            let index: usize = fields[0].parse().expect("basis index");
            assert_eq!(F::cantor_basis(index), read(fields[1]), "{line}");
            true
        }
        _ => false,
    }
}

#[test]
fn the_published_tower_vectors_still_hold() {
    let mut level = 0usize;
    let mut records = 0usize;
    for line in TOWER.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts[0] == "level" {
            level = parts[1].parse().expect("level width");
            continue;
        }
        let (kind, rest) = (parts[0], &parts[1..]);
        let known = match level {
            8 => check::<BinaryField8>(kind, rest, line),
            16 => check::<BinaryField16>(kind, rest, line),
            32 => check::<BinaryField32>(kind, rest, line),
            64 => check::<BinaryField64>(kind, rest, line),
            128 => check::<BinaryField128>(kind, rest, line),
            other => panic!("no level of {other} bits"),
        };
        assert!(known, "unknown record kind in {line}");
        records += 1;
    }
    // Eight products and four inverses at each of five levels, plus the whole Cantor basis.
    assert_eq!(records, 5 * 12 + 128);
}

#[test]
fn a_narrower_level_reads_the_same_cantor_vectors_truncated() {
    // The fixture lists the basis once, so this is what lets a narrower level consume it.
    for i in 0..64 {
        let wide = BinaryField128::cantor_basis(i).to_repr();
        assert_eq!(BinaryField64::cantor_basis(i).to_repr() as u128, wide);
    }
    for i in 0..32 {
        let wide = BinaryField128::cantor_basis(i).to_repr();
        assert_eq!(BinaryField32::cantor_basis(i).to_repr() as u128, wide);
    }
}

#[test]
fn the_fixture_would_notice_the_defining_polynomial_changing() {
    // The recurrence the header states is what makes the products reproducible elsewhere.
    // Squaring the level's own generator is where a different polynomial first shows.
    let top = BinaryField16::from_repr(0x0100);
    let below = BinaryField16::from_repr(0x0010);
    assert_eq!(top * top, below * top + BinaryField16::ONE);

    // The same identity one level up, so the recursion and not one hardcoded level is pinned.
    let top = BinaryField32::from_repr(0x0001_0000);
    let below = BinaryField32::from_repr(0x0000_0100);
    assert_eq!(top * top, below * top + BinaryField32::ONE);
}
