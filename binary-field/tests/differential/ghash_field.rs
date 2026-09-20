//! The polynomial-basis field against another implementation of the same field.
//!
//! Comparing this crate only against itself cannot catch a uniformly wrong convention.
//!
//! Only one external vector covered this field before, so this adds a second source.

use p3_binary_field::Ghash128;

const FLOCK_PRODUCTS: &str = include_str!("vectors/flock-ghash-products.txt");

/// One element of a line of the fixture, read as a big-endian 128-bit hex integer.
fn element(text: &str) -> Ghash128 {
    let raw = u128::from_str_radix(text, 16).expect("hexadecimal element");
    Ghash128::from_le_bytes(raw.to_le_bytes())
}

#[test]
fn flock_products_agree_element_for_element() {
    // Flock stores the coefficient of x^i at bit i, the same way this crate does.
    // So the only conversion is the one the fixture already did, joining its two words.
    let mut checked = 0;
    for line in FLOCK_PRODUCTS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let a = element(fields.next().expect("left operand"));
        let b = element(fields.next().expect("right operand"));
        let product = element(fields.next().expect("product"));
        assert!(fields.next().is_none(), "trailing field in {line}");

        assert_eq!(a * b, product, "{line}");
        checked += 1;
    }
    assert_eq!(checked, 128, "the fixture lost records");
}

#[test]
fn the_flock_fixture_would_notice_a_wrong_modulus() {
    // The reduction tail is what a transcription error would most plausibly disturb.
    // Reducing by the neighbouring pentanomial instead must break the fixture immediately.
    let wrong = |a: u128, b: u128| {
        let (mut acc, mut b) = (0u128, b);
        let mut a = a;
        while b != 0 {
            if b & 1 == 1 {
                acc ^= a;
            }
            b >>= 1;
            let top = a >> 127;
            a <<= 1;
            if top == 1 {
                // 0x87 is the real tail; 0x86 differs in one bit.
                a ^= 0x86;
            }
        }
        acc
    };
    let line = FLOCK_PRODUCTS
        .lines()
        .find(|l| l.starts_with("ece4"))
        .expect("a pseudorandom record");
    let mut fields = line.split_whitespace();
    let a = u128::from_str_radix(fields.next().unwrap(), 16).unwrap();
    let b = u128::from_str_radix(fields.next().unwrap(), 16).unwrap();
    let product = u128::from_str_radix(fields.next().unwrap(), 16).unwrap();
    assert_ne!(wrong(a, b), product);
}
