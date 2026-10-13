#[cfg(target_endian = "little")]
use core::cell::{Cell, RefCell};

use p3_binary_field::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128, Poly64, TowerLevel,
};
use p3_field::PrimeCharacteristicRing;
use p3_keccak::Keccak256Hash;
use p3_symmetric::{CryptographicHasher, SerializingHasher};

fn values<F: TowerLevel>(len: usize) -> Vec<F> {
    (0..len)
        .map(|i| {
            F::from_le_byte_iter((0..F::NUM_BYTES).map(|j| {
                (i as u8)
                    .wrapping_mul(73)
                    .wrapping_add((j as u8).wrapping_mul(29))
            }))
        })
        .collect()
}

fn check_byte_view<F: TowerLevel>() {
    let values = values::<F>(19);
    for range in [0..0, 0..19, 1..18, 19..19] {
        let input = &values[range];
        let expected: Vec<_> = F::into_byte_stream(input.iter().copied())
            .into_iter()
            .collect();
        if cfg!(target_endian = "little") {
            let bytes = F::as_serialized_bytes(input).expect("little-endian byte view");
            assert_eq!(bytes, expected);
            assert_eq!(bytes.as_ptr(), input.as_ptr().cast::<u8>());
        } else {
            assert!(F::as_serialized_bytes(input).is_none());
        }
    }
}

#[test]
fn byte_views_match_the_existing_serialization() {
    check_byte_view::<BinaryField2>();
    check_byte_view::<BinaryField4>();
    check_byte_view::<BinaryField8>();
    check_byte_view::<BinaryField16>();
    check_byte_view::<BinaryField32>();
    check_byte_view::<BinaryField64>();
    check_byte_view::<BinaryField128>();
    check_byte_view::<Poly64>();
}

#[derive(Clone)]
#[cfg(target_endian = "little")]
struct BorrowCheckingHasher<'a> {
    next_input: &'a Cell<*const u8>,
    row_bytes: usize,
    calls: &'a RefCell<Vec<usize>>,
}

#[cfg(target_endian = "little")]
impl CryptographicHasher<u8, [u8; 32]> for BorrowCheckingHasher<'_> {
    // A non-power-of-two lane count catches groups rounded to a byte boundary instead.
    const LANES: usize = 3;

    fn hash_iter<I: IntoIterator<Item = u8>>(&self, input: I) -> [u8; 32] {
        Keccak256Hash.hash_iter(input)
    }

    fn hash_many(&self, input: &[u8], out: &mut [[u8; 32]]) {
        assert_eq!(input.as_ptr(), self.next_input.get());
        assert_eq!(input.len(), self.row_bytes * out.len());
        self.next_input
            .set(input.as_ptr().wrapping_add(input.len()));
        self.calls.borrow_mut().push(out.len());
        Keccak256Hash.hash_many(input, out);
    }
}

#[cfg(target_endian = "little")]
fn check_borrowed_batch<F: TowerLevel>(width: usize, rows: usize, groups: &[usize]) {
    // A subslice also checks that the view starts at the requested element.
    let storage = values::<F>(width * rows + 1);
    let input = &storage[1..];
    let next_input = Cell::new(input.as_ptr().cast());
    let calls = RefCell::new(Vec::new());
    let hasher = SerializingHasher::new(BorrowCheckingHasher {
        next_input: &next_input,
        row_bytes: width * F::NUM_BYTES,
        calls: &calls,
    });
    let expected: Vec<_> = input
        .chunks(width)
        .map(|row| hasher.hash_slice(row))
        .collect();
    let mut actual = vec![[0; 32]; rows];
    hasher.hash_many(input, &mut actual);
    assert_eq!(actual, expected);
    assert_eq!(*calls.borrow(), groups);
}

#[cfg(target_endian = "little")]
#[test]
fn batched_hashing_borrows_storage_in_whole_lane_groups() {
    // 400-byte rows fit 20 times in 8 KiB, rounded down to six three-lane groups.
    check_borrowed_batch::<BinaryField128>(25, 40, &[18, 18, 4]);
    check_borrowed_batch::<Poly64>(50, 40, &[18, 18, 4]);
    check_borrowed_batch::<BinaryField128>(1, 2, &[2]);
    check_borrowed_batch::<Poly64>(1, 2, &[2]);
}

#[cfg(target_endian = "little")]
#[test]
fn borrowed_wide_rows_preserve_lane_groups_and_remainders() {
    // 2800-byte rows fit individually, but one lane group exceeds 8 KiB.
    check_borrowed_batch::<BinaryField128>(175, 7, &[3, 3, 1]);
    check_borrowed_batch::<Poly64>(350, 7, &[3, 3, 1]);
    // Each row alone exceeds the target.
    check_borrowed_batch::<BinaryField128>(513, 4, &[3, 1]);
    check_borrowed_batch::<Poly64>(1025, 4, &[3, 1]);
}

#[test]
fn serialization_preserves_empty_batches() {
    let hasher = SerializingHasher::new(Keccak256Hash);
    hasher.hash_many(&[Poly64::ONE], &mut []);
    let mut actual = [[0; 32]; 5];
    hasher.hash_many(&[] as &[Poly64], &mut actual);
    assert_eq!(actual, [Keccak256Hash.hash_iter([]); 5]);
}

#[test]
#[should_panic(expected = "whole multiple")]
fn serialization_rejects_ragged_batches() {
    let hasher = SerializingHasher::new(Keccak256Hash);
    hasher.hash_many(&[Poly64::ONE; 3], &mut [[0; 32]; 2]);
}
