//! Columns over a narrow alphabet, read as the packed elements a commitment holds.
//!
//! ```text
//!     1-bit column, one element per bit   ->  127 of 128 coordinates wasted
//!     1-bit column, packed                ->    0 of 128 coordinates wasted
//!     32-bit column, packed               ->    0 of 128 coordinates wasted
//! ```
//!
//! # Why packing moves no bits
//!
//! A tower level holds its elements in the basis its bytes already define.
//! Coordinate `j` of a value is bit `j` of its little-endian byte string.
//!
//! One run of narrow cells and the wide elements holding it are the same bytes, in order.
//! Packing is therefore one copy, and unpacking is that copy back.
//!
//! # What the packing means
//!
//! Cell `c` owns coordinates `c * a .. (c + 1) * a`, and `d` of them form one element.
//!
//! ```text
//!     a = 1,  d = 128   ->  element w holds cells 128w .. 128w + 128
//!     a = 32, d = 128   ->  element w holds cells   4w ..   4w + 4
//! ```
//!
//! Which ring switch opens such a commitment depends on the alphabet, not on anything here.
//!
//! # Orientation
//!
//! A bit-sliced trace often arrives one block per row, with a lane per column.
//! A commitment wants the transpose of that, every bit of one column contiguous.
//!
//! The square bit transpose turns one into the other in `log2(d)` word passes.
//! Reading one bit per source lane would cost `d^2` bit extractions instead.

use alloc::vec::Vec;
use core::marker::PhantomData;
use core::{ptr, slice};

use p3_binary_field::{
    BinaryField8, BinaryField16, BinaryField32, BinaryField64, BinaryField128, Gf2, PackedGf2,
    TowerLevel, Underlier,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::poly::Poly;
use p3_sumcheck::layout::Table;
use p3_util::log2_strict_usize;
use thiserror::Error;

/// A value whose bytes are a run of `F_2` coordinates, lowest coordinate first.
///
/// # Safety
///
/// An implementor is a transparent wrapper over an unsigned integer, or over a block of them.
/// Every bit pattern of its bytes is a value, so a run has no padding and no invalid state.
///
/// The coordinate count is the size in bits, with nothing left over:
///
/// ```text
///     COORDINATES == 8 * size_of::<Self>()
/// ```
///
/// A value carrying more bytes than that would read past the source of a copy.
/// On a little-endian target coordinate `j` of a run is bit `j % 8` of its byte `j / 8`.
pub unsafe trait Coordinates: Copy + Send + Sync + 'static {
    /// Coordinates one value of this type holds.
    const COORDINATES: usize;
}

/// Pin the part of one implementor's contract a constant can state: its two counts agree.
const fn check_coordinates<A: Coordinates>() {
    assert!(
        A::COORDINATES == 8 * size_of::<A>(),
        "a coordinate run must be exactly the bits of its bytes"
    );
}

// SAFETY: each level below wraps an integer it fills, whose bits are its own coordinates.
//
//     at or above a byte  ->  the integer is full, so a run has no padding
//     below a byte        ->  the integer is partly unused, so no run qualifies
//
// A one-bit alphabet therefore arrives bit-sliced rather than one element per byte.
unsafe impl Coordinates for BinaryField8 {
    const COORDINATES: usize = 8;
}

unsafe impl Coordinates for BinaryField16 {
    const COORDINATES: usize = 16;
}

unsafe impl Coordinates for BinaryField32 {
    const COORDINATES: usize = 32;
}

unsafe impl Coordinates for BinaryField64 {
    const COORDINATES: usize = 64;
}

unsafe impl Coordinates for BinaryField128 {
    const COORDINATES: usize = 128;
}

// SAFETY: a packing is exactly its backing block, whose contract pins size and alignment.
// Lane `j` is bit `j` of the byte view, which is the convention fixed above.
unsafe impl<U: Underlier> Coordinates for PackedGf2<U> {
    const COORDINATES: usize = U::BITS;
}

/// Read a run of cells as the wide elements holding the same coordinates.
///
/// One copy at the wide alignment, and no arithmetic. Reusing the source buffer is unsound.
///
/// # Panics
///
/// Panics unless the run's coordinates fill a whole number of wide elements.
#[must_use]
pub fn pack<A, EF>(cells: &[A]) -> Vec<EF>
where
    A: Coordinates,
    EF: Coordinates,
{
    const {
        check_coordinates::<A>();
        check_coordinates::<EF>();
        assert!(
            cfg!(target_endian = "little"),
            "packing a coordinate run needs a little-endian target"
        );
    }
    let coordinates = cells.len() * A::COORDINATES;
    assert_eq!(
        coordinates % EF::COORDINATES,
        0,
        "a packed run must fill whole elements"
    );
    let len = coordinates / EF::COORDINATES;
    let mut packed = Vec::<EF>::with_capacity(len);

    // SAFETY: each side is a padding-free run of its own coordinates, by both contracts.
    // The constant block above pins each one's count at exactly the bits of its bytes.
    //
    // The two cover the same coordinate count, hence the same byte count.
    //
    // Every bit pattern of the destination is a value, so nothing stays uninitialised.
    // The destination was reserved for exactly this many elements, and nothing aliases it.
    unsafe {
        ptr::copy_nonoverlapping(
            cells.as_ptr().cast::<u8>(),
            packed.as_mut_ptr().cast::<u8>(),
            len * size_of::<EF>(),
        );
        packed.set_len(len);
    }
    packed
}

/// Read a run of wide elements back as the cells whose coordinates they hold.
///
/// The inverse of packing, so a round trip is the identity, and it panics on a partial cell.
#[must_use]
pub fn unpack<A, EF>(elements: &[EF]) -> Vec<A>
where
    A: Coordinates,
    EF: Coordinates,
{
    const {
        check_coordinates::<A>();
        check_coordinates::<EF>();
        assert!(
            cfg!(target_endian = "little"),
            "unpacking a coordinate run needs a little-endian target"
        );
    }
    let coordinates = elements.len() * EF::COORDINATES;
    assert_eq!(
        coordinates % A::COORDINATES,
        0,
        "an unpacked run must fill whole cells"
    );
    let len = coordinates / A::COORDINATES;
    let mut cells = Vec::<A>::with_capacity(len);

    // SAFETY: the argument above, with the two roles exchanged.
    unsafe {
        ptr::copy_nonoverlapping(
            elements.as_ptr().cast::<u8>(),
            cells.as_mut_ptr().cast::<u8>(),
            len * size_of::<A>(),
        );
        cells.set_len(len);
    }
    cells
}

/// The byte view of a run of cells, which is the same bytes the packing holds.
#[must_use]
pub const fn coordinate_bytes<A: Coordinates>(cells: &[A]) -> &[u8] {
    const {
        check_coordinates::<A>();
        assert!(
            cfg!(target_endian = "little"),
            "viewing coordinate bytes needs a little-endian target"
        );
    }

    // SAFETY: by the trait contract the run is exactly this many initialised bytes.
    // It has no padding and no invalid pattern, and every bit pattern of a byte is valid.
    //
    // The byte slice borrows the same region for the same lifetime.
    unsafe { slice::from_raw_parts(cells.as_ptr().cast::<u8>(), size_of_val(cells)) }
}

/// Equal-height columns over a narrow alphabet, held as the elements a commitment takes.
///
/// # Overview
///
/// Each column packs on its own, so a column boundary lands on an element boundary.
///
/// ```text
///     elements   [ column 0 ][ column 1 ] ... [ column k-1 ]
///     column     2^arity elements, each holding d / a cells
/// ```
///
/// That is what a stacked commitment expects of a source table.
/// Nothing moves between the two.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackedStack<A, EF> {
    /// Every column's elements, lowest column first.
    elements: Vec<EF>,
    /// Elements one column occupies.
    column_len: usize,
    /// Marker for the alphabet the cells came from.
    _alphabet: PhantomData<A>,
}

impl<A, EF> PackedStack<A, EF>
where
    A: Coordinates,
    EF: Coordinates + TowerLevel,
{
    /// Cells one wide element absorbs.
    pub const CELLS_PER_ELEMENT: usize = EF::COORDINATES / A::COORDINATES;

    /// Pack equal-height columns, lowest column first.
    ///
    /// # Errors
    ///
    /// - The alphabet is wider than the level packing it, so no cell fits.
    /// - No column was supplied, so there is nothing to commit.
    /// - The columns have different heights, so the layout has no common arity.
    /// - A column's cells do not fill a whole number of elements.
    /// - A packed column is no power-of-two length, so it covers no hypercube.
    pub fn from_columns(columns: &[&[A]]) -> Result<Self, PackError> {
        if A::COORDINATES > EF::COORDINATES {
            return Err(PackError::CellWiderThanElement {
                cell: A::COORDINATES,
                element: EF::COORDINATES,
            });
        }
        let Some((&first, rest)) = columns.split_first() else {
            return Err(PackError::NoColumns);
        };
        if let Some(other) = rest.iter().find(|column| column.len() != first.len()) {
            return Err(PackError::RaggedColumns {
                expected: first.len(),
                actual: other.len(),
            });
        }
        if !first.len().is_multiple_of(Self::CELLS_PER_ELEMENT) {
            return Err(PackError::PartialElement {
                cells: first.len(),
                per_element: Self::CELLS_PER_ELEMENT,
            });
        }

        let column_len = first.len() / Self::CELLS_PER_ELEMENT;
        if column_len == 0 || !column_len.is_power_of_two() {
            return Err(PackError::NotAHypercube {
                elements: column_len,
            });
        }

        // One copy per column, so the whole stack is one sweep over the source.
        let mut elements = Vec::with_capacity(column_len * columns.len());
        for column in columns {
            elements.extend(pack::<A, EF>(column));
        }

        Ok(Self {
            elements,
            column_len,
            _alphabet: PhantomData,
        })
    }

    /// Every column's elements, lowest column first.
    #[must_use]
    pub fn elements(&self) -> &[EF] {
        &self.elements
    }

    /// Source columns the stack holds.
    #[must_use]
    pub const fn num_columns(&self) -> usize {
        self.elements.len() / self.column_len
    }

    /// Variables one packed column's multilinear has.
    #[must_use]
    pub const fn column_num_variables(&self) -> usize {
        log2_strict_usize(self.column_len)
    }

    /// Cells one source column held.
    #[must_use]
    pub const fn column_num_cells(&self) -> usize {
        self.column_len * Self::CELLS_PER_ELEMENT
    }

    /// One packed column's elements, panicking on a column the stack does not hold.
    #[must_use]
    pub fn column(&self, index: usize) -> &[EF] {
        assert!(index < self.num_columns(), "column index out of range");
        &self.elements[index * self.column_len..][..self.column_len]
    }

    /// Which element of the stack, and which cell inside it, one source cell landed in.
    ///
    /// This is the view an opening reads, since a claim lands on the committed polynomial.
    ///
    /// # Panics
    ///
    /// Panics if the column or the cell is not one the stack holds.
    #[must_use]
    pub fn locate(&self, column: usize, cell: usize) -> (usize, usize) {
        assert!(column < self.num_columns(), "column index out of range");
        assert!(cell < self.column_num_cells(), "cell index out of range");
        let element = column * self.column_len + cell / Self::CELLS_PER_ELEMENT;
        (element, cell % Self::CELLS_PER_ELEMENT)
    }

    /// One source cell, read back out of the element holding it.
    #[must_use]
    pub fn cell(&self, column: usize, cell: usize) -> A {
        let (element, inside) = self.locate(column, cell);
        unpack::<A, EF>(&self.elements[element..=element])[inside]
    }

    /// One source column, read back in full.
    #[must_use]
    pub fn unpack_column(&self, index: usize) -> Vec<A> {
        unpack::<A, EF>(self.column(index))
    }

    /// The stack as the multilinear a single-column commitment holds.
    pub fn into_poly(self) -> Poly<EF> {
        Poly::new(self.elements)
    }

    /// The stack as one source table of a commitment's stacked layout.
    ///
    /// Row `i` of the table is packed column `i`, the orientation that layout reads.
    pub fn into_table(self) -> Table<EF> {
        Table::new(RowMajorMatrix::new(self.elements, self.column_len))
    }
}

impl<U, EF> PackedStack<PackedGf2<U>, EF>
where
    U: Underlier,
    EF: Coordinates + TowerLevel,
{
    /// Pack a bit-sliced block whose lanes are columns rather than rows.
    ///
    /// ```text
    ///     in     row r    ->  lanes are columns 0 .. d
    ///     out    column c ->  lanes are the rows of one block
    /// ```
    ///
    /// One square bit transpose per block of `d` rows, so whole words move at a time.
    ///
    /// # Arguments
    ///
    /// - `rows`: one block per row, in row order, covering whole blocks.
    /// - `num_columns`: leading lanes to keep, the rest being unused width.
    ///
    /// # Errors
    ///
    /// - The rows are no whole number of square blocks, or ask for more than a block's lanes.
    /// - The transposed columns do not pack, for any of the reasons columns can fail to.
    pub fn from_bit_rows(rows: &[PackedGf2<U>], num_columns: usize) -> Result<Self, PackError> {
        let width = PackedGf2::<U>::WIDTH;
        if !rows.len().is_multiple_of(width) {
            return Err(PackError::PartialBitBlock {
                rows: rows.len(),
                width,
            });
        }
        if num_columns > width {
            return Err(PackError::TooManyColumns {
                requested: num_columns,
                width,
            });
        }

        // Column `c` takes lane `c` of every transposed block, so it grows one block at a time.
        let blocks = rows.len() / width;
        let mut columns = alloc::vec![Vec::with_capacity(blocks); num_columns];
        let mut block = alloc::vec![PackedGf2::<U>::from(Gf2::ZERO); width];

        for source in rows.chunks_exact(width) {
            // The transpose works in place, so the block is copied before it runs.
            block.copy_from_slice(source);
            PackedGf2::transpose(&mut block);

            for (column, &transposed) in columns.iter_mut().zip(&block) {
                column.push(transposed);
            }
        }

        let views: Vec<&[PackedGf2<U>]> = columns.iter().map(alloc::vec::Vec::as_slice).collect();
        Self::from_columns(&views)
    }
}

/// Why a run of columns could not be packed.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum PackError {
    /// The alphabet is wider than the level packing it, so no cell fits.
    #[error("a {cell}-coordinate cell does not fit a {element}-coordinate element")]
    CellWiderThanElement {
        /// Coordinates one cell holds.
        cell: usize,
        /// Coordinates one element holds.
        element: usize,
    },

    /// No column was supplied, so there is nothing to commit.
    #[error("a packed stack needs at least one column")]
    NoColumns,

    /// The columns have different heights, so the layout has no common arity.
    #[error("a column holds {actual} cells, expected {expected}")]
    RaggedColumns {
        /// Cells the first column holds.
        expected: usize,
        /// Cells the offending column holds.
        actual: usize,
    },

    /// A column's cells do not fill a whole number of elements.
    #[error("{cells} cells do not fill whole elements of {per_element} cells each")]
    PartialElement {
        /// Cells one column holds.
        cells: usize,
        /// Cells one element absorbs.
        per_element: usize,
    },

    /// A packed column is no power-of-two length, so it covers no hypercube.
    #[error("a packed column holds {elements} elements, which is no hypercube")]
    NotAHypercube {
        /// Elements one packed column holds.
        elements: usize,
    },

    /// The row count is no whole number of square blocks.
    #[error("{rows} bit-sliced rows are no whole number of {width}-row blocks")]
    PartialBitBlock {
        /// Rows supplied.
        rows: usize,
        /// Rows one square block holds.
        width: usize,
    },

    /// More columns are asked for than a block has lanes.
    #[error("{requested} columns exceed the {width} lanes a block holds")]
    TooManyColumns {
        /// Columns requested.
        requested: usize,
        /// Lanes one block holds.
        width: usize,
    },
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{PackedGf2x8, PackedGf2x64, PackedGf2x128};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = BinaryField128;

    /// Coordinate `j` of a run of cells, read one bit at a time.
    fn coordinate<A: Coordinates>(cells: &[A], index: usize) -> bool {
        let bytes = coordinate_bytes(cells);
        (bytes[index / 8] >> (index % 8)) & 1 == 1
    }

    #[test]
    fn packing_preserves_every_coordinate_in_order() {
        //     cells      32 words  *  32 = 1024 coordinates
        //     elements    8 wide    * 128 = 1024 coordinates
        let mut rng = SmallRng::seed_from_u64(0xC0DE);
        let cells: Vec<BinaryField32> = (0..32).map(|_| rng.random()).collect();
        let packed = pack::<BinaryField32, EF>(&cells);

        assert_eq!(packed.len(), 8);
        for index in 0..32 * 32 {
            assert_eq!(
                coordinate(&packed, index),
                coordinate(&cells, index),
                "coordinate {index}"
            );
        }
    }

    #[test]
    fn the_reinterpretation_agrees_with_the_field_basis_accessor() {
        // Invariant: the copy this module performs is the tower basis composition.
        //
        //     packed   one memcpy over the whole run
        //     basis    one element at a time, through the field's own accessor
        //
        // The accessor composes arithmetically, so agreement is evidence rather than a tautology.
        let mut rng = SmallRng::seed_from_u64(0xBA515);

        let words: Vec<BinaryField32> = (0..64).map(|_| rng.random()).collect();
        let by_basis: Vec<EF> = words
            .as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| EF::from_basis_coefficients_fn(|index| chunk[index]))
            .collect();
        assert_eq!(pack::<BinaryField32, EF>(&words), by_basis);

        let halves: Vec<BinaryField64> = (0..64).map(|_| rng.random()).collect();
        let by_basis: Vec<EF> = halves
            .as_chunks::<2>()
            .0
            .iter()
            .map(|chunk| EF::from_basis_coefficients_fn(|index| chunk[index]))
            .collect();
        assert_eq!(pack::<BinaryField64, EF>(&halves), by_basis);
    }

    #[test]
    fn a_round_trip_through_the_packing_is_the_identity() {
        // Both directions are the same copy, so neither may lose or reorder a coordinate.
        let mut rng = SmallRng::seed_from_u64(0xB1A5);

        let words: Vec<BinaryField64> = (0..64).map(|_| rng.random()).collect();
        assert_eq!(unpack::<BinaryField64, EF>(&pack::<_, EF>(&words)), words);

        let bytes: Vec<BinaryField8> = (0..256).map(|_| rng.random()).collect();
        assert_eq!(unpack::<BinaryField8, EF>(&pack::<_, EF>(&bytes)), bytes);

        let blocks: Vec<PackedGf2x64> = (0..16)
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect();
        assert_eq!(unpack::<PackedGf2x64, EF>(&pack::<_, EF>(&blocks)), blocks);
    }

    #[test]
    fn a_packed_bit_block_carries_the_lanes_it_was_given() {
        // Invariant: lane `l` of block `b` is coordinate `64b + l` of the packed run.
        //
        // This is what lets a bit-sliced column commit without a per-bit pass.
        let mut rng = SmallRng::seed_from_u64(0xB175);
        let blocks: Vec<PackedGf2x64> = (0..8)
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect();
        let packed = pack::<PackedGf2x64, EF>(&blocks);

        assert_eq!(packed.len(), 4);
        for (b, block) in blocks.iter().enumerate() {
            for lane in 0..PackedGf2x64::WIDTH {
                let set = block.get(lane) == Gf2::ONE;
                assert_eq!(coordinate(&packed, 64 * b + lane), set, "b={b} lane={lane}");
            }
        }
    }

    #[test]
    fn the_stack_lays_columns_out_back_to_back() {
        //     column   8 words at 4 per element, so arity 1
        //     stack    three columns, 6 elements, column 0 first
        let mut rng = SmallRng::seed_from_u64(0x57AC);
        let columns: Vec<Vec<BinaryField32>> = (0..3)
            .map(|_| (0..8).map(|_| rng.random()).collect())
            .collect();
        let views: Vec<&[BinaryField32]> = columns.iter().map(Vec::as_slice).collect();

        let stack = PackedStack::<BinaryField32, EF>::from_columns(&views).unwrap();

        assert_eq!(stack.num_columns(), 3);
        assert_eq!(stack.column_num_variables(), 1);
        assert_eq!(stack.column_num_cells(), 8);
        assert_eq!(stack.elements().len(), 6);

        // Each column reads back what went in, cell by cell and in bulk.
        for (index, column) in columns.iter().enumerate() {
            assert_eq!(&stack.unpack_column(index), column);
            for (cell, &value) in column.iter().enumerate() {
                assert_eq!(stack.cell(index, cell), value, "column={index} cell={cell}");
            }
        }

        // The located element is the one holding the cell, at the offset reported.
        let (element, inside) = stack.locate(2, 5);
        assert_eq!(element, 2 * 2 + 1);
        assert_eq!(inside, 1);
    }

    #[test]
    fn the_stacked_table_keeps_one_row_per_packed_column() {
        // The stacked layout reads a source table's rows as its polynomials.
        let mut rng = SmallRng::seed_from_u64(0x7AB1);
        let columns: Vec<Vec<BinaryField64>> = (0..2)
            .map(|_| (0..8).map(|_| rng.random()).collect())
            .collect();
        let views: Vec<&[BinaryField64]> = columns.iter().map(Vec::as_slice).collect();
        let stack = PackedStack::<BinaryField64, EF>::from_columns(&views).unwrap();

        let table = stack.clone().into_table();
        assert_eq!(table.num_polys(), 2);
        assert_eq!(table.num_variables(), 2);
        for index in 0..2 {
            assert_eq!(table.poly(index).as_slice(), stack.column(index));
        }
    }

    /// Exchange rows and columns of a bit-sliced block one lane at a time.
    fn transpose_reference<U: Underlier>(
        rows: &[PackedGf2<U>],
        num_columns: usize,
    ) -> Vec<Vec<PackedGf2<U>>> {
        let width = PackedGf2::<U>::WIDTH;
        (0..num_columns)
            .map(|column| {
                rows.chunks_exact(width)
                    .map(|block| PackedGf2::<U>::from_fn(|lane| block[lane].get(column)))
                    .collect()
            })
            .collect()
    }

    #[test]
    fn a_row_major_bit_block_packs_as_its_own_transpose() {
        // Invariant: column `c` of the stack is lane `c` of every source row.
        //
        //     rows      128 blocks in,  lane = column
        //     columns    40 runs out,   lane = row within a block
        //
        // The reference reads one bit per source lane, sharing nothing with the transpose.
        let mut rng = SmallRng::seed_from_u64(0x7A05);
        let rows: Vec<PackedGf2x64> = (0..128)
            .map(|_| PackedGf2x64::new(rng.random::<u64>()))
            .collect();

        let stack = PackedStack::<PackedGf2x64, EF>::from_bit_rows(&rows, 40).unwrap();
        let expected = transpose_reference(&rows, 40);

        assert_eq!(stack.num_columns(), 40);
        for (column, blocks) in expected.iter().enumerate() {
            assert_eq!(&stack.unpack_column(column), blocks, "column={column}");
        }
    }

    #[test]
    fn a_row_major_block_is_refused_unless_it_is_square_and_wide_enough() {
        // A partial block has no transpose, and a column past the lanes has no source.
        let rows = alloc::vec![PackedGf2x8::from(Gf2::ZERO); 12];
        assert_eq!(
            PackedStack::<PackedGf2x8, EF>::from_bit_rows(&rows, 8).unwrap_err(),
            PackError::PartialBitBlock { rows: 12, width: 8 }
        );

        let rows = alloc::vec![PackedGf2x8::from(Gf2::ZERO); 128];
        assert_eq!(
            PackedStack::<PackedGf2x8, EF>::from_bit_rows(&rows, 9).unwrap_err(),
            PackError::TooManyColumns {
                requested: 9,
                width: 8,
            }
        );
    }

    #[test]
    fn a_stack_that_describes_no_commitment_is_refused() {
        // No column at all.
        assert_eq!(
            PackedStack::<BinaryField32, EF>::from_columns(&[]).unwrap_err(),
            PackError::NoColumns
        );

        // Columns of different heights have no common arity.
        let long = alloc::vec![BinaryField32::ZERO; 8];
        let short = alloc::vec![BinaryField32::ZERO; 4];
        assert_eq!(
            PackedStack::<BinaryField32, EF>::from_columns(&[&long, &short]).unwrap_err(),
            PackError::RaggedColumns {
                expected: 8,
                actual: 4,
            }
        );

        // Six words fill one element of four and leave two over.
        let ragged = alloc::vec![BinaryField32::ZERO; 6];
        assert_eq!(
            PackedStack::<BinaryField32, EF>::from_columns(&[&ragged]).unwrap_err(),
            PackError::PartialElement {
                cells: 6,
                per_element: 4,
            }
        );

        // Twelve words fill three elements, which is no hypercube.
        let odd = alloc::vec![BinaryField32::ZERO; 12];
        assert_eq!(
            PackedStack::<BinaryField32, EF>::from_columns(&[&odd]).unwrap_err(),
            PackError::NotAHypercube { elements: 3 }
        );

        // A cell wider than the element packing it fits nowhere.
        let wide = alloc::vec![PackedGf2x128::from(Gf2::ZERO); 4];
        assert_eq!(
            PackedStack::<PackedGf2x128, BinaryField64>::from_columns(&[&wide]).unwrap_err(),
            PackError::CellWiderThanElement {
                cell: 128,
                element: 64,
            }
        );
    }

    proptest! {
        /// Every alphabet this module packs, over every shape a column can take.
        ///
        /// Sixteen cells is the smallest run filling one element of the widest level.
        #[test]
        fn packing_is_a_bijection_on_coordinates(
            log_cells in 4usize..=9,
            seed: u64,
        ) {
            let mut rng = SmallRng::seed_from_u64(seed);
            let cells = 1usize << log_cells;

            // A one-bit alphabet, arriving bit-sliced.
            let bits: Vec<PackedGf2x8> = (0..cells)
                .map(|_| PackedGf2x8::new(rng.random::<u8>()))
                .collect();
            prop_assert_eq!(unpack::<PackedGf2x8, EF>(&pack::<_, EF>(&bits)), bits);

            // A byte alphabet, where sixteen cells fill one element.
            let bytes: Vec<BinaryField8> = (0..cells).map(|_| rng.random()).collect();
            prop_assert_eq!(unpack::<BinaryField8, EF>(&pack::<_, EF>(&bytes)), bytes);

            // A word alphabet, where four cells fill one element.
            let words: Vec<BinaryField32> = (0..cells).map(|_| rng.random()).collect();
            prop_assert_eq!(&unpack::<BinaryField32, EF>(&pack::<_, EF>(&words)), &words);

            // The coordinate strings agree position for position, on the widest of the three.
            let packed = pack::<BinaryField32, EF>(&words);
            for index in 0..cells * 32 {
                prop_assert_eq!(coordinate(&packed, index), coordinate(&words, index));
            }
        }
    }
}
