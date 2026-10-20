//! A circuit's R1CS matrices as structured CCS operators, walked instead of stored.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_binary_field::BitCoordinates;
use p3_ccs::{BlockDiagonal, Ccs, Constraint, LinearOperator, PublicRegion, Term};
use p3_challenger::fs::TranscriptField;
use p3_field::ExtensionField;
use p3_multilinear_util::poly::Poly;

use crate::circuit::Circuit;

/// Encoding tag.
const TAG: u8 = 0x10;

/// Which R1CS matrix of a circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    /// Left factor.
    A,
    /// Right factor.
    B,
    /// Output, the identity since every row commits its slot.
    C,
}

impl Side {
    /// Statement encoding byte.
    const fn tag(self) -> u8 {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
        }
    }
}

/// One R1CS matrix of one circuit instance, over a characteristic-two field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitMatrix {
    /// The gate list.
    circuit: Arc<Circuit>,
    /// The matrix.
    side: Side,
}

impl CircuitMatrix {
    /// One matrix of a circuit.
    #[must_use]
    pub const fn new(circuit: Arc<Circuit>, side: Side) -> Self {
        Self { circuit, side }
    }

    /// The gate list.
    #[must_use]
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }

    /// The matrix.
    #[must_use]
    pub const fn side(&self) -> Side {
        self.side
    }
}

/// The batched R1CS of a circuit: `A o B - C = 0` with every matrix block-diagonal.
pub type CircuitCcs<F> = Ccs<F, BlockDiagonal<F, CircuitMatrix>>;

/// The R1CS of `2^log_instances` instances of a circuit as a CCS.
///
/// The relation is homogeneous, so it encodes the circuit only once the constant cells are pinned to one.
#[must_use]
pub fn ccs<F: BitCoordinates>(circuit: Arc<Circuit>, log_instances: usize) -> CircuitCcs<F> {
    let matrix =
        |circuit, side| BlockDiagonal::new(log_instances, CircuitMatrix::new(circuit, side));
    let [a, b] = [Side::A, Side::B].map(|side| matrix(Arc::clone(&circuit), side));
    Ccs::new(
        vec![a, b, matrix(circuit, Side::C)],
        vec![Constraint::new(vec![
            Term::new(F::ONE, [0, 1]),
            Term::new(F::ONE, [2]),
        ])],
    )
    .expect("three matrices of one shape and a degree-two constraint")
}

/// The constant wire of each of `2^log_instances` instances as public cells, all one.
#[must_use]
pub fn constant_cells<F: BitCoordinates>(
    circuit: &Circuit,
    log_instances: usize,
) -> PublicRegion<F> {
    PublicRegion::new(
        circuit.const_pos(),
        circuit.k_log(),
        vec![F::ONE; 1 << log_instances],
    )
}

impl<F: BitCoordinates> LinearOperator<F> for CircuitMatrix {
    fn log_rows(&self) -> usize {
        self.circuit.k_log()
    }

    fn log_cols(&self) -> usize {
        self.circuit.k_log()
    }

    fn apply(&self, z: &[F], out: &mut [F]) {
        assert_eq!(out.len(), z.len(), "square matrix");
        match self.side {
            // C is the identity.
            Side::C => out.copy_from_slice(z),
            // A forward walk writes one factor; rows no gate commits stay zero.
            Side::A => {
                out.fill(F::ZERO);
                self.circuit.walk_rows(z, |s, a, _| out[s] = a);
            }
            Side::B => {
                out.fill(F::ZERO);
                self.circuit.walk_rows(z, |s, _, b| out[s] = b);
            }
        }
    }

    fn add_transpose<EF: ExtensionField<F>>(&self, weights: &[EF], scale: EF, out: &mut [EF]) {
        assert_eq!(out.len(), weights.len(), "square matrix");
        // C moves the weights in place; a backward walk moves them for A and B.
        match self.side {
            Side::C => out
                .iter_mut()
                .zip(weights)
                .for_each(|(o, &w)| *o += scale * w),
            Side::A => self
                .circuit
                .add_fold_transpose(scale, EF::ZERO, weights, out),
            Side::B => self
                .circuit
                .add_fold_transpose(EF::ZERO, scale, weights, out),
        }
    }

    fn evaluate<EF: ExtensionField<F>>(&self, row: &[EF], col: &[EF]) -> EF {
        // M~(r, y) = <eq(r, .), M eq(y, .)>, by one forward walk of the column weights.
        let row_eq = Poly::new_from_point(row, EF::ONE);
        let col_eq = Poly::new_from_point(col, EF::ONE);
        let row_eq = row_eq.as_slice();
        match self.side {
            Side::C => row_eq
                .iter()
                .zip(col_eq.as_slice())
                .map(|(&r, &c)| r * c)
                .sum(),
            Side::A | Side::B => {
                let mut sum = EF::ZERO;
                self.circuit.walk_rows(col_eq.as_slice(), |s, a, b| {
                    sum += row_eq[s] * if self.side == Side::A { a } else { b };
                });
                sum
            }
        }
    }

    fn encode(&self, out: &mut Vec<u8>)
    where
        F: TranscriptField,
    {
        // Tag, side, then the fixed-width circuit digest.
        out.push(TAG);
        out.push(self.side.tag());
        out.extend_from_slice(self.circuit.digest());
    }
}
