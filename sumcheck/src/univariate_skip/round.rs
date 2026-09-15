//! One univariate-skip round: its message, its verifier check, and the binding it leaves behind.

use alloc::vec::Vec;

use p3_binary_field::TowerLevel;
use p3_field::{ExtensionField, Field, batch_multiplicative_inverse};
use p3_maybe_rayon::prelude::*;
use p3_multilinear_util::poly::Poly;

use super::composition::Composition;
use super::domain::{SkipDomain, SkipDomainError};
use super::lde::{CHUNK_BITS, CompressedLde, CompressedLdeError, TABLE_ROWS};

/// The round message does not carry one value per transmitted point.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[error("round message length mismatch: expected {expected}, got {actual}")]
pub struct MessageLenMismatch {
    /// Number of values this round transmits.
    pub expected: usize,
    /// Number of values the message carries.
    pub actual: usize,
}

/// Reasons a skip round cannot be set up.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SkipRoundError {
    /// The domain the round would run over is not realisable.
    #[error(transparent)]
    Domain(#[from] SkipDomainError),
    /// The extension table the round would use is not realisable.
    #[error(transparent)]
    Lde(#[from] CompressedLdeError),
}

/// The fixed setup of one univariate-skip round.
///
/// # Overview
///
/// A zerocheck over `m` variables normally spends `m` rounds.
///
/// Each of them crosses a bit-valued hypercube into the large field.
///
/// The first crossing is the expensive one.
///
/// It halves the cell count, but grows every remaining cell from a bit to a field element.
///
/// A skip round pays that crossing once for `k` variables instead of once per variable.
///
/// ```text
///     plain:  m rounds, the first widening 2^(m-1) cells to extension elements
///     skip:   1 round over a subspace, then m - k ordinary rounds
/// ```
///
/// # Protocol
///
/// Reading the skipped coordinates as subspace points turns each trace row into a polynomial.
///
/// The message is that polynomial summed over rows, under the zerocheck's equality weights:
///
/// ```text
///     P(y) = sum_x eq(r, x) * C( f_0(x, y), ..., f_{n-1}(x, y) )
/// ```
///
/// - On the subspace, `P` is the zerocheck's own claim, so it vanishes there and is not sent.
/// - Off it, `P` is transmitted, and the verifier reads it back at one challenge.
///
/// See Gruen, *Some Improvements for the PIOP for ZeroCheck*, and Bünz, Rothblum, Wang, *Flock*,
/// Section 4.2.
///
/// # Soundness
///
/// The verifier defines the round polynomial itself.
///
/// It is the one of degree below the extension size that vanishes on the subspace.
///
/// Off the subspace it matches the transmitted values.
///
/// - Vanishing on the subspace is imposed, never checked, which is what makes it a zerocheck.
/// - The residual sumcheck then pins that polynomial to the real one at a random challenge.
/// - A prover who sends anything else is caught except with probability `2^(k+e) / |EF|`.
///
/// Allowing degree up to the extension size costs the verifier nothing beyond that term.
///
/// It also saves it a consistency check on the honest degree bound.
///
/// # What the caller still owes
///
/// Two obligations sit outside this round, and neither is discharged here.
///
/// **The zerocheck point.**
/// This transcript neither absorbs nor draws the equality point the message is weighted by.
///
/// The surrounding protocol draws it from its own transcript, after the witness is committed.
///
/// A point chosen before the commitment, or chosen by the prover, makes the reduction vacuous.
///
/// **The final opening is not a plain multilinear evaluation.**
/// The residual rounds end on the rows read at the skip challenge, which unfold as
///
/// ```text
///     f(rho, lam) = sum_c L_c(lam) * f~(rho, c)
/// ```
///
/// an inner product of the witness's partial evaluation with the subspace's Lagrange vector.
///
/// - It is one claim, but not one that a commitment can open directly.
/// - Reducing it to a single evaluation `f~(rho, tau)` takes a further `k`-round degree-two
///   sumcheck, or any equivalent reduction.
/// - A caller that hands a commitment the bound rows evaluated at the residual point is proving
///   a different statement.
#[derive(Debug, Clone)]
pub struct SkipRound<F> {
    /// Where the round polynomial vanishes, and where it is transmitted.
    domain: SkipDomain<F>,
    /// The tabulated map from a row's bits to its transmitted values.
    lde: CompressedLde<F>,
    /// The formal derivative of the extension's vanishing polynomial, constant on the extension.
    derivative_on_extension: F,
}

impl<F: TowerLevel> SkipRound<F> {
    /// Set up a round that skips `log_size` variables of a composition of the given degree.
    ///
    /// # Errors
    ///
    /// - When no domain of that shape fits in the field.
    /// - When the subspace is too narrow to chunk a row into bytes.
    pub fn new(log_size: usize, degree: usize) -> Result<Self, SkipRoundError> {
        Self::from_domain(SkipDomain::for_degree(log_size, degree)?)
    }

    /// Set up a round over an explicit domain.
    ///
    /// # Errors
    ///
    /// Returns an error when the subspace is too narrow to chunk a row into bytes.
    pub fn from_domain(domain: SkipDomain<F>) -> Result<Self, SkipRoundError> {
        let lde = CompressedLde::new(&domain)?;

        // The extension's own vanishing polynomial is what the verifier interpolates against.
        // Its derivative is constant there, exactly as the subspace's is on the subspace.
        let derivative_on_extension = domain
            .subspace()
            .iter()
            .chain(domain.transmitted())
            .copied()
            .filter(|point| !point.is_zero())
            .product::<F>();

        Ok(Self {
            domain,
            lde,
            derivative_on_extension,
        })
    }

    /// The domain the round runs over.
    #[must_use]
    pub const fn domain(&self) -> &SkipDomain<F> {
        &self.domain
    }

    /// Number of variables this round binds in one go.
    #[must_use]
    pub const fn log_size(&self) -> usize {
        self.domain.log_size()
    }

    /// Number of field elements the round message carries.
    #[must_use]
    pub const fn num_transmitted(&self) -> usize {
        self.domain.num_transmitted()
    }

    /// Number of bytes one packed row occupies.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.lde.num_chunks()
    }

    /// Extend packed rows of one polynomial onto the transmitted points.
    ///
    /// # Arguments
    ///
    /// - `packed`: the rows back to back, least significant bit first within each byte.
    /// - `out`: receives one block of transmitted values per row.
    ///
    /// # Panics
    ///
    /// Panics if the two slices disagree on how many rows there are.
    pub fn extend_rows(&self, packed: &[u8], out: &mut [F])
    where
        F: Send + Sync,
    {
        self.lde.extend_batch(packed, out);
    }

    /// Form the round message straight from the packed witness, row by row.
    ///
    /// # Overview
    ///
    /// The unstreamed path materialises every operand's extension, then the composed values.
    ///
    /// That is `rows * transmitted` subfield elements per operand before anything is weighed.
    ///
    /// Streaming keeps only one row's worth of scratch per thread:
    ///
    /// ```text
    ///     per row:  extend each operand  ->  compose  ->  weigh  ->  accumulate
    /// ```
    ///
    /// The message is the only thing that survives the row.
    ///
    /// # Arguments
    ///
    /// - `operands`: one packed witness per operand the constraint reads.
    /// - `eq`: the zerocheck's equality weight for each row.
    /// - `composition`: the constraint being proved.
    ///
    /// # Panics
    ///
    /// Panics if the operand count disagrees with the constraint's arity.
    /// Panics if any operand's packed rows do not match the equality weights.
    #[must_use]
    pub fn stream_round_message<EF, C>(
        &self,
        operands: &[&[u8]],
        eq: &[EF],
        composition: &C,
    ) -> Vec<EF>
    where
        EF: ExtensionField<F>,
        F: Send + Sync,
        C: Composition<F> + Sync,
    {
        let arity = composition.arity();
        assert_eq!(operands.len(), arity, "one packed witness per operand");

        let stride = self.num_transmitted();
        let row_bytes = self.row_bytes();
        for rows in operands {
            assert_eq!(
                rows.len(),
                eq.len() * row_bytes,
                "one packed row per equality weight"
            );
        }

        // Rows are independent contributions to the same sum.
        //
        // Each thread carries its own message accumulator and its own row scratch.
        //
        // Nothing of row-by-transmitted size is ever allocated.
        (0..eq.len())
            .into_par_iter()
            .par_fold_reduce(
                || {
                    (
                        EF::zero_vec(stride),
                        F::zero_vec(arity * stride),
                        alloc::vec![F::ZERO; arity],
                    )
                },
                |(mut message, mut extended, mut tuple), row| {
                    // Extend this row of every operand onto the transmitted points.
                    for (operand, rows) in operands.iter().enumerate() {
                        let packed = &rows[row * row_bytes..][..row_bytes];
                        self.lde
                            .extend(packed, &mut extended[operand * stride..][..stride]);
                    }

                    // Read the constraint at each transmitted point and weigh the row once.
                    let weight = eq[row];
                    for (point, entry) in message.iter_mut().enumerate() {
                        for (operand, value) in tuple.iter_mut().enumerate() {
                            *value = extended[operand * stride + point];
                        }
                        *entry += weight * composition.eval(&tuple);
                    }

                    (message, extended, tuple)
                },
                |(mut left, extended, tuple), (right, _, _)| {
                    // Addition is associative, so regrouping the splits cannot change the sum.
                    for (entry, value) in left.iter_mut().zip(right) {
                        *entry += value;
                    }
                    (left, extended, tuple)
                },
            )
            .0
    }

    /// Weigh composed row values by the equality polynomial to form the round message.
    ///
    /// The streaming path never materialises the composed values, so a prover wants that one.
    ///
    /// This one serves callers already holding them, and is that path's reference.
    ///
    /// # Arguments
    ///
    /// - `composed`: the constraint's value at every transmitted point of every row.
    /// - `eq`: the zerocheck's equality weight for each row.
    ///
    /// # Returns
    ///
    /// The round polynomial's value at each transmitted point.
    ///
    /// # Panics
    ///
    /// Panics if the composed values are not one block per equality weight.
    #[must_use]
    pub fn round_message<EF>(&self, composed: &[F], eq: &[EF]) -> Vec<EF>
    where
        EF: ExtensionField<F>,
    {
        let stride = self.num_transmitted();
        assert_eq!(composed.len(), eq.len() * stride, "one block per row");

        // Rows are independent contributions to the same sum.
        //
        // Each thread keeps a private accumulator, and the partials are added at the end.
        composed
            .par_chunks_exact(stride)
            .zip(eq.par_iter())
            .par_fold_reduce(
                || EF::zero_vec(stride),
                |mut accumulator, (row, &weight)| {
                    // One row scales its whole block by that row's equality weight.
                    for (entry, &value) in accumulator.iter_mut().zip(row) {
                        *entry += weight * value;
                    }
                    accumulator
                },
                |mut left, right| {
                    // Addition is associative, so regrouping the splits cannot change the sum.
                    for (entry, value) in left.iter_mut().zip(right) {
                        *entry += value;
                    }
                    left
                },
            )
    }

    /// Read the round polynomial back at the verifier's challenge.
    ///
    /// # Algorithm
    ///
    /// The polynomial is pinned by its values on the whole extension.
    ///
    /// Those are zero on the subspace, and the transmitted values elsewhere.
    ///
    /// Interpolating a subspace has the barycentric form
    ///
    /// ```text
    ///     P(lam) = Z_T(lam) / Z_T'(T) * sum_j P(t_j) / (lam + t_j)
    /// ```
    ///
    /// and the subspace terms drop out because the polynomial is zero there.
    ///
    /// # Errors
    ///
    /// Returns an error when the message does not carry one value per transmitted point.
    ///
    /// The message comes from a proof, so the width is reported rather than asserted.
    pub fn evaluate<EF>(&self, message: &[EF], lambda: EF) -> Result<EF, MessageLenMismatch>
    where
        EF: ExtensionField<F>,
    {
        if message.len() != self.num_transmitted() {
            return Err(MessageLenMismatch {
                expected: self.num_transmitted(),
                actual: message.len(),
            });
        }

        // A challenge landing on a domain point reads the value there rather than dividing by zero.
        //
        // The domain lives in the base field, so a challenge outside it cannot land on a point.
        // Narrowing once keeps the search itself in the base field.
        if let Some(base) = lambda.as_base() {
            if let Some(index) = self.domain.transmitted().iter().position(|&t| t == base) {
                return Ok(message[index]);
            }
            if self.domain.subspace().contains(&base) {
                return Ok(EF::ZERO);
            }
        }

        // Off the domain every difference is invertible, so the barycentric form applies.
        //
        // The derivative is a base-field constant.
        //
        // Inverting and applying it there is cheaper than widening it first.
        let scale = vanishing_at(self.domain.log_extended(), lambda)
            * self.derivative_on_extension.inverse();

        // One inversion plus a few products per point, rather than one inversion each.
        let offsets = self
            .domain
            .transmitted()
            .iter()
            .map(|&point| lambda + point)
            .collect::<Vec<_>>();
        let inverses = batch_multiplicative_inverse(&offsets);

        Ok(scale
            * message
                .iter()
                .zip(&inverses)
                .map(|(&value, &inverse)| value * inverse)
                .sum::<EF>())
    }

    /// Build the table that binds packed rows at the verifier's challenge.
    ///
    /// Every row of every committed polynomial is bound through this one table.
    ///
    /// Its cost is therefore paid once per round rather than once per row.
    #[must_use]
    pub fn selector<EF>(&self, lambda: EF) -> RowSelector<EF>
    where
        EF: ExtensionField<F>,
    {
        RowSelector::new(&self.domain, lambda)
    }

    /// The Lagrange vector of the skipped subspace at one challenge, on its own.
    ///
    /// # Overview
    ///
    /// Entry `c` is the weight the skipped point `c` carries in a bound row.
    ///
    /// The byte table above is this vector summed over subsets.
    ///
    /// Only the prover reads rows through it.
    ///
    /// A verifier needs the vector alone, and this is how it gets it without paying for the
    /// 32 KiB of partial sums it would never touch.
    pub fn lagrange<EF>(&self, lambda: EF) -> Poly<EF>
    where
        EF: ExtensionField<F>,
    {
        Poly::new(lagrange_basis(&self.domain, lambda))
    }
}

/// The Lagrange basis of a subspace, read at one large-field point.
///
/// # Algorithm
///
/// Off the subspace the barycentric form applies:
///
/// ```text
///     L_c(lam) = Z_S(lam) / Z_S'(S) / (lam + s_c)
/// ```
///
/// A challenge landing on a subspace point makes that basis value one and the rest zero.
///
/// The barycentric form cannot express that, so that case is handled on its own.
///
/// # Performance
///
/// One inversion for the whole vector, plus a few products per point.
fn lagrange_basis<F, EF>(domain: &SkipDomain<F>, lambda: EF) -> Vec<EF>
where
    F: TowerLevel,
    EF: ExtensionField<F>,
{
    // The subspace lives in the base field, so narrowing once keeps the search there too.
    let hit = lambda
        .as_base()
        .and_then(|base| domain.subspace().iter().position(|&s| s == base));

    hit.map_or_else(
        || {
            // The derivative is a base-field constant.
            //
            // Inverting and applying it there is cheaper than widening it first.
            let scale =
                vanishing_at(domain.log_size(), lambda) * domain.derivative_on_subspace().inverse();

            // One inversion plus a few products per point, rather than one inversion each.
            let offsets = domain
                .subspace()
                .iter()
                .map(|&s| lambda + s)
                .collect::<Vec<_>>();
            batch_multiplicative_inverse(&offsets)
                .into_iter()
                .map(|inverse| scale * inverse)
                .collect::<Vec<_>>()
        },
        |hit| {
            (0..domain.size())
                .map(|index| if index == hit { EF::ONE } else { EF::ZERO })
                .collect::<Vec<_>>()
        },
    )
}

/// Evaluate the vanishing polynomial of a dimension-`log_size` subspace at a large-field point.
fn vanishing_at<EF: Field>(log_size: usize, x: EF) -> EF {
    // Square-and-add the recursion once per dimension, starting from the identity map.
    let mut value = x;
    for _ in 0..log_size {
        value = value.square() + value;
    }
    value
}

/// The table that reads packed rows at one fixed large-field point.
///
/// # Overview
///
/// After the skip round the residual sumcheck works over the rows read at the challenge:
///
/// ```text
///     f(x, lam) = sum_c bit_c(row x) * L_c(lam)
/// ```
///
/// The basis values are the same for every row.
///
/// They tabulate against a row's bytes exactly as the extension table does.
///
/// # Performance
///
/// The table is `2^k / 8 * 256` large-field elements.
///
/// That is 32 KiB for a six-bit skip over a 128-bit field.
///
/// Reading one row is then one lookup and one addition per byte, with no multiplications.
#[derive(Debug, Clone)]
pub struct RowSelector<EF> {
    /// One row of partial sums per chunk position, each indexed by that chunk's byte.
    table: Vec<EF>,
    /// Number of chunk positions, which is the number of bytes in a packed row.
    num_chunks: usize,
    /// The basis value at each subspace point, as a multilinear over the skipped variables.
    ///
    /// The table above is this vector summed over byte-sized subsets.
    ///
    /// The opening reduction needs the unaggregated form, and both must come from one place:
    /// two computations of the same Lagrange vector could disagree and nothing would catch it.
    lagrange: Poly<EF>,
}

impl<EF: Field> RowSelector<EF> {
    /// Tabulate the Lagrange basis of the subspace, read at one point.
    fn new<F>(domain: &SkipDomain<F>, lambda: EF) -> Self
    where
        F: TowerLevel,
        EF: ExtensionField<F>,
    {
        let num_chunks = domain.size() / CHUNK_BITS;

        // The basis value at each subspace point, read at the challenge.
        //
        // Shared with the verifier's own path, so the two can never disagree.
        let basis = lagrange_basis(domain, lambda);

        // Each chunk position gets the sums of every subset of the eight basis values it covers.
        //
        // Peeling the lowest set bit turns each entry into one addition.
        let mut table = EF::zero_vec(num_chunks * TABLE_ROWS);
        for (chunk, rows) in table.as_chunks_mut::<TABLE_ROWS>().0.iter_mut().enumerate() {
            for value in 1..1usize << CHUNK_BITS {
                let bit = value.trailing_zeros() as usize;
                rows[value] = rows[value & (value - 1)] + basis[chunk * CHUNK_BITS + bit];
            }
        }

        Self {
            table,
            num_chunks,
            lagrange: Poly::new(basis),
        }
    }

    /// Number of bytes one packed row occupies.
    #[must_use]
    pub const fn row_bytes(&self) -> usize {
        self.num_chunks
    }

    /// The Lagrange vector of the subspace, read at this round's challenge.
    ///
    /// Entry `c` is the weight the skipped point `c` carries in a bound row.
    ///
    /// The opening reduction consumes it, and takes it from here rather than recomputing it.
    pub const fn lagrange(&self) -> &Poly<EF> {
        &self.lagrange
    }

    /// Read one packed row at the point this table was built for.
    ///
    /// # Panics
    ///
    /// Panics if the row is not the tabulated number of bytes.
    #[must_use]
    pub fn read(&self, row: &[u8]) -> EF {
        assert_eq!(row.len(), self.num_chunks, "one byte per chunk");

        // Each byte selects the sum of the basis values its bits pick out.
        row.iter()
            .enumerate()
            .map(|(chunk, &byte)| self.table[chunk * TABLE_ROWS + usize::from(byte)])
            .sum()
    }

    /// Read every packed row of one polynomial, giving the multilinear the residual rounds fold.
    ///
    /// # Panics
    ///
    /// Panics if the input is not a whole number of rows.
    pub fn bind(&self, packed: &[u8]) -> Poly<EF>
    where
        EF: Send + Sync,
    {
        assert_eq!(packed.len() % self.num_chunks, 0, "whole rows");

        // Rows are independent, so the read splits across threads with no coordination.
        Poly::new(
            packed
                .par_chunks_exact(self.num_chunks)
                .map(|row| self.read(row))
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use p3_binary_field::{BinaryField8, BinaryField128};
    use p3_field::PrimeCharacteristicRing;
    use proptest::prelude::*;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::univariate_skip::Conjunction;

    /// The subspace domain lives in a byte field, and challenges in the 128-bit field above it.
    type F = BinaryField8;
    type EF = BinaryField128;

    /// Six skipped variables, matching the configuration a product-form zerocheck runs.
    const LOG_SKIP: usize = 6;

    /// A bit-valued witness satisfying the conjunction constraint, packed row by row.
    struct Witness {
        /// Packed rows of the first operand.
        a: Vec<u8>,
        /// Packed rows of the second operand.
        b: Vec<u8>,
        /// Packed rows of their conjunction.
        c: Vec<u8>,
    }

    /// Draw a witness whose cells satisfy the conjunction constraint everywhere.
    ///
    /// The conjunction of bit-valued multilinears is their product.
    ///
    /// The expression `a*b + c` therefore vanishes on the whole hypercube.
    fn random_witness(rng: &mut SmallRng, num_rows: usize, row_bytes: usize) -> Witness {
        // Two free operands, and the third pinned to their bitwise conjunction.
        let a = (0..num_rows * row_bytes)
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let b = (0..num_rows * row_bytes)
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let c = a.iter().zip(&b).map(|(&x, &y)| x & y).collect::<Vec<_>>();
        Witness { a, b, c }
    }

    /// The composition the round proves, at every transmitted point of every row.
    ///
    /// Characteristic two makes subtraction addition, so the constraint reads `a*b + c`.
    fn compose(round: &SkipRound<F>, witness: &Witness, num_rows: usize) -> Vec<F> {
        // Extend all three operands onto the transmitted points, staying in the byte field.
        let stride = round.num_transmitted();
        let extended = [&witness.a, &witness.b, &witness.c].map(|packed| {
            let mut out = F::zero_vec(num_rows * stride);
            round.extend_rows(packed, &mut out);
            out
        });

        // Combine them pointwise into the constraint's own values.
        (0..num_rows * stride)
            .map(|index| extended[0][index] * extended[1][index] + extended[2][index])
            .collect()
    }

    /// Read one packed bit-valued polynomial as its values over the hypercube.
    fn unpack(packed: &[u8]) -> Vec<F> {
        // Bit `i` of byte `i / 8` is the value at hypercube index `i`.
        (0..packed.len() * CHUNK_BITS)
            .map(|index| {
                let bit = (packed[index / CHUNK_BITS] >> (index % CHUNK_BITS)) & 1;
                if bit == 1 { F::ONE } else { F::ZERO }
            })
            .collect()
    }

    /// Weigh the bound rows by the equality polynomial, the way the residual sumcheck will.
    fn direct_sum(bound: &[Poly<EF>; 3], eq: &Poly<EF>, num_rows: usize) -> EF {
        // One term per row, each the constraint read at the challenge and scaled by its weight.
        (0..num_rows)
            .map(|row| {
                let a = bound[0].as_slice()[row];
                let b = bound[1].as_slice()[row];
                let c = bound[2].as_slice()[row];
                eq.as_slice()[row] * (a * b + c)
            })
            .sum()
    }

    #[test]
    fn streaming_the_message_agrees_with_materialising_it() {
        // Invariant: the streaming path is only a cheaper route to the same message.
        //
        //     materialised: rows * transmitted subfield values per operand, then weighed
        //     streamed:     one row's scratch per thread, weighed as it goes
        //
        // Fixture state: 2^5 rows of 2^6 bits, an odd-sized batch of three operands.
        let mut rng = SmallRng::seed_from_u64(0x57EA);
        let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
        let num_rows = 1 << 5;
        let witness = random_witness(&mut rng, num_rows, round.row_bytes());

        let r = (0..5).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
        let eq = Poly::new_from_point(&r, EF::ONE);

        let materialised = {
            let composed = compose(&round, &witness, num_rows);
            round.round_message::<EF>(&composed, eq.as_slice())
        };

        let streamed = round.stream_round_message::<EF, _>(
            &[&witness.a, &witness.b, &witness.c],
            eq.as_slice(),
            &Conjunction,
        );

        assert_eq!(streamed, materialised);
    }

    #[test]
    fn the_round_message_agrees_with_the_bound_polynomials() {
        // Invariant: what the verifier reads off the message is the sum the rounds then prove.
        //
        // This is the identity the whole reduction rests on.
        //
        //     evaluate(message, lam)  ==  sum_x eq(r, x) * (A(x)*B(x) + C(x))
        //
        // with A, B, C the committed rows read at lam.
        //
        // Fixture state: 2^4 rows of 2^6 bits, so a 10-variable zerocheck.
        let mut rng = SmallRng::seed_from_u64(0x5C1);
        let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
        let num_rows = 1 << 4;
        let witness = random_witness(&mut rng, num_rows, round.row_bytes());

        // The zerocheck challenge covers the variables the residual rounds will bind.
        let r = (0..4).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
        let eq = Poly::new_from_point(&r, EF::ONE);

        // Prover side: extend, compose, and weigh the rows.
        let composed = compose(&round, &witness, num_rows);
        let message = round.round_message::<EF>(&composed, eq.as_slice());

        // Verifier side: read the message back at a random challenge.
        let lambda = rng.random::<EF>();
        let read = round.evaluate(&message, lambda).unwrap();

        // One table binds every row of every committed polynomial at that same challenge.
        let selector = round.selector::<EF>(lambda);
        let bound = [&witness.a, &witness.b, &witness.c].map(|packed| selector.bind(packed));

        assert_eq!(read, direct_sum(&bound, &eq, num_rows));
    }

    /// The true round polynomial read on the skipped subspace, straight from the witness bits.
    ///
    /// On a subspace point a row's interpolating polynomial is just the bit stored there.
    ///
    /// The value is therefore the weighted sum of the raw constraint bits in that column.
    fn true_values_on_subspace(
        witness: &Witness,
        eq: &Poly<EF>,
        num_rows: usize,
        row_bytes: usize,
    ) -> Vec<EF> {
        // The constraint on bits is the conjunction exclusive-ored against the claimed output.
        let residue = (0..witness.a.len())
            .map(|index| (witness.a[index] & witness.b[index]) ^ witness.c[index])
            .collect::<Vec<_>>();

        // Column y collects its bit from every row, weighted by that row's equality weight.
        (0..row_bytes * CHUNK_BITS)
            .map(|column| {
                (0..num_rows)
                    .filter(|&row| {
                        let byte = residue[row * row_bytes + column / CHUNK_BITS];
                        (byte >> (column % CHUNK_BITS)) & 1 == 1
                    })
                    .map(|row| eq.as_slice()[row])
                    .sum()
            })
            .collect()
    }

    #[test]
    fn the_round_polynomial_vanishes_on_the_subspace_exactly_when_the_constraint_holds() {
        // Invariant: on the skipped subspace the round polynomial is the zerocheck's own claim.
        //
        // The verifier imposes that vanishing rather than checking it.
        //
        // It is the premise the whole round rests on, so it is pinned independently here.
        //
        // Fixture state: 2^3 rows of 2^6 bits, with the constraint satisfied everywhere.
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
        let num_rows = 1 << 3;
        let mut witness = random_witness(&mut rng, num_rows, round.row_bytes());

        let r = (0..3).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
        let eq = Poly::new_from_point(&r, EF::ONE);

        let satisfied = true_values_on_subspace(&witness, &eq, num_rows, round.row_bytes());
        assert!(satisfied.iter().all(|&value| value == EF::ZERO));

        // Mutation: flip one bit of the claimed conjunction.
        //
        //     c[cell] ^= 1   ->  the constraint is one at that cell, zero elsewhere
        //
        // Exactly the column holding that cell must stop vanishing.
        witness.c[0] ^= 1;
        let broken = true_values_on_subspace(&witness, &eq, num_rows, round.row_bytes());
        assert_eq!(broken[0], eq.as_slice()[0]);
        assert!(broken[1..].iter().all(|&value| value == EF::ZERO));
    }

    #[test]
    fn a_broken_constraint_breaks_the_identity_the_verifier_checks() {
        // The reconstruction is zero on the subspace whatever the prover sends.
        //
        // A broken witness therefore cannot show up there.
        //
        // It shows up instead as a gap between the message and the residual sum:
        //
        //     honest:  evaluate(message, lam)  ==  sum_x eq(r,x) * constraint(x, lam)
        //     broken:  the two differ, and the residual sumcheck rejects
        let mut rng = SmallRng::seed_from_u64(0xBADC0DE);
        let round = SkipRound::<F>::new(LOG_SKIP, 2).unwrap();
        let num_rows = 1 << 3;
        let mut witness = random_witness(&mut rng, num_rows, round.row_bytes());
        witness.c[0] ^= 1;

        let r = (0..3).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
        let eq = Poly::new_from_point(&r, EF::ONE);
        let composed = compose(&round, &witness, num_rows);
        let message = round.round_message::<EF>(&composed, eq.as_slice());

        let lambda = rng.random::<EF>();
        let selector = round.selector::<EF>(lambda);
        let bound = [&witness.a, &witness.b, &witness.c].map(|packed| selector.bind(packed));

        assert_ne!(
            round.evaluate(&message, lambda).unwrap(),
            direct_sum(&bound, &eq, num_rows)
        );
    }

    #[test]
    fn reading_the_message_on_a_transmitted_point_returns_that_value() {
        // A challenge landing on the domain has no barycentric form.
        //
        // The branch that reads the stored value directly is what keeps the evaluation total.
        let mut rng = SmallRng::seed_from_u64(0xD0A1);
        let round = SkipRound::<F>::new(4, 2).unwrap();
        let message = (0..round.num_transmitted())
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();

        for (index, &point) in round.domain().transmitted().iter().enumerate() {
            assert_eq!(
                round.evaluate(&message, EF::from(point)).unwrap(),
                message[index]
            );
        }
    }

    #[test]
    fn reading_the_message_on_the_subspace_returns_zero() {
        // This is the imposed vanishing written out as code, and the one place it appears.
        //
        // A random challenge reaches it with negligible probability.
        //
        // Replacing the zero with a stored entry passes every other test, so only this pins it.
        //
        //     nonzero message, challenge on any subspace point  ->  zero
        let mut rng = SmallRng::seed_from_u64(0x2E80);
        let round = SkipRound::<F>::new(4, 2).unwrap();
        let message = (0..round.num_transmitted())
            .map(|_| rng.random::<EF>())
            .collect::<Vec<_>>();
        assert!(message.iter().any(|&value| value != EF::ZERO));

        for &s in round.domain().subspace() {
            assert_eq!(
                round.evaluate(&message, EF::from(s)).unwrap(),
                EF::ZERO,
                "s={s:?}"
            );
        }
    }

    #[test]
    fn binding_on_a_subspace_point_recovers_the_original_bits() {
        // The rows are the values of a polynomial on the subspace.
        //
        // Reading one back at a subspace point must return the bit stored there.
        //
        //     selector at s_c  ->  the row's bit c, embedded in the large field
        let mut rng = SmallRng::seed_from_u64(0xB175);
        let round = SkipRound::<F>::new(4, 2).unwrap();
        let num_rows = 4;
        let packed = (0..num_rows * round.row_bytes())
            .map(|_| rng.random::<u8>())
            .collect::<Vec<_>>();
        let bits = unpack(&packed);

        for (column, &s) in round.domain().subspace().iter().enumerate() {
            let bound = round.selector::<EF>(EF::from(s)).bind(&packed);
            for row in 0..num_rows {
                let expected = bits[row * round.domain().size() + column];
                assert_eq!(bound.as_slice()[row], EF::from(expected), "c={column}");
            }
        }
    }

    #[test]
    fn rejects_a_skip_the_domain_or_the_table_cannot_host() {
        // A dimension-2 subspace is four points, so a row is not a whole byte.
        assert!(matches!(
            SkipRound::<F>::new(2, 2).unwrap_err(),
            SkipRoundError::Lde(_)
        ));
        // A degree-4 composition needs two extra dimensions, which a byte field cannot add to 7.
        assert!(matches!(
            SkipRound::<F>::new(7, 4).unwrap_err(),
            SkipRoundError::Domain(_)
        ));
    }

    proptest! {
        #[test]
        fn the_message_agrees_with_the_bound_polynomials_at_every_shape(
            seed: u64,
            log_skip in 3usize..=6,
            log_rows in 1usize..=4,
        ) {
            // Invariant: the defining identity holds at every skip width and trace height.
            //
            // It is not special to the product-form configuration.
            let mut rng = SmallRng::seed_from_u64(seed);
            let round = SkipRound::<F>::new(log_skip, 2).unwrap();
            let num_rows = 1 << log_rows;
            let witness = random_witness(&mut rng, num_rows, round.row_bytes());

            let r = (0..log_rows).map(|_| rng.random::<EF>()).collect::<Vec<_>>();
            let eq = Poly::new_from_point(&r, EF::ONE);
            let composed = compose(&round, &witness, num_rows);
            let message = round.round_message::<EF>(&composed, eq.as_slice());

            let lambda = rng.random::<EF>();
            let selector = round.selector::<EF>(lambda);
            let bound = [&witness.a, &witness.b, &witness.c].map(|p| selector.bind(p));

            prop_assert_eq!(
                round.evaluate(&message, lambda).unwrap(),
                direct_sum(&bound, &eq, num_rows)
            );
        }

        #[test]
        fn binding_is_each_row_read_at_the_challenge(seed: u64) {
            // Invariant: the tabulated read is the same map as the plain Lagrange evaluation.
            let mut rng = SmallRng::seed_from_u64(seed);
            let round = SkipRound::<F>::new(3, 2).unwrap();
            let num_rows = 4;
            let packed = (0..num_rows * round.row_bytes())
                .map(|_| rng.random::<u8>())
                .collect::<Vec<_>>();
            let bits = unpack(&packed);
            let lambda = rng.random::<EF>();

            let bound = round.selector::<EF>(lambda).bind(&packed);

            // The reference walks each row through the plain Lagrange formula.
            let size = round.domain().size();
            for row in 0..num_rows {
                let values = bits[row * size..][..size]
                    .iter()
                    .map(|&bit| EF::from(bit))
                    .collect::<Vec<_>>();
                prop_assert_eq!(
                    bound.as_slice()[row],
                    lagrange_reference(round.domain(), &values, lambda)
                );
            }
        }
    }

    /// Evaluate the polynomial taking the given large-field values on the subspace, at a point.
    fn lagrange_reference(domain: &SkipDomain<F>, values: &[EF], at: EF) -> EF {
        // A query on a subspace point reads the stored value rather than dividing by zero.
        if let Some(index) = at
            .as_base()
            .and_then(|base| domain.subspace().iter().position(|&s| s == base))
        {
            return values[index];
        }

        // Off the subspace the barycentric form applies, the derivative being constant throughout.
        let scale = vanishing_at(domain.log_size(), at)
            * EF::from(domain.derivative_on_subspace()).inverse();
        scale
            * values
                .iter()
                .zip(domain.subspace())
                .map(|(&value, &s)| value * (at + s).inverse())
                .sum::<EF>()
    }
}
