//! Private current-row AIR over a Cantor subspace, with monomial arithmetic.
//!
//! Interpolate each trace column as T0 and draw an independent scalar r for
//! T=T0+rW. Commit T and each unsplit Q=C(T,p)/W together, using independent
//! PCS tails. All commitments precede one common challenge z outside H.
//! T(z) is jointly uniform and Q(z) is determined by those values and p.
//!
//! The INTERACTIVE algebra/query error is at most the PCS bound plus
//! D/(2^128-N), where D=max(delta*(M-1), N+M-1) uses the committed message
//! degree M-1, not the honest trace/quotient degrees. Compilation checks
//! D*2^(s+2)<=2^128-N and selects q=4*(s+2). Hash binding/hiding, salt
//! guessing, Fiat–Shamir/RO-query/grinding and simulator programming losses
//! remain unproved. This is a reference prototype, not a production proof.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirLayout, BaseEntry, BaseLeaf, SymbolicExpression, SymbolicVariable};
use p3_challenger::{CanObserve, CanSample};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use rand::{CryptoRng, RngExt};
use serde::{Deserialize, Serialize};

use super::polynomial::{add, evaluate, exact_divide, interpolate, multiply, scale, vanishing};
use super::{
    Challenger, CurrentRowAirBuilder, ReferencePcsConfig, ReferenceZkPcs, Val, ZkCommitment,
    ZkError, ZkOpeningProof, challenger, observe_bytes, observe_usize,
};

type Expr = SymbolicExpression<Val>;
const VERSION: &[u8] = b"p3-binary-reference-zk-air-v1";

/// Checked trace geometry and requested interactive algebra/query target.
#[derive(Debug, Clone)]
pub struct ZkAirConfig {
    log_trace_height: usize,
    security_bits: usize,
}

impl ZkAirConfig {
    /// Trace logarithm 1..=8 and target 1..=96 bits. Compilation additionally
    /// limits width to 16, augmented constraints to 32, algebraic degree to 4,
    /// public values to 1024, and each expanded expression to 4096 nodes/128
    /// levels. The PCS allocation/work limits also apply to the whole batch.
    pub fn new(log_trace_height: usize, security_bits: usize) -> Result<Self, ZkError> {
        if !(1..=8).contains(&log_trace_height) || !(1..=96).contains(&security_bits) {
            return Err(ZkError::InvalidConfig(
                "AIR height or security target out of range",
            ));
        }
        Ok(Self {
            log_trace_height,
            security_bits,
        })
    }
}

/// Only the salted commitment and hiding PCS opening are transmitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZkAirProof {
    pub commitment: ZkCommitment,
    pub opening: ZkOpeningProof,
}

/// Compiled, verifier-owned static shape, symbolic constraints and PCS geometry.
///
/// Only current-row static AIR is supported: override the default
/// `main_next_row_columns` declaration with an empty vector. The interactive
/// error is the reference PCS bound plus D/(2^128-N), using committed degree
/// M-1 and D=max(delta*(M-1), N+M-1). Hash and Fiat–Shamir losses remain
/// unproved; this experimental target is not a production security claim.
#[derive(Debug, Clone)]
pub struct ReferenceZkAir {
    config: ZkAirConfig,
    width: usize,
    public_count: usize,
    constraints: Vec<Expr>,
    statement: Vec<u8>,
    vanishing: Vec<Val>,
    pcs: ReferenceZkPcs,
}

impl ReferenceZkAir {
    /// Compile current Main(offset=0), Public and Constant expressions only.
    /// Unsupported metadata and actual leaves are rejected. Leading Boolean
    /// columns receive explicit x*(x+1) constraints before sizing or binding.
    ///
    /// Permutation AIRs are rejected by the builder's capabilities, even when
    /// an empty permutation loop would emit no symbolic constraints:
    ///
    /// ```compile_fail,E0277
    /// use p3_air::{Air, BaseAir, PermutationAirBuilder};
    /// use p3_binary_pcs::zk::{ReferenceZkAir, Val, ZkAirConfig};
    /// struct EmptyPermutationLoop;
    /// impl BaseAir<Val> for EmptyPermutationLoop {
    ///     fn width(&self) -> usize { 1 }
    ///     fn main_next_row_columns(&self) -> Vec<usize> { vec![] }
    /// }
    /// impl<AB: PermutationAirBuilder<F = Val>> Air<AB> for EmptyPermutationLoop {
    ///     fn eval(&self, builder: &mut AB) {
    ///         for value in builder.permutation_values().to_vec() {
    ///             builder.assert_zero_ext(value);
    ///         }
    ///     }
    /// }
    /// let _ = ReferenceZkAir::compile(
    ///     &EmptyPermutationLoop, ZkAirConfig::new(1, 1).unwrap(),
    /// );
    /// ```
    ///
    /// Indexed permutation access is also rejected before evaluating the AIR:
    ///
    /// ```compile_fail,E0277
    /// use p3_air::{Air, BaseAir, PermutationAirBuilder, WindowAccess};
    /// use p3_binary_pcs::zk::{ReferenceZkAir, Val, ZkAirConfig};
    /// struct IndexedPermutation;
    /// impl BaseAir<Val> for IndexedPermutation {
    ///     fn width(&self) -> usize { 1 }
    ///     fn main_next_row_columns(&self) -> Vec<usize> { vec![] }
    /// }
    /// impl<AB: PermutationAirBuilder<F = Val>> Air<AB> for IndexedPermutation {
    ///     fn eval(&self, builder: &mut AB) {
    ///         let permutation = builder.permutation();
    ///         builder.assert_zero_ext(permutation.current_slice()[0]);
    ///     }
    /// }
    /// let _ = ReferenceZkAir::compile(
    ///     &IndexedPermutation, ZkAirConfig::new(1, 1).unwrap(),
    /// );
    /// ```
    pub fn compile<A: Air<CurrentRowAirBuilder>>(
        air: &A,
        config: ZkAirConfig,
    ) -> Result<Self, ZkError> {
        let width = air.width();
        let public_count = air.num_public_values();
        if !(1..=16).contains(&width) || public_count > 1024 {
            return Err(ZkError::InvalidConfig(
                "AIR width or public count out of range",
            ));
        }
        if air.preprocessed_width() != 0
            || air.preprocessed_trace().is_some()
            || air.num_periodic_columns() != 0
            || !air.periodic_columns().is_empty()
            || !air.periodic_periods().is_empty()
            || !air.periodic_statement().is_empty()
            || !air.main_next_row_columns().is_empty()
            || !air.preprocessed_next_row_columns().is_empty()
            || !air.public_boundary_io().is_empty()
        {
            return Err(ZkError::UnsupportedAir(
                "preprocessing, periodic, next-row or public-boundary declaration",
            ));
        }
        let boolean_columns = air.boolean_columns();
        if boolean_columns > width || (air.assumes_boolean_trace() && boolean_columns != width) {
            return Err(ZkError::UnsupportedAir(
                "inconsistent Boolean-column metadata",
            ));
        }
        let layout = AirLayout {
            main_width: width,
            num_public_values: public_count,
            ..Default::default()
        };
        let mut builder = CurrentRowAirBuilder::new(layout);
        air.eval(&mut builder);
        if builder.has_extension_constraints() {
            return Err(ZkError::UnsupportedAir("extension constraints"));
        }
        let mut constraints = builder.base_constraints();
        if constraints
            .len()
            .checked_add(boolean_columns)
            .is_none_or(|n| n > 32)
        {
            return Err(ZkError::InvalidConfig("too many augmented AIR constraints"));
        }
        for column in 0..boolean_columns {
            let x: Expr =
                SymbolicVariable::<Val>::new(BaseEntry::Main { offset: 0 }, column).into();
            constraints.push(x.clone() * (x + Val::ONE));
        }
        let mut max_degree = 0;
        for expr in &mut constraints {
            let (normal, degree) = normalize(expr, width, public_count, 0, &mut 4096)?;
            max_degree = max_degree.max(degree);
            *expr = normal;
        }
        let n = 1usize << config.log_trace_height;
        // deg T<=N; deg Q<=delta*N-N. Geometry never depends on witness degree.
        let message_len = (max_degree.saturating_sub(1) * n + 1).max(n + 1);
        let d = (max_degree * (message_len - 1)).max(n + message_len - 1);
        let denominator = u128::MAX - (n as u128 - 1);
        if (d as u128)
            .checked_mul(1u128 << (config.security_bits + 2))
            .is_none_or(|v| v > denominator)
        {
            return Err(ZkError::InvalidConfig(
                "committed-degree AIR soundness bound exceeds target",
            ));
        }
        let pcs_config = ReferencePcsConfig::new(message_len, 4 * (config.security_bits + 2))?;
        pcs_config.check_width(width + constraints.len())?;
        // Normalization removes cached degrees and pointer-sharing differences;
        // postcard's existing symbolic expression encoding fixes the AST bytes.
        let statement = postcard::to_allocvec(&constraints).map_err(|_| ZkError::Serialization)?;
        Ok(Self {
            vanishing: vanishing(config.log_trace_height),
            config,
            width,
            public_count,
            constraints,
            statement,
            pcs: ReferenceZkPcs::new(pcs_config),
        })
    }

    /// Prove a fixed-shape trace, using fresh private randomness for every mask
    /// and PCS commitment. Invalid shapes/public lengths fail before RNG draws.
    pub fn prove<R: CryptoRng>(
        &self,
        trace: RowMajorMatrix<Val>,
        public_values: &[Val],
        rng: &mut R,
    ) -> Result<ZkAirProof, ZkError> {
        self.check_public(public_values)?;
        let n = 1usize << self.config.log_trace_height;
        // Do not call height() on a possibly malformed/zero-width matrix.
        if trace.width != self.width || trace.values.len() != n * self.width {
            return Err(ZkError::InvalidShape("trace does not match compiled shape"));
        }
        let mut polynomials: Vec<Vec<Val>> = (0..self.width)
            .map(|column| {
                let values: Vec<_> = (0..n)
                    .map(|row| trace.values[row * self.width + column])
                    .collect();
                let unmasked = interpolate(&values, self.config.log_trace_height);
                add(&unmasked, &scale(&self.vanishing, rng.random()))
            })
            .collect();
        drop(trace);
        for constraint in &self.constraints {
            let numerator =
                polynomial_expression(constraint, &polynomials[..self.width], public_values);
            let quotient = exact_divide(numerator, &self.vanishing).ok_or(
                ZkError::InvalidWitness("constraint does not vanish on trace domain"),
            )?;
            polynomials.push(quotient);
        }
        let batch_width = polynomials.len();
        let m = self.pcs.config().message_len();
        let mut messages = vec![Val::ZERO; m * batch_width];
        for (column, polynomial) in polynomials.iter().enumerate() {
            for (row, &coefficient) in polynomial.iter().enumerate() {
                messages[row * batch_width + column] = coefficient;
            }
        }
        let (commitment, data) = self
            .pcs
            .commit(RowMajorMatrix::new(messages, batch_width), rng)?;
        let mut transcript = self.transcript(public_values);
        self.pcs.observe_commitment(&commitment, &mut transcript);
        let z = self.sample_point(&mut transcript);
        let opening = self.pcs.open(data, z, &mut transcript, rng)?;
        Ok(ZkAirProof {
            commitment,
            opening,
        })
    }

    /// Verify exact compiled shapes, all quotient identities, then PCS checks.
    pub fn verify(&self, proof: &ZkAirProof, public_values: &[Val]) -> Result<(), ZkError> {
        self.check_public(public_values)?;
        let batch_width = self.width + self.constraints.len();
        self.pcs.validate_commitment(&proof.commitment)?;
        if proof.commitment.num_polynomials != batch_width {
            return Err(ZkError::InvalidShape(
                "commitment batch does not match compiled AIR",
            ));
        }
        self.pcs.validate_proof(batch_width, &proof.opening)?;
        let mut transcript = self.transcript(public_values);
        self.pcs
            .observe_commitment(&proof.commitment, &mut transcript);
        let z = self.sample_point(&mut transcript);
        let wz = evaluate(&self.vanishing, z);
        let claims = &proof.opening.claimed_values;
        for (index, constraint) in self.constraints.iter().enumerate() {
            if scalar_expression(constraint, &claims[..self.width], public_values)
                != wz * claims[self.width + index]
            {
                return Err(ZkError::InvalidOpening("AIR quotient identity"));
            }
        }
        self.pcs
            .verify(&proof.commitment, z, &proof.opening, &mut transcript)?;
        Ok(())
    }

    const fn check_public(&self, values: &[Val]) -> Result<(), ZkError> {
        if values.len() != self.public_count {
            return Err(ZkError::InvalidShape("public input count mismatch"));
        }
        Ok(())
    }

    fn transcript(&self, public_values: &[Val]) -> Challenger {
        let mut transcript = challenger();
        observe_bytes(&mut transcript, VERSION);
        for value in [
            self.config.log_trace_height,
            self.config.security_bits,
            self.width,
            self.public_count,
            self.constraints.len(),
        ] {
            observe_usize(&mut transcript, value);
        }
        observe_bytes(&mut transcript, &self.statement);
        observe_usize(&mut transcript, public_values.len());
        transcript.observe_slice(public_values);
        transcript
    }

    fn sample_point(&self, transcript: &mut Challenger) -> Val {
        // Public rejection sampling: conditional uniformity over F\H. The
        // identical fixed procedure is used by prover and verifier.
        loop {
            let point: Val = transcript.sample();
            if evaluate(&self.vanishing, point) != Val::ZERO {
                return point;
            }
        }
    }
}

fn normalize(
    expr: &Expr,
    width: usize,
    public_count: usize,
    depth: usize,
    remaining: &mut usize,
) -> Result<(Expr, usize), ZkError> {
    if depth >= 128 || *remaining == 0 {
        return Err(ZkError::UnsupportedAir(
            "expression exceeds reference resource limits",
        ));
    }
    *remaining -= 1;
    let result = match expr {
        Expr::Leaf(BaseLeaf::Constant(c)) => (Expr::Leaf(BaseLeaf::Constant(*c)), 0),
        Expr::Leaf(BaseLeaf::Variable(v)) => {
            let degree = match v.entry {
                BaseEntry::Main { offset: 0 } if v.index < width => 1,
                BaseEntry::Public if v.index < public_count => 0,
                _ => {
                    return Err(ZkError::UnsupportedAir(
                        "unsupported or out-of-range symbolic variable",
                    ));
                }
            };
            (Expr::Leaf(BaseLeaf::Variable(*v)), degree)
        }
        Expr::Leaf(_) => return Err(ZkError::UnsupportedAir("row selectors")),
        Expr::Neg { x, .. } => {
            let (x, degree) = normalize(x, width, public_count, depth + 1, remaining)?;
            (
                Expr::Neg {
                    x: Arc::new(x),
                    degree_multiple: degree,
                },
                degree,
            )
        }
        Expr::Add { x, y, .. } | Expr::Sub { x, y, .. } | Expr::Mul { x, y, .. } => {
            let (x, dx) = normalize(x, width, public_count, depth + 1, remaining)?;
            let (y, dy) = normalize(y, width, public_count, depth + 1, remaining)?;
            let degree = if matches!(expr, Expr::Mul { .. }) {
                dx + dy
            } else {
                dx.max(dy)
            };
            let (x, y) = (Arc::new(x), Arc::new(y));
            let normal = match expr {
                Expr::Add { .. } => Expr::Add {
                    x,
                    y,
                    degree_multiple: degree,
                },
                Expr::Sub { .. } => Expr::Sub {
                    x,
                    y,
                    degree_multiple: degree,
                },
                _ => Expr::Mul {
                    x,
                    y,
                    degree_multiple: degree,
                },
            };
            (normal, degree)
        }
    };
    if result.1 > 4 {
        return Err(ZkError::UnsupportedAir(
            "algebraic constraint degree exceeds four",
        ));
    }
    Ok(result)
}

fn polynomial_expression(expr: &Expr, main: &[Vec<Val>], public: &[Val]) -> Vec<Val> {
    match expr {
        Expr::Leaf(BaseLeaf::Constant(c)) => vec![*c],
        Expr::Leaf(BaseLeaf::Variable(v)) => match v.entry {
            BaseEntry::Main { .. } => main[v.index].clone(),
            _ => vec![public[v.index]],
        },
        Expr::Add { x, y, .. } | Expr::Sub { x, y, .. } => add(
            &polynomial_expression(x, main, public),
            &polynomial_expression(y, main, public),
        ),
        Expr::Mul { x, y, .. } => multiply(
            &polynomial_expression(x, main, public),
            &polynomial_expression(y, main, public),
        ),
        Expr::Neg { x, .. } => polynomial_expression(x, main, public),
        Expr::Leaf(_) => unreachable!("compiled AIR leaves were validated"),
    }
}

fn scalar_expression(expr: &Expr, main: &[Val], public: &[Val]) -> Val {
    match expr {
        Expr::Leaf(BaseLeaf::Constant(c)) => *c,
        Expr::Leaf(BaseLeaf::Variable(v)) => match v.entry {
            BaseEntry::Main { .. } => main[v.index],
            _ => public[v.index],
        },
        Expr::Add { x, y, .. } | Expr::Sub { x, y, .. } => {
            scalar_expression(x, main, public) + scalar_expression(y, main, public)
        }
        Expr::Mul { x, y, .. } => {
            scalar_expression(x, main, public) * scalar_expression(y, main, public)
        }
        Expr::Neg { x, .. } => scalar_expression(x, main, public),
        Expr::Leaf(_) => unreachable!("compiled AIR leaves were validated"),
    }
}
