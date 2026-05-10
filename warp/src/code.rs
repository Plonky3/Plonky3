//! Reed-Solomon code for WARP.
//!
//! WARP is generic over any linear code, but for this implementation we only
//! support smooth Reed-Solomon codes evaluated on a multiplicative subgroup of
//! `F^*` of order `n`.
//!
//! Two layouts are supported:
//!
//! - [`ReedSolomonEncoding::Coefficient`]: treat the witness `w` as the
//!   coefficient vector of a degree-`<k` polynomial, zero-pad it to length `n`,
//!   and DFT it.
//! - [`ReedSolomonEncoding::Systematic`]: treat `w` as evaluations on the
//!   canonical subgroup of size `k`, and compute its low-degree extension to
//!   the canonical subgroup of size `n`.
//!
//! The systematic layout is the one needed by WARP's efficient §6.3
//! twin-constraint prover: the paper notes that, for systematic codes, the
//! folded witness `ŵ` is equivalent to a fixed subset of the folded codeword
//! `F̂`. It is also the natural layout for a WHIR-native finalizer, because the
//! final PESAT check can be reduced to openings of the same Reed-Solomon
//! oracle instead of proving a separate coefficient/codeword consistency proof.
//!
//! The corresponding multilinear extension `f̂` of the codeword is then the
//! standard MLE of the evaluation vector — exactly what
//! [`p3_multilinear_util::poly::Poly`] handles.

use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_dft::TwoAdicSubgroupDft;
use p3_field::{BasedVectorSpace, ExtensionField, Field, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_multilinear_util::point::Point;
use p3_multilinear_util::poly::Poly;

/// Encoding layout for the WARP Reed-Solomon code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReedSolomonEncoding {
    /// Message entries are polynomial coefficients.
    Coefficient,
    /// Message entries are evaluations on the size-`k` subgroup and are
    /// embedded in the size-`n` codeword at indices `i * (n / k)`.
    Systematic,
}

/// A smooth Reed-Solomon code with message length `2^log_msg` and codeword
/// length `2^(log_msg + log_inv_rate)`.
///
/// This is intentionally a thin wrapper around a Plonky3 DFT implementation.
/// See [`ReedSolomonEncoding`] for the available message layouts.
///
/// # Layout
///
/// - The codeword is the evaluation of the message-as-polynomial on the
///   smooth domain `L = {ω^i : i ∈ [n]}` for the codeword-length-th root of
///   unity ω.
/// - As a multilinear polynomial in `log n` variables, the evaluation vector
///   is interpreted via the Plonky3 hypercube indexing convention used by
///   [`Poly::new_from_point`](p3_multilinear_util::poly::Poly::new_from_point).
#[derive(Clone, Debug)]
pub struct ReedSolomonCode<F, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    log_msg: usize,
    log_inv_rate: usize,
    encoding: ReedSolomonEncoding,
    dft: Dft,
    _ph: PhantomData<F>,
}

impl<F, Dft> ReedSolomonCode<F, Dft>
where
    F: TwoAdicField,
    Dft: TwoAdicSubgroupDft<F>,
{
    /// Create a Reed-Solomon code with the given log message length and log
    /// inverse rate, using coefficient-form encoding.
    ///
    /// # Panics
    ///
    /// - `log_msg + log_inv_rate` must not exceed `F::TWO_ADICITY` (the
    ///   smooth-domain root of unity must exist in `F`).
    pub fn new(log_msg: usize, log_inv_rate: usize, dft: Dft) -> Self {
        Self::new_with_encoding(log_msg, log_inv_rate, dft, ReedSolomonEncoding::Coefficient)
    }

    /// Create a Reed-Solomon code with coefficient-form encoding.
    pub fn new_coefficient(log_msg: usize, log_inv_rate: usize, dft: Dft) -> Self {
        Self::new_with_encoding(log_msg, log_inv_rate, dft, ReedSolomonEncoding::Coefficient)
    }

    /// Create a Reed-Solomon code with systematic encoding.
    pub fn new_systematic(log_msg: usize, log_inv_rate: usize, dft: Dft) -> Self {
        Self::new_with_encoding(log_msg, log_inv_rate, dft, ReedSolomonEncoding::Systematic)
    }

    /// Create a Reed-Solomon code with an explicit encoding layout.
    pub fn new_with_encoding(
        log_msg: usize,
        log_inv_rate: usize,
        dft: Dft,
        encoding: ReedSolomonEncoding,
    ) -> Self {
        assert!(
            log_msg + log_inv_rate <= F::TWO_ADICITY,
            "RS domain size 2^{} exceeds field 2-adicity {}",
            log_msg + log_inv_rate,
            F::TWO_ADICITY,
        );
        Self {
            log_msg,
            log_inv_rate,
            encoding,
            dft,
            _ph: PhantomData,
        }
    }

    /// Returns the log message length `log k`.
    #[inline]
    pub const fn log_msg_len(&self) -> usize {
        self.log_msg
    }

    /// Returns the log codeword length `log n = log k + log(1/ρ)`.
    #[inline]
    pub const fn log_codeword_len(&self) -> usize {
        self.log_msg + self.log_inv_rate
    }

    /// Returns the message length `k = 2^log_msg`.
    #[inline]
    pub const fn msg_len(&self) -> usize {
        1 << self.log_msg
    }

    /// Returns the codeword length `n = 2^log_codeword_len`.
    #[inline]
    pub const fn codeword_len(&self) -> usize {
        1 << (self.log_msg + self.log_inv_rate)
    }

    /// Returns the log inverse rate `log(1/ρ)`.
    #[inline]
    pub const fn log_inv_rate(&self) -> usize {
        self.log_inv_rate
    }

    /// Returns the encoding layout.
    #[inline]
    pub const fn encoding(&self) -> ReedSolomonEncoding {
        self.encoding
    }

    /// Returns true when this code uses systematic encoding.
    #[inline]
    pub const fn is_systematic(&self) -> bool {
        matches!(self.encoding, ReedSolomonEncoding::Systematic)
    }

    /// Return the codeword index containing message symbol `message_index` in
    /// systematic mode.
    ///
    /// For the canonical nested two-adic domains used by Plonky3, the size-`k`
    /// subgroup is embedded in the size-`n` subgroup at stride `n / k`.
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Systematic`].
    /// - `message_index < msg_len()`.
    pub fn systematic_codeword_index(&self, message_index: usize) -> usize {
        assert!(
            self.is_systematic(),
            "systematic_codeword_index requires systematic encoding"
        );
        assert!(message_index < self.msg_len(), "message index out of range");
        message_index << self.log_inv_rate
    }

    /// Lift a point in the message MLE domain to the codeword MLE subspace
    /// carrying the systematic symbols.
    ///
    /// In systematic mode, the message entry at index `i` is stored at
    /// codeword index `i << log_inv_rate`. With Plonky3's big-endian Boolean
    /// hypercube indexing this means the message variables are the prefix
    /// variables and the added inverse-rate variables are fixed to zero:
    ///
    /// ```text
    ///     W_hat(y) = F_hat(y, 0, ..., 0).
    /// ```
    ///
    /// This is the bridge between WARP's systematic twin-constraint statement
    /// (Construction 6.3) and WHIR's constrained-RS opening statements
    /// `Z * eq(z, .)`: any terminal witness opening can be expressed as an
    /// opening of the same RS codeword oracle.
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Systematic`].
    /// - `message_point.len() == log_msg_len()`.
    pub fn systematic_message_point<EF: Field>(&self, message_point: &[EF]) -> Point<EF> {
        assert!(
            self.is_systematic(),
            "systematic_message_point requires systematic encoding"
        );
        assert_eq!(
            message_point.len(),
            self.log_msg_len(),
            "message point dimension"
        );
        let mut codeword_point = Vec::with_capacity(self.log_codeword_len());
        codeword_point.extend_from_slice(message_point);
        codeword_point.extend(EF::zero_vec(self.log_inv_rate()));
        Point::new(codeword_point)
    }

    /// Return the Boolean codeword point containing `message_index` in
    /// systematic mode.
    ///
    /// This is equivalent to
    /// `Point::hypercube(systematic_codeword_index(message_index), log_n)`,
    /// but documents the RS layout boundary used by WARP/WHIR integration.
    pub fn systematic_message_index_point<EF: Field>(&self, message_index: usize) -> Point<EF> {
        let message_point = Point::<EF>::hypercube(message_index, self.log_msg_len());
        self.systematic_message_point(message_point.as_slice())
    }

    /// Return the linear weights expressing one systematic-codeword entry as
    /// a weighted sum of the message evaluations.
    ///
    /// In systematic mode the message `w` is interpreted as evaluations of a
    /// degree-`< k` polynomial on the size-`k` two-adic subgroup, and the
    /// codeword is its low-degree extension to the size-`n` subgroup. For a
    /// codeword-domain point `x = omega_n^index`, this returns Lagrange
    /// weights `lambda_i(x)` such that:
    ///
    /// ```text
    ///     C(w)[index] = sum_i lambda_i(x) * w[i].
    /// ```
    ///
    /// This is the algebraic bridge used by the WHIR-backed WARP root proof to
    /// avoid committing the already RS-encoded WARP codeword as a second WHIR
    /// message. The WHIR commitment is to `w`; WARP's shift query is compiled
    /// into this linear Sigma claim over `w`.
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Systematic`].
    /// - `codeword_index < codeword_len()`.
    pub fn systematic_codeword_index_weights<EF: ExtensionField<F>>(
        &self,
        codeword_index: usize,
    ) -> Vec<EF> {
        assert!(
            self.is_systematic(),
            "systematic_codeword_index_weights requires systematic encoding"
        );
        assert!(
            codeword_index < self.codeword_len(),
            "codeword index out of range"
        );

        let stride = 1 << self.log_inv_rate();
        if codeword_index.is_multiple_of(stride) {
            let message_index = codeword_index / stride;
            let mut weights = EF::zero_vec(self.msg_len());
            weights[message_index] = EF::ONE;
            return weights;
        }

        let x = F::two_adic_generator(self.log_codeword_len()).exp_u64(codeword_index as u64);
        let x_ext = EF::from(x);
        let vanish = EF::from(x.exp_power_of_2(self.log_msg_len()) - F::ONE);
        let inv_k = EF::from(F::from_usize(self.msg_len()).inverse());
        let message_generator = F::two_adic_generator(self.log_msg_len());

        let mut weights = Vec::with_capacity(self.msg_len());
        let mut domain_point = F::ONE;
        for _ in 0..self.msg_len() {
            let denominator = x_ext - EF::from(domain_point);
            weights.push(vanish * EF::from(domain_point) * inv_k * denominator.inverse());
            domain_point *= message_generator;
        }
        weights
    }

    /// Return the RS-domain point used at a codeword index.
    ///
    /// The WHIR paper's smooth Reed-Solomon code evaluates one degree-`< k`
    /// polynomial on the size-`n` two-adic domain. WARP's RS-specialized code
    /// uses the same object for its codeword oracle. This helper exposes the
    /// shared domain point `omega_n^index` so WARP codeword openings can be
    /// compiled as constrained-RS claims, not as claims against a second code.
    ///
    /// # Panics
    ///
    /// - `codeword_index < codeword_len()`.
    pub fn rs_domain_point(&self, codeword_index: usize) -> F {
        assert!(
            codeword_index < self.codeword_len(),
            "codeword index out of range"
        );
        F::two_adic_generator(self.log_codeword_len()).exp_u64(codeword_index as u64)
    }

    /// Return the coefficient-form RS weights for one codeword entry.
    ///
    /// In coefficient layout, the message is the coefficient vector
    /// `w = (w_0, ..., w_{k-1})` and the codeword entry at
    /// `x = omega_n^index` is
    ///
    /// ```text
    ///     C(w)[index] = sum_{j=0}^{k-1} w_j x^j.
    /// ```
    ///
    /// These are exactly WHIR's select weights for the initial RS polynomial.
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Coefficient`].
    /// - `codeword_index < codeword_len()`.
    pub fn coefficient_codeword_index_weights<EF: ExtensionField<F>>(
        &self,
        codeword_index: usize,
    ) -> Vec<EF> {
        assert!(
            matches!(self.encoding, ReedSolomonEncoding::Coefficient),
            "coefficient_codeword_index_weights requires coefficient encoding"
        );
        let x = EF::from(self.rs_domain_point(codeword_index));
        x.powers().take(self.msg_len()).collect()
    }

    /// Return the RS-message weights for one codeword entry.
    ///
    /// This is the layout-polymorphic WARP/WHIR boundary:
    ///
    /// - coefficient RS uses WHIR select/monomial weights `x^j`;
    /// - systematic RS uses the Lagrange basis on the size-`k` subgroup.
    ///
    /// Both cases are the same single RS code `C`; only the chosen coordinates
    /// for `C^{-1}` differ.
    pub fn codeword_index_weights<EF: ExtensionField<F>>(&self, codeword_index: usize) -> Vec<EF> {
        match self.encoding {
            ReedSolomonEncoding::Coefficient => {
                self.coefficient_codeword_index_weights(codeword_index)
            }
            ReedSolomonEncoding::Systematic => {
                self.systematic_codeword_index_weights(codeword_index)
            }
        }
    }

    /// Return the linear weights expressing an arbitrary codeword-MLE opening
    /// as a weighted sum of the systematic message evaluations.
    ///
    /// For systematic RS encoding, the codeword vector is
    ///
    /// ```text
    ///     f = DFT_n(pad(IDFT_k(w))).
    /// ```
    ///
    /// A codeword-MLE claim at point `z` is an inner product
    /// `eq_z · f`. The adjoint map is therefore
    /// `IDFT_k(DFT_n(eq_z)[..k])`, which avoids materializing the dense
    /// `n x k` LDE matrix. This is the general version of
    /// [`Self::systematic_codeword_index_weights`] used for WARP's final
    /// accumulator claim `f_hat(alpha) = mu`.
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Systematic`].
    /// - `codeword_point.len() == log_codeword_len()`.
    pub fn systematic_codeword_mle_weights<EF>(&self, codeword_point: &[EF]) -> Vec<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert!(
            self.is_systematic(),
            "systematic_codeword_mle_weights requires systematic encoding"
        );
        assert_eq!(
            codeword_point.len(),
            self.log_codeword_len(),
            "codeword MLE point dimension"
        );

        let eq = Poly::<EF>::new_from_point(codeword_point, EF::ONE);
        self.systematic_codeword_query_weights(eq.as_slice())
    }

    /// Return the RS-message weights for an arbitrary codeword-MLE query.
    ///
    /// For a codeword-domain query vector `q`, this computes the adjoint
    /// `C^T q` in the coordinates selected by [`ReedSolomonEncoding`]. This is
    /// still one RS code: the operation is only the verifier-side way to
    /// express WARP's `u in C` and `u_hat(alpha)=mu` obligations over the WHIR
    /// initial polynomial committed to `C^{-1}(u)`.
    pub fn codeword_mle_weights<EF>(&self, codeword_point: &[EF]) -> Vec<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert_eq!(
            codeword_point.len(),
            self.log_codeword_len(),
            "codeword MLE point dimension"
        );
        let eq = Poly::<EF>::new_from_point(codeword_point, EF::ONE);
        self.codeword_query_weights(eq.as_slice())
    }

    /// Apply the adjoint of the systematic RS encoder to a codeword-domain
    /// linear query.
    ///
    /// If `A` is the systematic encoder from message evaluations to codeword
    /// evaluations, this returns `A^T q` for `q` over the codeword domain. It
    /// is the reusable primitive behind WARP's WHIR compiler: many codeword
    /// openings can first be batched into one sparse/dense query `q`, then
    /// lowered to a single linear-Sigma claim over the committed message.
    ///
    /// Algebraically, for systematic encoding
    ///
    /// ```text
    ///     A = DFT_n · pad · IDFT_k,
    ///     A^T q = IDFT_k(DFT_n(q)[..k]).
    /// ```
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Systematic`].
    /// - `codeword_query.len() == codeword_len()`.
    pub fn systematic_codeword_query_weights<EF>(&self, codeword_query: &[EF]) -> Vec<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert!(
            self.is_systematic(),
            "systematic_codeword_query_weights requires systematic encoding"
        );
        assert_eq!(
            codeword_query.len(),
            self.codeword_len(),
            "codeword query length mismatch"
        );

        let mut spectral = self.dft.dft_algebra(codeword_query.to_vec());
        spectral.truncate(self.msg_len());
        self.dft.idft_algebra(spectral)
    }

    /// Apply the adjoint of this RS encoder to a codeword-domain query.
    ///
    /// This generalizes [`Self::systematic_codeword_query_weights`] to both RS
    /// coordinate layouts:
    ///
    /// - coefficient RS: `C = DFT_n · pad`, hence `C^T q = DFT_n(q)[..k]`;
    /// - systematic RS: `C = DFT_n · pad · IDFT_k`, hence
    ///   `C^T q = IDFT_k(DFT_n(q)[..k])`.
    pub fn codeword_query_weights<EF>(&self, codeword_query: &[EF]) -> Vec<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert_eq!(
            codeword_query.len(),
            self.codeword_len(),
            "codeword query length mismatch"
        );
        let mut spectral = self.dft.dft_algebra(codeword_query.to_vec());
        spectral.truncate(self.msg_len());
        match self.encoding {
            ReedSolomonEncoding::Coefficient => spectral,
            ReedSolomonEncoding::Systematic => self.dft.idft_algebra(spectral),
        }
    }

    /// Batched version of [`Self::systematic_codeword_query_weights`].
    ///
    /// The input matrix has one codeword-domain query per column. The output
    /// matrix has one message-domain adjoint query per corresponding column.
    pub fn systematic_codeword_query_weights_batch<EF>(
        &self,
        codeword_queries: RowMajorMatrix<EF>,
    ) -> RowMajorMatrix<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert!(
            self.is_systematic(),
            "systematic_codeword_query_weights_batch requires systematic encoding"
        );
        assert_eq!(
            codeword_queries.height(),
            self.codeword_len(),
            "codeword query height mismatch"
        );

        let width = codeword_queries.width();
        let mut spectral = self.dft.dft_algebra_batch(codeword_queries);
        spectral.values.truncate(self.msg_len() * width);
        self.dft
            .idft_algebra_batch(RowMajorMatrix::new(spectral.values, width))
    }

    /// Batched version of [`Self::codeword_query_weights`].
    pub fn codeword_query_weights_batch<EF>(
        &self,
        codeword_queries: RowMajorMatrix<EF>,
    ) -> RowMajorMatrix<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert_eq!(
            codeword_queries.height(),
            self.codeword_len(),
            "codeword query height mismatch"
        );
        let width = codeword_queries.width();
        let mut spectral = self.dft.dft_algebra_batch(codeword_queries);
        spectral.values.truncate(self.msg_len() * width);
        match self.encoding {
            ReedSolomonEncoding::Coefficient => RowMajorMatrix::new(spectral.values, width),
            ReedSolomonEncoding::Systematic => self
                .dft
                .idft_algebra_batch(RowMajorMatrix::new(spectral.values, width)),
        }
    }

    /// Extract the systematic message from a codeword.
    ///
    /// This is valid for systematic encoding because message entry `i` is stored
    /// at codeword index `i << log_inv_rate`.
    pub fn systematic_message_from_codeword<T: Clone>(&self, codeword: &[T]) -> Vec<T> {
        assert!(
            self.is_systematic(),
            "systematic_message_from_codeword requires systematic encoding"
        );
        assert_eq!(
            codeword.len(),
            self.codeword_len(),
            "codeword length mismatch"
        );
        (0..self.msg_len())
            .map(|i| codeword[self.systematic_codeword_index(i)].clone())
            .collect()
    }

    /// Recover the RS message coordinates from a codeword.
    ///
    /// For systematic RS this extracts the embedded subgroup evaluations. For
    /// coefficient RS this performs one inverse DFT over the size-`n` domain
    /// and truncates to the degree-`< k` coefficient vector. This is used by
    /// the WHIR-backed WARP root proof to make the accumulator oracle and WHIR
    /// oracle share one RS code instead of committing an already encoded
    /// codeword as a fresh WHIR message.
    pub fn message_from_codeword<V>(&self, codeword: &[V]) -> Vec<V>
    where
        V: BasedVectorSpace<F> + Clone + Send + Sync,
    {
        assert_eq!(
            codeword.len(),
            self.codeword_len(),
            "codeword length mismatch"
        );
        match self.encoding {
            ReedSolomonEncoding::Systematic => (0..self.msg_len())
                .map(|i| codeword[self.systematic_codeword_index(i)].clone())
                .collect(),
            ReedSolomonEncoding::Coefficient => {
                let mut message = self.dft.idft_algebra(codeword.to_vec());
                message.truncate(self.msg_len());
                message
            }
        }
    }

    /// Encode a base-field message `w ∈ F^k` into a codeword `f ∈ F^n`.
    ///
    /// # Panics
    ///
    /// - `w.len()` must equal `msg_len()`.
    pub fn encode(&self, w: &[F]) -> Vec<F> {
        assert_eq!(w.len(), self.msg_len(), "message length mismatch");
        self.encode_batch_matrix(RowMajorMatrix::new_col(w.to_vec()))
            .values
    }

    /// Encode a batch of base-field messages arranged as matrix columns.
    ///
    /// The input matrix has height `k` and width equal to the batch size; the
    /// returned matrix has height `n` and the same width. This is the reusable
    /// primitive used by WARP's stacked fresh-codeword commitment and mirrors
    /// the column-batched DFT style used by WHIR's committer.
    ///
    /// # Panics
    ///
    /// - `messages.height()` must equal `msg_len()`.
    pub fn encode_batch_matrix(&self, messages: RowMajorMatrix<F>) -> RowMajorMatrix<F> {
        assert_eq!(messages.height(), self.msg_len(), "message height mismatch");
        let width = messages.width();
        let out = match self.encoding {
            ReedSolomonEncoding::Coefficient => {
                // Zero-pad coefficient vector to codeword length and DFT.
                let mut padded = F::zero_vec(self.codeword_len() * width);
                for row in 0..self.msg_len() {
                    let src = &messages.values[row * width..(row + 1) * width];
                    let dst = &mut padded[row * width..(row + 1) * width];
                    dst.copy_from_slice(src);
                }
                self.dft
                    .dft_batch(RowMajorMatrix::new(padded, width))
                    .to_row_major_matrix()
            }
            ReedSolomonEncoding::Systematic => {
                // Low-degree extend evaluations on H_k to evaluations on H_n.
                self.dft
                    .lde_batch(messages, self.log_inv_rate)
                    .to_row_major_matrix()
            }
        };
        debug_assert_eq!(out.height(), self.codeword_len());
        debug_assert_eq!(out.width(), width);
        out
    }

    /// Encode a batch of base-field messages supplied as separate vectors.
    ///
    /// The output rows are codeword positions and columns are input messages.
    pub fn encode_batch(&self, messages: &[Vec<F>]) -> RowMajorMatrix<F> {
        assert!(!messages.is_empty(), "empty RS batch");
        let width = messages.len();
        let mut values = F::zero_vec(self.msg_len() * width);
        for (col, message) in messages.iter().enumerate() {
            assert_eq!(message.len(), self.msg_len(), "message length mismatch");
            for row in 0..self.msg_len() {
                values[row * width + col] = message[row];
            }
        }
        self.encode_batch_matrix(RowMajorMatrix::new(values, width))
    }

    /// Encode an extension-field message `w ∈ EF^k` into a codeword `f ∈ EF^n`.
    pub fn encode_algebra<EF: ExtensionField<F>>(&self, w: &[EF]) -> Vec<EF> {
        assert_eq!(w.len(), self.msg_len(), "message length mismatch");
        self.encode_algebra_batch_matrix(RowMajorMatrix::new_col(w.to_vec()))
            .values
    }

    /// Evaluate a systematic message on one codeword coset only.
    ///
    /// For systematic encoding, the length-`n` domain decomposes into
    /// `n / k` cosets of the message subgroup:
    ///
    /// ```text
    ///     H_n = ⋃_{r=0}^{n/k-1} ω_n^r H_k.
    /// ```
    ///
    /// Full encoding computes every coset. Verifier reductions often need only
    /// a few non-systematic codeword indices, so this helper reuses the same
    /// Plonky3 coset-LDE primitive as the full encoder but returns only the
    /// requested coset. Coset `0` is the systematic message itself.
    ///
    /// # Panics
    ///
    /// - This code must use [`ReedSolomonEncoding::Systematic`].
    /// - `w.len() == msg_len()`.
    /// - `coset_index < 2^log_inv_rate`.
    pub fn encode_algebra_systematic_coset<EF>(&self, w: &[EF], coset_index: usize) -> Vec<EF>
    where
        EF: ExtensionField<F> + Send + Sync,
    {
        assert!(
            self.is_systematic(),
            "encode_algebra_systematic_coset requires systematic encoding"
        );
        assert_eq!(w.len(), self.msg_len(), "message length mismatch");
        let stride = 1 << self.log_inv_rate();
        assert!(coset_index < stride, "coset index out of range");

        if coset_index == 0 {
            return w.to_vec();
        }

        let shift = F::two_adic_generator(self.log_codeword_len()).exp_u64(coset_index as u64);
        self.dft.coset_lde_algebra(w.to_vec(), 0, shift)
    }

    /// Encode a batch of extension-field messages arranged as matrix columns.
    ///
    /// This is the extension-field analogue of [`Self::encode_batch_matrix`],
    /// used by final decider/root code when the accumulated witness lives over
    /// `F̂`.
    pub fn encode_algebra_batch_matrix<EF: ExtensionField<F>>(
        &self,
        messages: RowMajorMatrix<EF>,
    ) -> RowMajorMatrix<EF> {
        assert_eq!(messages.height(), self.msg_len(), "message height mismatch");
        let width = messages.width();
        let out = match self.encoding {
            ReedSolomonEncoding::Coefficient => {
                let mut padded = EF::zero_vec(self.codeword_len() * width);
                for row in 0..self.msg_len() {
                    let src = &messages.values[row * width..(row + 1) * width];
                    let dst = &mut padded[row * width..(row + 1) * width];
                    dst.copy_from_slice(src);
                }
                self.dft
                    .dft_algebra_batch(RowMajorMatrix::new(padded, width))
                    .to_row_major_matrix()
            }
            ReedSolomonEncoding::Systematic => self
                .dft
                .lde_algebra_batch(messages, self.log_inv_rate)
                .to_row_major_matrix(),
        };
        debug_assert_eq!(out.height(), self.codeword_len());
        debug_assert_eq!(out.width(), width);
        out
    }

    /// Encode a batch of extension-field messages supplied as separate vectors.
    pub fn encode_algebra_batch<EF: ExtensionField<F>>(
        &self,
        messages: &[Vec<EF>],
    ) -> RowMajorMatrix<EF> {
        assert!(!messages.is_empty(), "empty RS batch");
        let width = messages.len();
        let mut values = EF::zero_vec(self.msg_len() * width);
        for (col, message) in messages.iter().enumerate() {
            assert_eq!(message.len(), self.msg_len(), "message length mismatch");
            for row in 0..self.msg_len() {
                values[row * width + col] = message[row];
            }
        }
        self.encode_algebra_batch_matrix(RowMajorMatrix::new(values, width))
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use p3_baby_bear::BabyBear;
    use p3_dft::Radix2DFTSmallBatch;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use p3_multilinear_util::poly::Poly;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn rs<const LOG_K: usize, const LOG_INV_RATE: usize>()
    -> ReedSolomonCode<F, Radix2DFTSmallBatch<F>> {
        ReedSolomonCode::new(LOG_K, LOG_INV_RATE, Radix2DFTSmallBatch::<F>::default())
    }

    fn systematic_rs<const LOG_K: usize, const LOG_INV_RATE: usize>()
    -> ReedSolomonCode<F, Radix2DFTSmallBatch<F>> {
        ReedSolomonCode::new_systematic(LOG_K, LOG_INV_RATE, Radix2DFTSmallBatch::<F>::default())
    }

    #[test]
    fn encode_zero_message_yields_zero_codeword() {
        let code = rs::<4, 2>();
        let w = vec![F::ZERO; code.msg_len()];
        let f = code.encode(&w);
        assert_eq!(f.len(), code.codeword_len());
        assert!(f.iter().all(|&x| x == F::ZERO));
    }

    #[test]
    fn encode_constant_message_polynomial_evaluates_constant() {
        // w = [c, 0, 0, ...] is the polynomial p(X) = c, which evaluates to c
        // everywhere on the smooth coset. Hence the codeword is all c.
        let code = rs::<4, 1>();
        let mut w = F::zero_vec(code.msg_len());
        w[0] = F::from_u64(7);
        let f = code.encode(&w);
        assert!(f.iter().all(|&x| x == F::from_u64(7)));
    }

    #[test]
    fn encode_extension_matches_lift_of_base_encode() {
        // For w in F (lifted to EF via embedding), encode_algebra and encode
        // should agree.
        let code = rs::<5, 2>();
        let w_base: Vec<F> = (0..code.msg_len())
            .map(|i| F::from_u64(i as u64 + 1))
            .collect();
        let w_ext: Vec<EF> = w_base.iter().map(|&x| EF::from(x)).collect();

        let f_base = code.encode(&w_base);
        let f_ext = code.encode_algebra(&w_ext);

        assert_eq!(f_base.len(), f_ext.len());
        for (b, e) in f_base.iter().zip(&f_ext) {
            assert_eq!(EF::from(*b), *e);
        }
    }

    #[test]
    fn batch_encode_matches_individual_encode() {
        let code = rs::<4, 2>();
        let messages: Vec<Vec<F>> = (0..3)
            .map(|j| {
                (0..code.msg_len())
                    .map(|i| F::from_u64((10 * j + i) as u64 + 1))
                    .collect()
            })
            .collect();

        let batched = code.encode_batch(&messages);
        assert_eq!(batched.height(), code.codeword_len());
        assert_eq!(batched.width(), messages.len());
        for (col, message) in messages.iter().enumerate() {
            let expected = code.encode(message);
            for row in 0..code.codeword_len() {
                assert_eq!(batched.values[row * messages.len() + col], expected[row]);
            }
        }
    }

    #[test]
    fn systematic_encode_embeds_message_at_subgroup_stride() {
        let code = systematic_rs::<4, 2>();
        let w: Vec<F> = (0..code.msg_len())
            .map(|i| F::from_u64(i as u64 + 11))
            .collect();

        let f = code.encode(&w);

        assert!(code.is_systematic());
        assert_eq!(f.len(), code.codeword_len());
        for (i, &expected) in w.iter().enumerate() {
            assert_eq!(f[code.systematic_codeword_index(i)], expected);
            let codeword_point =
                Point::<F>::hypercube(code.systematic_codeword_index(i), code.log_codeword_len());
            assert_eq!(
                code.systematic_message_index_point::<F>(i),
                codeword_point,
                "systematic Boolean point must match codeword index"
            );
        }
    }

    #[test]
    fn systematic_codeword_index_weights_match_lde() {
        let code = systematic_rs::<4, 2>();
        let w: Vec<F> = (0..code.msg_len())
            .map(|i| F::from_u64(7 * i as u64 + 3))
            .collect();
        let codeword = code.encode(&w);

        for (index, &expected) in codeword.iter().enumerate() {
            let weights = code.systematic_codeword_index_weights::<EF>(index);
            let actual: EF = weights
                .iter()
                .zip(&w)
                .map(|(&weight, &value)| weight * EF::from(value))
                .sum();
            assert_eq!(
                actual,
                EF::from(expected),
                "systematic Lagrange weights must reproduce codeword index {index}"
            );
        }
    }

    #[test]
    fn systematic_coset_encoding_matches_full_encoding() {
        let code = systematic_rs::<5, 2>();
        let w: Vec<EF> = (0..code.msg_len())
            .map(|i| EF::from(F::from_u64(19 * i as u64 + 5)))
            .collect();
        let full = code.encode_algebra(&w);
        let stride = 1 << code.log_inv_rate();

        for residue in 0..stride {
            let coset = code.encode_algebra_systematic_coset(&w, residue);
            assert_eq!(coset.len(), code.msg_len());
            for row in 0..code.msg_len() {
                assert_eq!(
                    coset[row],
                    full[residue + row * stride],
                    "coset {residue}, row {row}"
                );
            }
        }
    }

    #[test]
    fn coefficient_codeword_index_weights_match_dft() {
        let code = rs::<4, 2>();
        let w: Vec<F> = (0..code.msg_len())
            .map(|i| F::from_u64(5 * i as u64 + 9))
            .collect();
        let codeword = code.encode(&w);

        for (index, &expected) in codeword.iter().enumerate() {
            let weights = code.codeword_index_weights::<EF>(index);
            let actual: EF = weights
                .iter()
                .zip(&w)
                .map(|(&weight, &value)| weight * EF::from(value))
                .sum();
            assert_eq!(
                actual,
                EF::from(expected),
                "coefficient RS weights must reproduce codeword index {index}"
            );
        }
    }

    #[test]
    fn systematic_codeword_mle_weights_match_encoded_mle() {
        let code = systematic_rs::<4, 1>();
        let w: Vec<EF> = (0..code.msg_len())
            .map(|i| EF::from(F::from_u64(11 * i as u64 + 9)))
            .collect();
        let codeword = code.encode_algebra(&w);
        let point: Vec<EF> = (0..code.log_codeword_len())
            .map(|i| EF::from(F::from_u64(5 * i as u64 + 13)))
            .collect();

        let expected = Poly::new(codeword).eval_ext::<F>(&Point::new(point.clone()));
        let weights = code.systematic_codeword_mle_weights::<EF>(&point);
        let actual: EF = weights
            .iter()
            .zip(&w)
            .map(|(&weight, &value)| weight * value)
            .sum();

        assert_eq!(actual, expected);
    }

    #[test]
    fn codeword_mle_weights_match_coefficient_encoded_mle() {
        let code = rs::<4, 1>();
        let w: Vec<EF> = (0..code.msg_len())
            .map(|i| EF::from(F::from_u64(13 * i as u64 + 7)))
            .collect();
        let codeword = code.encode_algebra(&w);
        let point: Vec<EF> = (0..code.log_codeword_len())
            .map(|i| EF::from(F::from_u64(3 * i as u64 + 19)))
            .collect();

        let expected = Poly::new(codeword).eval_ext::<F>(&Point::new(point.clone()));
        let weights = code.codeword_mle_weights::<EF>(&point);
        let actual: EF = weights
            .iter()
            .zip(&w)
            .map(|(&weight, &value)| weight * value)
            .sum();

        assert_eq!(actual, expected);
    }

    #[test]
    fn systematic_message_from_codeword_extracts_stride_entries() {
        let code = systematic_rs::<4, 2>();
        let w: Vec<EF> = (0..code.msg_len())
            .map(|i| EF::from(F::from_u64(i as u64 + 17)))
            .collect();
        let codeword = code.encode_algebra(&w);

        assert_eq!(code.systematic_message_from_codeword(&codeword), w);
        assert_eq!(code.message_from_codeword(&codeword), w);
    }

    #[test]
    fn coefficient_message_from_codeword_decodes_coefficients() {
        let code = rs::<4, 2>();
        let w: Vec<EF> = (0..code.msg_len())
            .map(|i| EF::from(F::from_u64(i as u64 + 23)))
            .collect();
        let codeword = code.encode_algebra(&w);

        assert_eq!(code.message_from_codeword(&codeword), w);
    }

    #[test]
    fn systematic_encode_extension_matches_lift_of_base_encode() {
        let code = systematic_rs::<5, 1>();
        let w_base: Vec<F> = (0..code.msg_len())
            .map(|i| F::from_u64(3 * i as u64 + 5))
            .collect();
        let w_ext: Vec<EF> = w_base.iter().map(|&x| EF::from(x)).collect();

        let f_base = code.encode(&w_base);
        let f_ext = code.encode_algebra(&w_ext);

        assert_eq!(f_base.len(), f_ext.len());
        for (b, e) in f_base.iter().zip(&f_ext) {
            assert_eq!(EF::from(*b), *e);
        }
    }

    #[test]
    fn systematic_batch_encode_algebra_matches_individual_encode() {
        let code = systematic_rs::<4, 2>();
        let messages: Vec<Vec<EF>> = (0..3)
            .map(|j| {
                (0..code.msg_len())
                    .map(|i| EF::from(F::from_u64((17 * j + i) as u64 + 3)))
                    .collect()
            })
            .collect();

        let batched = code.encode_algebra_batch(&messages);
        assert_eq!(batched.height(), code.codeword_len());
        assert_eq!(batched.width(), messages.len());
        for (col, message) in messages.iter().enumerate() {
            let expected = code.encode_algebra(message);
            for row in 0..code.codeword_len() {
                assert_eq!(batched.values[row * messages.len() + col], expected[row]);
            }
        }
    }

    #[test]
    fn systematic_message_subspace_matches_message_mle() {
        use rand::RngExt;

        let code = systematic_rs::<4, 2>();
        let mut rng = SmallRng::seed_from_u64(0x575952);
        let w: Vec<EF> = (0..code.msg_len()).map(|_| rng.random()).collect();
        let f = code.encode_algebra::<EF>(&w);

        let w_poly = Poly::<EF>::new(w);
        let f_poly = Poly::<EF>::new(f);

        for _ in 0..8 {
            let message_point = Point::<EF>::rand(&mut rng, code.log_msg_len());
            let codeword_point = code.systematic_message_point(message_point.as_slice());
            assert_eq!(
                w_poly.eval_ext::<F>(&message_point),
                f_poly.eval_ext::<F>(&codeword_point),
                "systematic RS subspace must expose the message MLE"
            );
        }
    }
}
