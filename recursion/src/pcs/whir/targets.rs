//! Circuit target representation of native WHIR multilinear proofs.
//!
//! This module is the first layer needed to recurse over WHIR proofs. It does
//! not yet verify the WHIR transcript; it allocates the native proof in the same
//! order in which the recursive verifier will later absorb and check it. Keeping
//! this proof shape independent from the FRI target types is important because
//! WHIR is a multilinear PCS: its proof is made of WHIR sumcheck messages, STIR
//! query openings, and optional per-round commitments, not univariate FRI query
//! paths.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use p3_circuit::CircuitBuilder;
use p3_commit::Mmcs;
use p3_field::{ExtensionField, Field};
use p3_multilinear_util::point::Point;
use p3_whir::pcs::proof::{QueryOpening, WhirProof, WhirRoundProof};
use p3_whir::pcs::verifier::WhirBatchedInitialVerifierOracle;
use p3_whir::sumcheck::SumcheckData;

use crate::Target;
use crate::traits::{Recursive, RecursiveMmcs, RecursivePrivateInput};

/// Native input consumed by one recursive WHIR verifier instance.
///
/// This is the benchmark-facing baseline object: a native WHIR commitment,
/// the multilinear opening claims for that commitment, and the native
/// `WhirProof` that proves those claims. WARP accumulation is intentionally not
/// represented here; a recursive N-WHIR baseline should allocate `N` of these
/// inputs and verify each WHIR transcript in the same circuit.
pub struct WhirProofVerificationInput<F, EF, MT, Comm>
where
    F: Send + Sync + Clone,
    MT: Mmcs<F>,
{
    /// Native WHIR commitment for the opened multilinear polynomial.
    pub commitment: Comm,
    /// Opening claims grouped by committed polynomial. Current `WhirPcs`
    /// supports one polynomial, but the outer vector matches
    /// `MultilinearPcs::verify`.
    pub opening_claims: Vec<Vec<(Point<EF>, EF)>>,
    /// Native WHIR proof for the opening claims.
    pub proof: WhirProof<F, EF, MT>,
    /// Deduplicated Fiat-Shamir STIR query indices for each intermediate
    /// WHIR round. These are verifier-recomputable metadata carried as private
    /// circuit witness data so the circuit can handle native WHIR's
    /// sort-and-dedup query schedule.
    pub round_query_index_bits: Vec<Vec<Vec<EF>>>,
    /// Deduplicated Fiat-Shamir STIR query indices for the final WHIR query
    /// phase.
    pub final_query_index_bits: Vec<Vec<EF>>,
}

/// Recursive target representation of one multilinear WHIR opening claim.
///
/// A claim is `f(z) = v`, with `z` a multilinear point and `v` the expected
/// extension-field evaluation. These targets are public because they are part
/// of the statement being recursively checked, just like the native verifier's
/// commitment and claimed values.
pub struct WhirOpeningClaimTargets<EF: Field> {
    /// Coordinates of the multilinear opening point.
    pub point: Vec<Target>,
    /// Claimed evaluation at `point`.
    pub value: Target,
    _phantom: PhantomData<EF>,
}

impl<EF: Field> Recursive<EF> for WhirOpeningClaimTargets<EF> {
    type Input = (Point<EF>, EF);

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        Self {
            point: circuit.alloc_public_inputs(input.0.num_variables(), "WHIR opening point"),
            value: circuit.alloc_public_input("WHIR opening value"),
            _phantom: PhantomData,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        input
            .0
            .as_slice()
            .iter()
            .copied()
            .chain([input.1])
            .collect()
    }
}

/// Recursive targets for one native WHIR proof verification.
///
/// This is the unit that an N-WHIR recursive aggregation circuit should repeat
/// `N` times. It deliberately mirrors `MultilinearPcs::verify`:
/// `(commitment, opening_claims, proof, challenger) -> Result<()>`.
pub struct WhirProofVerificationTargets<
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    Comm,
    RecMmcs: RecursiveMmcs<F, EF>,
> {
    /// Recursive commitment targets.
    pub commitment: Comm,
    /// Recursive opening-claim targets.
    pub opening_claims: Vec<Vec<WhirOpeningClaimTargets<EF>>>,
    /// Recursive WHIR proof targets.
    pub proof: WhirProofTargets<F, EF, RecMmcs>,
    /// Private little-endian bits for deduplicated intermediate query indices.
    pub round_query_index_bits: Vec<Vec<Vec<Target>>>,
    /// Private little-endian bits for deduplicated final query indices.
    pub final_query_index_bits: Vec<Vec<Target>>,
}

impl<F, EF, RecMmcs> Recursive<EF>
    for WhirProofVerificationTargets<F, EF, RecMmcs::Commitment, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone,
{
    type Input = WhirProofVerificationInput<
        F,
        EF,
        RecMmcs::Input,
        <RecMmcs::Commitment as Recursive<EF>>::Input,
    >;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let commitment = RecMmcs::Commitment::new(circuit, &input.commitment);
        let opening_claims = input
            .opening_claims
            .iter()
            .map(|claims| {
                claims
                    .iter()
                    .map(|claim| WhirOpeningClaimTargets::new(circuit, claim))
                    .collect()
            })
            .collect();
        // Allocation order must match `get_values`: statement commitment,
        // opening claims, proof payload, then query-index metadata.
        let proof = WhirProofTargets::new_with_initial_commitment(
            circuit,
            &input.proof,
            input
                .proof
                .initial_commitment
                .as_ref()
                .map(|_| commitment.clone()),
        );
        Self {
            commitment,
            opening_claims,
            // The native verifier parses the initial WHIR commitment from the proof
            // transcript and checks it equals the statement commitment. In-circuit we
            // represent the same equality by reusing this one target set in the proof
            // target. Allocating a second public commitment and connecting it would
            // create duplicate public witness creators after circuit rewriting.
            proof,
            round_query_index_bits: input
                .round_query_index_bits
                .iter()
                .map(|round| {
                    round
                        .iter()
                        .map(|bits| {
                            circuit.alloc_public_inputs(bits.len(), "WHIR query index bits")
                        })
                        .collect()
                })
                .collect(),
            final_query_index_bits: input
                .final_query_index_bits
                .iter()
                .map(|bits| circuit.alloc_public_inputs(bits.len(), "WHIR final query index bits"))
                .collect(),
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        RecMmcs::Commitment::get_values(&input.commitment)
            .into_iter()
            .chain(
                input
                    .opening_claims
                    .iter()
                    .flat_map(|claims| claims.iter().flat_map(WhirOpeningClaimTargets::get_values)),
            )
            .chain(
                WhirProofTargets::<F, EF, RecMmcs>::get_values_without_initial_commitment(
                    &input.proof,
                ),
            )
            .chain(
                input
                    .round_query_index_bits
                    .iter()
                    .flatten()
                    .flatten()
                    .copied(),
            )
            .chain(input.final_query_index_bits.iter().flatten().copied())
            .collect()
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        RecMmcs::Commitment::get_private_values(&input.commitment)
            .into_iter()
            .chain(
                WhirProofTargets::<F, EF, RecMmcs>::get_private_values_without_initial_commitment(
                    &input.proof,
                ),
            )
            .collect()
    }
}

impl<F, EF, RecMmcs> WhirProofVerificationTargets<F, EF, RecMmcs::Commitment, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone + RecursivePrivateInput<EF>,
    RecMmcs::Proof: RecursivePrivateInput<EF>,
{
    pub fn new_private_proof(
        circuit: &mut CircuitBuilder<EF>,
        input: &WhirProofVerificationInput<
            F,
            EF,
            RecMmcs::Input,
            <RecMmcs::Commitment as Recursive<EF>>::Input,
        >,
    ) -> Self {
        let commitment = RecMmcs::Commitment::new(circuit, &input.commitment);
        let opening_claims = input
            .opening_claims
            .iter()
            .map(|claims| {
                claims
                    .iter()
                    .map(|claim| WhirOpeningClaimTargets::new(circuit, claim))
                    .collect()
            })
            .collect();
        let proof = WhirProofTargets::new_with_initial_commitment_private_witness(
            circuit,
            &input.proof,
            input
                .proof
                .initial_commitment
                .as_ref()
                .map(|_| commitment.clone()),
        );
        Self {
            commitment,
            opening_claims,
            proof,
            round_query_index_bits: input
                .round_query_index_bits
                .iter()
                .map(|round| {
                    round
                        .iter()
                        .map(|bits| {
                            circuit.alloc_private_inputs(bits.len(), "WHIR query index bits")
                        })
                        .collect()
                })
                .collect(),
            final_query_index_bits: input
                .final_query_index_bits
                .iter()
                .map(|bits| circuit.alloc_private_inputs(bits.len(), "WHIR final query index bits"))
                .collect(),
        }
    }

    pub fn public_values_for_private_proof(
        input: &WhirProofVerificationInput<
            F,
            EF,
            RecMmcs::Input,
            <RecMmcs::Commitment as Recursive<EF>>::Input,
        >,
    ) -> Vec<EF> {
        RecMmcs::Commitment::get_values(&input.commitment)
            .into_iter()
            .chain(
                input
                    .opening_claims
                    .iter()
                    .flat_map(|claims| claims.iter().flat_map(WhirOpeningClaimTargets::get_values)),
            )
            .collect()
    }

    pub fn private_values_for_private_proof(
        input: &WhirProofVerificationInput<
            F,
            EF,
            RecMmcs::Input,
            <RecMmcs::Commitment as Recursive<EF>>::Input,
        >,
    ) -> Vec<EF> {
        RecMmcs::Commitment::get_private_values(&input.commitment)
            .into_iter()
            .chain(
                WhirProofTargets::<F, EF, RecMmcs>::private_witness_values_without_initial_commitment(
                    &input.proof,
                ),
            )
            .chain(
                input
                    .round_query_index_bits
                    .iter()
                    .flatten()
                    .flatten()
                    .copied(),
            )
            .chain(input.final_query_index_bits.iter().flatten().copied())
            .collect()
    }
}

/// Recursive target representation of one initial oracle participating in a
/// batched WHIR proof.
///
/// Native WHIR can prove one virtual initial polynomial
/// `sum_i coeff_i f_i`. The committed inputs may be independent base or
/// extension oracles, or a shared base-field matrix root with several columns.
/// This target type preserves that shape so the recursive verifier can observe
/// the same roots and use the same coefficients as the native verifier.
pub enum WhirBatchedInitialOracleTargets<Comm> {
    /// One base-field initial oracle.
    Base { coeff: Target, root: Comm },
    /// One extension-field initial oracle.
    Extension { coeff: Target, root: Comm },
    /// Several base-field columns committed under one shared root.
    SharedBase { coeffs: Vec<Target>, root: Comm },
    /// Several extension-field columns committed under one shared root.
    SharedExtension { coeffs: Vec<Target>, root: Comm },
}

impl<EF, Comm> Recursive<EF> for WhirBatchedInitialOracleTargets<Comm>
where
    EF: Field,
    Comm: Recursive<EF>,
{
    type Input = WhirBatchedInitialVerifierOracle<EF, Comm::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        match input {
            WhirBatchedInitialVerifierOracle::Base { root, .. } => Self::Base {
                coeff: circuit.alloc_public_input("WHIR batched initial base coefficient"),
                root: Comm::new(circuit, root),
            },
            WhirBatchedInitialVerifierOracle::Extension { root, .. } => Self::Extension {
                coeff: circuit.alloc_public_input("WHIR batched initial extension coefficient"),
                root: Comm::new(circuit, root),
            },
            WhirBatchedInitialVerifierOracle::SharedBase { coeffs, root } => Self::SharedBase {
                coeffs: circuit
                    .alloc_public_inputs(coeffs.len(), "WHIR batched shared-base coefficients"),
                root: Comm::new(circuit, root),
            },
            WhirBatchedInitialVerifierOracle::SharedExtension { coeffs, root } => {
                Self::SharedExtension {
                    coeffs: circuit.alloc_public_inputs(
                        coeffs.len(),
                        "WHIR batched shared-extension coefficients",
                    ),
                    root: Comm::new(circuit, root),
                }
            }
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        match input {
            WhirBatchedInitialVerifierOracle::Base { coeff, root }
            | WhirBatchedInitialVerifierOracle::Extension { coeff, root } => {
                let mut values = vec![*coeff];
                values.extend(Comm::get_values(root));
                values
            }
            WhirBatchedInitialVerifierOracle::SharedBase { coeffs, root } => coeffs
                .iter()
                .copied()
                .chain(Comm::get_values(root))
                .collect(),
            WhirBatchedInitialVerifierOracle::SharedExtension { coeffs, root } => coeffs
                .iter()
                .copied()
                .chain(Comm::get_values(root))
                .collect(),
        }
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        match input {
            WhirBatchedInitialVerifierOracle::Base { root, .. }
            | WhirBatchedInitialVerifierOracle::Extension { root, .. }
            | WhirBatchedInitialVerifierOracle::SharedBase { root, .. }
            | WhirBatchedInitialVerifierOracle::SharedExtension { root, .. } => {
                Comm::get_private_values(root)
            }
        }
    }
}

/// Recursive target representation of a WHIR sumcheck transcript.
///
/// WHIR sends two extension-field values per sumcheck variable, namely
/// `h_i(0)` and `h_i(infinity)`. The verifier reconstructs `h_i(1)` from the
/// running sum constraint, absorbs the two sent values into Fiat-Shamir, checks
/// the optional grinding witness, and samples the folding challenge.
pub struct WhirSumcheckDataTargets<F: Field, EF: ExtensionField<F>> {
    /// Per-round `[h_i(0), h_i(infinity)]` targets.
    pub polynomial_evaluations: Vec<[Target; 2]>,
    /// Optional proof-of-work witnesses, one per sumcheck round when enabled.
    pub pow_witnesses: Vec<Target>,
    _phantom: PhantomData<(F, EF)>,
}

impl<F: Field, EF: ExtensionField<F>> Recursive<EF> for WhirSumcheckDataTargets<F, EF> {
    type Input = SumcheckData<F, EF>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let polynomial_evaluations = (0..input.polynomial_evaluations.len())
            .map(|_| circuit.alloc_public_input_array("WHIR sumcheck evaluations"))
            .collect();
        let pow_witnesses =
            circuit.alloc_public_inputs(input.pow_witnesses.len(), "WHIR sumcheck PoW witnesses");

        Self {
            polynomial_evaluations,
            pow_witnesses,
            _phantom: PhantomData,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        input
            .polynomial_evaluations
            .iter()
            .flat_map(|[c0, c_inf]| [*c0, *c_inf])
            .chain(input.pow_witnesses.iter().map(|w| EF::from(*w)))
            .collect()
    }
}

impl<F: Field, EF: ExtensionField<F>> WhirSumcheckDataTargets<F, EF> {
    pub fn new_private_witness(
        circuit: &mut CircuitBuilder<EF>,
        input: &SumcheckData<F, EF>,
    ) -> Self {
        let polynomial_evaluations = (0..input.polynomial_evaluations.len())
            .map(|_| circuit.alloc_private_input_array("WHIR sumcheck evaluations"))
            .collect();
        let pow_witnesses =
            circuit.alloc_private_inputs(input.pow_witnesses.len(), "WHIR sumcheck PoW witnesses");

        Self {
            polynomial_evaluations,
            pow_witnesses,
            _phantom: PhantomData,
        }
    }

    pub fn private_witness_values(input: &SumcheckData<F, EF>) -> Vec<EF> {
        Self::get_values(input)
    }
}

/// Recursive target representation of a WHIR STIR query opening.
///
/// The variants match native WHIR exactly:
/// - `Base` opens one base-field oracle row,
/// - `Extension` opens one extension-field folded oracle row,
/// - `SharedBase` opens several base-field matrices under one shared root,
/// - `Batched` groups multiple independently committed initial oracles.
///
/// The Merkle/MMCS proof itself is delegated to `RecMmcs::Proof`, so this type
/// can be reused with any recursive Merkle backend used by WHIR.
pub enum WhirQueryOpeningTargets<F: Field, EF: ExtensionField<F>, RecMmcs: RecursiveMmcs<F, EF>> {
    /// Opening of a base-field row, lifted into the challenge field in-circuit.
    Base {
        /// Opened base values, represented as lifted extension-field targets.
        values: Vec<Target>,
        /// Recursive MMCS proof for the opened row.
        proof: RecMmcs::Proof,
    },
    /// Opening of an extension-field row.
    Extension {
        /// Opened extension values.
        values: Vec<Target>,
        /// Recursive MMCS proof for the opened row.
        proof: RecMmcs::Proof,
    },
    /// Opening of several base-field rows under one shared MMCS root.
    SharedBase {
        /// Opened rows, each represented as lifted extension-field targets.
        values: Vec<Vec<Target>>,
        /// Recursive MMCS proof for the shared opened row.
        proof: RecMmcs::Proof,
    },
    /// Opening of several extension-field rows under one shared MMCS root.
    SharedExtension {
        /// Opened extension rows.
        values: Vec<Vec<Target>>,
        /// Recursive MMCS proof for the shared opened row.
        proof: RecMmcs::Proof,
    },
    /// Batched openings against several initial commitments.
    Batched {
        /// Inner openings in native proof order.
        openings: Vec<WhirQueryOpeningTargets<F, EF, RecMmcs>>,
    },
}

impl<F, EF, RecMmcs> Recursive<EF> for WhirQueryOpeningTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
{
    type Input = QueryOpening<F, EF, <RecMmcs::Input as Mmcs<F>>::Proof>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        match input {
            QueryOpening::Base { values, proof } => Self::Base {
                values: circuit.alloc_public_inputs(values.len(), "WHIR base query values"),
                proof: RecMmcs::Proof::new(circuit, proof),
            },
            QueryOpening::Extension { values, proof } => Self::Extension {
                values: circuit.alloc_public_inputs(values.len(), "WHIR extension query values"),
                proof: RecMmcs::Proof::new(circuit, proof),
            },
            QueryOpening::SharedBase { values, proof } => Self::SharedBase {
                values: values
                    .iter()
                    .map(|row| circuit.alloc_public_inputs(row.len(), "WHIR shared base query row"))
                    .collect(),
                proof: RecMmcs::Proof::new(circuit, proof),
            },
            QueryOpening::SharedExtension { values, proof } => Self::SharedExtension {
                values: values
                    .iter()
                    .map(|row| {
                        circuit.alloc_public_inputs(row.len(), "WHIR shared extension query row")
                    })
                    .collect(),
                proof: RecMmcs::Proof::new(circuit, proof),
            },
            QueryOpening::Batched { openings } => Self::Batched {
                openings: openings
                    .iter()
                    .map(|opening| Self::new(circuit, opening))
                    .collect(),
            },
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        match input {
            QueryOpening::Base { values, proof } => values
                .iter()
                .map(|v| EF::from(*v))
                .chain(RecMmcs::Proof::get_values(proof))
                .collect(),
            QueryOpening::Extension { values, proof } => values
                .iter()
                .copied()
                .chain(RecMmcs::Proof::get_values(proof))
                .collect(),
            QueryOpening::SharedBase { values, proof } => values
                .iter()
                .flat_map(|row| row.iter().map(|v| EF::from(*v)))
                .chain(RecMmcs::Proof::get_values(proof))
                .collect(),
            QueryOpening::SharedExtension { values, proof } => values
                .iter()
                .flat_map(|row| row.iter().copied())
                .chain(RecMmcs::Proof::get_values(proof))
                .collect(),
            QueryOpening::Batched { openings } => {
                openings.iter().flat_map(Self::get_values).collect()
            }
        }
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        match input {
            QueryOpening::Base { proof, .. }
            | QueryOpening::Extension { proof, .. }
            | QueryOpening::SharedBase { proof, .. }
            | QueryOpening::SharedExtension { proof, .. } => {
                RecMmcs::Proof::get_private_values(proof)
            }
            QueryOpening::Batched { openings } => {
                openings.iter().flat_map(Self::get_private_values).collect()
            }
        }
    }
}

impl<F, EF, RecMmcs> WhirQueryOpeningTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Proof: RecursivePrivateInput<EF>,
{
    pub fn new_private_witness(
        circuit: &mut CircuitBuilder<EF>,
        input: &QueryOpening<F, EF, <RecMmcs::Input as Mmcs<F>>::Proof>,
    ) -> Self {
        match input {
            QueryOpening::Base { values, proof } => Self::Base {
                values: circuit.alloc_private_inputs(values.len(), "WHIR base query values"),
                proof: RecMmcs::Proof::new_private_input(circuit, proof),
            },
            QueryOpening::Extension { values, proof } => Self::Extension {
                values: circuit.alloc_private_inputs(values.len(), "WHIR extension query values"),
                proof: RecMmcs::Proof::new_private_input(circuit, proof),
            },
            QueryOpening::SharedBase { values, proof } => Self::SharedBase {
                values: values
                    .iter()
                    .map(|row| {
                        circuit.alloc_private_inputs(row.len(), "WHIR shared base query row")
                    })
                    .collect(),
                proof: RecMmcs::Proof::new_private_input(circuit, proof),
            },
            QueryOpening::SharedExtension { values, proof } => Self::SharedExtension {
                values: values
                    .iter()
                    .map(|row| {
                        circuit.alloc_private_inputs(row.len(), "WHIR shared extension query row")
                    })
                    .collect(),
                proof: RecMmcs::Proof::new_private_input(circuit, proof),
            },
            QueryOpening::Batched { openings } => Self::Batched {
                openings: openings
                    .iter()
                    .map(|opening| Self::new_private_witness(circuit, opening))
                    .collect(),
            },
        }
    }

    pub fn private_witness_values(
        input: &QueryOpening<F, EF, <RecMmcs::Input as Mmcs<F>>::Proof>,
    ) -> Vec<EF> {
        match input {
            QueryOpening::Base { values, proof } => values
                .iter()
                .map(|v| EF::from(*v))
                .chain(RecMmcs::Proof::get_private_input_values(proof))
                .collect(),
            QueryOpening::Extension { values, proof } => values
                .iter()
                .copied()
                .chain(RecMmcs::Proof::get_private_input_values(proof))
                .collect(),
            QueryOpening::SharedBase { values, proof } => values
                .iter()
                .flat_map(|row| row.iter().map(|v| EF::from(*v)))
                .chain(RecMmcs::Proof::get_private_input_values(proof))
                .collect(),
            QueryOpening::SharedExtension { values, proof } => values
                .iter()
                .flat_map(|row| row.iter().copied())
                .chain(RecMmcs::Proof::get_private_input_values(proof))
                .collect(),
            QueryOpening::Batched { openings } => openings
                .iter()
                .flat_map(Self::private_witness_values)
                .collect(),
        }
    }
}

/// Recursive target representation of one WHIR folding round.
pub struct WhirRoundProofTargets<F: Field, EF: ExtensionField<F>, RecMmcs: RecursiveMmcs<F, EF>> {
    /// Optional round commitment. It is present for rounds that commit a folded
    /// codeword and absent for protocol shapes that skip a commitment.
    pub commitment: Option<RecMmcs::Commitment>,
    /// Out-of-domain answers sent before this round's sumcheck challenge.
    pub ood_answers: Vec<Target>,
    /// Commitment-round proof-of-work witness.
    pub pow_witness: Target,
    /// STIR query openings for this round.
    pub queries: Vec<WhirQueryOpeningTargets<F, EF, RecMmcs>>,
    /// Sumcheck messages for this round.
    pub sumcheck: WhirSumcheckDataTargets<F, EF>,
}

impl<F, EF, RecMmcs> Recursive<EF> for WhirRoundProofTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone,
{
    type Input = WhirRoundProof<F, EF, RecMmcs::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        let commitment = input
            .commitment
            .as_ref()
            .map(|commitment| RecMmcs::Commitment::new(circuit, commitment));
        let ood_answers =
            circuit.alloc_public_inputs(input.ood_answers.len(), "WHIR round OOD answers");
        let pow_witness = circuit.alloc_public_input("WHIR round PoW witness");
        let queries = input
            .queries
            .iter()
            .map(|query| WhirQueryOpeningTargets::new(circuit, query))
            .collect();
        let sumcheck = WhirSumcheckDataTargets::new(circuit, &input.sumcheck);

        Self {
            commitment,
            ood_answers,
            pow_witness,
            queries,
            sumcheck,
        }
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        input
            .commitment
            .iter()
            .flat_map(RecMmcs::Commitment::get_values)
            .chain(input.ood_answers.iter().copied())
            .chain([EF::from(input.pow_witness)])
            .chain(
                input
                    .queries
                    .iter()
                    .flat_map(WhirQueryOpeningTargets::<F, EF, RecMmcs>::get_values),
            )
            .chain(WhirSumcheckDataTargets::<F, EF>::get_values(
                &input.sumcheck,
            ))
            .collect()
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        input
            .commitment
            .iter()
            .flat_map(RecMmcs::Commitment::get_private_values)
            .chain(
                input
                    .queries
                    .iter()
                    .flat_map(WhirQueryOpeningTargets::<F, EF, RecMmcs>::get_private_values),
            )
            .chain(WhirSumcheckDataTargets::<F, EF>::get_private_values(
                &input.sumcheck,
            ))
            .collect()
    }
}

impl<F, EF, RecMmcs> WhirRoundProofTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone + RecursivePrivateInput<EF>,
    RecMmcs::Proof: RecursivePrivateInput<EF>,
{
    pub fn new_private_witness(
        circuit: &mut CircuitBuilder<EF>,
        input: &WhirRoundProof<F, EF, RecMmcs::Input>,
    ) -> Self {
        let commitment = input
            .commitment
            .as_ref()
            .map(|commitment| RecMmcs::Commitment::new_private_input(circuit, commitment));
        let ood_answers =
            circuit.alloc_private_inputs(input.ood_answers.len(), "WHIR round OOD answers");
        let pow_witness = circuit.alloc_private_input("WHIR round PoW witness");
        let queries = input
            .queries
            .iter()
            .map(|query| WhirQueryOpeningTargets::new_private_witness(circuit, query))
            .collect();
        let sumcheck = WhirSumcheckDataTargets::new_private_witness(circuit, &input.sumcheck);

        Self {
            commitment,
            ood_answers,
            pow_witness,
            queries,
            sumcheck,
        }
    }

    pub fn private_witness_values(input: &WhirRoundProof<F, EF, RecMmcs::Input>) -> Vec<EF> {
        input
            .commitment
            .iter()
            .flat_map(RecMmcs::Commitment::get_private_input_values)
            .chain(input.ood_answers.iter().copied())
            .chain([EF::from(input.pow_witness)])
            .chain(
                input
                    .queries
                    .iter()
                    .flat_map(WhirQueryOpeningTargets::<F, EF, RecMmcs>::private_witness_values),
            )
            .chain(WhirSumcheckDataTargets::<F, EF>::private_witness_values(
                &input.sumcheck,
            ))
            .collect()
    }
}

/// Recursive target representation of a full native WHIR proof.
pub struct WhirProofTargets<F: Field, EF: ExtensionField<F>, RecMmcs: RecursiveMmcs<F, EF>> {
    /// Optional initial codeword commitment.
    pub initial_commitment: Option<RecMmcs::Commitment>,
    /// Initial out-of-domain answers.
    pub initial_ood_answers: Vec<Target>,
    /// Initial sumcheck messages.
    pub initial_sumcheck: WhirSumcheckDataTargets<F, EF>,
    /// Per-folding-round proofs.
    pub rounds: Vec<WhirRoundProofTargets<F, EF, RecMmcs>>,
    /// Optional final polynomial sent in the clear.
    pub final_poly: Option<Vec<Target>>,
    /// Final proof-of-work witness.
    pub final_pow_witness: Target,
    /// Final STIR query openings.
    pub final_queries: Vec<WhirQueryOpeningTargets<F, EF, RecMmcs>>,
    /// Optional direct final sumcheck.
    pub final_sumcheck: Option<WhirSumcheckDataTargets<F, EF>>,
}

impl<F, EF, RecMmcs> Recursive<EF> for WhirProofTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone,
{
    type Input = WhirProof<F, EF, RecMmcs::Input>;

    fn new(circuit: &mut CircuitBuilder<EF>, input: &Self::Input) -> Self {
        Self::new_with_initial_commitment(circuit, input, None)
    }

    fn get_values(input: &Self::Input) -> Vec<EF> {
        input
            .initial_commitment
            .iter()
            .flat_map(RecMmcs::Commitment::get_values)
            .chain(Self::get_values_without_initial_commitment(input))
            .collect()
    }

    fn get_private_values(input: &Self::Input) -> Vec<EF> {
        input
            .initial_commitment
            .iter()
            .flat_map(RecMmcs::Commitment::get_private_values)
            .chain(Self::get_private_values_without_initial_commitment(input))
            .collect()
    }
}

impl<F, EF, RecMmcs> WhirProofTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone,
{
    /// Allocate a WHIR proof target, optionally reusing a caller-owned target for
    /// the initial commitment.
    ///
    /// This is used by `WhirProofVerificationTargets` so the public statement
    /// commitment and the proof's initial transcript commitment are represented
    /// by the same witness IDs. The transcript still observes the commitment in
    /// exactly the same position as native WHIR; we only avoid duplicate target
    /// allocation for an equality that is static in the recursive statement.
    pub fn new_with_initial_commitment(
        circuit: &mut CircuitBuilder<EF>,
        input: &WhirProof<F, EF, RecMmcs::Input>,
        initial_commitment_override: Option<RecMmcs::Commitment>,
    ) -> Self {
        let initial_commitment = input.initial_commitment.as_ref().map(|commitment| {
            initial_commitment_override
                .clone()
                .unwrap_or_else(|| RecMmcs::Commitment::new(circuit, commitment))
        });
        let initial_ood_answers = circuit
            .alloc_public_inputs(input.initial_ood_answers.len(), "WHIR initial OOD answers");
        let initial_sumcheck = WhirSumcheckDataTargets::new(circuit, &input.initial_sumcheck);
        let rounds = input
            .rounds
            .iter()
            .map(|round| WhirRoundProofTargets::new(circuit, round))
            .collect();
        let final_poly = input.final_poly.as_ref().map(|poly| {
            circuit.alloc_public_inputs(poly.num_evals(), "WHIR final polynomial evaluations")
        });
        let final_pow_witness = circuit.alloc_public_input("WHIR final PoW witness");
        let final_queries = input
            .final_queries
            .iter()
            .map(|query| WhirQueryOpeningTargets::new(circuit, query))
            .collect();
        let final_sumcheck = input
            .final_sumcheck
            .as_ref()
            .map(|sumcheck| WhirSumcheckDataTargets::new(circuit, sumcheck));

        Self {
            initial_commitment,
            initial_ood_answers,
            initial_sumcheck,
            rounds,
            final_poly,
            final_pow_witness,
            final_queries,
            final_sumcheck,
        }
    }

    pub fn get_values_without_initial_commitment(
        input: &WhirProof<F, EF, RecMmcs::Input>,
    ) -> Vec<EF> {
        input
            .initial_ood_answers
            .iter()
            .copied()
            .chain(WhirSumcheckDataTargets::<F, EF>::get_values(
                &input.initial_sumcheck,
            ))
            .chain(
                input
                    .rounds
                    .iter()
                    .flat_map(WhirRoundProofTargets::<F, EF, RecMmcs>::get_values),
            )
            .chain(
                input
                    .final_poly
                    .iter()
                    .flat_map(|poly| poly.as_slice().iter().copied()),
            )
            .chain([EF::from(input.final_pow_witness)])
            .chain(
                input
                    .final_queries
                    .iter()
                    .flat_map(WhirQueryOpeningTargets::<F, EF, RecMmcs>::get_values),
            )
            .chain(
                input
                    .final_sumcheck
                    .iter()
                    .flat_map(WhirSumcheckDataTargets::<F, EF>::get_values),
            )
            .collect()
    }

    pub fn get_private_values_without_initial_commitment(
        input: &WhirProof<F, EF, RecMmcs::Input>,
    ) -> Vec<EF> {
        input
            .rounds
            .iter()
            .flat_map(WhirRoundProofTargets::<F, EF, RecMmcs>::get_private_values)
            .chain(
                input
                    .final_queries
                    .iter()
                    .flat_map(WhirQueryOpeningTargets::<F, EF, RecMmcs>::get_private_values),
            )
            .chain(
                input
                    .final_sumcheck
                    .iter()
                    .flat_map(WhirSumcheckDataTargets::<F, EF>::get_private_values),
            )
            .collect()
    }
}

impl<F, EF, RecMmcs> WhirProofTargets<F, EF, RecMmcs>
where
    F: Field + Send + Sync + Clone,
    EF: ExtensionField<F>,
    RecMmcs: RecursiveMmcs<F, EF>,
    RecMmcs::Commitment: Clone + RecursivePrivateInput<EF>,
    RecMmcs::Proof: RecursivePrivateInput<EF>,
{
    /// Allocate a WHIR proof transcript as private witness data, optionally
    /// reusing a caller-owned public target for the initial commitment.
    pub fn new_with_initial_commitment_private_witness(
        circuit: &mut CircuitBuilder<EF>,
        input: &WhirProof<F, EF, RecMmcs::Input>,
        initial_commitment_override: Option<RecMmcs::Commitment>,
    ) -> Self {
        let initial_commitment = input.initial_commitment.as_ref().map(|commitment| {
            initial_commitment_override
                .clone()
                .unwrap_or_else(|| RecMmcs::Commitment::new_private_input(circuit, commitment))
        });
        let initial_ood_answers = circuit
            .alloc_private_inputs(input.initial_ood_answers.len(), "WHIR initial OOD answers");
        let initial_sumcheck =
            WhirSumcheckDataTargets::new_private_witness(circuit, &input.initial_sumcheck);
        let rounds = input
            .rounds
            .iter()
            .map(|round| WhirRoundProofTargets::new_private_witness(circuit, round))
            .collect();
        let final_poly = input.final_poly.as_ref().map(|poly| {
            circuit.alloc_private_inputs(poly.num_evals(), "WHIR final polynomial evaluations")
        });
        let final_pow_witness = circuit.alloc_private_input("WHIR final PoW witness");
        let final_queries = input
            .final_queries
            .iter()
            .map(|query| WhirQueryOpeningTargets::new_private_witness(circuit, query))
            .collect();
        let final_sumcheck = input
            .final_sumcheck
            .as_ref()
            .map(|sumcheck| WhirSumcheckDataTargets::new_private_witness(circuit, sumcheck));

        Self {
            initial_commitment,
            initial_ood_answers,
            initial_sumcheck,
            rounds,
            final_poly,
            final_pow_witness,
            final_queries,
            final_sumcheck,
        }
    }

    pub fn private_witness_values_without_initial_commitment(
        input: &WhirProof<F, EF, RecMmcs::Input>,
    ) -> Vec<EF> {
        input
            .initial_ood_answers
            .iter()
            .copied()
            .chain(WhirSumcheckDataTargets::<F, EF>::private_witness_values(
                &input.initial_sumcheck,
            ))
            .chain(
                input
                    .rounds
                    .iter()
                    .flat_map(WhirRoundProofTargets::<F, EF, RecMmcs>::private_witness_values),
            )
            .chain(
                input
                    .final_poly
                    .iter()
                    .flat_map(|poly| poly.as_slice().iter().copied()),
            )
            .chain([EF::from(input.final_pow_witness)])
            .chain(
                input
                    .final_queries
                    .iter()
                    .flat_map(WhirQueryOpeningTargets::<F, EF, RecMmcs>::private_witness_values),
            )
            .chain(
                input
                    .final_sumcheck
                    .iter()
                    .flat_map(WhirSumcheckDataTargets::<F, EF>::private_witness_values),
            )
            .collect()
    }
}
