//! Typed Fiat-Shamir transcript for the bus multiset argument.

use alloc::vec;

use p3_challenger::FieldChallenger;
use p3_challenger::fs::{
    DomainSeparator, FieldToFieldCodec, FieldUnit, Hierarchy, Interaction, InteractionPattern,
    Kind, Length, ProverState, TranscriptField, VerifierState,
};
use p3_field::ExtensionField;

use super::BusChallenges;
use crate::{BusDirection, BusPlan};

/// Version byte bound into the protocol seed.
const VERSION: u8 = 1;

/// Protocol name bound into the protocol seed.
const NAME: &[u8] = b"p3-bus-argument";

/// Label of the tuple-fingerprint challenge point.
const FINGERPRINT: &str = "fingerprint";

/// Label of the random shift applied to every tuple fingerprint.
const OFFSET: &str = "offset";

/// Label of the delegated product-tree reduction.
const PRODUCT: &str = "product_gkr";

/// Sponge alphabet of a challenger native to the base field.
type Alphabet<F> = FieldUnit<F>;

/// Type-level marker for the delegated product-tree reduction.
struct ProductReduction;

impl BusPlan {
    /// Describe the statement-derived challenge and delegation schedule.
    fn interaction_pattern<F, EF>(&self) -> InteractionPattern
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The public layout fixes both challenge length and the delegated reduction shape.
        InteractionPattern::new(vec![
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                FINGERPRINT,
                Length::Fixed(self.security_geometry().tuple_variables()),
            ),
            Interaction::algebra::<F, EF>(
                Hierarchy::Atomic,
                Kind::Challenge,
                OFFSET,
                Length::Scalar,
            ),
            Interaction::marker::<ProductReduction>(Hierarchy::Begin, Kind::Protocol, PRODUCT),
            Interaction::marker::<ProductReduction>(Hierarchy::End, Kind::Protocol, PRODUCT),
        ])
        .expect("one matched product-reduction bracket is well formed")
    }

    /// Bind every public dimension that changes a tuple or tree position.
    fn domain_separator<F, EF>(&self) -> DomainSeparator<Alphabet<F>>
    where
        F: TranscriptField,
        EF: ExtensionField<F>,
    {
        // The seed commits to the full verifier-derived statement before any challenge.
        let mut separator =
            DomainSeparator::new(VERSION, NAME, self.interaction_pattern::<F, EF>());
        separator
            .instance(&(self.domains().len() as u64).to_be_bytes())
            .instance(&(self.payload_slots() as u64).to_be_bytes())
            .instance(&(self.domain_slots() as u64).to_be_bytes())
            .instance(&(self.fingerprint_width() as u64).to_be_bytes());

        // Domain names and identities prevent two named buses from sharing one tuple space.
        for domain in self.domains() {
            separator
                .instance(&(domain.name.len() as u64).to_be_bytes())
                .instance(domain.name.as_bytes())
                .instance(&(domain.payload_width as u64).to_be_bytes())
                .instance(&(domain.identity as u64).to_be_bytes());
        }

        // Physical block order determines the product-tree leaf address of every declaration.
        for direction in BusDirection::ALL {
            let blocks = self.blocks(direction);
            separator.instance(&(blocks.len() as u64).to_be_bytes());
            for block in blocks {
                separator
                    .instance(&(block.bus as u64).to_be_bytes())
                    .instance(&(block.owner.air as u64).to_be_bytes())
                    .instance(&(block.owner.declaration as u64).to_be_bytes())
                    .instance(&(block.log_height as u64).to_be_bytes())
                    .instance(&(block.offset as u64).to_be_bytes());
            }
        }
        separator
    }
}

/// Prover-side driver for the bus argument transcript.
pub(super) struct BusProverTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player borrowing the surrounding challenger.
    state: ProverState<&'a mut C, Alphabet<F>>,
    /// Number of coordinates in the tuple-fingerprint point.
    fingerprint_variables: usize,
    /// Challenge-field marker used by the typed codec.
    _ef: core::marker::PhantomData<EF>,
}

impl<'a, C, F, EF> BusProverTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    /// Seed the transcript from the complete public bus layout.
    pub(super) fn new(challenger: &'a mut C, plan: &BusPlan) -> Self {
        // Every later draw is scoped to the exact tuple and tree geometry.
        Self {
            state: ProverState::new(challenger, &plan.domain_separator::<F, EF>()),
            fingerprint_variables: plan.security_geometry().tuple_variables(),
            _ef: core::marker::PhantomData,
        }
    }

    /// Sample the challenges defining every bus leaf factor.
    pub(super) fn challenges(&mut self) -> BusChallenges<EF> {
        // The tuple point precedes the independent product shift.
        let fingerprint = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                FINGERPRINT,
                self.fingerprint_variables,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        let offset = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OFFSET)
            .into_inner();
        BusChallenges {
            fingerprint,
            offset,
        }
    }

    /// Execute the delegated product proof inside its transcript bracket.
    pub(super) fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // The nested proof owns every product-reduction message and challenge.
        self.state.begin_protocol::<ProductReduction>(PRODUCT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductReduction>(PRODUCT);
        output
    }

    /// Finish after every shape-derived transcript step has been played.
    pub(super) fn finish(self) {
        // The proof object carries every message, so the typed wire is empty.
        assert!(self.state.finalize().is_empty());
    }

    /// Disable completeness checking after a checked honest-prover input error.
    pub(super) fn abort(&mut self) {
        // No transcript output is consumed after an honest-prover shape failure.
        self.state.abort();
    }
}

/// Verifier-side replay of the bus argument transcript.
pub(super) struct BusVerifierTranscript<'a, C, F: TranscriptField, EF> {
    /// Pattern player borrowing the surrounding challenger over an empty wire.
    state: VerifierState<'static, &'a mut C, Alphabet<F>>,
    /// Number of coordinates in the tuple-fingerprint point.
    fingerprint_variables: usize,
    /// Challenge-field marker used by the typed codec.
    _ef: core::marker::PhantomData<EF>,
}

impl<'a, C, F, EF> BusVerifierTranscript<'a, C, F, EF>
where
    F: TranscriptField,
    EF: ExtensionField<F>,
    C: FieldChallenger<F>,
{
    /// Seed the replay from the complete public bus layout.
    pub(super) fn new(challenger: &'a mut C, plan: &BusPlan) -> Self {
        // This protocol reads no wire values outside the delegated proof object.
        Self {
            state: VerifierState::new(challenger, &plan.domain_separator::<F, EF>(), &[]),
            fingerprint_variables: plan.security_geometry().tuple_variables(),
            _ef: core::marker::PhantomData,
        }
    }

    /// Replay the challenges defining every bus leaf factor.
    pub(super) fn challenges(&mut self) -> BusChallenges<EF> {
        // The verifier redraws the exact statement-derived challenge count.
        let fingerprint = self
            .state
            .challenge_extensions::<F, EF, FieldToFieldCodec<F>>(
                FINGERPRINT,
                self.fingerprint_variables,
            )
            .into_iter()
            .map(|value| value.into_inner())
            .collect();
        let offset = self
            .state
            .challenge_extension::<F, EF, FieldToFieldCodec<F>>(OFFSET)
            .into_inner();
        BusChallenges {
            fingerprint,
            offset,
        }
    }

    /// Execute delegated product verification inside its transcript bracket.
    pub(super) fn product<R>(&mut self, run: impl FnOnce(&mut C) -> R) -> R {
        // Nested verification replays every product message before leaving the bracket.
        self.state.begin_protocol::<ProductReduction>(PRODUCT);
        let output = run(self.state.challenger_mut());
        self.state.end_protocol::<ProductReduction>(PRODUCT);
        output
    }

    /// Finish after every shape-derived transcript step has been replayed.
    pub(super) fn finish(self) {
        // Complete replay consumes the empty typed wire exactly.
        self.state
            .finalize()
            .expect("the bus argument reads an empty wire");
    }

    /// Disable outer completeness checking after delegated verification rejects.
    pub(super) fn abort(&mut self) {
        // The caller returns the nested verification error immediately.
        self.state.abort();
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;
    use alloc::vec::Vec;

    use p3_air::symbolic::{BaseEntry, SymbolicVariable};
    use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro128Plus;

    use super::*;
    use crate::{BusActivation, BusPlanInput, SymbolicBusInteraction};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;
    type Challenger = DuplexChallenger<F, Poseidon2BabyBear<16>, 16, 8>;

    fn challenger() -> Challenger {
        let mut rng = Xoroshiro128Plus::seed_from_u64(0xB055_700D);
        DuplexChallenger::new(Poseidon2BabyBear::new_from_rng_128(&mut rng))
    }

    fn interaction(name: &str, direction: BusDirection, width: usize) -> SymbolicBusInteraction<F> {
        SymbolicBusInteraction {
            bus_name: name.to_string(),
            direction,
            fields: (0..width)
                .map(|index| SymbolicVariable::new(BaseEntry::Main { offset: 0 }, index).into())
                .collect(),
            activation: BusActivation::Always,
        }
    }

    // Length-delimited chunk, written out independently of the encoder under test.
    fn chunk(bytes: &[u8]) -> Vec<u8> {
        let mut out = (bytes.len() as u32).to_be_bytes().to_vec();
        out.extend_from_slice(bytes);
        out
    }

    fn number(value: u64) -> Vec<u8> {
        chunk(&value.to_be_bytes())
    }

    fn label(plan: &BusPlan) -> Vec<u8> {
        plan.domain_separator::<F, EF>().instance_label().to_vec()
    }

    // Two plans collide only when both the pattern shape and the instance label agree.
    fn seeds_agree(left: &BusPlan, right: &BusPlan) -> bool {
        let left = left.domain_separator::<F, EF>();
        let right = right.domain_separator::<F, EF>();
        left.pattern().pattern_hash() == right.pattern().pattern_hash()
            && left.instance_label() == right.instance_label()
    }

    #[test]
    fn the_instance_label_spells_out_every_public_dimension() {
        // Two named buses of unequal payload width over two tables of unequal height.
        let tall = [
            interaction("alpha", BusDirection::Push, 2),
            interaction("beta", BusDirection::Pull, 1),
        ];
        let short = [interaction("beta", BusDirection::Push, 1)];
        let plan = BusPlan::build(&[
            BusPlanInput {
                log_height: 2,
                interactions: &tall,
            },
            BusPlanInput {
                log_height: 1,
                interactions: &short,
            },
        ])
        .unwrap()
        .unwrap();

        // Two names need two identity bits, and two payload slots fill the four-wide tuple.
        let mut expected = Vec::new();
        expected.extend(number(2));
        expected.extend(number(2));
        expected.extend(number(2));
        expected.extend(number(4));

        // Lexicographic order gives alpha identity one and beta identity two.
        expected.extend(number(5));
        expected.extend(chunk(b"alpha"));
        expected.extend(number(2));
        expected.extend(number(1));
        expected.extend(number(4));
        expected.extend(chunk(b"beta"));
        expected.extend(number(1));
        expected.extend(number(2));

        // Push carries the tall alpha block at leaf zero and the short beta block at leaf four.
        expected.extend(number(2));
        for field in [0, 0, 0, 2, 0] {
            expected.extend(number(field));
        }
        for field in [1, 1, 0, 1, 4] {
            expected.extend(number(field));
        }

        // Pull carries the tall beta block alone.
        expected.extend(number(1));
        for field in [1, 0, 1, 2, 0] {
            expected.extend(number(field));
        }

        assert_eq!(label(&plan), expected);
    }

    #[test]
    fn one_step_from_a_plan_never_shares_its_seed() {
        // Baseline: one table, one push and one pull declaration on the same named bus.
        let base = [
            interaction("alpha", BusDirection::Push, 2),
            interaction("alpha", BusDirection::Pull, 2),
        ];
        let plan = |interactions: &[SymbolicBusInteraction<F>], log_height| {
            BusPlan::build(&[BusPlanInput {
                log_height,
                interactions,
            }])
            .unwrap()
            .unwrap()
        };
        let baseline = plan(&base, 3);

        // Each of these moves exactly one public dimension away from the baseline.
        let renamed = [
            interaction("omega", BusDirection::Push, 2),
            interaction("omega", BusDirection::Pull, 2),
        ];
        let widened = [
            interaction("alpha", BusDirection::Push, 3),
            interaction("alpha", BusDirection::Pull, 3),
        ];
        let reordered = [
            interaction("alpha", BusDirection::Pull, 2),
            interaction("alpha", BusDirection::Push, 2),
        ];
        let split = [
            interaction("alpha", BusDirection::Push, 2),
            interaction("omega", BusDirection::Pull, 2),
        ];
        for changed in [
            plan(&renamed, 3),
            plan(&widened, 3),
            plan(&reordered, 3),
            plan(&split, 3),
            plan(&base, 4),
        ] {
            assert!(!seeds_agree(&baseline, &changed));
            assert_ne!(baseline, changed);
        }

        // Splitting the same declarations across two tables moves the owning AIR.
        let one = [interaction("alpha", BusDirection::Push, 2)];
        let other = [interaction("alpha", BusDirection::Pull, 2)];
        let two_tables = BusPlan::build(&[
            BusPlanInput {
                log_height: 3,
                interactions: &one,
            },
            BusPlanInput {
                log_height: 3,
                interactions: &other,
            },
        ])
        .unwrap()
        .unwrap();
        assert!(!seeds_agree(&baseline, &two_tables));
    }

    #[test]
    fn every_distinct_plan_in_a_sweep_lands_on_its_own_seed() {
        // Sweep every layout reachable from two names, two widths, two heights, two tables.
        let names = ["alpha", "beta"];
        let mut seen: Vec<(BusPlan, Vec<u8>)> = Vec::new();
        for push_name in names {
            for pull_name in names {
                for width in 1..=3 {
                    for log_height in 0..3 {
                        for second_table in [false, true] {
                            let push = [interaction(push_name, BusDirection::Push, width)];
                            let pull = [interaction(pull_name, BusDirection::Pull, width)];
                            let both = [push[0].clone(), pull[0].clone()];
                            let inputs: Vec<BusPlanInput<'_, F>> = if second_table {
                                alloc::vec![
                                    BusPlanInput {
                                        log_height,
                                        interactions: &push,
                                    },
                                    BusPlanInput {
                                        log_height,
                                        interactions: &pull,
                                    },
                                ]
                            } else {
                                alloc::vec![BusPlanInput {
                                    log_height,
                                    interactions: &both,
                                }]
                            };
                            let plan = BusPlan::build(&inputs).unwrap().unwrap();
                            let bytes = label(&plan);
                            for (other, other_bytes) in &seen {
                                if other == &plan {
                                    assert_eq!(other_bytes, &bytes);
                                } else {
                                    assert_ne!(other_bytes, &bytes);
                                }
                            }
                            seen.push((plan, bytes));
                        }
                    }
                }
            }
        }
        assert_eq!(seen.len(), 72);
    }

    // Eight leaves per side make the product reduction genuinely challenge-dependent.
    fn balanced_witness(_: &BusChallenges<EF>) -> [Vec<EF>; 2] {
        let side = (0..8)
            .map(|value| EF::from_u8(value + 3))
            .collect::<Vec<_>>();
        [side.clone(), side]
    }

    #[test]
    fn a_product_proof_for_one_named_bus_is_rejected_under_another() {
        // Both plans have identical geometry, so only the bus name separates them.
        let for_bus = |name: &str| {
            let interactions = [
                interaction(name, BusDirection::Push, 1),
                interaction(name, BusDirection::Pull, 1),
            ];
            BusPlan::build(&[BusPlanInput {
                log_height: 3,
                interactions: &interactions,
            }])
            .unwrap()
            .unwrap()
        };
        let alpha = for_bus("alpha");
        let omega = for_bus("omega");
        assert_eq!(alpha.product_shape(), omega.product_shape());
        assert_eq!(
            alpha.security_geometry().non_padding_leaf_counts(),
            omega.security_geometry().non_padding_leaf_counts()
        );

        let (proof, output) = alpha
            .prove::<F, EF, _>(balanced_witness, &mut challenger())
            .unwrap();
        let replayed = alpha
            .verify::<F, EF, _>(&proof, &mut challenger())
            .expect("the proof verifies under the plan that produced it");
        assert_eq!(replayed, output);

        // The same bytes carry no claim about a differently named bus.
        omega
            .verify::<F, EF, _>(&proof, &mut challenger())
            .expect_err("a claim about one named bus is not a claim about another");
    }

    #[test]
    fn a_product_proof_for_one_table_layout_is_rejected_under_another() {
        // Identical declarations, one moved to a second table of the same height.
        let together = [
            interaction("alpha", BusDirection::Push, 1),
            interaction("alpha", BusDirection::Pull, 1),
        ];
        let one = [interaction("alpha", BusDirection::Push, 1)];
        let other = [interaction("alpha", BusDirection::Pull, 1)];
        let single = BusPlan::build(&[BusPlanInput {
            log_height: 3,
            interactions: &together,
        }])
        .unwrap()
        .unwrap();
        let paired = BusPlan::build(&[
            BusPlanInput {
                log_height: 3,
                interactions: &one,
            },
            BusPlanInput {
                log_height: 3,
                interactions: &other,
            },
        ])
        .unwrap()
        .unwrap();
        assert_eq!(single.product_shape(), paired.product_shape());

        let (proof, _) = single
            .prove::<F, EF, _>(balanced_witness, &mut challenger())
            .unwrap();
        paired
            .verify::<F, EF, _>(&proof, &mut challenger())
            .expect_err("a claim about one table layout is not a claim about another");
    }
}
