//! Poseidon2 configuration types and execution closures.

use alloc::format;
use alloc::sync::Arc;
use alloc::vec::Vec;

use p3_field::Field;
use serde::{Deserialize, Serialize};

use crate::CircuitBuilderError;
use crate::builder::NpoLoweringContext;
use crate::types::{ExprId, WitnessId};

/// Poseidon2 configuration used as a stable operation key and parameter source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub enum Poseidon2Config {
    /// BabyBear with extension degree D=1 (base field challenges), width 16.
    BabyBearD1Width16,
    BabyBearD4Width16,
    BabyBearD4Width24,
    /// KoalaBear with extension degree D=1 (base field challenges), width 16.
    KoalaBearD1Width16,
    KoalaBearD4Width16,
    KoalaBearD4Width24,
    /// Goldilocks with extension degree D=2, width 8 (matches Poseidon2Goldilocks<8>).
    GoldilocksD2Width8,
}

impl Poseidon2Config {
    pub const fn d(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::KoalaBearD1Width16 => 1,
            Self::GoldilocksD2Width8 => 2,
            Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::KoalaBearD4Width16
            | Self::KoalaBearD4Width24 => 4,
        }
    }

    pub const fn width(self) -> usize {
        match self {
            Self::BabyBearD1Width16
            | Self::BabyBearD4Width16
            | Self::KoalaBearD1Width16
            | Self::KoalaBearD4Width16 => 16,
            Self::BabyBearD4Width24 | Self::KoalaBearD4Width24 => 24,
            Self::GoldilocksD2Width8 => 8,
        }
    }

    /// Rate in extension field elements (WIDTH / D for D=4, or WIDTH for D=1).
    pub const fn rate_ext(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::KoalaBearD1Width16 => 8,
            Self::BabyBearD4Width16 | Self::KoalaBearD4Width16 => 2,
            Self::BabyBearD4Width24 | Self::KoalaBearD4Width24 => 4,
            Self::GoldilocksD2Width8 => 2,
        }
    }

    pub const fn rate(self) -> usize {
        self.rate_ext() * self.d()
    }

    /// Capacity in extension field elements.
    pub const fn capacity_ext(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::KoalaBearD1Width16 => 8,
            Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::KoalaBearD4Width16
            | Self::KoalaBearD4Width24 => 2,
            Self::GoldilocksD2Width8 => 2,
        }
    }

    pub const fn sbox_degree(self) -> u64 {
        match self {
            Self::BabyBearD1Width16 | Self::BabyBearD4Width16 | Self::BabyBearD4Width24 => 7,
            Self::KoalaBearD1Width16 | Self::KoalaBearD4Width16 | Self::KoalaBearD4Width24 => 3,
            Self::GoldilocksD2Width8 => 7,
        }
    }

    pub const fn sbox_registers(self) -> usize {
        match self {
            Self::BabyBearD1Width16
            | Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::GoldilocksD2Width8 => 1,
            Self::KoalaBearD1Width16 | Self::KoalaBearD4Width16 | Self::KoalaBearD4Width24 => 0,
        }
    }

    pub const fn half_full_rounds(self) -> usize {
        match self {
            Self::BabyBearD1Width16
            | Self::BabyBearD4Width16
            | Self::BabyBearD4Width24
            | Self::KoalaBearD1Width16
            | Self::KoalaBearD4Width16
            | Self::KoalaBearD4Width24
            | Self::GoldilocksD2Width8 => 4,
        }
    }

    pub const fn partial_rounds(self) -> usize {
        match self {
            Self::BabyBearD1Width16 | Self::BabyBearD4Width16 => 13,
            Self::BabyBearD4Width24 => 21,
            Self::KoalaBearD1Width16 | Self::KoalaBearD4Width16 => 20,
            Self::KoalaBearD4Width24 => 23,
            Self::GoldilocksD2Width8 => 22,
        }
    }

    pub const fn width_ext(self) -> usize {
        self.rate_ext() + self.capacity_ext()
    }

    /// Check that input and output counts match this config's expected layout.
    ///
    /// - For D=1: `add_poseidon2_perm` always supplies `width_ext + 2` input slots (MMCS slots may
    ///   be empty when `merkle_path` is false). `add_poseidon2_perm_base` uses exactly `width` inputs.
    /// - For D=1 with Merkle (`merkle_path`): inputs must be `width_ext + 2`.
    /// - For D>1: expects `width_ext + 2` inputs and `rate_ext` or `width_ext` outputs.
    ///
    /// # Errors
    ///
    /// Returns `NonPrimitiveOpArity` if counts do not match.
    pub fn validate_io_counts(
        self,
        input_count: usize,
        output_count: usize,
        merkle_path: bool,
    ) -> Result<(), CircuitBuilderError> {
        let is_d1 = self.d() == 1;
        let inputs_ok = if is_d1 {
            if merkle_path {
                input_count == self.width_ext() + 2
            } else {
                input_count == self.width() || input_count == self.width_ext() + 2
            }
        } else {
            input_count == self.width_ext() + 2
        };
        if !inputs_ok {
            let expected = if is_d1 {
                if merkle_path {
                    format!("{} inputs", self.width_ext() + 2)
                } else {
                    format!(
                        "{} or {} inputs for D=1 mode",
                        self.width(),
                        self.width_ext() + 2
                    )
                }
            } else {
                format!("{} inputs", self.width_ext() + 2)
            };
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Poseidon2Perm",
                expected,
                got: input_count,
            });
        }

        let valid_output_count = if is_d1 {
            output_count == self.rate() || output_count == self.width()
        } else {
            output_count == self.rate_ext() || output_count == self.width_ext()
        };
        if !valid_output_count {
            return Err(CircuitBuilderError::NonPrimitiveOpArity {
                op: "Poseidon2Perm",
                expected: if is_d1 {
                    format!("{} or {} outputs for D=1 mode", self.rate(), self.width())
                } else {
                    format!(
                        "{} or {} outputs for D>1 mode",
                        self.rate_ext(),
                        self.width_ext()
                    )
                },
                got: output_count,
            });
        }

        Ok(())
    }

    /// Convert input expressions to witness indices according to this config's layout.
    ///
    /// - D=1 with `width` inputs (`add_poseidon2_perm_base`): flat slots.
    /// - Otherwise: `width_ext` limb slots, then MMCS index accumulator and direction bit.
    pub fn lower_inputs<F: Field>(
        self,
        input_exprs: &[Vec<ExprId>],
        ctx: &NpoLoweringContext<'_, F>,
        merkle_path: bool,
    ) -> Result<Vec<Vec<WitnessId>>, CircuitBuilderError> {
        if self.d() == 1 && !merkle_path && input_exprs.len() == self.width() {
            return ctx.lower_expr_slots(input_exprs, "Poseidon2Perm", "D=1 input");
        }

        let width_ext = self.width_ext();
        let mut widx =
            ctx.lower_expr_slots(&input_exprs[..width_ext], "Poseidon2Perm", "input limb")?;

        let [mmcs_sum] = ctx
            .lower_expr_slots(
                &input_exprs[width_ext..=width_ext],
                "Poseidon2Perm",
                "mmcs_index_sum",
            )?
            .try_into()
            .expect("single-element slice must yield single-element vec");
        widx.push(mmcs_sum);

        let [mmcs_bit] = ctx
            .lower_expr_slots(
                &input_exprs[width_ext + 1..=width_ext + 1],
                "Poseidon2Perm",
                "mmcs_bit",
            )?
            .try_into()
            .expect("single-element slice must yield single-element vec");
        widx.push(mmcs_bit);

        Ok(widx)
    }

    /// Stable string name for this config variant, used to build `NpoTypeId`.
    pub const fn variant_name(self) -> &'static str {
        match self {
            Self::BabyBearD1Width16 => "baby_bear_d1_w16",
            Self::BabyBearD4Width16 => "baby_bear_d4_w16",
            Self::BabyBearD4Width24 => "baby_bear_d4_w24",
            Self::KoalaBearD1Width16 => "koala_bear_d1_w16",
            Self::KoalaBearD4Width16 => "koala_bear_d4_w16",
            Self::KoalaBearD4Width24 => "koala_bear_d4_w24",
            Self::GoldilocksD2Width8 => "goldilocks_d2_w8",
        }
    }

    /// Parse a `Poseidon2Config` from a variant name string.
    pub fn from_variant_name(name: &str) -> Option<Self> {
        match name {
            "baby_bear_d1_w16" => Some(Self::BabyBearD1Width16),
            "baby_bear_d4_w16" => Some(Self::BabyBearD4Width16),
            "baby_bear_d4_w24" => Some(Self::BabyBearD4Width24),
            "koala_bear_d1_w16" => Some(Self::KoalaBearD1Width16),
            "koala_bear_d4_w16" => Some(Self::KoalaBearD4Width16),
            "koala_bear_d4_w24" => Some(Self::KoalaBearD4Width24),
            "goldilocks_d2_w8" => Some(Self::GoldilocksD2Width8),
            _ => None,
        }
    }
}

/// Poseidon2 permutation execution closure.
///
/// Takes `width_ext` field elements and returns `width_ext` output elements.
/// For D=1 mode, `width_ext == width` and the elements are base field values.
pub type Poseidon2PermExec<F> = Arc<dyn Fn(&[F]) -> Vec<F> + Send + Sync>;

/// Config data stored inside `NpoConfig` for Poseidon2 operations.
///
/// Stored behind `NpoConfig(Arc<dyn Any>)`, so cloning happens at the Arc level.
pub struct Poseidon2PermConfigData<F> {
    pub exec: Poseidon2PermExec<F>,
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use hashbrown::HashMap;
    use p3_test_utils::baby_bear_params::BabyBear;

    use super::*;
    use crate::builder::NpoLoweringContext;

    type F = BabyBear;

    #[test]
    fn validate_io_counts_d4_w16_ok() {
        let cfg = Poseidon2Config::BabyBearD4Width16;
        // width_ext=4, rate_ext=2
        assert!(cfg.validate_io_counts(4 + 2, 2, false).is_ok()); // inputs=6, outputs=rate
        assert!(cfg.validate_io_counts(6, 4, true).is_ok()); // outputs=width
    }

    #[test]
    fn validate_io_counts_d1_w16_ok() {
        let cfg = Poseidon2Config::BabyBearD1Width16;
        assert!(cfg.validate_io_counts(16, 8, false).is_ok());
        assert!(cfg.validate_io_counts(16, 16, false).is_ok());
        assert!(cfg.validate_io_counts(18, 8, false).is_ok());
        assert!(cfg.validate_io_counts(18, 16, false).is_ok());
    }

    #[test]
    fn validate_io_counts_d1_w16_merkle_ok() {
        let cfg = Poseidon2Config::BabyBearD1Width16;
        assert!(cfg.validate_io_counts(18, 8, true).is_ok());
        assert!(cfg.validate_io_counts(18, 16, true).is_ok());
    }

    #[test]
    fn validate_io_counts_d1_merkle_wrong_input_len_errors() {
        let cfg = Poseidon2Config::BabyBearD1Width16;
        let Err(CircuitBuilderError::NonPrimitiveOpArity { expected, got, .. }) =
            cfg.validate_io_counts(16, 8, true)
        else {
            panic!("expected NonPrimitiveOpArity");
        };
        assert_eq!(expected, "18 inputs");
        assert_eq!(got, 16);
    }

    #[test]
    fn validate_io_counts_wrong_inputs_errors() {
        let cfg = Poseidon2Config::BabyBearD4Width16;
        let Err(CircuitBuilderError::NonPrimitiveOpArity { op, expected, got }) =
            cfg.validate_io_counts(3, 2, false)
        else {
            panic!("expected NonPrimitiveOpArity");
        };
        assert_eq!(op, "Poseidon2Perm");
        assert_eq!(expected, "6 inputs");
        assert_eq!(got, 3);
    }

    #[test]
    fn validate_io_counts_wrong_outputs_errors() {
        let cfg = Poseidon2Config::BabyBearD4Width16;
        let Err(CircuitBuilderError::NonPrimitiveOpArity { op, expected, got }) =
            cfg.validate_io_counts(6, 3, false)
        else {
            panic!("expected NonPrimitiveOpArity");
        };
        assert_eq!(op, "Poseidon2Perm");
        assert_eq!(expected, "2 or 4 outputs for D>1 mode");
        assert_eq!(got, 3);
    }

    #[test]
    fn validate_io_counts_d1_wrong_outputs_errors() {
        let cfg = Poseidon2Config::BabyBearD1Width16;
        let Err(CircuitBuilderError::NonPrimitiveOpArity { op, expected, got }) =
            cfg.validate_io_counts(16, 5, false)
        else {
            panic!("expected NonPrimitiveOpArity");
        };
        assert_eq!(op, "Poseidon2Perm");
        assert_eq!(expected, "8 or 16 outputs for D=1 mode");
        assert_eq!(got, 5);
    }

    #[test]
    fn lower_inputs_d4_produces_correct_structure() {
        let cfg = Poseidon2Config::BabyBearD4Width16;

        let mut map = HashMap::new();
        for i in 0u32..6 {
            map.insert(ExprId(i), WitnessId(100 + i));
        }
        let mut counter = 200u32;
        let mut alloc = |_: usize| {
            let id = WitnessId(counter);
            counter += 1;
            id
        };
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let input_exprs: Vec<Vec<ExprId>> = (0u32..6).map(|i| vec![ExprId(i)]).collect();
        let result = cfg.lower_inputs(&input_exprs, &ctx, false).unwrap();

        let expected: Vec<Vec<WitnessId>> = (0u32..6).map(|i| vec![WitnessId(100 + i)]).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn lower_inputs_d1_produces_flat_structure() {
        let cfg = Poseidon2Config::BabyBearD1Width16;

        let mut map = HashMap::new();
        for i in 0u32..16 {
            map.insert(ExprId(i), WitnessId(i));
        }
        let mut counter = 100u32;
        let mut alloc = |_: usize| {
            let id = WitnessId(counter);
            counter += 1;
            id
        };
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let input_exprs: Vec<Vec<ExprId>> = (0u32..16).map(|i| vec![ExprId(i)]).collect();
        let result = cfg.lower_inputs(&input_exprs, &ctx, false).unwrap();

        let expected: Vec<Vec<WitnessId>> = (0u32..16).map(|i| vec![WitnessId(i)]).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn lower_inputs_d1_merkle_matches_ext_layout() {
        let cfg = Poseidon2Config::BabyBearD1Width16;
        let mut map = HashMap::new();
        for i in 0u32..18 {
            map.insert(ExprId(i), WitnessId(100 + i));
        }
        let mut counter = 300u32;
        let mut alloc = |_: usize| {
            let id = WitnessId(counter);
            counter += 1;
            id
        };
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let input_exprs: Vec<Vec<ExprId>> = (0u32..18).map(|i| vec![ExprId(i)]).collect();
        let result = cfg.lower_inputs(&input_exprs, &ctx, true).unwrap();

        let expected: Vec<Vec<WitnessId>> = (0u32..18).map(|i| vec![WitnessId(100 + i)]).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn lower_inputs_d1_non_merkle_18_slot_layout() {
        let cfg = Poseidon2Config::BabyBearD1Width16;
        let mut map = HashMap::new();
        for i in 0u32..18 {
            map.insert(ExprId(i), WitnessId(200 + i));
        }
        let mut counter = 400u32;
        let mut alloc = |_: usize| {
            let id = WitnessId(counter);
            counter += 1;
            id
        };
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let input_exprs: Vec<Vec<ExprId>> = (0u32..18).map(|i| vec![ExprId(i)]).collect();
        let result = cfg.lower_inputs(&input_exprs, &ctx, false).unwrap();

        let expected: Vec<Vec<WitnessId>> = (0u32..18).map(|i| vec![WitnessId(200 + i)]).collect();
        assert_eq!(result, expected);
    }

    #[test]
    fn lower_inputs_d4_with_empty_slots() {
        let cfg = Poseidon2Config::BabyBearD4Width16;

        let mut map = HashMap::new();
        map.insert(ExprId(10), WitnessId(50));
        map.insert(ExprId(11), WitnessId(51));
        let mut counter = 200u32;
        let mut alloc = |_: usize| {
            let id = WitnessId(counter);
            counter += 1;
            id
        };
        let ctx = NpoLoweringContext::<F>::new(&mut map, &mut alloc);

        let mut input_exprs: Vec<Vec<ExprId>> = vec![vec![]; cfg.width_ext()];
        input_exprs.push(vec![ExprId(10)]);
        input_exprs.push(vec![ExprId(11)]);

        let result = cfg.lower_inputs(&input_exprs, &ctx, false).unwrap();

        assert_eq!(
            result,
            vec![
                vec![],
                vec![],
                vec![],
                vec![],
                vec![WitnessId(50)],
                vec![WitnessId(51)],
            ]
        );
    }

    #[test]
    fn variant_name_roundtrip_all_configs() {
        let configs = [
            Poseidon2Config::BabyBearD1Width16,
            Poseidon2Config::BabyBearD4Width16,
            Poseidon2Config::BabyBearD4Width24,
            Poseidon2Config::KoalaBearD1Width16,
            Poseidon2Config::KoalaBearD4Width16,
            Poseidon2Config::KoalaBearD4Width24,
            Poseidon2Config::GoldilocksD2Width8,
        ];
        for cfg in configs {
            let name = cfg.variant_name();
            let parsed = Poseidon2Config::from_variant_name(name);
            assert_eq!(parsed, Some(cfg), "roundtrip failed for {name}");
        }
    }

    #[test]
    fn from_variant_name_unknown_returns_none() {
        assert_eq!(Poseidon2Config::from_variant_name("unknown"), None);
    }
}
