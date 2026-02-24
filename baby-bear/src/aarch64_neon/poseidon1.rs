use p3_monty_31::PartialRoundParametersNeon;

use crate::{BabyBearParameters, BabyBearPoseidon1Parameters};

impl PartialRoundParametersNeon<BabyBearParameters, 16> for BabyBearPoseidon1Parameters {}
impl PartialRoundParametersNeon<BabyBearParameters, 24> for BabyBearPoseidon1Parameters {}
