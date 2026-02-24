use p3_monty_31::PartialRoundParametersAVX512;

use crate::{BabyBearParameters, BabyBearPoseidon1Parameters};

impl PartialRoundParametersAVX512<BabyBearParameters, 16> for BabyBearPoseidon1Parameters {}
impl PartialRoundParametersAVX512<BabyBearParameters, 24> for BabyBearPoseidon1Parameters {}
