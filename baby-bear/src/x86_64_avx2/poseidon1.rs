use p3_monty_31::PartialRoundParametersAVX2;

use crate::{BabyBearParameters, BabyBearPoseidon1Parameters};

impl PartialRoundParametersAVX2<BabyBearParameters, 16> for BabyBearPoseidon1Parameters {}
impl PartialRoundParametersAVX2<BabyBearParameters, 24> for BabyBearPoseidon1Parameters {}
