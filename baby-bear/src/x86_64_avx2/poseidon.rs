use p3_monty_31::PartialRoundParametersAVX2;

use crate::{BabyBearParameters, BabyBearPoseidonParameters};

impl PartialRoundParametersAVX2<BabyBearParameters, 16> for BabyBearPoseidonParameters {}
impl PartialRoundParametersAVX2<BabyBearParameters, 24> for BabyBearPoseidonParameters {}
