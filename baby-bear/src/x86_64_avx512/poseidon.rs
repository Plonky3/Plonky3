use p3_monty_31::PartialRoundParametersAVX512;

use crate::{BabyBearParameters, BabyBearPoseidonParameters};

impl PartialRoundParametersAVX512<BabyBearParameters, 16> for BabyBearPoseidonParameters {}
impl PartialRoundParametersAVX512<BabyBearParameters, 24> for BabyBearPoseidonParameters {}
