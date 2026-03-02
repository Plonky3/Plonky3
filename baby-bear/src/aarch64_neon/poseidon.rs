use p3_monty_31::PartialRoundParametersNeon;

use crate::{BabyBearParameters, BabyBearPoseidonParameters};

impl PartialRoundParametersNeon<BabyBearParameters, 16> for BabyBearPoseidonParameters {}
impl PartialRoundParametersNeon<BabyBearParameters, 24> for BabyBearPoseidonParameters {}
