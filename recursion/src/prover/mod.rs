use p3_uni_stark::{Proof, StarkGenericConfig, Val, prove as base_prove};

use crate::air::AluAir;
use crate::air::alu::air::FieldOperation;
use crate::air::asic::Asic;
use crate::circuit_builder::gates::event::AllEvents;
pub struct RecursiveProof<SC: StarkGenericConfig> {
    pub add_air: AluAir<1>,
    pub sub_air: AluAir<1>,
    pub add_proof: Proof<SC>,
    pub sub_proof: Proof<SC>,
}

pub fn prove<SC>(
    config: &SC,
    asic: Asic<Val<SC>>,
    all_events: AllEvents<Val<SC>>,
) -> RecursiveProof<SC>
where
    SC: StarkGenericConfig,
{
    let traces = asic.generate_trace(&all_events);

    let add_air: AluAir<1> = AluAir {
        op: FieldOperation::Add,
    };
    let sub_air: AluAir<1> = AluAir {
        op: FieldOperation::Sub,
    };

    let add_proof = base_prove(config, &add_air, traces[0].clone(), &vec![]);
    let sub_proof = base_prove(config, &sub_air, traces[1].clone(), &vec![]);

    RecursiveProof {
        add_air,
        sub_air,
        add_proof,
        sub_proof,
    }
}
