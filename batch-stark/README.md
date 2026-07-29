# p3-batch-stark

Batched STARK proving and verification atop `p3-uni-stark`: proves several
AIR instances (of possibly different heights) under a single commitment and
a shared FRI opening, with optional cross-instance lookups.

```rust,ignore
use p3_batch_stark::{prove_batch, verify_batch, ProverData, StarkInstance};

let instances = vec![
    StarkInstance { air: &air1, trace: trace1, public_values: pv1, lookups: vec![] },
    StarkInstance { air: &air2, trace: trace2, public_values: pv2, lookups: vec![] },
];

let prover_data = ProverData::from_instances(&config, &instances);
let common = &prover_data.common;
let proof = prove_batch(&config, &instances, &prover_data);
verify_batch(&config, &[air1, air2], &proof, &[pv1, pv2], common)?;
```

Part of [Plonky3](https://github.com/Plonky3/Plonky3), dual-licensed under MIT and Apache 2.0.
