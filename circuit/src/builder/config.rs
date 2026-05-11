use hashbrown::HashMap;

use crate::ops::{NpoConfig, NpoTypeId};

/// Configuration for the circuit builder.
#[derive(Debug)]
pub struct BuilderConfig {
    /// Enabled non-primitive operation types with their respective configuration.
    enabled_ops: HashMap<NpoTypeId, NpoConfig>,
}

impl Default for BuilderConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl BuilderConfig {
    /// Creates a new builder configuration.
    pub fn new() -> Self {
        Self {
            enabled_ops: HashMap::new(),
        }
    }

    /// Enables a non-primitive operation type with its configuration.
    pub fn enable_op(&mut self, op: NpoTypeId, cfg: NpoConfig) {
        self.enabled_ops.insert(op, cfg);
    }

    /// Checks whether an operation type is enabled.
    pub fn is_op_enabled(&self, op: &NpoTypeId) -> bool {
        self.enabled_ops.contains_key(op)
    }

    /// Consumes the config and returns the enabled operations map.
    pub fn into_enabled_ops(self) -> HashMap<NpoTypeId, NpoConfig> {
        self.enabled_ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Poseidon2Config;

    #[test]
    fn test_builder_config_default() {
        let config = BuilderConfig::default();
        assert!(!config.is_op_enabled(&NpoTypeId::poseidon2_perm(
            Poseidon2Config::BabyBearD4Width16,
        )));
    }

    #[test]
    fn test_builder_config_enable_op() {
        let mut config = BuilderConfig::new();

        config.enable_op(
            NpoTypeId::poseidon2_perm(Poseidon2Config::BabyBearD4Width16),
            NpoConfig::new(()),
        );

        assert!(config.is_op_enabled(&NpoTypeId::poseidon2_perm(
            Poseidon2Config::BabyBearD4Width16,
        )));
    }
}
