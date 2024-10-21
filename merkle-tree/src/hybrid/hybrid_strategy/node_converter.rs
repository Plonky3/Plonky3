pub struct NodeConversor256BabyBearBytes {}

impl NodeConverter<[BabyBear; 8], [u8; 32]> for NodeConversor256BabyBearBytes {
    fn to_n1(&self, input: [BabyBear; 8]) -> [u8; 32] {
        todo!()
    }
}
