// #[derive(Debug, Serialize, Deserialize)]
// pub struct MerkleTree<F, W, M, const DIGEST_ELEMS: usize> {
//     pub(crate) leaves: Vec<M>,
//     // Enable serialization for this type whenever the underlying array type supports it (len 1-32).
//     #[serde(bound(serialize = "[W; DIGEST_ELEMS]: Serialize"))]
//     // Enable deserialization for this type whenever the underlying array type supports it (len 1-32).
//     #[serde(bound(deserialize = "[W; DIGEST_ELEMS]: Deserialize<'de>"))]
//     pub(crate) digest_layers: Vec<Vec<[W; DIGEST_ELEMS]>>,
//     _phantom: PhantomData<F>,
// }

// #[instrument(name = "build merkle tree", level = "debug", skip_all,
//              fields(dimensions = alloc::format!("{:?}", leaves.iter().map(|l| l.dimensions()).collect::<Vec<_>>())))]
// pub fn new<P, PW, H, C>(h: &H, c: &C, leaves: Vec<M>) -> Self
// where
//     P: PackedValue<Value = F>,
//     PW: PackedValue<Value = W>,
//     H: CryptographicHasher<F, [W; DIGEST_ELEMS]>,
//     H: CryptographicHasher<P, [PW; DIGEST_ELEMS]>,
//     H: Sync,
//     C: PseudoCompressionFunction<[W; DIGEST_ELEMS], 2>,
//     C: PseudoCompressionFunction<[PW; DIGEST_ELEMS], 2>,
//     C: Sync,
