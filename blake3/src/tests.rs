use alloc::vec;
use alloc::vec::Vec;

use blake3::{BLOCK_LEN, CHUNK_LEN, OUT_LEN};
use hex_literal::hex;
use p3_symmetric::CryptographicHasher;
use proptest::prelude::*;

use crate::batch::{self, Kernel, Mode};
use crate::{Blake3, LANES};

/// Spec table 3: flag of the keyed hash mode.
const KEYED_HASH: u32 = 1 << 4;

/// Spec table 3: flag of the context string hash in key derivation.
const DERIVE_KEY_CONTEXT: u32 = 1 << 5;

/// Spec table 3: flag of the key material hash in key derivation.
const DERIVE_KEY_MATERIAL: u32 = 1 << 6;

/// Key of the official test vectors.
const KEY: &[u8; 32] = b"whats the Elvish word for friend";

/// Context string of the official test vectors.
const CONTEXT: &[u8] = b"BLAKE3 2019-12-27 16:29:52 test vectors context";

/// One official test vector, cut to the 32-byte digest.
struct Vector {
    len: usize,
    hash: [u8; OUT_LEN],
    keyed_hash: [u8; OUT_LEN],
    derive_key: [u8; OUT_LEN],
}

/// `test_vectors.json` from the BLAKE3 repository.
///
/// The input of each vector is the byte sequence 0, 1, ..., 250, repeated.
const VECTORS: &[Vector] = &[
    Vector {
        len: 0,
        hash: hex!("af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
        keyed_hash: hex!("92b2b75604ed3c761f9d6f62392c8a9227ad0ea3f09573e783f1498a4ed60d26"),
        derive_key: hex!("2cc39783c223154fea8dfb7c1b1660f2ac2dcbd1c1de8277b0b0dd39b7e50d7d"),
    },
    Vector {
        len: 1,
        hash: hex!("2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
        keyed_hash: hex!("6d7878dfff2f485635d39013278ae14f1454b8c0a3a2d34bc1ab38228a80c95b"),
        derive_key: hex!("b3e2e340a117a499c6cf2398a19ee0d29cca2bb7404c73063382693bf66cb06c"),
    },
    Vector {
        len: 2,
        hash: hex!("7b7015bb92cf0b318037702a6cdd81dee41224f734684c2c122cd6359cb1ee63"),
        keyed_hash: hex!("5392ddae0e0a69d5f40160462cbd9bd889375082ff224ac9c758802b7a6fd20a"),
        derive_key: hex!("1f166565a7df0098ee65922d7fea425fb18b9943f19d6161e2d17939356168e6"),
    },
    Vector {
        len: 3,
        hash: hex!("e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f"),
        keyed_hash: hex!("39e67b76b5a007d4921969779fe666da67b5213b096084ab674742f0d5ec62b9"),
        derive_key: hex!("440aba35cb006b61fc17c0529255de438efc06a8c9ebf3f2ddac3b5a86705797"),
    },
    Vector {
        len: 4,
        hash: hex!("f30f5ab28fe047904037f77b6da4fea1e27241c5d132638d8bedce9d40494f32"),
        keyed_hash: hex!("7671dde590c95d5ac9616651ff5aa0a27bee5913a348e053b8aa9108917fe070"),
        derive_key: hex!("f46085c8190d69022369ce1a18880e9b369c135eb93f3c63550d3e7630e91060"),
    },
    Vector {
        len: 5,
        hash: hex!("b40b44dfd97e7a84a996a91af8b85188c66c126940ba7aad2e7ae6b385402aa2"),
        keyed_hash: hex!("73ac69eecf286894d8102018a6fc729f4b1f4247d3703f69bdc6a5fe3e0c8461"),
        derive_key: hex!("1f24eda69dbcb752847ec3ebb5dd42836d86e58500c7c98d906ecd82ed9ae47f"),
    },
    Vector {
        len: 6,
        hash: hex!("06c4e8ffb6872fad96f9aaca5eee1553eb62aed0ad7198cef42e87f6a616c844"),
        keyed_hash: hex!("82d3199d0013035682cc7f2a399d4c212544376a839aa863a0f4c91220ca7a6d"),
        derive_key: hex!("be96b30b37919fe4379dfbe752ae77b4f7e2ab92f7ff27435f76f2f065f6a5f4"),
    },
    Vector {
        len: 7,
        hash: hex!("3f8770f387faad08faa9d8414e9f449ac68e6ff0417f673f602a646a891419fe"),
        keyed_hash: hex!("af0a7ec382aedc0cfd626e49e7628bc7a353a4cb108855541a5651bf64fbb28a"),
        derive_key: hex!("dc3b6485f9d94935329442916b0d059685ba815a1fa2a14107217453a7fc9f0e"),
    },
    Vector {
        len: 8,
        hash: hex!("2351207d04fc16ade43ccab08600939c7c1fa70a5c0aaca76063d04c3228eaeb"),
        keyed_hash: hex!("be2f5495c61cba1bb348a34948c004045e3bd4dae8f0fe82bf44d0da245a0600"),
        derive_key: hex!("2b166978cef14d9d438046c720519d8b1cad707e199746f1562d0c87fbd32940"),
    },
    Vector {
        len: 63,
        hash: hex!("e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"),
        keyed_hash: hex!("bb1eb5d4afa793c1ebdd9fb08def6c36d10096986ae0cfe148cd101170ce37ae"),
        derive_key: hex!("b6451e30b953c206e34644c6803724e9d2725e0893039cfc49584f991f451af3"),
    },
    Vector {
        len: 64,
        hash: hex!("4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"),
        keyed_hash: hex!("ba8ced36f327700d213f120b1a207a3b8c04330528586f414d09f2f7d9ccb7e6"),
        derive_key: hex!("a5c4a7053fa86b64746d4bb688d06ad1f02a18fce9afd3e818fefaa7126bf73e"),
    },
    Vector {
        len: 65,
        hash: hex!("de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"),
        keyed_hash: hex!("c0a4edefa2d2accb9277c371ac12fcdbb52988a86edc54f0716e1591b4326e72"),
        derive_key: hex!("51fd05c3c1cfbc8ed67d139ad76f5cf8236cd2acd26627a30c104dfd9d3ff8a8"),
    },
    Vector {
        len: 127,
        hash: hex!("d81293fda863f008c09e92fc382a81f5a0b4a1251cba1634016a0f86a6bd640d"),
        keyed_hash: hex!("c64200ae7dfaf35577ac5a9521c47863fb71514a3bcad18819218b818de85818"),
        derive_key: hex!("c91c090ceee3a3ac81902da31838012625bbcd73fcb92e7d7e56f78deba4f0c3"),
    },
    Vector {
        len: 128,
        hash: hex!("f17e570564b26578c33bb7f44643f539624b05df1a76c81f30acd548c44b45ef"),
        keyed_hash: hex!("b04fe15577457267ff3b6f3c947d93be581e7e3a4b018679125eaf86f6a628ec"),
        derive_key: hex!("81720f34452f58a0120a58b6b4608384b5c51d11f39ce97161a0c0e442ca0225"),
    },
    Vector {
        len: 129,
        hash: hex!("683aaae9f3c5ba37eaaf072aed0f9e30bac0865137bae68b1fde4ca2aebdcb12"),
        keyed_hash: hex!("d4a64dae6cdccbac1e5287f54f17c5f985105457c1a2ec1878ebd4b57e20d38f"),
        derive_key: hex!("938d2d4435be30eafdbb2b7031f7857c98b04881227391dc40db3c7b21f41fc1"),
    },
    Vector {
        len: 1023,
        hash: hex!("10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
        keyed_hash: hex!("c951ecdf03288d0fcc96ee3413563d8a6d3589547f2c2fb36d9786470f1b9d6e"),
        derive_key: hex!("74a16c1c3d44368a86e1ca6df64be6a2f64cce8f09220787450722d85725dea5"),
    },
    Vector {
        len: 1024,
        hash: hex!("42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
        keyed_hash: hex!("75c46f6f3d9eb4f55ecaaee480db732e6c2105546f1e675003687c31719c7ba4"),
        derive_key: hex!("7356cd7720d5b66b6d0697eb3177d9f8d73a4a5c5e968896eb6a689684302706"),
    },
    Vector {
        len: 1025,
        hash: hex!("d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
        keyed_hash: hex!("357dc55de0c7e382c900fd6e320acc04146be01db6a8ce7210b7189bd664ea69"),
        derive_key: hex!("effaa245f065fbf82ac186839a249707c3bddf6d3fdda22d1b95a3c970379bcb"),
    },
    Vector {
        len: 2048,
        hash: hex!("e776b6028c7cd22a4d0ba182a8bf62205d2ef576467e838ed6f2529b85fba24a"),
        keyed_hash: hex!("879cf1fa2ea0e79126cb1063617a05b6ad9d0b696d0d757cf053439f60a99dd1"),
        derive_key: hex!("7b2945cb4fef70885cc5d78a87bf6f6207dd901ff239201351ffac04e1088a23"),
    },
    Vector {
        len: 2049,
        hash: hex!("5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030"),
        keyed_hash: hex!("9f29700902f7c86e514ddc4df1e3049f258b2472b6dd5267f61bf13983b78dd5"),
        derive_key: hex!("2ea477c5515cc3dd606512ee72bb3e0e758cfae7232826f35fb98ca1bcbdf273"),
    },
    Vector {
        len: 3072,
        hash: hex!("b98cb0ff3623be03326b373de6b9095218513e64f1ee2edd2525c7ad1e5cffd2"),
        keyed_hash: hex!("044a0e7b172a312dc02a4c9a818c036ffa2776368d7f528268d2e6b5df191770"),
        derive_key: hex!("050df97f8c2ead654d9bb3ab8c9178edcd902a32f8495949feadcc1e0480c46b"),
    },
    Vector {
        len: 3073,
        hash: hex!("7124b49501012f81cc7f11ca069ec9226cecb8a2c850cfe644e327d22d3e1cd3"),
        keyed_hash: hex!("68dede9bef00ba89e43f31a6825f4cf433389fedae75c04ee9f0cf16a427c95a"),
        derive_key: hex!("72613c9ec9ff7e40f8f5c173784c532ad852e827dba2bf85b2ab4b76f7079081"),
    },
    Vector {
        len: 4096,
        hash: hex!("015094013f57a5277b59d8475c0501042c0b642e531b0a1c8f58d2163229e969"),
        keyed_hash: hex!("befc660aea2f1718884cd8deb9902811d332f4fc4a38cf7c7300d597a081bfc0"),
        derive_key: hex!("1e0d7f3db8c414c97c6307cbda6cd27ac3b030949da8e23be1a1a924ad2f25b9"),
    },
    Vector {
        len: 4097,
        hash: hex!("9b4052b38f1c5fc8b1f9ff7ac7b27cd242487b3d890d15c96a1c25b8aa0fb995"),
        keyed_hash: hex!("00df940cd36bb9fa7cbbc3556744e0dbc8191401afe70520ba292ee3ca80abbc"),
        derive_key: hex!("aca51029626b55fda7117b42a7c211f8c6e9ba4fe5b7a8ca922f34299500ead8"),
    },
    Vector {
        len: 5120,
        hash: hex!("9cadc15fed8b5d854562b26a9536d9707cadeda9b143978f319ab34230535833"),
        keyed_hash: hex!("2c493e48e9b9bf31e0553a22b23503c0a3388f035cece68eb438d22fa1943e20"),
        derive_key: hex!("7a7acac8a02adcf3038d74cdd1d34527de8a0fcc0ee3399d1262397ce5817f60"),
    },
    Vector {
        len: 5121,
        hash: hex!("628bd2cb2004694adaab7bbd778a25df25c47b9d4155a55f8fbd79f2fe154cff"),
        keyed_hash: hex!("6ccf1c34753e7a044db80798ecd0782a8f76f33563accaddbfbb2e0ea4b2d024"),
        derive_key: hex!("b07f01e518e702f7ccb44a267e9e112d403a7b3f4883a47ffbed4b48339b3c34"),
    },
    Vector {
        len: 6144,
        hash: hex!("3e2e5b74e048f3add6d21faab3f83aa44d3b2278afb83b80b3c35164ebeca205"),
        keyed_hash: hex!("3d6b6d21281d0ade5b2b016ae4034c5dec10ca7e475f90f76eac7138e9bc8f1d"),
        derive_key: hex!("2a95beae63ddce523762355cf4b9c1d8f131465780a391286a5d01abb5683a15"),
    },
    Vector {
        len: 6145,
        hash: hex!("f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f"),
        keyed_hash: hex!("9ac301e9e39e45e3250a7e3b3df701aa0fb6889fbd80eeecf28dbc6300fbc539"),
        derive_key: hex!("379bcc61d0051dd489f686c13de00d5b14c505245103dc040d9e4dd1facab8e5"),
    },
    Vector {
        len: 7168,
        hash: hex!("61da957ec2499a95d6b8023e2b0e604ec7f6b50e80a9678b89d2628e99ada77a"),
        keyed_hash: hex!("b42835e40e9d4a7f42ad8cc04f85a963a76e18198377ed84adddeaecacc6f3fc"),
        derive_key: hex!("11c37a112765370c94a51415d0d651190c288566e295d505defdad895dae2237"),
    },
    Vector {
        len: 7169,
        hash: hex!("a003fc7a51754a9b3c7fae0367ab3d782dccf28855a03d435f8cfe74605e7817"),
        keyed_hash: hex!("ed9b1a922c046fdb3d423ae34e143b05ca1bf28b710432857bf738bcedbfa511"),
        derive_key: hex!("554b0a5efea9ef183f2f9b931b7497995d9eb26f5c5c6dad2b97d62fc5ac31d9"),
    },
    Vector {
        len: 8192,
        hash: hex!("aae792484c8efe4f19e2ca7d371d8c467ffb10748d8a5a1ae579948f718a2a63"),
        keyed_hash: hex!("dc9637c8845a770b4cbf76b8daec0eebf7dc2eac11498517f08d44c8fc00d58a"),
        derive_key: hex!("ad01d7ae4ad059b0d33baa3c01319dcf8088094d0359e5fd45d6aeaa8b2d0c3d"),
    },
    Vector {
        len: 8193,
        hash: hex!("bab6c09cb8ce8cf459261398d2e7aef35700bf488116ceb94a36d0f5f1b7bc3b"),
        keyed_hash: hex!("954a2a75420c8d6547e3ba5b98d963e6fa6491addc8c023189cc519821b4a1f5"),
        derive_key: hex!("af1e0346e389b17c23200270a64aa4e1ead98c61695d917de7d5b00491c9b0f1"),
    },
    Vector {
        len: 16384,
        hash: hex!("f875d6646de28985646f34ee13be9a576fd515f76b5b0a26bb324735041ddde4"),
        keyed_hash: hex!("9e9fc4eb7cf081ea7c47d1807790ed211bfec56aa25bb7037784c13c4b707b0d"),
        derive_key: hex!("160e18b5878cd0df1c3af85eb25a0db5344d43a6fbd7a8ef4ed98d0714c3f7e1"),
    },
    Vector {
        len: 31744,
        hash: hex!("62b6960e1a44bcc1eb1a611a8d6235b6b4b78f32e7abc4fb4c6cdcce94895c47"),
        keyed_hash: hex!("efa53b389ab67c593dba624d898d0f7353ab99e4ac9d42302ee64cbf9939a419"),
        derive_key: hex!("39772aef80e0ebe60596361e45b061e8f417429d529171b6764468c22928e28e"),
    },
    Vector {
        len: 102400,
        hash: hex!("bc3e3d41a1146b069abffad3c0d44860cf664390afce4d9661f7902e7943e085"),
        keyed_hash: hex!("1c35d1a5811083fd7119f5d5d1ba027b4d01c0c6c49fb6ff2cf75393ea5db4a7"),
        derive_key: hex!("4652cff7a3f385a6103b5c260fc1593e13c778dbe608efb092fe7ee69df6e9c6"),
    },
];

/// The input of an official test vector.
fn vector_input(len: usize) -> Vec<u8> {
    (0..len).map(|i| (i % 251) as u8).collect()
}

/// A deterministic xorshift stream.
///
/// Its period keeps every message of a batch distinct, so a lane swap cannot hide.
fn stream(len: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect()
}

/// Read a 32-byte key as eight little-endian words.
fn key_words(key: &[u8; 32]) -> [u32; 8] {
    let (words, _) = key.as_chunks::<4>();
    core::array::from_fn(|w| u32::from_le_bytes(words[w]))
}

/// Hash `count` copies of one message through `kernel`.
///
/// Every lane must agree, and the common digest is returned.
fn digest(kernel: Kernel, mode: Mode, message: &[u8], count: usize) -> [u8; OUT_LEN] {
    let input = message.repeat(count);
    let mut digests = vec![[0u8; OUT_LEN]; count];
    kernel.hash_many(mode, &input, message.len(), &mut digests);
    assert!(
        digests.iter().all(|d| d == &digests[0]),
        "{kernel:?}: lanes disagree"
    );
    digests[0]
}

/// Batch sizes around every register boundary of `kernel`.
///
/// One lane, a full register, a full group, and a group plus a short register.
const fn counts(kernel: Kernel) -> [usize; 4] {
    [
        1,
        kernel.width,
        kernel.lanes,
        kernel.lanes + kernel.width + 1,
    ]
}

/// Hash a batch through `kernel`, one digest per `len`-byte message.
fn hash_many(kernel: Kernel, messages: &[u8], len: usize, count: usize) -> Vec<[u8; OUT_LEN]> {
    let mut digests = vec![[0u8; OUT_LEN]; count];
    kernel.hash_many(Mode::HASH, messages, len, &mut digests);
    digests
}

/// Plain digests of each message on its own, from the upstream crate.
fn upstream(messages: &[u8], len: usize, count: usize) -> Vec<[u8; OUT_LEN]> {
    (0..count)
        .map(|k| *blake3::hash(&messages[k * len..][..len]).as_bytes())
        .collect()
}

#[test]
fn hash_many_matches_the_official_vectors() {
    for kernel in batch::supported() {
        for v in VECTORS {
            let input = vector_input(v.len);
            for count in counts(kernel) {
                assert_eq!(
                    digest(kernel, Mode::HASH, &input, count),
                    v.hash,
                    "{kernel:?}, len {}, count {count}",
                    v.len
                );
            }
        }
    }
}

#[test]
fn keyed_mode_matches_the_official_vectors() {
    // The keyed mode starts every node from the key and adds one flag.
    let mode = Mode {
        key: key_words(KEY),
        flags: KEYED_HASH,
    };
    for kernel in batch::supported() {
        for v in VECTORS {
            let input = vector_input(v.len);
            assert_eq!(
                digest(kernel, mode, &input, kernel.lanes),
                v.keyed_hash,
                "{kernel:?}, len {}",
                v.len
            );
        }
    }
}

#[test]
fn key_derivation_matches_the_official_vectors() {
    // The context string hashes to a key, which then keys the material.
    let context = Mode {
        key: batch::IV,
        flags: DERIVE_KEY_CONTEXT,
    };
    for kernel in batch::supported() {
        let material = Mode {
            key: key_words(&digest(kernel, context, CONTEXT, 1)),
            flags: DERIVE_KEY_MATERIAL,
        };
        for v in VECTORS {
            let input = vector_input(v.len);
            assert_eq!(
                digest(kernel, material, &input, kernel.lanes),
                v.derive_key,
                "{kernel:?}, len {}",
                v.len
            );
        }
    }
}

#[test]
fn single_message_paths_match_the_official_vectors() {
    for v in VECTORS {
        let input = vector_input(v.len);
        assert_eq!(Blake3.hash_slice(&input), v.hash, "slice, len {}", v.len);
        assert_eq!(
            Blake3.hash_iter(input.iter().copied()),
            v.hash,
            "iter, len {}",
            v.len
        );

        // Pieces split on and around block and chunk boundaries.
        let pieces = input.chunks(BLOCK_LEN - 1);
        assert_eq!(
            Blake3.hash_iter_slices(pieces),
            v.hash,
            "slices, len {}",
            v.len
        );
    }
}

#[test]
fn hash_many_matches_upstream_across_shapes() {
    // Every length that changes the shape of the batched path:
    //
    // - no bytes, and less than one block: only the last block runs;
    // - on a block boundary: the last block is full;
    // - one chunk, one chunk plus a byte: the first parent node;
    // - three chunks: an unbalanced tree;
    // - eight chunks plus a byte: a deep left spine and a lone right chunk.
    let lengths = [
        0,
        1,
        3,
        4,
        5,
        BLOCK_LEN - 1,
        BLOCK_LEN,
        BLOCK_LEN + 1,
        CHUNK_LEN - 1,
        CHUNK_LEN,
        CHUNK_LEN + 1,
        CHUNK_LEN + BLOCK_LEN,
        3 * CHUNK_LEN,
        8 * CHUNK_LEN + 1,
    ];
    for kernel in batch::supported() {
        for len in lengths {
            for count in 1..=2 * kernel.lanes + 1 {
                let messages = stream(len * count, 0x2545_f491_4f6c_dd1d);
                assert_eq!(
                    hash_many(kernel, &messages, len, count),
                    upstream(&messages, len, count),
                    "{kernel:?}, len {len}, count {count}"
                );
            }
        }
    }
}

#[test]
fn bytes_past_a_message_do_not_leak_into_its_digest() {
    // The last block is read in place, past the message end, then masked.
    //
    // Flipping every byte of the next message must leave the first digest alone.
    let len = BLOCK_LEN + 3;
    for kernel in batch::supported() {
        let mut messages = stream(2 * len, 7);
        let before = hash_many(kernel, &messages, len, 2);

        messages[len..].iter_mut().for_each(|b| *b = !*b);
        let after = hash_many(kernel, &messages, len, 2);

        assert_eq!(before[0], after[0], "{kernel:?}");
        assert_ne!(before[1], after[1], "{kernel:?}");
    }
}

#[test]
fn hash_many_reads_nothing_when_no_digests_are_requested() {
    // A message length cannot be derived from zero digests, so the input is left untouched.
    Blake3.hash_many(&[1, 2, 3], &mut []);
}

#[test]
#[should_panic(expected = "must be a whole multiple")]
fn hash_many_rejects_ragged_input() {
    // 5 bytes cannot split into 2 equal messages.
    let mut digests = [[0u8; OUT_LEN]; 2];
    Blake3.hash_many(&[1, 2, 3, 4, 5], &mut digests);
}

proptest! {
    #[test]
    fn hash_many_matches_upstream_on_random_batches(
        len in 0usize..=5 * CHUNK_LEN,
        count in 1usize..=2 * LANES + 1,
        seed in any::<u64>(),
    ) {
        let messages = stream(len * count, seed);
        let expected = upstream(&messages, len, count);
        for kernel in batch::supported() {
            prop_assert_eq!(&hash_many(kernel, &messages, len, count), &expected, "{:?}", kernel);
        }

        // The public entry point runs the widest of them.
        let mut digests = vec![[0u8; OUT_LEN]; count];
        Blake3.hash_many(&messages, &mut digests);
        prop_assert_eq!(digests, expected);
    }
}
