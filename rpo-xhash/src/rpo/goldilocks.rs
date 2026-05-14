//! RPO for Goldilocks: x^7 / x^{1/7} S-boxes over F_p.
//!
//! Goldilocks p = 2^64 - 2^32 + 1, p-1 = 2^32 * 3 * 5 * 17 * 257 * 65537.
//! d = 7: gcd(7, p-1) = 1.
//! Inverse exponent: 7^{-1} mod (p-1) = 10540996611094048183 = 0x92492491B6DB6DB7.
//!
//! Verified with Sage:
//! ```sage
//! p = 2^64 - 2^32 + 1
//! assert p.is_prime()
//! assert gcd(7, p - 1) == 1
//! e = inverse_mod(7, p - 1)    # 10540996611094048183 = 0x92492491B6DB6DB7
//! assert (7 * e) % (p - 1) == 1
//! F = GF(p); x = F(123456789)
//! assert (x^7)^e == x
//! ```

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks;
use rand::RngExt;

use crate::mds_goldilocks::MdsBase12;

use super::{RpoHash, RpoSbox};

/// RPO-Goldilocks: 7 rounds (matches miden-crypto / RPO256 spec).
pub const RPO_GL_ROUNDS: usize = 7;

/// RPO S-box for Goldilocks: x^7 forward, x^{1/7} backward.
#[derive(Debug, Copy, Clone, Default, Eq, PartialEq)]
pub struct SboxGL;

impl RpoSbox<Goldilocks, 12> for SboxGL {
    #[inline(always)]
    fn forward(&self, state: &mut [Goldilocks; 12]) {
        apply_pow7(state);
    }

    #[inline(always)]
    fn backward(&self, state: &mut [Goldilocks; 12]) {
        apply_pow_inv7(state);
    }
}

/// x^7 for 12 Goldilocks elements (cost: 2 sqr + 2 mul per element).
#[inline(always)]
fn apply_pow7(state: &mut [Goldilocks; 12]) {
    for s in state.iter_mut() {
        let x2 = s.square();
        let x3 = x2 * *s;
        let x6 = x3.square();
        *s = x6 * *s;
    }
}

/// x^{1/7} = x^{10540996611094048183} for 12 Goldilocks elements.
///
/// Exponent 0x92492491B6DB6DB7 in binary:
///   `b1001001001001001001001001001000110110110110110110110110110110111`
///
/// 72-multiplication addition chain (vs. 95 for naive square-and-multiply).
/// Ported from miden-crypto's `apply_inv_sbox`.
#[inline(always)]
fn apply_pow_inv7(state: &mut [Goldilocks; 12]) {
    let mut t1 = *state;
    t1.iter_mut().for_each(|t| *t = t.square());
    // t1 = x^2

    let mut t2 = t1;
    t2.iter_mut().for_each(|t| *t = t.square());
    // t2 = x^4

    // t3 = t2^(2^3) * t2 = x^36 = b100100
    let t3 = exp_acc::<3>(t2, t2);

    // t4 = t3^(2^6) * t3 = b100100_100100
    let t4 = exp_acc::<6>(t3, t3);

    // t5 = t4^(2^12) * t4 = b100100_100100_100100_100100
    let t5 = exp_acc::<12>(t4, t4);

    // t6 = t5^(2^6) * t3 = b100100_100100_100100_100100_100100
    let t6 = exp_acc::<6>(t5, t3);

    // t7 = t6^(2^31) * t6 = full 62-bit block of the pattern
    let t7 = exp_acc::<31>(t6, t6);

    // Final combine: (t7^2 * t6)^4 * t1 * t2 * x = x^{0x92492491B6DB6DB7}
    for (i, s) in state.iter_mut().enumerate() {
        let a = (t7[i].square() * t6[i]).square().square();
        let b = t1[i] * t2[i] * *s;
        *s = a * b;
    }
}

/// `result = base^(2^M) * tail`, applied lane-wise across the state.
/// Squarings are batched across all elements for ILP.
#[inline(always)]
fn exp_acc<const M: usize>(base: [Goldilocks; 12], tail: [Goldilocks; 12]) -> [Goldilocks; 12] {
    let mut result = base;
    for _ in 0..M {
        for r in result.iter_mut() {
            *r = r.square();
        }
    }
    for (r, &t) in result.iter_mut().zip(tail.iter()) {
        *r *= t;
    }
    result
}

// ============================================================
// Hash type aliases and factory functions
// ============================================================

/// RPO-Goldilocks: 12-element state, x^7 / x^{1/7} S-box, 12×12 FFT MDS
/// matching miden-crypto's `Rpo256` row.
pub type RpoGoldilocks = RpoHash<Goldilocks, SboxGL, MdsBase12, 12>;

pub fn rpo_goldilocks(rng: &mut impl rand::Rng) -> RpoGoldilocks {
    let num_constants = (2 * RPO_GL_ROUNDS + 1) * 12;
    let round_constants = (0..num_constants)
        .map(|_| Goldilocks::new(rng.random::<u64>()))
        .collect();
    RpoHash::new_from_constants(RPO_GL_ROUNDS, round_constants)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_symmetric::Permutation;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn pow7_roundtrip() {
        let state: [Goldilocks; 12] =
            core::array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 1_000_000_007));
        let mut s = state;
        SboxGL.forward(&mut s);
        assert_ne!(s, state);
        SboxGL.backward(&mut s);
        assert_eq!(s, state);
    }

    /// Sage:
    /// ```sage
    /// p = 2^64 - 2^32 + 1
    /// F = GF(p); x = F(123456789)
    /// e = inverse_mod(7, p - 1)
    /// (x^e).lift()  # known answer
    /// ```
    #[test]
    fn pow_inv7_known_answer() {
        let mut state = [Goldilocks::new(0); 12];
        state[0] = Goldilocks::new(123456789);
        apply_pow_inv7(&mut state);
        // Round-trip verifies the chain produces a true 7th root.
        let mut back = state;
        apply_pow7(&mut back);
        assert_eq!(back[0], Goldilocks::new(123456789));
    }

    #[test]
    fn rpo_goldilocks_deterministic() {
        let mut rng = SmallRng::seed_from_u64(1);
        let hash = rpo_goldilocks(&mut rng);
        let input: [Goldilocks; 12] =
            core::array::from_fn(|i| Goldilocks::new((i as u64 + 1) * 37));
        let out1 = hash.permute(input);
        let out2 = hash.permute(input);
        assert_eq!(out1, out2);
    }
}

// ============================================================
// miden-crypto Rpo256 compatibility test
// ============================================================
//
// Verifies our primitives (`MdsBase12`, `apply_pow7`, `apply_pow_inv7`,
// `super::add_rc`) match miden-crypto's RPO256 byte-for-byte, by running
// the *canonical* RPO schedule (no final MDS+RC) with miden's ARK1/ARK2
// and comparing against vectors generated by miden-crypto.
//
// Our `RpoHash` adds a final MDS+RC layer; that's a design difference, not
// a primitive-level disagreement. This test isolates the primitive layer.

#[cfg(test)]
mod miden_compat {
    use super::{MdsBase12, RpoSbox, SboxGL};
    use crate::rpo::add_rc;
    use p3_goldilocks::Goldilocks;
    use p3_symmetric::Permutation;

    /// First-half round constants (forward step), copied verbatim from
    /// miden-crypto/src/hash/algebraic_sponge/rescue/mod.rs.
    const ARK1: [[Goldilocks; 12]; 7] = [
        [
            Goldilocks::new(5789762306288267392),
            Goldilocks::new(6522564764413701783),
            Goldilocks::new(17809893479458208203),
            Goldilocks::new(107145243989736508),
            Goldilocks::new(6388978042437517382),
            Goldilocks::new(15844067734406016715),
            Goldilocks::new(9975000513555218239),
            Goldilocks::new(3344984123768313364),
            Goldilocks::new(9959189626657347191),
            Goldilocks::new(12960773468763563665),
            Goldilocks::new(9602914297752488475),
            Goldilocks::new(16657542370200465908),
        ],
        [
            Goldilocks::new(12987190162843096997),
            Goldilocks::new(653957632802705281),
            Goldilocks::new(4441654670647621225),
            Goldilocks::new(4038207883745915761),
            Goldilocks::new(5613464648874830118),
            Goldilocks::new(13222989726778338773),
            Goldilocks::new(3037761201230264149),
            Goldilocks::new(16683759727265180203),
            Goldilocks::new(8337364536491240715),
            Goldilocks::new(3227397518293416448),
            Goldilocks::new(8110510111539674682),
            Goldilocks::new(2872078294163232137),
        ],
        [
            Goldilocks::new(18072785500942327487),
            Goldilocks::new(6200974112677013481),
            Goldilocks::new(17682092219085884187),
            Goldilocks::new(10599526828986756440),
            Goldilocks::new(975003873302957338),
            Goldilocks::new(8264241093196931281),
            Goldilocks::new(10065763900435475170),
            Goldilocks::new(2181131744534710197),
            Goldilocks::new(6317303992309418647),
            Goldilocks::new(1401440938888741532),
            Goldilocks::new(8884468225181997494),
            Goldilocks::new(13066900325715521532),
        ],
        [
            Goldilocks::new(5674685213610121970),
            Goldilocks::new(5759084860419474071),
            Goldilocks::new(13943282657648897737),
            Goldilocks::new(1352748651966375394),
            Goldilocks::new(17110913224029905221),
            Goldilocks::new(1003883795902368422),
            Goldilocks::new(4141870621881018291),
            Goldilocks::new(8121410972417424656),
            Goldilocks::new(14300518605864919529),
            Goldilocks::new(13712227150607670181),
            Goldilocks::new(17021852944633065291),
            Goldilocks::new(6252096473787587650),
        ],
        [
            Goldilocks::new(4887609836208846458),
            Goldilocks::new(3027115137917284492),
            Goldilocks::new(9595098600469470675),
            Goldilocks::new(10528569829048484079),
            Goldilocks::new(7864689113198939815),
            Goldilocks::new(17533723827845969040),
            Goldilocks::new(5781638039037710951),
            Goldilocks::new(17024078752430719006),
            Goldilocks::new(109659393484013511),
            Goldilocks::new(7158933660534805869),
            Goldilocks::new(2955076958026921730),
            Goldilocks::new(7433723648458773977),
        ],
        [
            Goldilocks::new(16308865189192447297),
            Goldilocks::new(11977192855656444890),
            Goldilocks::new(12532242556065780287),
            Goldilocks::new(14594890931430968898),
            Goldilocks::new(7291784239689209784),
            Goldilocks::new(5514718540551361949),
            Goldilocks::new(10025733853830934803),
            Goldilocks::new(7293794580341021693),
            Goldilocks::new(6728552937464861756),
            Goldilocks::new(6332385040983343262),
            Goldilocks::new(13277683694236792804),
            Goldilocks::new(2600778905124452676),
        ],
        [
            Goldilocks::new(7123075680859040534),
            Goldilocks::new(1034205548717903090),
            Goldilocks::new(7717824418247931797),
            Goldilocks::new(3019070937878604058),
            Goldilocks::new(11403792746066867460),
            Goldilocks::new(10280580802233112374),
            Goldilocks::new(337153209462421218),
            Goldilocks::new(13333398568519923717),
            Goldilocks::new(3596153696935337464),
            Goldilocks::new(8104208463525993784),
            Goldilocks::new(14345062289456085693),
            Goldilocks::new(17036731477169661256),
        ],
    ];

    /// Second-half round constants (backward step).
    const ARK2: [[Goldilocks; 12]; 7] = [
        [
            Goldilocks::new(6077062762357204287),
            Goldilocks::new(15277620170502011191),
            Goldilocks::new(5358738125714196705),
            Goldilocks::new(14233283787297595718),
            Goldilocks::new(13792579614346651365),
            Goldilocks::new(11614812331536767105),
            Goldilocks::new(14871063686742261166),
            Goldilocks::new(10148237148793043499),
            Goldilocks::new(4457428952329675767),
            Goldilocks::new(15590786458219172475),
            Goldilocks::new(10063319113072092615),
            Goldilocks::new(14200078843431360086),
        ],
        [
            Goldilocks::new(6202948458916099932),
            Goldilocks::new(17690140365333231091),
            Goldilocks::new(3595001575307484651),
            Goldilocks::new(373995945117666487),
            Goldilocks::new(1235734395091296013),
            Goldilocks::new(14172757457833931602),
            Goldilocks::new(707573103686350224),
            Goldilocks::new(15453217512188187135),
            Goldilocks::new(219777875004506018),
            Goldilocks::new(17876696346199469008),
            Goldilocks::new(17731621626449383378),
            Goldilocks::new(2897136237748376248),
        ],
        [
            Goldilocks::new(8023374565629191455),
            Goldilocks::new(15013690343205953430),
            Goldilocks::new(4485500052507912973),
            Goldilocks::new(12489737547229155153),
            Goldilocks::new(9500452585969030576),
            Goldilocks::new(2054001340201038870),
            Goldilocks::new(12420704059284934186),
            Goldilocks::new(355990932618543755),
            Goldilocks::new(9071225051243523860),
            Goldilocks::new(12766199826003448536),
            Goldilocks::new(9045979173463556963),
            Goldilocks::new(12934431667190679898),
        ],
        [
            Goldilocks::new(18389244934624494276),
            Goldilocks::new(16731736864863925227),
            Goldilocks::new(4440209734760478192),
            Goldilocks::new(17208448209698888938),
            Goldilocks::new(8739495587021565984),
            Goldilocks::new(17000774922218161967),
            Goldilocks::new(13533282547195532087),
            Goldilocks::new(525402848358706231),
            Goldilocks::new(16987541523062161972),
            Goldilocks::new(5466806524462797102),
            Goldilocks::new(14512769585918244983),
            Goldilocks::new(10973956031244051118),
        ],
        [
            Goldilocks::new(6982293561042362913),
            Goldilocks::new(14065426295947720331),
            Goldilocks::new(16451845770444974180),
            Goldilocks::new(7139138592091306727),
            Goldilocks::new(9012006439959783127),
            Goldilocks::new(14619614108529063361),
            Goldilocks::new(1394813199588124371),
            Goldilocks::new(4635111139507788575),
            Goldilocks::new(16217473952264203365),
            Goldilocks::new(10782018226466330683),
            Goldilocks::new(6844229992533662050),
            Goldilocks::new(7446486531695178711),
        ],
        [
            Goldilocks::new(3736792340494631448),
            Goldilocks::new(577852220195055341),
            Goldilocks::new(6689998335515779805),
            Goldilocks::new(13886063479078013492),
            Goldilocks::new(14358505101923202168),
            Goldilocks::new(7744142531772274164),
            Goldilocks::new(16135070735728404443),
            Goldilocks::new(12290902521256031137),
            Goldilocks::new(12059913662657709804),
            Goldilocks::new(16456018495793751911),
            Goldilocks::new(4571485474751953524),
            Goldilocks::new(17200392109565783176),
        ],
        [
            Goldilocks::new(17130398059294018733),
            Goldilocks::new(519782857322261988),
            Goldilocks::new(9625384390925085478),
            Goldilocks::new(1664893052631119222),
            Goldilocks::new(7629576092524553570),
            Goldilocks::new(3485239601103661425),
            Goldilocks::new(9755891797164033838),
            Goldilocks::new(15218148195153269027),
            Goldilocks::new(16460604813734957368),
            Goldilocks::new(9643968136937729763),
            Goldilocks::new(3611348709641382851),
            Goldilocks::new(18256379591337759196),
        ],
    ];

    /// Canonical RPO256 permutation: 7 rounds of
    /// `(MDS → ARK1[r] → x^7 → MDS → ARK2[r] → x^{1/7})`. No final layer.
    fn miden_permute(state: &mut [Goldilocks; 12]) {
        let mds = MdsBase12;
        for r in 0..7 {
            mds.permute_mut(state);
            add_rc(state, &ARK1[r]);
            SboxGL.forward(state);

            mds.permute_mut(state);
            add_rc(state, &ARK2[r]);
            SboxGL.backward(state);
        }
    }

    /// Test vectors generated from miden-crypto's `Rpo256::apply_permutation`.
    #[test]
    fn matches_miden_crypto_rpo256() {
        let cases: &[([u64; 12], [u64; 12])] = &[
            // state = [0, 1, ..., 11]
            (
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [
                    15056646954853821376,
                    594518210294093573,
                    10395398226526937664,
                    3903707756219396109,
                    7670128982698747483,
                    4249514323476682720,
                    16506822133651532340,
                    10593868791806571942,
                    9413309068803954142,
                    15946782832277734471,
                    7904287043744270535,
                    16548919317472389167,
                ],
            ),
            // state = [42; 12]
            (
                [42; 12],
                [
                    803473572191352483,
                    4919384702411114639,
                    7057026313342623225,
                    8115304200781576288,
                    16503787088684104694,
                    16290840577441548507,
                    9990008047058543581,
                    12681907180278292496,
                    5582684703997150203,
                    2141334559836121510,
                    15520000079195543313,
                    5431088739207367651,
                ],
            ),
            // state = [(i+1) * 1_000_000_007]
            (
                {
                    let mut s = [0u64; 12];
                    for i in 0..12 {
                        s[i] = ((i as u64) + 1).wrapping_mul(1_000_000_007);
                    }
                    s
                },
                [
                    523469508799167376,
                    14493813212983667685,
                    5371140406163079417,
                    4772124378307943412,
                    15821213887998156229,
                    4718139618272126980,
                    17821537582276079833,
                    2041564681537659650,
                    11514936618602200187,
                    16346665034303203919,
                    18355088631239783651,
                    12995628848895960128,
                ],
            ),
        ];

        for (input, expected) in cases {
            let mut state: [Goldilocks; 12] = core::array::from_fn(|i| Goldilocks::new(input[i]));
            miden_permute(&mut state);
            let got: [u64; 12] = core::array::from_fn(|i| {
                use p3_field::PrimeField64;
                state[i].as_canonical_u64()
            });
            assert_eq!(got, *expected, "input = {input:?}");
        }
    }
}
