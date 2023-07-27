use alloc::boxed::Box;
use alloc::vec;

use p3_code::{LinearCodeFamily, SLCodeRegistry};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::sparse::CsrMatrix;
use rand::distributions::{Distribution, Standard};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::macros::{brakedown, brakedown_to_rs};
use crate::BrakedownCode;

pub fn fast_registry<F>() -> impl LinearCodeFamily<F, RowMajorMatrix<F>>
where
    F: Field,
    Standard: Distribution<F>,
{
    let height_4 = brakedown_to_rs!(16, 2, 0, 2, 4, 0);

    let height_5 = brakedown_to_rs!(32, 4, 0, 5, 8, 0);

    let height_6 = brakedown_to_rs!(64, 8, 1, 11, 16, 1);

    #[rustfmt::skip]
    let height_7 = brakedown!(128, 16, 7, 22, 32, 11,
        brakedown_to_rs!(16, 2, 0, 2, 4, 0));

    #[rustfmt::skip]
    let height_8 = brakedown!(256, 31, 11, 44, 65, 16,
        brakedown_to_rs!(31, 4, 0, 5, 8, 0));

    #[rustfmt::skip]
    let height_9 = brakedown!(512, 62, 10, 88, 131, 23,
        brakedown_to_rs!(62, 8, 1, 11, 15, 1));

    #[rustfmt::skip]
    let height_10 = brakedown!(1024, 123, 9, 175, 263, 24,
        brakedown_to_rs!(123, 15, 7, 21, 31, 11));

    #[rustfmt::skip]
    let height_11 = brakedown!(2048, 246, 9, 351, 526, 21,
        brakedown!(246, 30, 11, 42, 63, 15,
            brakedown_to_rs!(30, 4, 0, 5, 7, 0)));

    #[rustfmt::skip]
    let height_12 = brakedown!(4096, 492, 8, 702, 1053, 21,
        brakedown!(492, 60, 10, 85, 125, 23,
            brakedown_to_rs!(60, 8, 1, 11, 14, 1)));

    #[rustfmt::skip]
    let height_13 = brakedown!(8192, 984, 8, 1405, 2105, 20,
        brakedown!(984, 119, 9, 170, 251, 24,
            brakedown_to_rs!(119, 15, 7, 21, 30, 11)));

    #[rustfmt::skip]
    let height_14 = brakedown!(16384, 1967, 8, 2810, 4211, 20,
        brakedown!(1967, 237, 9, 338, 505, 23,
            brakedown!(237, 29, 11, 41, 60, 15,
                brakedown_to_rs!(29, 4, 0, 5, 7, 0))));

    #[rustfmt::skip]
    let height_15 = brakedown!(32768, 3933, 8, 5618, 8425, 20,
        brakedown!(3933, 472, 8, 674, 1011, 21,
            brakedown!(472, 57, 10, 81, 121, 22,
                brakedown_to_rs!(57, 7, 1, 10, 14, 1))));

    #[rustfmt::skip]
    let height_16 = brakedown!(65536, 7865, 8, 11235, 16851, 19,
        brakedown!(7865, 944, 8, 1348, 2022, 21,
            brakedown!(944, 114, 9, 162, 242, 23,
                brakedown_to_rs!(114, 14, 7, 20, 28, 11))));

    #[rustfmt::skip]
    let height_17 = brakedown!(131072, 15729, 8, 22470, 33703, 19,
        brakedown!(15729, 1888, 8, 2697, 4044, 20,
            brakedown!(1888, 227, 9, 324, 485, 23,
                brakedown!(227, 28, 11, 40, 57, 15,
                    brakedown_to_rs!(28, 4, 0, 5, 7, 0)))));

    #[rustfmt::skip]
    let height_18 = brakedown!(262144, 31458, 8, 44940, 67407, 19,
        brakedown!(31458, 3775, 8, 5392, 8090, 20,
            brakedown!(3775, 453, 8, 647, 970, 22,
                brakedown!(453, 55, 10, 78, 116, 21,
                    brakedown_to_rs!(55, 7, 1, 10, 13, 1)))));

    #[rustfmt::skip]
    let height_19 = brakedown!(524288, 62915, 7, 89878, 134816, 19,
        brakedown!(62915, 7550, 8, 10785, 16178, 20,
            brakedown!(7550, 906, 8, 1294, 1941, 21,
                brakedown!(906, 109, 9, 155, 233, 23,
                    brakedown_to_rs!(109, 14, 7, 20, 26, 11)))));

    #[rustfmt::skip]
    let height_20 = brakedown!(1048576, 125830, 8, 179757, 269632, 19,
        brakedown!(125830, 15100, 8, 21571, 32356, 19,
            brakedown!(15100, 1812, 8, 2588, 3883, 20,
                brakedown!(1812, 218, 9, 311, 465, 22,
                    brakedown!(218, 27, 11, 38, 55, 15,
                        brakedown_to_rs!(27, 4, 0, 5, 6, 0))))));

    #[rustfmt::skip]
    let height_21 = brakedown!(2097152, 251659, 8, 359512, 539267, 19,
        brakedown!(251659, 30200, 8, 43142, 64711, 19,
            brakedown!(30200, 3624, 8, 5177, 7765, 20,
                brakedown!(3624, 435, 8, 621, 932, 21,
                    brakedown!(435, 53, 10, 75, 111, 21,
                        brakedown_to_rs!(53, 7, 1, 10, 12, 1))))));

    #[rustfmt::skip]
    let height_22 = brakedown!(4194304, 503317, 8, 719024, 1078534, 19,
        brakedown!(503317, 60399, 7, 86284, 129423, 19,
            brakedown!(60399, 7248, 8, 10354, 15531, 19,
                brakedown!(7248, 870, 8, 1242, 1864, 21,
                    brakedown!(870, 105, 9, 150, 222, 27,
                        brakedown_to_rs!(105, 13, 7, 18, 27, 11))))));

    #[rustfmt::skip]
    let height_23 = brakedown!(8388608, 1006633, 8, 1438047, 2157070, 19,
        brakedown!(1006633, 120796, 8, 172565, 258849, 19,
            brakedown!(120796, 14496, 8, 20708, 31061, 19,
                brakedown!(14496, 1740, 8, 2485, 3727, 20,
                    brakedown!(1740, 209, 9, 298, 447, 24,
                        brakedown!(209, 26, 10, 37, 52, 14,
                            brakedown_to_rs!(26, 4, 0, 5, 6, 0)))))));

    #[rustfmt::skip]
    let height_24 = brakedown!(16777216, 2013266, 8, 2876094, 4314141, 19,
        brakedown!(2013266, 241592, 8, 345131, 517697, 19,
            brakedown!(241592, 28992, 8, 41417, 62122, 19,
                brakedown!(28992, 3480, 8, 4971, 7454, 20,
                    brakedown!(3480, 418, 8, 597, 894, 22,
                        brakedown!(418, 51, 10, 72, 107, 20,
                            brakedown_to_rs!(51, 7, 1, 10, 11, 1)))))));

    #[rustfmt::skip]
    let height_25 = brakedown!(33554432, 4026532, 8, 5752188, 8628282, 19,
        brakedown!(4026532, 483184, 8, 690262, 1035394, 19,
            brakedown!(483184, 57983, 7, 82832, 124246, 19,
                brakedown!(57983, 6958, 8, 9940, 14909, 20,
                    brakedown!(6958, 835, 8, 1192, 1790, 21,
                        brakedown!(835, 101, 9, 144, 213, 26,
                            brakedown_to_rs!(101, 13, 6, 18, 25, 9)))))));

    SLCodeRegistry::new(vec![
        Box::new(height_4),
        Box::new(height_5),
        Box::new(height_6),
        Box::new(height_7),
        Box::new(height_8),
        Box::new(height_9),
        Box::new(height_10),
        Box::new(height_11),
        Box::new(height_12),
        Box::new(height_13),
        Box::new(height_14),
        Box::new(height_15),
        Box::new(height_16),
        Box::new(height_17),
        Box::new(height_18),
        Box::new(height_19),
        Box::new(height_20),
        Box::new(height_21),
        Box::new(height_22),
        Box::new(height_23),
        Box::new(height_24),
        Box::new(height_25),
    ])
}
