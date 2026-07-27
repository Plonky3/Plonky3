use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use p3_field_testing::bench_packed_extension_field;

bench_packed_extension_field! {
    BabyBear,
    quartic = BinomialExtensionField<BabyBear, 4>,
    quintic = BinomialExtensionField<BabyBear, 5>,
    octic = BinomialExtensionField<BabyBear, 8>,
}
