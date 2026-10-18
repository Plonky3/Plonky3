use core::fmt::Debug;

use p3_binary_field::{
    BinaryField2, BinaryField4, BinaryField8, BinaryField16, BinaryField32, BinaryField64,
    BinaryField128, Poly64, TowerLevel,
};
use p3_field::Packable;

fn check_transpose<T: Packable + Debug>(make: impl Fn(usize) -> T) {
    for (width, height) in [(0, 0), (1, 1), (3, 5), (16, 16), (17, 19), (129, 131)] {
        let input: Vec<_> = (0..width * height).map(&make).collect();
        let mut expected = vec![T::default(); input.len()];
        for r in 0..height {
            for c in 0..width {
                expected[c * height + r] = input[r * width + c];
            }
        }
        let mut output = vec![T::default(); input.len()];
        T::transpose(&input, &mut output, width, height);
        assert_eq!(output, expected);
        output.fill(T::default());
        T::transpose_rows(
            height,
            width,
            |r| &input[r * width..(r + 1) * width],
            &mut output,
        );
        assert_eq!(output, expected);
    }
}

#[test]
fn tower_transposes_preserve_elements_across_tiles_and_edges() {
    macro_rules! check {
        ($field:ty, $repr:ty) => {
            check_transpose(|i| <$field>::from_repr((i as $repr).wrapping_mul(<$repr>::MAX / 3)));
        };
    }
    check!(BinaryField2, u8);
    check!(BinaryField4, u8);
    check!(BinaryField8, u8);
    check!(BinaryField16, u16);
    check!(BinaryField32, u32);
    check!(BinaryField64, u64);
    check!(BinaryField128, u128);
}

#[test]
fn poly64_transposes_preserve_elements_across_tiles_and_edges() {
    check_transpose(|i| Poly64::new((i as u64).wrapping_mul(u64::MAX / 3)));
}
