use core::ops::{Add, Div, Mul, Neg, Rem, Sub};

use crate::field::Field;

/// Computes a multiplicative subgroup whose order is known in advance.
pub fn cyclic_subgroup_known_order<F: Field>(
    generator: F,
    order: usize,
) -> impl Iterator<Item = F> + Clone {
    generator.powers().take(order)
}

/// Computes a coset of a multiplicative subgroup whose order is known in advance.
pub fn cyclic_subgroup_coset_known_order<F: Field>(
    generator: F,
    shift: F,
    order: usize,
) -> impl Iterator<Item = F> + Clone {
    cyclic_subgroup_known_order(generator, order).map(move |x| x * shift)
}

pub fn bezout<Int>(a: Int, b: Int) -> (Int, Int, Int)
where
    Int: Copy
        + Eq
        + Rem<Output = Int>
        + Div<Output = Int>
        + Sub<Output = Int>
        + Mul<Output = Int>
        + Add<Output = Int>
        + From<u8>,
{
    if a == Int::from(0) {
        return (b, Int::from(0), Int::from(1));
    }
    let (q, r) = (b / a, b % a); // hopefully the compiler gets the hint that this is a single division
    let (g, x, y) = bezout(r, a); // TODO: compare perf vs while loop
    (g, y - q * x, x)
}

pub fn inverse<Int>(a: Int, m: Int) -> Option<Int>
where
    Int: Copy
        + Eq
        + Rem<Output = Int>
        + Div<Output = Int>
        + Sub<Output = Int>
        + Mul<Output = Int>
        + Add<Output = Int>
        + From<u8>
        + PartialOrd,
{
    let (g, x, _) = bezout(a, m);
    // g==1 means a,m relatively prime
    if g == Int::from(1) {
        if x < Int::from(0) {
            Some(x + m)
        } else {
            Some(x)
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bezout() {
        assert_eq!(bezout(240, 46), (2, -9, 47));
        assert_eq!(bezout(17, 23), (1, -4, 3));
        assert_eq!(bezout(0, 5), (5, 0, 1));
        assert_eq!(bezout(5, 0), (5, 1, 0));
    }

    #[test]
    fn test_inverse() {
        assert_eq!(inverse(3, 11), Some(4));
        assert_eq!(inverse(17, 23), Some(19));
        assert_eq!(inverse(0, 5), None);
        assert_eq!(inverse(5, 0), None);
        assert_eq!(inverse(4, 8), None);
    }
    #[test]
    fn test_inverse_itypes() {
        //assert_eq!(inverse::<i8>(17, 23), Some(19));
        assert_eq!(inverse::<i16>(17, 23), Some(19));
        assert_eq!(inverse::<i32>(17, 23), Some(19));
        assert_eq!(inverse::<i64>(17, 23), Some(19));
        assert_eq!(inverse::<i128>(17, 23), Some(19));
        assert_eq!(inverse::<isize>(17, 23), Some(19));
    }
    #[test]
    fn test_bezout_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a: i64 = rng.gen::<i32>().abs() as i64;
            let b: i64 = rng.gen::<i32>().abs() as i64;
            let (g, x, y) = bezout::<i64>(a as i64, b as i64);
            assert_eq!(g, a * x + b * y);
        }
    }
    #[test]
    fn test_inverse_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a: i64 = (rng.gen::<i32>().abs() as i64) >> 1;
            let m: i64 = rng.gen::<i32>().abs() as i64;
            if m > 0 && a > 0 && a < m {
                let inv = inverse::<i64>(a, m);
                if let Some(inv) = inv {
                    assert_eq!(a * inv % m, 1);
                }
            }
        }
    }
    #[test]
    fn test_inverse_m31_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a: i32 = rng.gen();
            let m: i32 = 1 << 31 - 1;
            if a < m && a > 0 {
                let inv = inverse(a, m);
                if let Some(inv) = inv {
                    assert_eq!((a as i64) * (inv as i64) % (m as i64), 1);
                }
            }
        }
    }
    #[test]
    fn test_inverse_goldilocks_random() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a: i128 = (rng.gen::<u64>() >> 1) as i128;
            let m: i128 = 0xFFFF_FFFF_0000_0001;
            if a < m {
                let inv = inverse(a, m);
                if let Some(inv) = inv {
                    assert_eq!((a as u128) * (inv as u128) % (m as u128), 1);
                    assert!(inv < m);
                }
            }
        }
    }
}
