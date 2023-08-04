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

pub fn gcd<Int>(a: Int, b: Int) -> (Int, Int, Int)
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
    let (g, x, y) = gcd(r, a); // TODO: compare perf vs while loop
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
    let (g, x, _) = gcd(a, m);

    if g == Int::from(1) {
        // a,m relatively prime
        if x < Int::from(0) {
            Some(x + m)
        } else {
            Some(x)
        }
    } else {
        None
    }
}

pub fn gcd_u<UInt, IInt>(a: UInt, b: UInt) -> (UInt, IInt, IInt)
where
    UInt: Copy
        + Eq
        + Rem<Output = UInt>
        + Div<Output = UInt>
        + Mul<Output = UInt>
        + Add<Output = UInt>
        + From<u8>
        + TryFrom<IInt>,
    IInt: Copy
        + Eq
        + Rem<Output = IInt>
        + Div<Output = IInt>
        + Sub<Output = IInt>
        + Mul<Output = IInt>
        + Add<Output = IInt>
        + From<u8>
        + TryFrom<UInt>,
{
    if a == UInt::from(0) {
        return (b, IInt::from(0), IInt::from(1));
    }
    if a == UInt::from(1) {
        return (UInt::from(1), IInt::from(1), IInt::from(0));
    }
    let (q, r) = (b / a, b % a); // hopefully the compiler gets the hint that this is a single division
    let (g, x, y) = gcd_u::<UInt, IInt>(r, a); // TODO: compare perf vs while loop
    let z = IInt::try_from(q).unwrap_or_else(|_| panic!("Failed to convert q to IInt"));
    (g, y - z * x, x)
}

pub fn inverse_u<UInt, IInt>(a: UInt, m: UInt) -> Option<UInt>
where
    UInt: Copy
        + Eq
        + Rem<Output = UInt>
        + Div<Output = UInt>
        + Mul<Output = UInt>
        + Add<Output = UInt>
        + Sub<Output = UInt>
        + From<u8>
        + TryFrom<IInt>,
    IInt: Copy
        + Eq
        + Rem<Output = IInt>
        + Div<Output = IInt>
        + Sub<Output = IInt>
        + Mul<Output = IInt>
        + Add<Output = IInt>
        + From<u8>
        + TryFrom<UInt>
        + PartialOrd<IInt>
        + Neg<Output = IInt>,
{
    let (g, x, _) = gcd_u::<UInt, IInt>(a, m);
    // g==1 means a,m relatively prime
    if g == UInt::from(1) {
        if x < IInt::from(0) {
            Some(
                m - (-x)
                    .try_into()
                    .unwrap_or_else(|_| panic!("Failed to convert q to UInt")),
            )
        } else {
            Some(
                x.try_into()
                    .unwrap_or_else(|_| panic!("Failed to convert q to UInt")),
            )
        }
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(240, 46), (2, -9, 47));
        assert_eq!(gcd(17, 23), (1, -4, 3));
        assert_eq!(gcd(0, 5), (5, 0, 1));
        assert_eq!(gcd(5, 0), (5, 1, 0));
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
            let (g, x, y) = gcd::<i64>(a as i64, b as i64);
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

    #[test]
    fn test_inverse_goldilocks_random_u() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let a: u64 = rng.gen::<u64>();
            let m: u64 = 0xFFFF_FFFF_0000_0001;
            if a < m {
                let inv = inverse_u::<u64, i64>(a, m);
                if let Some(inv) = inv {
                    assert_eq!((a as u128) * (inv as u128) % (m as u128), 1);
                    assert!(inv < m);
                }
            }
        }
    }
}
