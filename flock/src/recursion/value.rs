//! Field values of a circuit under construction: constants, or wires with their native value.
//!
//! Additions are XORs and cost no product; every product is emitted as it is formed.

use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::fmt::{self, Debug};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use p3_binary_field::{Gf2, Poly64, Poly192};
use p3_field::{Algebra, Dup, PrimeCharacteristicRing};
use spin::{Mutex, Once};

use super::gates::{BinaryElement, Gates, limbs_of};
use crate::circuit::Wire;

/// The circuit a value's wires belong to.
pub(crate) type Shared = Arc<Mutex<Gates>>;

/// A value of `GF(2^64)` or `GF(2^192)` in a circuit.
#[derive(Clone)]
pub struct Bin<E>(Repr<E>);

/// A committed-field value of a circuit.
pub type Gf64 = Bin<Poly64>;

/// A challenge-field value of a circuit.
pub type Gf192 = Bin<Poly192>;

/// The forms of a value.
#[derive(Clone)]
enum Repr<E> {
    /// A constant of the circuit.
    Const(E),
    /// Wires, one per coordinate bit.
    Wired(Arc<Wired<E>>),
    /// A sum of products whose Karatsuba recombination waits until the value is read.
    Lazy(Arc<Lazy<E>>),
}

/// Wires and the native value they carry.
struct Wired<E> {
    /// The native value.
    value: E,
    /// One wire per coordinate bit.
    wires: Vec<Wire>,
    /// The circuit.
    gates: Shared,
    /// The Karatsuba leaves of the value's limbs, formed at its first product.
    limbs: Once<Vec<Vec<Wire>>>,
    /// The Karatsuba leaves of the pairwise limb sums, formed at its first full product.
    sums: Once<Vec<Vec<Wire>>>,
}

impl<E> Wired<E> {
    /// The value as a product factor.
    fn leaves(&self) -> Leaves<'_> {
        Leaves {
            wires: &self.wires,
            limbs: &self.limbs,
            sums: &self.sums,
        }
    }
}

/// Summed leaf products, recombined once when read.
struct Lazy<E> {
    /// The native value.
    value: E,
    /// The circuit.
    gates: Shared,
    /// Summed leaf products per product slot.
    slots: Vec<Option<Vec<Wire>>>,
    /// The value, recombined at its first read.
    wires: Once<Vec<Wire>>,
}

impl<E: Lowering> Bin<E> {
    /// The value the wires carry, natively `value`.
    pub(crate) fn wired(gates: &Shared, wires: Vec<Wire>, value: E) -> Self {
        debug_assert_eq!(wires.len(), E::BITS, "one wire per bit");
        Self(Repr::Wired(Arc::new(Wired {
            value,
            wires,
            gates: Arc::clone(gates),
            limbs: Once::new(),
            sums: Once::new(),
        })))
    }

    /// The native value.
    pub fn value(&self) -> E {
        match &self.0 {
            Repr::Const(c) => *c,
            Repr::Wired(w) => w.value,
            Repr::Lazy(l) => l.value,
        }
    }

    /// The constant behind the value, if it is one.
    pub(crate) const fn as_const(&self) -> Option<&E> {
        match &self.0 {
            Repr::Const(c) => Some(c),
            _ => None,
        }
    }

    /// The circuit of the value's wires, if it has any.
    pub(crate) fn gates(&self) -> Option<&Shared> {
        match &self.0 {
            Repr::Const(_) => None,
            Repr::Wired(w) => Some(&w.gates),
            Repr::Lazy(l) => Some(&l.gates),
        }
    }

    /// One wire per coordinate bit, constants as the constant wires.
    ///
    /// A lazy value must have been recombined, which [`Self::settled`] does.
    pub(crate) fn wires(&self, gates: &Gates) -> Vec<Wire> {
        match &self.0 {
            Repr::Const(c) => gates.constant_bits(c),
            Repr::Wired(w) => w.wires.clone(),
            Repr::Lazy(l) => l.wires.get().expect("a settled value").clone(),
        }
    }

    /// The value with any pending recombination done, as a constant or wires.
    pub(crate) fn settled(self) -> Self {
        match &self.0 {
            Repr::Lazy(l) => {
                let wires = l
                    .wires
                    .call_once(|| E::finish(&mut l.gates.lock(), &l.slots))
                    .clone();
                Self::wired(&l.gates, wires, l.value)
            }
            _ => self,
        }
    }
}

impl<E: Debug> Debug for Bin<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Repr::Const(c) => f.debug_tuple("Const").field(c).finish(),
            Repr::Wired(w) => f.debug_tuple("Wired").field(&w.value).finish(),
            Repr::Lazy(l) => f.debug_tuple("Lazy").field(&l.value).finish(),
        }
    }
}

/// The field operations a value type lowers to.
pub trait Lowering: BinaryElement {
    /// Product slots of a lazy value.
    const SLOTS: usize;

    /// The Karatsuba leaves of the limbs.
    fn limb_leaves(gates: &mut Gates, x: &[Wire]) -> Vec<Vec<Wire>>;

    /// The Karatsuba leaves of the pairwise limb sums, from the limbs' leaves.
    fn sum_leaves(gates: &mut Gates, limbs: &[Vec<Wire>]) -> Vec<Vec<Wire>>;

    /// The product slots of `x y`, leaves read through the accessors.
    fn product_slots(gates: &mut Gates, x: &Leaves<'_>, y: &Leaves<'_>) -> Vec<Option<Vec<Wire>>>;

    /// The value of summed product slots.
    fn finish(gates: &mut Gates, slots: &[Option<Vec<Wire>>]) -> Vec<Wire>;

    /// `c x` on wires, free.
    fn mul_const_wires(gates: &mut Gates, c: Self, x: &[Wire]) -> Vec<Wire>;

    /// `x^2` on wires, free.
    fn square_wires(gates: &mut Gates, x: &[Wire]) -> Vec<Wire>;
}

/// A factor of a product: its wires and its cached leaves.
pub struct Leaves<'a> {
    /// The wires.
    wires: &'a [Wire],
    /// The limbs' leaves, formed on first use.
    limbs: &'a Once<Vec<Vec<Wire>>>,
    /// The pairwise sums' leaves, formed on first use.
    sums: &'a Once<Vec<Vec<Wire>>>,
}

impl Leaves<'_> {
    /// Whether every limb past the first is a structural zero.
    fn embedded(&self) -> bool {
        self.wires[64..].iter().all(|w| w.is_zero())
    }

    /// The limbs' leaves.
    fn limbs<E: Lowering>(&self, gates: &mut Gates) -> &Vec<Vec<Wire>> {
        self.limbs.call_once(|| E::limb_leaves(gates, self.wires))
    }

    /// The pairwise sums' leaves.
    fn sums<E: Lowering>(&self, gates: &mut Gates) -> &Vec<Vec<Wire>> {
        let limbs = self.limbs::<E>(gates).clone();
        self.sums.call_once(|| E::sum_leaves(gates, &limbs))
    }
}

impl Lowering for Poly64 {
    const SLOTS: usize = 1;

    fn limb_leaves(gates: &mut Gates, x: &[Wire]) -> Vec<Vec<Wire>> {
        vec![gates.expand(x)]
    }

    fn sum_leaves(_gates: &mut Gates, _limbs: &[Vec<Wire>]) -> Vec<Vec<Wire>> {
        Vec::new()
    }

    fn product_slots(gates: &mut Gates, x: &Leaves<'_>, y: &Leaves<'_>) -> Vec<Option<Vec<Wire>>> {
        gates.current.mul64 += 1;
        let ex = x.limbs::<Self>(gates)[0].clone();
        let ey = y.limbs::<Self>(gates)[0].clone();
        vec![Some(gates.leaf_products(&ex, &ey))]
    }

    fn finish(gates: &mut Gates, slots: &[Option<Vec<Wire>>]) -> Vec<Wire> {
        slots[0]
            .as_ref()
            .map_or_else(|| vec![Wire::ZERO; 64], |leaves| gates.finish64(leaves))
    }

    fn mul_const_wires(gates: &mut Gates, c: Self, x: &[Wire]) -> Vec<Wire> {
        gates.mul64_const(c, x)
    }

    fn square_wires(gates: &mut Gates, x: &[Wire]) -> Vec<Wire> {
        gates.square64(x)
    }
}

impl Lowering for Poly192 {
    const SLOTS: usize = 9;

    fn limb_leaves(gates: &mut Gates, x: &[Wire]) -> Vec<Vec<Wire>> {
        x.as_chunks::<64>()
            .0
            .iter()
            .map(|limb| gates.expand(limb))
            .collect()
    }

    fn sum_leaves(gates: &mut Gates, limbs: &[Vec<Wire>]) -> Vec<Vec<Wire>> {
        // Expansion is linear, so a sum's leaves are the sum of the leaves.
        [(0, 1), (0, 2), (1, 2)]
            .iter()
            .map(|&(i, j)| gates.xor(&limbs[i], &limbs[j]))
            .collect()
    }

    fn product_slots(gates: &mut Gates, x: &Leaves<'_>, y: &Leaves<'_>) -> Vec<Option<Vec<Wire>>> {
        let (x, y) = if x.embedded() { (y, x) } else { (x, y) };
        let mut slots = vec![None; 9];
        if y.embedded() {
            // A committed-field factor: three limb products, one per coefficient.
            gates.current.mul_mixed += 1;
            let ey = y.limbs::<Self>(gates)[0].clone();
            let ex = x.limbs::<Self>(gates).clone();
            for (l, ex) in ex.iter().enumerate() {
                slots[6 + l] = Some(gates.leaf_products(ex, &ey));
            }
        } else {
            gates.current.mul192 += 1;
            let ex: Vec<Vec<Wire>> = x
                .limbs::<Self>(gates)
                .iter()
                .chain(x.sums::<Self>(gates))
                .cloned()
                .collect();
            let ey: Vec<Vec<Wire>> = y
                .limbs::<Self>(gates)
                .iter()
                .chain(y.sums::<Self>(gates))
                .cloned()
                .collect();
            for (k, (ex, ey)) in ex.iter().zip(&ey).enumerate() {
                slots[k] = Some(gates.leaf_products(ex, ey));
            }
        }
        slots
    }

    fn finish(gates: &mut Gates, slots: &[Option<Vec<Wire>>]) -> Vec<Wire> {
        let slots: &[Option<Vec<Wire>>; 9] = slots.try_into().expect("nine slots");
        gates.finish192(slots)
    }

    fn mul_const_wires(gates: &mut Gates, c: Self, x: &[Wire]) -> Vec<Wire> {
        gates.mul192_const(c, x)
    }

    fn square_wires(gates: &mut Gates, x: &[Wire]) -> Vec<Wire> {
        gates.square192(x)
    }
}

impl<E: Lowering> Add for Bin<E> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        match (&self.0, &rhs.0) {
            (Repr::Const(a), Repr::Const(b)) => Self(Repr::Const(*a + *b)),
            (Repr::Const(z), _) if *z == E::ZERO => rhs,
            (_, Repr::Const(z)) if *z == E::ZERO => self,
            (Repr::Lazy(a), Repr::Lazy(b)) => {
                // Sums of products stay unrecombined: their leaf products add.
                let slots = {
                    let mut gates = a.gates.lock();
                    a.slots
                        .iter()
                        .zip(&b.slots)
                        .map(|(x, y)| match (x, y) {
                            (Some(x), Some(y)) => Some(gates.xor(x, y)),
                            (Some(v), None) | (None, Some(v)) => Some(v.clone()),
                            (None, None) => None,
                        })
                        .collect()
                };
                Self(Repr::Lazy(Arc::new(Lazy {
                    value: a.value + b.value,
                    gates: Arc::clone(&a.gates),
                    slots,
                    wires: Once::new(),
                })))
            }
            _ => {
                let (lhs, rhs) = (self.settled(), rhs.settled());
                let shared = Arc::clone(
                    lhs.gates()
                        .or_else(|| rhs.gates())
                        .expect("a wired operand"),
                );
                let value = lhs.value() + rhs.value();
                let wires = {
                    let mut gates = shared.lock();
                    let (x, y) = (lhs.wires(&gates), rhs.wires(&gates));
                    gates.xor(&x, &y)
                };
                Self::wired(&shared, wires, value)
            }
        }
    }
}

impl<E: Lowering> Mul for Bin<E> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let (lhs, rhs) = (self.settled(), rhs.settled());
        match (&lhs.0, &rhs.0) {
            (Repr::Const(a), Repr::Const(b)) => Self(Repr::Const(*a * *b)),
            (Repr::Const(c), _) | (_, Repr::Const(c)) if *c == E::ZERO => Self::ZERO,
            (Repr::Const(c), _) if *c == E::ONE => rhs,
            (_, Repr::Const(c)) if *c == E::ONE => lhs,
            (Repr::Const(c), Repr::Wired(w)) | (Repr::Wired(w), Repr::Const(c)) => {
                let wires = E::mul_const_wires(&mut w.gates.lock(), *c, &w.wires);
                Self::wired(&w.gates, wires, *c * w.value)
            }
            (Repr::Wired(a), Repr::Wired(b)) if Arc::ptr_eq(a, b) => lhs.square(),
            (Repr::Wired(a), Repr::Wired(b)) => {
                let slots = E::product_slots(&mut a.gates.lock(), &a.leaves(), &b.leaves());
                Self(Repr::Lazy(Arc::new(Lazy {
                    value: a.value * b.value,
                    gates: Arc::clone(&a.gates),
                    slots,
                    wires: Once::new(),
                })))
            }
            _ => unreachable!("settled values are constants or wires"),
        }
    }
}

impl<E: Lowering> Sub for Bin<E> {
    type Output = Self;

    // Characteristic two: subtraction is addition.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, rhs: Self) -> Self {
        self + rhs
    }
}

impl<E: Lowering> Neg for Bin<E> {
    type Output = Self;

    fn neg(self) -> Self {
        self
    }
}

impl<E: Lowering> AddAssign for Bin<E> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.dup() + rhs;
    }
}

impl<E: Lowering> SubAssign for Bin<E> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.dup() - rhs;
    }
}

impl<E: Lowering> MulAssign for Bin<E> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.dup() * rhs;
    }
}

impl<E: Lowering> Sum for Bin<E> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, x| acc + x)
    }
}

impl<E: Lowering> Product for Bin<E> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ONE, |acc, x| acc * x)
    }
}

impl<E: Lowering> Default for Bin<E> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<E: Lowering> Dup for Bin<E> {
    #[inline]
    fn dup(&self) -> Self {
        self.clone()
    }
}

impl<E: Lowering> From<E> for Bin<E> {
    fn from(value: E) -> Self {
        Self(Repr::Const(value))
    }
}

impl<E: Lowering + Debug> PrimeCharacteristicRing for Bin<E> {
    type PrimeSubfield = Gf2;

    const ZERO: Self = Self(Repr::Const(E::ZERO));
    const ONE: Self = Self(Repr::Const(E::ONE));
    const TWO: Self = Self(Repr::Const(E::TWO));
    const NEG_ONE: Self = Self(Repr::Const(E::NEG_ONE));

    fn from_prime_subfield(f: Gf2) -> Self {
        Self(Repr::Const(E::from_prime_subfield(f)))
    }

    fn square(&self) -> Self {
        match &self.dup().settled().0 {
            Repr::Const(c) => Self(Repr::Const(c.square())),
            Repr::Wired(w) => {
                let wires = E::square_wires(&mut w.gates.lock(), &w.wires);
                Self::wired(&w.gates, wires, w.value.square())
            }
            Repr::Lazy(_) => unreachable!("a settled value"),
        }
    }
}

/// Ring operations with a right-hand side lifted into the value type.
macro_rules! mixed_impls {
    ($rhs:ty) => {
        impl Add<$rhs> for Gf192 {
            type Output = Self;

            fn add(self, rhs: $rhs) -> Self {
                self + Self::from(rhs)
            }
        }

        impl Sub<$rhs> for Gf192 {
            type Output = Self;

            fn sub(self, rhs: $rhs) -> Self {
                self - Self::from(rhs)
            }
        }

        impl Mul<$rhs> for Gf192 {
            type Output = Self;

            fn mul(self, rhs: $rhs) -> Self {
                self * Self::from(rhs)
            }
        }

        impl AddAssign<$rhs> for Gf192 {
            fn add_assign(&mut self, rhs: $rhs) {
                *self += Self::from(rhs);
            }
        }

        impl SubAssign<$rhs> for Gf192 {
            fn sub_assign(&mut self, rhs: $rhs) {
                *self -= Self::from(rhs);
            }
        }

        impl MulAssign<$rhs> for Gf192 {
            fn mul_assign(&mut self, rhs: $rhs) {
                *self *= Self::from(rhs);
            }
        }
    };
}

mixed_impls!(Gf64);
mixed_impls!(Poly64);

impl Add<Poly64> for Gf64 {
    type Output = Self;

    fn add(self, rhs: Poly64) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<Poly64> for Gf64 {
    type Output = Self;

    fn sub(self, rhs: Poly64) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<Poly64> for Gf64 {
    type Output = Self;

    fn mul(self, rhs: Poly64) -> Self {
        self * Self::from(rhs)
    }
}

impl AddAssign<Poly64> for Gf64 {
    fn add_assign(&mut self, rhs: Poly64) {
        *self += Self::from(rhs);
    }
}

impl SubAssign<Poly64> for Gf64 {
    fn sub_assign(&mut self, rhs: Poly64) {
        *self -= Self::from(rhs);
    }
}

impl MulAssign<Poly64> for Gf64 {
    fn mul_assign(&mut self, rhs: Poly64) {
        *self *= Self::from(rhs);
    }
}

impl Add<Poly192> for Gf192 {
    type Output = Self;

    fn add(self, rhs: Poly192) -> Self {
        self + Self::from(rhs)
    }
}

impl Sub<Poly192> for Gf192 {
    type Output = Self;

    fn sub(self, rhs: Poly192) -> Self {
        self - Self::from(rhs)
    }
}

impl Mul<Poly192> for Gf192 {
    type Output = Self;

    fn mul(self, rhs: Poly192) -> Self {
        self * Self::from(rhs)
    }
}

impl AddAssign<Poly192> for Gf192 {
    fn add_assign(&mut self, rhs: Poly192) {
        *self += Self::from(rhs);
    }
}

impl SubAssign<Poly192> for Gf192 {
    fn sub_assign(&mut self, rhs: Poly192) {
        *self -= Self::from(rhs);
    }
}

impl MulAssign<Poly192> for Gf192 {
    fn mul_assign(&mut self, rhs: Poly192) {
        *self *= Self::from(rhs);
    }
}

impl From<Poly64> for Gf192 {
    fn from(value: Poly64) -> Self {
        Self(Repr::Const(Poly192::from(value)))
    }
}

impl From<Gf64> for Gf192 {
    /// The committed field in the challenge field: the low limb, the others structural zeros.
    fn from(value: Gf64) -> Self {
        match value.settled().0 {
            Repr::Const(c) => Self::from(c),
            Repr::Wired(w) => {
                let mut wires = w.wires.clone();
                wires.resize(192, Wire::ZERO);
                Self::wired(&w.gates, wires, Poly192::from(w.value))
            }
            Repr::Lazy(_) => unreachable!("a settled value"),
        }
    }
}

impl Algebra<Poly64> for Gf64 {}

impl Algebra<Poly192> for Gf192 {}

impl Algebra<Poly64> for Gf192 {}

impl Algebra<Gf64> for Gf192 {}

impl Gf192 {
    /// The three committed-field coordinates.
    pub(crate) fn limbs(&self) -> [Gf64; 3] {
        match &self.dup().settled().0 {
            Repr::Const(c) => {
                let limbs = limbs_of(c);
                core::array::from_fn(|l| Gf64::from(limbs[l]))
            }
            Repr::Wired(w) => {
                let limbs = limbs_of(&w.value);
                core::array::from_fn(|l| {
                    Gf64::wired(&w.gates, w.wires[64 * l..64 * l + 64].to_vec(), limbs[l])
                })
            }
            Repr::Lazy(_) => unreachable!("a settled value"),
        }
    }

    /// The value with the given committed-field coordinates.
    pub(crate) fn from_limbs(limbs: &[Gf64; 3]) -> Self {
        let limbs = limbs.clone().map(Bin::settled);
        let value = Poly192::new(core::array::from_fn(|l| limbs[l].value()));
        limbs.iter().find_map(Bin::gates).map_or_else(
            || Self::from(value),
            |shared| {
                let wires = {
                    let gates = shared.lock();
                    limbs.iter().flat_map(|l| l.wires(&gates)).collect()
                };
                Self::wired(shared, wires, value)
            },
        )
    }
}

impl<E: Lowering> Bin<E> {
    /// `bit * self`, one product per wire.
    pub(crate) fn scaled(&self, bit: Wire, bit_value: bool, gates: &Shared) -> Self {
        let value = if bit_value { self.value() } else { E::ZERO };
        let settled = self.dup().settled();
        let wires = {
            let mut g = gates.lock();
            let x = settled.wires(&g);
            g.scale(bit, &x)
        };
        Self::wired(gates, wires, value)
    }

    /// Whether both values are the same wires.
    pub(crate) fn ptr_eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Repr::Wired(a), Repr::Wired(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}
