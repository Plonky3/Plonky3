use alloc::vec::Vec;

use itertools::Itertools;
use p3_field::{
    batch_multiplicative_inverse, cyclic_subgroup_coset_known_order, ExtensionField, Field,
    TwoAdicField,
};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;
use p3_util::{log2_ceil_usize, log2_strict_usize};

/// Given a `PolynomialSpace`, `S`, and a subset `R`, a Lagrange Selector `P_R` is
/// a polynomial which is not equal to `0` for every element in `R` but is equal
/// to `0` for every element of `S` not in `R`.
///
/// This struct contains either a single or a collection of evaluations of
/// several lagrange selector over a particular `PolynomialSpace`.
///
/// The Lagrange selector is normalized if it is equal to `1` for every element in `R`.
/// The LagrangeSelectors given here are not normalized.
#[derive(Debug)]
pub struct LagrangeSelectors<T> {
    /// A Lagrange selector corresponding to the first point in the space.
    pub is_first_row: T,
    /// A Lagrange selector corresponding to the last point in the space.
    pub is_last_row: T,
    /// A Lagrange selector corresponding the subset of all but the last point.
    pub is_transition: T,
    /// The inverse of the zerofier which is a lagrange selector corresponding to the empty set
    pub inv_zeroifier: T,
}

/// Fixing a field `F`, `PolynomialSpace<Val = F>` is an abstract indexed subset of `F^n`
/// with some additional algebraic structure.
///
/// We do not expect `PolynomialSpace` to store this subset, instead it usually contains
/// some associated data which allows it to generate the subset or pieces of it.
///
/// Each `PolynomialSpace` should be part of a generic family of similar spaces for some
/// collection of sizes (usually powers of two). Any space other than at the smallest size
/// should be decomposable into a disjoint collection of smaller spaces. Additionally, the
/// set of all `PolynomialSpace` of a given size should form a disjoint partition of some
/// subset of `F^n` which supports a group structure.
///
/// The canonical example of a `PolynomialSpace` is a coset `gH` of
/// a two-adic subgroup `H` of the multiplicative group `F*`. This satisfies the properties
/// above as cosets partition the group and decompose as `gH = g(H^2) u gh(H^2)` for `h` a
/// generator of `H`.
pub trait PolynomialSpace: Copy {
    /// The base field `F`.
    type Val: Field;

    /// The number of elements of the space.
    fn size(&self) -> usize;

    /// The first point in the space.
    fn first_point(&self) -> Self::Val;

    /// An algebraic function which takes the i'th element of the space and returns
    /// the (i+1)'th evaluated on the given point.
    ///
    /// When `PolynomialSpace` corresponds to a coset, `gH` this
    /// function is multiplication by `h` for a chosen generator `h` of `H`.
    ///
    /// This function may not exist for other classes of `PolynomialSpace` in which
    /// case this will return `None`.
    fn next_point<Ext: ExtensionField<Self::Val>>(&self, x: Ext) -> Option<Ext>;

    /// Return another `PolynomialSpace` with size `min_size` disjoint from this space.
    ///
    /// This fixes a canonical choice for prover/verifier determinism and LDE caching.
    fn create_disjoint_domain(&self, min_size: usize) -> Self;

    /// Split the `PolynomialSpace` into `num_chunks` smaller `PolynomialSpaces` of equal size.
    ///
    /// `num_chunks` must divide `self.size()` (which usually forces it to be a power of 2.)
    fn split_domains(&self, num_chunks: usize) -> Vec<Self>;

    /// Split a set of polynomial evaluations over this `PolynomialSpace` into a vector
    /// of polynomial evaluations over each `PolynomialSpace` generated from `split_domains`.
    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Self::Val>,
    ) -> Vec<RowMajorMatrix<Self::Val>>;

    /// Compute the zerofier of the space, evaluated at the given point.
    fn zp_at_point<Ext: ExtensionField<Self::Val>>(&self, point: Ext) -> Ext;

    /// Compute several lagrange selectors at a given point.
    ///
    /// Note that these may be unnormalized.
    fn selectors_at_point<Ext: ExtensionField<Self::Val>>(
        &self,
        point: Ext,
    ) -> LagrangeSelectors<Ext>;

    /// Compute several lagrange selectors at all points of the given disjoint `PolynomialSpace`.
    ///
    /// Note that these may be unnormalized.
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Self::Val>>;
}

/// A coset of the form `gH` where `H` is the unique multiplicative subgroup of order `n = 2^{log_n}`.
///
/// Fixing a generator `h` of `H`, we index this subgroup by `{g, gh, gh^2, ..., gh^{-1}}`.
/// We refer to `g` as the "shift" of the coset.
#[derive(Copy, Clone, Debug)]
pub struct TwoAdicMultiplicativeCoset<Val: TwoAdicField> {
    /// The log of the order of the subgroup.
    pub log_n: usize,
    /// The shift defining the coset.
    pub shift: Val,
}

impl<Val: TwoAdicField> TwoAdicMultiplicativeCoset<Val> {
    /// Return the element `h` which generates the subgroup `H`.
    fn gen(&self) -> Val {
        Val::two_adic_generator(self.log_n)
    }
}

impl<Val: TwoAdicField> PolynomialSpace for TwoAdicMultiplicativeCoset<Val> {
    type Val = Val;

    fn size(&self) -> usize {
        1 << self.log_n
    }

    fn first_point(&self) -> Self::Val {
        self.shift
    }

    /// Getting the next point corresponds to multiplication by the generator.
    fn next_point<Ext: ExtensionField<Val>>(&self, x: Ext) -> Option<Ext> {
        Some(x * self.gen())
    }

    /// Given the coset `gH`, return the disjoint coset `gfK` where `f`
    /// is a fixed generator of `F^*` and `K` is the unique two-adic subgroup
    /// of with size `2^(ceil(log_2(min_size)))`.
    fn create_disjoint_domain(&self, min_size: usize) -> Self {
        // We provide a short proof that these cosets are always disjoint:
        //
        // Assume without loss of generality that `|H| <= min_size <= |K|`.
        // Then we know that `gH` is entirely contained in `gK`. As cosets are
        // either equal or disjoint, this means that `gH` is disjoint from `g'K`
        // for every `g'` not contained in `gK`. As `f` is a generator of `F^*`
        // it does not lie in `K` and so `gf` cannot lie in `gK`.
        //
        // Thus `gH` and `gfK` are disjoint.
        Self {
            log_n: log2_ceil_usize(min_size),
            shift: self.shift * Val::GENERATOR,
        }
    }

    /// Given the coset `gH` and generator `h` of `H`, let `K = H^{num_chunks}`
    /// be the unique group of order `|H|/num_chunks`.
    ///
    /// Then we decompose `gH` into `gK, ghK, gh^2K, ..., gh^{num_chunks}K`.
    fn split_domains(&self, num_chunks: usize) -> Vec<Self> {
        let log_chunks = log2_strict_usize(num_chunks);
        (0..num_chunks)
            .map(|i| Self {
                log_n: self.log_n - log_chunks,
                shift: self.shift * self.gen().exp_u64(i as u64),
            })
            .collect()
    }

    fn split_evals(
        &self,
        num_chunks: usize,
        evals: RowMajorMatrix<Self::Val>,
    ) -> Vec<RowMajorMatrix<Self::Val>> {
        // todo less copy
        (0..num_chunks)
            .map(|i| {
                evals
                    .as_view()
                    .vertically_strided(num_chunks, i)
                    .to_row_major_matrix()
            })
            .collect()
    }

    /// Compute the zerofier polynomial at the given point:
    ///
    /// `Z_{gH}(X) = g^{-|H|}\prod_{h \in H} (X - gh) = (g^{-1}X)^|H| - 1`
    fn zp_at_point<Ext: ExtensionField<Val>>(&self, point: Ext) -> Ext {
        (point * self.shift.inverse()).exp_power_of_2(self.log_n) - Ext::ONE
    }

    /// Compute some lagrange selectors at the given point:
    ///
    /// Defining the zerofier by `Z_{gH}(X) = g^{-|H|}\prod_{h \in H} (X - gh) = (g^{-1}X)^|H| - 1` return:
    /// - `Z_{gH}(X)/(g^{-1}X - 1)`: The lagrange selector of the point `g`.
    /// - `Z_{gH}(X)/(g^{-1}X - h^{-1})`: The lagrange selector of the point `gh^{-1}` where `h` is the generator of `H`.
    /// - `(g^{-1}X - h^{-1})`: The lagrange selector of the subset consisting of everything but the point `gh^{-1}`.
    /// - `1/Z_{gH}(X)`: The inverse of the zerofier.
    fn selectors_at_point<Ext: ExtensionField<Val>>(&self, point: Ext) -> LagrangeSelectors<Ext> {
        let unshifted_point = point * self.shift.inverse();
        let z_h = unshifted_point.exp_power_of_2(self.log_n) - Ext::ONE;
        LagrangeSelectors {
            is_first_row: z_h / (unshifted_point - Ext::ONE),
            is_last_row: z_h / (unshifted_point - self.gen().inverse()),
            is_transition: unshifted_point - self.gen().inverse(),
            inv_zeroifier: z_h.inverse(),
        }
    }

    /// Compute the lagrange selectors of our space at every point in the coset.
    ///
    /// This will error if our space is not the group `H` and if the given
    /// coset is not disjoint from `H`.
    fn selectors_on_coset(&self, coset: Self) -> LagrangeSelectors<Vec<Val>> {
        assert_eq!(self.shift, Val::ONE);
        assert_ne!(coset.shift, Val::ONE);
        assert!(coset.log_n >= self.log_n);
        let rate_bits = coset.log_n - self.log_n;

        let s_pow_n = coset.shift.exp_power_of_2(self.log_n);
        // evals of Z_H(X) = X^n - 1
        let evals = Val::two_adic_generator(rate_bits)
            .powers()
            .take(1 << rate_bits)
            .map(|x| s_pow_n * x - Val::ONE)
            .collect_vec();

        let xs = cyclic_subgroup_coset_known_order(coset.gen(), coset.shift, 1 << coset.log_n)
            .collect_vec();

        let single_point_selector = |i: u64| {
            let coset_i = self.gen().exp_u64(i);
            let denoms = xs.iter().map(|&x| x - coset_i).collect_vec();
            let invs = batch_multiplicative_inverse(&denoms);
            evals
                .iter()
                .cycle()
                .zip(invs)
                .map(|(&z_h, inv)| z_h * inv)
                .collect_vec()
        };

        let subgroup_last = self.gen().inverse();

        LagrangeSelectors {
            is_first_row: single_point_selector(0),
            is_last_row: single_point_selector((1 << self.log_n) - 1),
            is_transition: xs.into_iter().map(|x| x - subgroup_last).collect(),
            inv_zeroifier: batch_multiplicative_inverse(&evals)
                .into_iter()
                .cycle()
                .take(1 << coset.log_n)
                .collect(),
        }
    }
}
