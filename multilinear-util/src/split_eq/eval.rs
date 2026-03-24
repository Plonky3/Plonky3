use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::evals::{PARALLEL_THRESHOLD, Poly};
use crate::split_eq::{EqMaybePacked, SplitEq};

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Evaluates a base-field polynomial against the split eq tables:
    /// ```text
    ///   Σ_{x ∈ {0,1}^k} eq(z, x) · poly(x)
    /// ```
    pub fn eval_base(&self, poly: &Poly<F>) -> EF {
        assert_eq!(poly.num_vars(), self.num_vars());

        if let Some(constant) = poly.as_constant() {
            return constant.into();
        }

        match &self.eq1 {
            EqMaybePacked::Unpacked(eq1) => {
                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    poly.0
                        .chunks(eq1.num_evals())
                        .zip_eq(self.eq0.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<EF>()
                                * w1
                        })
                        .sum::<EF>()
                } else {
                    poly.0
                        .par_chunks(eq1.num_evals())
                        .zip_eq(self.eq0.0.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<EF>()
                                * w1
                        })
                        .sum::<EF>()
                }
            }

            EqMaybePacked::Packed(eq1) => {
                let poly = F::Packing::pack_slice(poly.as_slice());
                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    let sum = poly
                        .chunks(eq1.num_evals())
                        .zip_eq(self.eq0.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<EF::ExtensionPacking>()
                                * w1
                        })
                        .sum::<EF::ExtensionPacking>();
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                } else {
                    let sum = poly
                        .par_chunks(eq1.num_evals())
                        .zip_eq(self.eq0.0.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<EF::ExtensionPacking>()
                                * w1
                        })
                        .sum::<EF::ExtensionPacking>();
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                }
            }
        }
    }

    /// Evaluates an extension-field polynomial against the split eq tables:
    /// ```text
    ///   Σ_{x ∈ {0,1}^k} eq(z, x) · poly(x)
    /// ```
    pub fn eval_ext(&self, poly: &Poly<EF>) -> EF {
        assert_eq!(poly.num_vars(), self.num_vars());

        if let Some(constant) = poly.as_constant() {
            return constant;
        }

        match &self.eq1 {
            EqMaybePacked::Unpacked(eq1) => {
                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    poly.0
                        .chunks(eq1.num_evals())
                        .zip_eq(self.eq0.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<EF>()
                                * w1
                        })
                        .sum::<EF>()
                } else {
                    poly.0
                        .par_chunks(eq1.num_evals())
                        .zip_eq(self.eq0.0.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| w0 * f)
                                .sum::<EF>()
                                * w1
                        })
                        .sum::<EF>()
                }
            }
            EqMaybePacked::Packed(eq1) => {
                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    let sum = poly
                        .0
                        .chunks(eq1.num_evals() * F::Packing::WIDTH)
                        .zip_eq(self.eq0.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .chunks(F::Packing::WIDTH)
                                .zip_eq(eq1.iter())
                                .map(|(chunk, &w0)| {
                                    EF::ExtensionPacking::from_ext_slice(chunk) * w0
                                })
                                .sum::<EF::ExtensionPacking>()
                                * w1
                        })
                        .sum::<EF::ExtensionPacking>();
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                } else {
                    let sum = poly
                        .0
                        .par_chunks(eq1.num_evals() * F::Packing::WIDTH)
                        .zip_eq(self.eq0.0.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .chunks(F::Packing::WIDTH)
                                .zip_eq(eq1.iter())
                                .map(|(chunk, &w0)| {
                                    EF::ExtensionPacking::from_ext_slice(chunk) * w0
                                })
                                .sum::<EF::ExtensionPacking>()
                                * w1
                        })
                        .sum::<EF::ExtensionPacking>();
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                }
            }
        }
    }

    /// Evaluates a packed extension-field polynomial against the split eq tables:
    /// ```text
    ///   Σ_{x ∈ {0,1}^k} eq(z, x) · poly(x)
    /// ```
    pub fn eval_packed(&self, poly: &Poly<EF::ExtensionPacking>) -> EF {
        assert_eq!(
            poly.num_vars() + log2_strict_usize(F::Packing::WIDTH),
            self.num_vars()
        );
        match &self.eq1 {
            EqMaybePacked::Packed(eq1) => {
                if (1 << (self.num_vars() - log2_strict_usize(F::Packing::WIDTH)))
                    < PARALLEL_THRESHOLD
                {
                    let sum = poly
                        .0
                        .chunks(eq1.num_evals())
                        .zip_eq(self.eq0.iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| f * w0)
                                .sum::<EF::ExtensionPacking>()
                                * w1
                        })
                        .sum::<EF::ExtensionPacking>();
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                } else {
                    let sum = poly
                        .0
                        .par_chunks(eq1.num_evals())
                        .zip_eq(self.eq0.0.par_iter())
                        .map(|(chunk, &w1)| {
                            chunk
                                .iter()
                                .zip_eq(eq1.iter())
                                .map(|(&f, &w0)| f * w0)
                                .sum::<EF::ExtensionPacking>()
                                * w1
                        })
                        .sum::<EF::ExtensionPacking>();
                    EF::ExtensionPacking::to_ext_iter([sum]).sum()
                }
            }
            EqMaybePacked::Unpacked(_) => self.eval_ext(&Poly::new(
                EF::ExtensionPacking::to_ext_iter(poly.iter().copied()).collect(),
            )),
        }
    }
}
