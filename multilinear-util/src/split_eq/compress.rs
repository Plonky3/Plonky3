use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::evals::{PARALLEL_THRESHOLD, Poly};
use crate::split_eq::{EqMaybePacked, SplitEq};

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Fixes the low variables of a multilinear polynomial using the split eq
    /// tables, returning a reduced polynomial over the remaining high variables.
    ///
    /// Given `poly` with `n` variables and split eq with `m ≤ n` variables, computes:
    /// ```text
    ///   out(x_hi) = Σ_{y_lo ∈ {0,1}^m} eq(point, y_lo) · poly(y_lo, x_hi)
    /// ```
    pub fn compress_lo(&self, poly: &Poly<F>) -> Poly<EF> {
        assert!(self.num_vars() <= poly.num_vars());
        let k_inner = poly.num_vars() - self.num_vars();

        match &self.eq1 {
            EqMaybePacked::Unpacked(eq1) => {
                let size_outer = poly.num_evals() / self.eq0.num_evals();
                let size_inner = size_outer / eq1.num_evals();

                if (1 << poly.num_vars()) < PARALLEL_THRESHOLD {
                    let mut out = Poly::<EF>::zero(k_inner);
                    poly.0
                        .chunks(size_outer)
                        .zip_eq(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks(size_inner)
                                .zip_eq(eq1.iter())
                                .for_each(|(chunk, &w1)| {
                                    let w = w0 * w1;
                                    out.0
                                        .iter_mut()
                                        .zip_eq(chunk.iter())
                                        .for_each(|(acc, &f)| *acc += w * f);
                                });
                        });
                    out
                } else {
                    poly.0
                        .par_chunks(size_outer)
                        .zip_eq(self.eq0.0.par_iter())
                        .par_fold_reduce(
                            || Poly::<EF>::zero(k_inner),
                            |mut acc, (chunk, &w0)| {
                                chunk.chunks(size_inner).zip_eq(eq1.iter()).for_each(
                                    |(chunk, &w1)| {
                                        let w = w0 * w1;
                                        acc.0
                                            .iter_mut()
                                            .zip_eq(chunk.iter())
                                            .for_each(|(acc, &f)| *acc += w * f);
                                    },
                                );
                                acc
                            },
                            |mut acc, part| {
                                acc.0
                                    .iter_mut()
                                    .zip_eq(part.iter())
                                    .for_each(|(acc, &part)| *acc += part);
                                acc
                            },
                        )
                }
            }
            EqMaybePacked::Packed(eq1) => {
                let size_outer = poly.num_evals() / self.eq0.num_evals();
                let size_inner = size_outer / (eq1.num_evals() * F::Packing::WIDTH);

                if (1 << poly.num_vars()) < PARALLEL_THRESHOLD {
                    let mut out = Poly::<EF>::zero(k_inner);
                    poly.0
                        .chunks(size_outer)
                        .zip_eq(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks(size_inner * F::Packing::WIDTH)
                                .zip_eq(eq1.iter())
                                .for_each(|(chunk, &w1)| {
                                    chunk
                                        .chunks(size_inner)
                                        .zip_eq(EF::ExtensionPacking::to_ext_iter([w1 * w0]))
                                        .for_each(|(chunk, w)| {
                                            out.0
                                                .iter_mut()
                                                .zip_eq(chunk.iter())
                                                .for_each(|(acc, &f)| *acc += w * f);
                                        });
                                });
                        });
                    out
                } else {
                    poly.0
                        .par_chunks(size_outer)
                        .zip_eq(self.eq0.0.par_iter())
                        .par_fold_reduce(
                            || Poly::<EF>::zero(k_inner),
                            |mut acc, (chunk, &w0)| {
                                chunk
                                    .chunks(size_inner * F::Packing::WIDTH)
                                    .zip_eq(eq1.iter())
                                    .for_each(|(chunk, &w1)| {
                                        chunk
                                            .chunks(size_inner)
                                            .zip_eq(EF::ExtensionPacking::to_ext_iter([w1 * w0]))
                                            .for_each(|(chunk, w)| {
                                                acc.0
                                                    .iter_mut()
                                                    .zip_eq(chunk.iter())
                                                    .for_each(|(acc, &f)| *acc += w * f);
                                            });
                                    });
                                acc
                            },
                            |mut acc, part| {
                                acc.0
                                    .iter_mut()
                                    .zip_eq(part.iter())
                                    .for_each(|(acc, &part)| *acc += part);
                                acc
                            },
                        )
                }
            }
        }
    }

    /// Like [`compress_lo`](Self::compress_lo), but returns the result in packed
    /// extension-field representation. Requires that `poly` has enough variables
    /// to fill at least one packed element after compression.
    ///
    /// ```text
    ///   out(x_hi) = Σ_{y_lo ∈ {0,1}^m} eq(point, y_lo) · poly(y_lo, x_hi)
    /// ```
    pub fn compress_lo_to_packed(&self, poly: &Poly<F>) -> Poly<EF::ExtensionPacking> {
        assert!(self.num_vars() <= poly.num_vars());
        assert!(poly.num_vars() >= (self.num_vars() + log2_strict_usize(F::Packing::WIDTH)));
        let k_pack = log2_strict_usize(F::Packing::WIDTH);
        let k_inner = poly.num_vars() - self.num_vars() - k_pack;

        match &self.eq1 {
            EqMaybePacked::Unpacked(eq1) => {
                let size_outer = poly.num_evals() / self.eq0.num_evals();
                let size_inner = size_outer / (eq1.num_evals() * F::Packing::WIDTH);

                if (1 << poly.num_vars()) < PARALLEL_THRESHOLD {
                    let mut out = Poly::<EF::ExtensionPacking>::zero(k_inner);
                    poly.0
                        .chunks(size_outer)
                        .zip_eq(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            let chunk = F::Packing::pack_slice(chunk);
                            chunk
                                .chunks(size_inner)
                                .zip_eq(eq1.iter())
                                .for_each(|(chunk, &w1)| {
                                    let w = EF::ExtensionPacking::from(w0 * w1);
                                    out.0
                                        .iter_mut()
                                        .zip_eq(chunk.iter())
                                        .for_each(|(acc, &f)| *acc += w * f);
                                });
                        });
                    out
                } else {
                    poly.0
                        .par_chunks(size_outer)
                        .zip_eq(self.eq0.0.par_iter())
                        .par_fold_reduce(
                            || Poly::zero(k_inner),
                            |mut acc, (chunk, &w0)| {
                                let chunk = F::Packing::pack_slice(chunk);
                                chunk.chunks(size_inner).zip_eq(eq1.iter()).for_each(
                                    |(chunk, &w1)| {
                                        let w = EF::ExtensionPacking::from(w0 * w1);
                                        acc.0
                                            .iter_mut()
                                            .zip_eq(chunk.iter())
                                            .for_each(|(acc, &f)| *acc += w * f);
                                    },
                                );
                                acc
                            },
                            |mut acc, part| {
                                acc.0
                                    .iter_mut()
                                    .zip_eq(part.iter())
                                    .for_each(|(acc, &part)| *acc += part);
                                acc
                            },
                        )
                }
            }
            EqMaybePacked::Packed(eq1) => {
                let size_outer = poly.num_evals() / self.eq0.num_evals();
                let size_inner = size_outer / (eq1.num_evals() * F::Packing::WIDTH);

                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    let mut out = Poly::zero(k_inner);
                    poly.0
                        .chunks(size_outer)
                        .zip_eq(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks(size_inner * F::Packing::WIDTH)
                                .zip_eq(eq1.iter())
                                .for_each(|(chunk, &w1)| {
                                    chunk
                                        .chunks(size_inner)
                                        .zip_eq(EF::ExtensionPacking::to_ext_iter([w1 * w0]))
                                        .for_each(|(chunk, w)| {
                                            let w = EF::ExtensionPacking::from(w);
                                            let chunk = F::Packing::pack_slice(chunk);
                                            out.0
                                                .iter_mut()
                                                .zip_eq(chunk.iter())
                                                .for_each(|(acc, &f)| *acc += w * f);
                                        });
                                });
                        });
                    out
                } else {
                    poly.0
                        .par_chunks(size_outer)
                        .zip_eq(self.eq0.0.par_iter())
                        .par_fold_reduce(
                            || Poly::zero(k_inner),
                            |mut acc, (chunk, &w0)| {
                                chunk
                                    .chunks(size_inner * F::Packing::WIDTH)
                                    .zip_eq(eq1.iter())
                                    .for_each(|(chunk, &w1)| {
                                        let w = EF::ExtensionPacking::to_ext_iter([w1 * w0]);
                                        chunk.chunks(size_inner).zip_eq(w).for_each(
                                            |(chunk, w)| {
                                                let w = EF::ExtensionPacking::from(w);
                                                let chunk = F::Packing::pack_slice(chunk);
                                                acc.0
                                                    .iter_mut()
                                                    .zip_eq(chunk.iter())
                                                    .for_each(|(acc, &f)| *acc += w * f);
                                            },
                                        );
                                    });
                                acc
                            },
                            |mut acc, part| {
                                acc.0
                                    .iter_mut()
                                    .zip_eq(part.iter())
                                    .for_each(|(acc, &part)| *acc += part);
                                acc
                            },
                        )
                }
            }
        }
    }

    /// Fixes the high variables of a multilinear polynomial using the split eq
    /// tables, returning a reduced polynomial over the remaining low variables.
    ///
    /// Given `poly` with `n` variables and split eq with `m ≤ n` variables, computes:
    /// ```text
    ///   out(x_lo) = Σ_{y_hi ∈ {0,1}^m} eq(point, y_hi) · poly(x_lo, y_hi)
    /// ```
    pub fn compress_hi(&self, poly: &Poly<F>) -> Poly<EF> {
        let mut out = Poly::zero(poly.num_vars() - self.num_vars());
        self.compress_hi_into(out.as_mut_slice(), poly);
        out
    }

    /// Like [`compress_hi`](Self::compress_hi), but writes into a pre-allocated buffer.
    pub fn compress_hi_into(&self, out: &mut [EF], poly: &Poly<F>) {
        assert!(self.num_vars() <= poly.num_vars());
        assert_eq!(out.len(), poly.num_evals() >> self.num_vars());

        match &self.eq1 {
            EqMaybePacked::Unpacked(eq1) => {
                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    out.iter_mut()
                        .zip_eq(poly.0.chunks(1 << self.num_vars()))
                        .for_each(|(out, chunk)| {
                            *out = chunk
                                .chunks(eq1.num_evals())
                                .zip_eq(self.eq0.iter())
                                .map(|(chunk, &w0)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq1.iter())
                                        .map(|(&f, &w1)| w1 * f)
                                        .sum::<EF>()
                                        * w0
                                })
                                .sum::<EF>();
                        });
                } else {
                    out.par_iter_mut()
                        .zip(poly.0.par_chunks(1 << self.num_vars()))
                        .for_each(|(out, chunk)| {
                            *out = chunk
                                .chunks(eq1.num_evals())
                                .zip_eq(self.eq0.iter())
                                .map(|(chunk, &w0)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq1.iter())
                                        .map(|(&f, &w1)| w1 * f)
                                        .sum::<EF>()
                                        * w0
                                })
                                .sum::<EF>();
                        });
                }
            }
            EqMaybePacked::Packed(eq1) => {
                if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
                    out.iter_mut()
                        .zip_eq(poly.0.chunks(1 << self.num_vars()))
                        .for_each(|(out, chunk)| {
                            let chunk = F::Packing::pack_slice(chunk);
                            let sum = chunk
                                .chunks(eq1.num_evals())
                                .zip_eq(self.eq0.iter())
                                .map(|(chunk, &w0)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq1.iter())
                                        .map(|(&f, &w1)| w1 * f)
                                        .sum::<EF::ExtensionPacking>()
                                        * w0
                                })
                                .sum::<EF::ExtensionPacking>();
                            *out = EF::ExtensionPacking::to_ext_iter([sum]).sum();
                        });
                } else {
                    out.par_iter_mut()
                        .zip(poly.0.par_chunks(1 << self.num_vars()))
                        .for_each(|(out, chunk)| {
                            let chunk = F::Packing::pack_slice(chunk);
                            let sum = chunk
                                .chunks(eq1.num_evals())
                                .zip_eq(self.eq0.iter())
                                .map(|(chunk, &w0)| {
                                    chunk
                                        .iter()
                                        .zip_eq(eq1.iter())
                                        .map(|(&f, &w1)| w1 * f)
                                        .sum::<EF::ExtensionPacking>()
                                        * w0
                                })
                                .sum::<EF::ExtensionPacking>();
                            *out = EF::ExtensionPacking::to_ext_iter([sum]).sum();
                        });
                }
            }
        }
    }
}
