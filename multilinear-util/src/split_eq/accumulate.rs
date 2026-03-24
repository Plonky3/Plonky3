use itertools::Itertools;
use p3_field::{ExtensionField, Field, PackedFieldExtension, PackedValue};
use p3_maybe_rayon::prelude::*;
use p3_util::log2_strict_usize;

use crate::evals::PARALLEL_THRESHOLD;
use crate::split_eq::{EqMaybePacked, SplitEq};

impl<F: Field, EF: ExtensionField<F>> SplitEq<F, EF> {
    /// Accumulates the eq table into `out`, using the factored `eq0 · eq1`
    /// representation. When `scale` is `None`, uses `1`.
    /// ```text
    ///   out[x] += scale · eq(point, x)   for all x ∈ {0,1}^k
    /// ```
    pub fn accumulate_into(&self, out: &mut [EF], scale: Option<EF>) {
        assert_eq!(log2_strict_usize(out.len()), self.num_vars());
        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            match (&self.eq1, scale) {
                (EqMaybePacked::Unpacked(eq1), Some(scale)) => {
                    out.chunks_mut(eq1.num_evals())
                        .zip(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w0 * w1 * scale);
                        });
                }
                (EqMaybePacked::Unpacked(eq1), None) => {
                    out.chunks_mut(eq1.num_evals())
                        .zip(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w0 * w1);
                        });
                }
                (EqMaybePacked::Packed(eq1), Some(scale)) => {
                    out.chunks_mut(eq1.num_evals() * F::Packing::WIDTH)
                        .zip(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks_mut(F::Packing::WIDTH)
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| {
                                    out.iter_mut()
                                        .zip_eq(EF::ExtensionPacking::to_ext_iter([w1]))
                                        .for_each(|(out, w1)| *out += w1 * w0 * scale);
                                });
                        });
                }
                (EqMaybePacked::Packed(eq1), None) => {
                    out.chunks_mut(eq1.num_evals() * F::Packing::WIDTH)
                        .zip(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks_mut(F::Packing::WIDTH)
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| {
                                    out.iter_mut()
                                        .zip_eq(EF::ExtensionPacking::to_ext_iter([w1]))
                                        .for_each(|(out, w1)| *out += w1 * w0);
                                });
                        });
                }
            }
        } else {
            match (&self.eq1, scale) {
                (EqMaybePacked::Unpacked(eq1), Some(scale)) => {
                    out.par_chunks_mut(eq1.num_evals())
                        .zip(self.eq0.0.par_iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w0 * w1 * scale);
                        });
                }
                (EqMaybePacked::Unpacked(eq1), None) => {
                    out.par_chunks_mut(eq1.num_evals())
                        .zip(self.eq0.0.par_iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w0 * w1);
                        });
                }
                (EqMaybePacked::Packed(eq1), Some(scale)) => {
                    out.par_chunks_mut(eq1.num_evals() * F::Packing::WIDTH)
                        .zip(self.eq0.0.par_iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks_mut(F::Packing::WIDTH)
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| {
                                    out.iter_mut()
                                        .zip_eq(EF::ExtensionPacking::to_ext_iter([w1]))
                                        .for_each(|(out, w1)| *out += w1 * w0 * scale);
                                });
                        });
                }
                (EqMaybePacked::Packed(eq1), None) => {
                    out.par_chunks_mut(eq1.num_evals() * F::Packing::WIDTH)
                        .zip(self.eq0.0.par_iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .chunks_mut(F::Packing::WIDTH)
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| {
                                    out.iter_mut()
                                        .zip_eq(EF::ExtensionPacking::to_ext_iter([w1]))
                                        .for_each(|(out, w1)| *out += w1 * w0);
                                });
                        });
                }
            }
        }
    }

    /// Like [`accumulate_into`](Self::accumulate_into), but writes into a packed
    /// extension-field buffer.
    pub fn accumulate_into_packed(&self, out: &mut [EF::ExtensionPacking], scale: Option<EF>) {
        assert_eq!(
            log2_strict_usize(F::Packing::WIDTH * out.len()),
            self.num_vars()
        );

        if (1 << self.num_vars()) < PARALLEL_THRESHOLD {
            match (&self.eq1, scale) {
                (EqMaybePacked::Packed(eq1), Some(scale)) => {
                    out.chunks_mut(eq1.num_evals())
                        .zip(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w1 * w0 * scale);
                        });
                }
                (EqMaybePacked::Packed(eq1), None) => {
                    out.chunks_mut(eq1.num_evals())
                        .zip(self.eq0.iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w1 * w0);
                        });
                }
                _ => unreachable!(),
            }
        } else {
            match (&self.eq1, scale) {
                (EqMaybePacked::Packed(eq1), Some(scale)) => {
                    out.par_chunks_mut(eq1.num_evals())
                        .zip(self.eq0.0.par_iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w1 * w0 * scale);
                        });
                }
                (EqMaybePacked::Packed(eq1), None) => {
                    out.par_chunks_mut(eq1.num_evals())
                        .zip(self.eq0.0.par_iter())
                        .for_each(|(chunk, &w0)| {
                            chunk
                                .iter_mut()
                                .zip(eq1.iter())
                                .for_each(|(out, &w1)| *out += w1 * w0);
                        });
                }
                _ => unreachable!(),
            }
        }
    }
}
