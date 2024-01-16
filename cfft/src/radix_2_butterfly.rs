use alloc::vec::Vec;
use p3_matrix::dense::{RowMajorMatrix};
use p3_matrix::Matrix;
use p3_field::{Field, ComplexExtension};
use crate::traits::CircleSubgroupFFT;
use crate::util::{cfft_domain, twin_coset_domain};

/// An O(N Log(N)) implementation of the CFT roughly analagous to the FFT approach to the DFT.
/// See the paper "Circle Stark".
#[derive(Default, Clone)]
pub struct Radix2CFT;

impl<Base: Field, Ext: ComplexExtension<Base>> CircleSubgroupFFT<Base, Ext> for Radix2CFT {
	type Evaluations  = RowMajorMatrix<Base>;

	fn cfft_batch(&self, mut mat: RowMajorMatrix<Base>) -> <Self as CircleSubgroupFFT<Base, Ext>>::Evaluations {
		let n = mat.height();
    	let n_u32: u32 = n.try_into().unwrap();
    	let log_n: usize = n.trailing_zeros().try_into().unwrap();
    	debug_assert_eq!(1_u32 << log_n, n_u32); // The Height better be a power of 2.

    	let twiddles = cfft_twiddles::<Base, Ext>(log_n);

    	for (i, twiddle) in twiddles.iter().enumerate() {
        let block_size = 1 << (log_n - i);
        let half_block_size = block_size >> 1;


        // CURRENTLY THIS IS TOTALLY BROKEN AND WILL NOT WORK FOR MATRICES OF WIDTH > 1.	
        for chunck in mat.values.chunks_mut(block_size) {
            // Need this if we didn't change the twiddles.
            // chunck[half_block_size..].reverse();
            for j in 0..half_block_size {
                let s = chunck[j].clone();
                let t = chunck[block_size - j - 1].clone();
                chunck[j] = s.clone() + t.clone();
                chunck[block_size - j - 1] = (s - t) * twiddle[j];
            }
        }
    }
    	mat
	}
	
	fn coset_cfft_batch(&self, _: RowMajorMatrix<Base>, _: Ext) -> <Self as CircleSubgroupFFT<Base, Ext>>::Evaluations {
		todo!()
	}
}


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

// Code for computing the twiddles.
// Currently, this requires us to recompute the twiddles every time we want to use the CFFT.
// This is somewhat expensive so, in the long run, it would be ideal to modify the structure so they can be precomputed.

/// Compute the twiddles for the CFFT.
/// Let N = 2^n be the size of our initial set. Then we start with the domain
/// {g, g^3, ..., g^{-3}, g^{-1}} for g a 2N'th root of unity.
/// The initial twiddle domain is the first half of the full domain.
/// In the first step our twiddles are the inverse imaginary parts and we simply halve the domain size.
/// In all subsequent steps our twiddles are the inverse real parts and we both halve the domain size and square every element.
pub fn cfft_twiddles<Base: Field, Ext: ComplexExtension<Base>>(log_n: usize) -> Vec<Vec<Base>> {
    let size = 1 << (log_n - 1);
    let init_domain = cfft_domain::<Base, Ext>(log_n, size); // Get the starting domain.

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect(); // After the first step we only need the real part.

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            if i == 0 {
                init_domain.iter().map(|x| x.imag().inverse()).collect() // The twiddles in step one are the inverse imaginary parts.
            } else {
                let output = working_domain.iter().map(|x| x.inverse()).collect(); // The twiddles in subsequent steps are the inverse real parts.
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect(); // When we square a point, the real part changes as x -> 2x^2 - 1.
                output
            }
        })
        .collect()
}

/// Compute the twiddles for the inverse CFFT.
/// The twiddles are essentially the same as in the CFFT case except we no longer need to take inverses.
pub fn cfft_inv_twiddles<Base: Field, Ext: ComplexExtension<Base>>(log_n: usize) -> Vec<Vec<Base>> {
    let size = 1 << (log_n - 1);
    let init_domain = cfft_domain::<Base, Ext>(log_n, size);

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect();

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            if i == 0 {
                init_domain.iter().map(|x| x.imag()).collect() // The twiddles in the outer step are the imaginary parts.
            } else {
                // Not sure if this is the cleanest was of doing this.

                let output = working_domain.clone(); // The twiddles in all other steps are the real parts.
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect();
                output
            }
        })
        .collect()
}

/// Compute the twiddles for the coset evaluation.
/// Unlike the previous cases, here we actually need to start with a given group element.
/// TODO: Explain what these twiddles are.
pub fn coset_eval_twiddles<Base: Field, Ext: ComplexExtension<Base>>(
    log_n: usize,
    coset_elem: Ext,
) -> Vec<Vec<Base>> {
    let size = 1 << (log_n - 1);
    let generator = Ext::circle_two_adic_generator(log_n - 1);

    let init_domain = twin_coset_domain::<Base, Ext>(generator, coset_elem, size);

    let mut working_domain: Vec<_> = init_domain
        .iter()
        .take(size / 2)
        .map(|x| x.real())
        .collect();

    (0..log_n)
        .map(|i| {
            let size = working_domain.len();
            if i == 0 {
                init_domain.iter().map(|x| x.imag()).collect()
            } else {
                let output = working_domain.clone();
                working_domain = working_domain
                    .iter()
                    .take(size / 2)
                    .map(|x| Base::two() * *x * *x - Base::one())
                    .collect();
                output
            }
        })
        .collect()
}