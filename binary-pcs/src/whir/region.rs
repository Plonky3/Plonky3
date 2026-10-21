//! Regions of one bit commitment, each an aligned power-of-two run of bits.

use alloc::vec::Vec;
use core::cmp::Reverse;

/// Where every region lies in a commitment of `2^num_variables` bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegionLayout {
    /// Variables of the whole commitment.
    pub num_variables: usize,
    /// The regions, in the order they were asked for.
    regions: Vec<Region>,
}

/// One region: `2^log_bits` bits from bit `offset_bits` on, the offset a multiple of the size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    /// Variables of the region.
    pub log_bits: usize,
    /// First bit of the region.
    pub offset_bits: usize,
}

impl RegionLayout {
    /// Place regions of `2^log_sizes[i]` bits, largest first, so each lands aligned to its size.
    ///
    /// The commitment is the smallest power of two holding them all.
    ///
    /// # Panics
    ///
    /// The regions do not fit the address space.
    #[must_use]
    pub fn allocate(log_sizes: &[usize]) -> Self {
        let mut regions = log_sizes
            .iter()
            .map(|&log_bits| Region {
                log_bits,
                offset_bits: 0,
            })
            .collect::<Vec<_>>();
        let mut order = regions.iter_mut().collect::<Vec<_>>();
        order.sort_by_key(|region| Reverse(region.log_bits));
        // Descending powers of two keep every running offset a multiple of the next size.
        let mut next = 0usize;
        for region in order {
            region.offset_bits = next;
            next = 1usize
                .checked_shl(region.log_bits as u32)
                .and_then(|size| next.checked_add(size))
                .expect("the regions fit the address space");
        }
        let total = next
            .checked_next_power_of_two()
            .expect("the regions fit the address space");
        Self {
            num_variables: total.trailing_zeros() as usize,
            regions,
        }
    }

    /// The regions, in the order they were asked for.
    #[must_use]
    pub fn regions(&self) -> &[Region] {
        &self.regions
    }
}

impl Region {
    /// The Boolean prefix that selects the region in a commitment of `2^total` bits, most significant first.
    ///
    /// # Panics
    ///
    /// The region does not lie inside the commitment.
    #[must_use]
    pub fn selector(&self, total: usize) -> Vec<bool> {
        let prefix = total
            .checked_sub(self.log_bits)
            .expect("the region lies inside the commitment");
        let slot = self.offset_bits >> self.log_bits;
        // A shift past the word reads zero bits.
        let bit = |at: usize| slot.checked_shr(at as u32).unwrap_or(0);
        assert_eq!(bit(prefix), 0, "the region lies inside the commitment");
        (0..prefix).rev().map(|at| bit(at) & 1 == 1).collect()
    }

    /// The packed words of the region, bit `j` of word `w` being bit `64 w + j`.
    ///
    /// # Panics
    ///
    /// The region is narrower than one word, or `all` does not hold it.
    #[must_use]
    pub fn words<'a>(&self, all: &'a [u64]) -> &'a [u64] {
        assert!(self.log_bits >= 6, "a region of whole words");
        &all[self.offset_bits >> 6..][..1 << (self.log_bits - 6)]
    }

    /// The packed words of the region, to write.
    ///
    /// # Panics
    ///
    /// The region is narrower than one word, or `all` does not hold it.
    #[must_use]
    pub fn words_mut<'a>(&self, all: &'a mut [u64]) -> &'a mut [u64] {
        assert!(self.log_bits >= 6, "a region of whole words");
        &mut all[self.offset_bits >> 6..][..1 << (self.log_bits - 6)]
    }

    /// Whether the region is aligned to its size and lies inside a commitment of `2^total` bits.
    #[must_use]
    pub const fn lies_within(&self, total: usize) -> bool {
        total < usize::BITS as usize
            && self.log_bits <= total
            && self.offset_bits & ((1 << self.log_bits) - 1) == 0
            && self.offset_bits >> self.log_bits < 1 << (total - self.log_bits)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use proptest::prelude::*;

    use super::*;

    #[test]
    fn regions_land_aligned_largest_first() {
        let layout = RegionLayout::allocate(&[6, 10, 8, 8]);
        assert_eq!(layout.num_variables, 11);
        let offsets = layout
            .regions()
            .iter()
            .map(|region| region.offset_bits)
            .collect::<Vec<_>>();
        assert_eq!(offsets, vec![1536, 0, 1024, 1280]);
        assert!(RegionLayout::allocate(&[]).regions().is_empty());
        assert_eq!(RegionLayout::allocate(&[12]).num_variables, 12);
    }

    #[test]
    fn a_selector_spells_the_slot() {
        let region = Region {
            log_bits: 8,
            offset_bits: 3 << 8,
        };
        assert_eq!(region.selector(11), vec![false, true, true]);
        let whole = Region {
            log_bits: 8,
            offset_bits: 0,
        };
        assert!(whole.selector(8).is_empty());
    }

    #[test]
    #[should_panic = "the region lies inside the commitment"]
    fn a_region_outside_the_commitment_has_no_selector() {
        let _ = Region {
            log_bits: 8,
            offset_bits: 4 << 8,
        }
        .selector(10);
    }

    #[test]
    fn region_words_are_its_bits() {
        let all = (0..64u64).collect::<Vec<_>>();
        let region = Region {
            log_bits: 9,
            offset_bits: 3 << 9,
        };
        assert_eq!(region.words(&all), &all[24..32]);
        let mut written = all.clone();
        region.words_mut(&mut written).fill(0);
        assert!(written[24..32].iter().all(|&w| w == 0));
        assert_eq!(written[..24], all[..24]);
        assert_eq!(written[32..], all[32..]);
    }

    #[test]
    fn a_region_lies_within_only_aligned_and_inside() {
        let region = |log_bits, offset_bits| Region {
            log_bits,
            offset_bits,
        };
        assert!(region(8, 3 << 8).lies_within(10));
        assert!(region(10, 0).lies_within(10));
        // Past the end, misaligned, or wider than the commitment.
        assert!(!region(8, 4 << 8).lies_within(10));
        assert!(!region(8, 1 << 7).lies_within(10));
        assert!(!region(11, 0).lies_within(10));
        assert!(!region(0, 0).lies_within(usize::BITS as usize));
    }

    proptest! {
        #[test]
        fn allocated_regions_are_disjoint_aligned_and_fit(log_sizes in prop::collection::vec(0usize..12, 0..8)) {
            let layout = RegionLayout::allocate(&log_sizes);
            let total = 1usize << layout.num_variables;
            let regions = layout.regions();
            prop_assert_eq!(regions.len(), log_sizes.len());
            for (region, &log) in regions.iter().zip(&log_sizes) {
                prop_assert_eq!(region.log_bits, log);
                prop_assert_eq!(region.offset_bits % (1 << log), 0);
                prop_assert!(region.lies_within(layout.num_variables));
                prop_assert!(region.offset_bits + (1 << log) <= total);
                // The selector names the region's slot.
                let slot = region.selector(layout.num_variables).iter().fold(0, |acc, &bit| (acc << 1) | usize::from(bit));
                prop_assert_eq!(slot << log, region.offset_bits);
            }
            for (i, a) in regions.iter().enumerate() {
                for b in &regions[i + 1..] {
                    let disjoint = a.offset_bits + (1 << a.log_bits) <= b.offset_bits || b.offset_bits + (1 << b.log_bits) <= a.offset_bits;
                    prop_assert!(disjoint);
                }
            }
            // The commitment is the smallest power of two holding every region.
            let used: usize = log_sizes.iter().map(|&log| 1usize << log).sum();
            prop_assert_eq!(total, used.next_power_of_two());
        }
    }
}
