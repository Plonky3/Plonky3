//! The cache budgets a transform cuts its passes to.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__cpuid, __cpuid_count};
#[cfg(target_arch = "x86_64")]
use core::sync::atomic::{AtomicU64, Ordering};

/// Bytes of cache one worker may fill with a single unit of work.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct CacheBudget {
    /// Bytes a unit may fill in one hardware thread's share of its core's private cache.
    ///
    /// It is half that share, since the unit's gather and scatter stream through the other half.
    pub(crate) private: usize,
    /// Bytes a unit may spill to in one hardware thread's share of the last-level cache.
    ///
    /// It is at least the private budget.
    pub(crate) shared: usize,
    /// Bytes of the whole last-level cache this thread sits in, or zero where unknown.
    pub(crate) last_level: usize,
}

impl CacheBudget {
    /// The budget where the cache sizes are unknown.
    ///
    /// - The private budget is half of one thread's share of a 1 MiB L2 that two SMT threads share.
    /// - The shared budget is 2 MiB, a common per-core slice of L3.
    /// - Apple silicon's cluster L2 holds more than both per core.
    /// - A budget below the real cache only forgoes some fused stages, which is the safe direction.
    /// - The last-level size is unknown, so no matrix is taken to stay in cache.
    pub(crate) const DEFAULT: Self = Self {
        private: 256 * 1024,
        shared: 2 * 1024 * 1024,
        last_level: 0,
    };

    /// The fewest bytes a detected L2 share is taken at.
    ///
    /// A smaller figure is a virtual machine's placeholder, not a cache.
    #[cfg(target_arch = "x86_64")]
    const MIN_PRIVATE: usize = 128 * 1024;

    /// The most private budgets a detected shared budget may span.
    #[cfg(target_arch = "x86_64")]
    const MAX_SPILL: usize = 8;

    /// The budget of the running machine, read once and then cached.
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn host() -> Self {
        // Zero means not read yet, since every budget has a non-zero private part.
        //
        // The three sizes sit in 21-bit fields of KiB, private lowest, which holds up to 2 GiB each.
        static CACHE: AtomicU64 = AtomicU64::new(0);
        const FIELD: u32 = 21;
        const MASK: u64 = (1 << FIELD) - 1;

        let packed = CACHE.load(Ordering::Relaxed);
        if packed != 0 {
            let field = |i: u32| (((packed >> (i * FIELD)) & MASK) as usize) << 10;
            return Self {
                private: field(0),
                shared: field(1),
                last_level: field(2),
            };
        }

        let budget = Self::from_cpuid().unwrap_or(Self::DEFAULT);
        let field = |bytes: usize| ((bytes >> 10) as u64).min(MASK);
        let packed = field(budget.private)
            | (field(budget.shared) << FIELD)
            | (field(budget.last_level) << (2 * FIELD));

        // Racing threads compute the same answer, so a plain store is enough.
        CACHE.store(packed, Ordering::Relaxed);

        // The budget is returned as the cache will give it back, so every call agrees.
        Self {
            private: (field(budget.private) as usize) << 10,
            shared: (field(budget.shared) as usize) << 10,
            last_level: (field(budget.last_level) as usize) << 10,
        }
    }

    /// The budget of the running machine.
    ///
    /// Other targets expose no cache sizes to user code.
    #[cfg(not(target_arch = "x86_64"))]
    pub(crate) const fn host() -> Self {
        Self::DEFAULT
    }

    /// The per-thread shares of L2 and L3, from the deterministic cache parameters leaf.
    ///
    /// Intel lists them at leaf 4, AMD and Hygon at leaf `0x8000_001D`, both in one layout.
    #[cfg(target_arch = "x86_64")]
    fn from_cpuid() -> Option<Self> {
        // "Auth"enticAMD and "Hygo"nGenuine, as the little-endian EBX of leaf 0.
        const AMD: u32 = 0x6874_7541;
        const HYGON: u32 = 0x6f67_7948;

        let vendor = __cpuid(0).ebx;
        let leaf = if vendor == AMD || vendor == HYGON {
            0x8000_001d
        } else {
            4
        };

        // The leaf must exist in its range, basic or extended.
        if __cpuid(leaf & 0x8000_0000).eax < leaf {
            return None;
        }

        let (mut l2, mut l3) = (None, None);
        for subleaf in 0..16 {
            let r = __cpuid_count(leaf, subleaf);

            // Type 0 ends the list, and type 2 is an instruction cache.
            match r.eax & 0x1f {
                0 => break,
                2 => continue,
                _ => {}
            }

            // Bytes = ways * partitions * line size * sets, each field stored minus one.
            let ways = (r.ebx >> 22) as usize + 1;
            let partitions = ((r.ebx >> 12) & 0x3ff) as usize + 1;
            let line = (r.ebx & 0xfff) as usize + 1;
            let sets = r.ecx as usize + 1;
            let bytes = ways * partitions * line * sets;

            // Logical processors sharing the cache, stored minus one.
            let sharing = ((r.eax >> 14) & 0xfff) as usize + 1;

            match (r.eax >> 5) & 0x7 {
                2 => l2 = Some(bytes / sharing),
                3 => l3 = Some((bytes, bytes / sharing)),
                _ => {}
            }
        }

        // Half the L2 share holds the unit, the other half the traffic streaming through it.
        let private = l2.filter(|&bytes| bytes >= Self::MIN_PRIVATE)? / 2;

        // A thread's L3 share is read on one cluster, and the pool may span a smaller one.
        //
        // So the shared budget is capped at a few private ones, which every cluster holds.
        let shared = l3
            .map_or(private, |(_, share)| share)
            .clamp(private, Self::MAX_SPILL * private);
        let last_level = l3.map_or(0, |(bytes, _)| bytes);
        Some(Self {
            private,
            shared,
            last_level,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CacheBudget;

    #[test]
    fn the_host_budget_is_stable_and_ordered() {
        // Invariant: the cached budget is what a fresh read gives.
        //
        // The second call reads the cache, the first one fills it.
        let first = CacheBudget::host();
        assert_eq!(CacheBudget::host(), first);

        // Every schedule assumes a shared budget no smaller than the private one.
        assert!(first.private > 0);
        assert!(first.shared >= first.private);
    }
}
