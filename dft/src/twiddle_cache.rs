use alloc::collections::BTreeMap;
use alloc::collections::btree_map::Entry;
use alloc::sync::Arc;

use spin::RwLock;

/// Memoized tables keyed by transform shape, shared by clones of a DFT.
///
/// A missing table is computed outside the lock. With the `parallel` feature its computation
/// runs on Rayon, whose workers may pick up another transform sharing this cache while they
/// wait, and that transform must be able to read the cache. Concurrent misses may compute the
/// same table twice; every caller receives the entry published first.
#[derive(Debug)]
pub(crate) struct TwiddleCache<K, V: ?Sized> {
    entries: RwLock<BTreeMap<K, Arc<V>>>,
}

impl<K, V: ?Sized> Default for TwiddleCache<K, V> {
    fn default() -> Self {
        Self {
            entries: RwLock::new(BTreeMap::new()),
        }
    }
}

impl<K: Ord, V: ?Sized> TwiddleCache<K, V> {
    /// Returns the table for `key`, computing and publishing it on a miss.
    pub(crate) fn get_or_compute(&self, key: K, compute: impl FnOnce() -> Arc<V>) -> Arc<V> {
        if let Some(value) = self.entries.read().get(&key) {
            return value.clone();
        }
        let value = compute();
        // Declare the guard after `value` so an unused table is dropped after unlocking.
        let mut entries = self.entries.write();
        match entries.entry(key) {
            Entry::Occupied(entry) => entry.get().clone(),
            Entry::Vacant(entry) => entry.insert(value).clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::sync::Weak;
    use core::sync::atomic::{AtomicBool, Ordering};

    use super::*;

    #[test]
    fn hit_reuses_table_without_computing() {
        let cache = TwiddleCache::default();
        let expected: Arc<[u64]> = alloc::vec![1, 2, 3].into();
        let first = cache.get_or_compute(4, || expected.clone());
        let second = cache.get_or_compute(4, || panic!("a cached table must not be recomputed"));

        assert!(Arc::ptr_eq(&first, &expected));
        assert!(Arc::ptr_eq(&second, &expected));
    }

    #[test]
    fn miss_computes_without_holding_the_lock() {
        let cache = TwiddleCache::<usize, u64>::default();
        let value = cache.get_or_compute(4, || {
            assert!(
                cache.entries.try_write().is_some(),
                "table construction must not hold a cache lock"
            );
            Arc::new(23)
        });

        assert_eq!(*value, 23);
    }

    #[test]
    fn miss_reuses_entry_published_during_computation() {
        // Model a transform that runs on the same cache while this table is being built.
        let cache = TwiddleCache::default();
        let inserted = Arc::new(17_u64);
        let result = cache.get_or_compute(4, || {
            // Fail fast here: a held lock would make the nested call below spin forever.
            assert!(
                cache.entries.try_write().is_some(),
                "table construction must not hold a cache lock"
            );
            let nested = cache.get_or_compute(4, || inserted.clone());
            assert!(Arc::ptr_eq(&nested, &inserted));
            Arc::new(23_u64)
        });

        assert!(Arc::ptr_eq(&result, &inserted));
        assert!(Arc::ptr_eq(
            cache.entries.read().get(&4).unwrap(),
            &inserted
        ));
    }

    #[test]
    fn miss_drops_unused_table_after_unlocking() {
        type Cache = TwiddleCache<usize, DropProbe>;

        #[derive(Default)]
        struct DropProbe {
            cache: Weak<Cache>,
            dropped: Arc<AtomicBool>,
        }

        impl Drop for DropProbe {
            fn drop(&mut self) {
                if let Some(cache) = self.cache.upgrade() {
                    assert!(
                        cache.entries.try_write().is_some(),
                        "discarded table must be dropped after unlocking the cache"
                    );
                }
                self.dropped.store(true, Ordering::Relaxed);
            }
        }

        let cache = Arc::new(Cache::default());
        let inserted = Arc::new(DropProbe::default());
        let dropped = Arc::new(AtomicBool::new(false));
        let result = cache.get_or_compute(4, || {
            assert!(
                cache.entries.try_write().is_some(),
                "table construction must not hold a cache lock"
            );
            cache.get_or_compute(4, || inserted.clone());
            Arc::new(DropProbe {
                cache: Arc::downgrade(&cache),
                dropped: dropped.clone(),
            })
        });

        assert!(Arc::ptr_eq(&result, &inserted));
        assert!(dropped.load(Ordering::Relaxed));
    }
}
