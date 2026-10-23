use alloc::collections::BTreeMap;
use alloc::sync::Arc;
use core::mem;

use spin::RwLock;

/// Memoized tables keyed by transform shape, shared by clones of a DFT.
///
/// A missing table is computed outside the lock. With the `parallel` feature its computation
/// runs on Rayon, whose workers may pick up another transform sharing this cache while they
/// wait, and that transform must be able to read the cache. Concurrent misses may compute the
/// same table twice; every caller receives the entry published first.
///
/// Publishing a new key into a full cache first evicts every entry, which bounds its memory.
#[derive(Debug)]
pub(crate) struct TwiddleCache<K, V: ?Sized> {
    entries: RwLock<BTreeMap<K, Arc<V>>>,
}

/// Maximum number of cached tables.
///
/// - Size-keyed caches hold at most one table per two-adic size, far below this.
/// - Coset-keyed caches hold one table per shift, so arbitrary shifts would otherwise grow without bound.
/// - 256 still covers every coset of an LDE with blowup up to 2^8.
const CAPACITY: usize = 256;

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
        if let Some(published) = entries.get(&key) {
            return published.clone();
        }
        let _evicted = if entries.len() >= CAPACITY {
            mem::take(&mut *entries)
        } else {
            BTreeMap::new()
        };
        entries.insert(key, value.clone());
        // Unlock before freeing the evicted tables.
        drop(entries);
        value
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

    #[test]
    fn full_cache_evicts_before_publishing() {
        // Fill the cache: keys 0..CAPACITY each hold their own key.
        let cache = TwiddleCache::<usize, usize>::default();
        for key in 0..CAPACITY {
            cache.get_or_compute(key, || Arc::new(key));
        }
        assert_eq!(cache.entries.read().len(), CAPACITY);

        // One more key evicts every entry, then publishes itself.
        //
        //     before: {0, 1, ..., 255}  ->  after: {256}
        let value = cache.get_or_compute(CAPACITY, || Arc::new(CAPACITY));
        assert_eq!(*value, CAPACITY);
        assert_eq!(cache.entries.read().len(), 1);

        // An evicted key is computed again on its next use.
        let recomputed = cache.get_or_compute(0, || Arc::new(42));
        assert_eq!(*recomputed, 42);
    }
}
