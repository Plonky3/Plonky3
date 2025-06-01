use alloc::vec::Vec;
use core::mem;
use core::ops::Index;

/// A linear key-value map backed by a `Vec`.
///
/// This map performs **O(n)** lookups and inserts.
/// It is suitable only for **small** sets of keys which
/// must implement `Eq`.
///
/// Internally stores key-value pairs in insertion order.
/// Duplicate key inserts overwrite the previous value.
///
/// # Performance
/// Avoid using this for more than a few keys. All core operations are linear.
#[derive(Debug)]
pub struct LinearMap<K, V>(
    /// The underlying storage for key-value pairs.
    Vec<(K, V)>,
);

impl<K, V> Default for LinearMap<K, V> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<K: Eq, V> LinearMap<K, V> {
    /// Creates a new empty `LinearMap`.
    pub fn new() -> Self {
        Default::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Gets a reference to the value associated with the key, if it exists.
    ///
    /// Returns `Some(&V)` if found, or `None` if not present.
    ///
    /// This is an **O(n)** operation.
    pub fn get(&self, k: &K) -> Option<&V> {
        self.0.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }

    /// Gets a mutable reference to the value associated with the key, if it exists.
    ///
    /// Returns `Some(&mut V)` if found, or `None` otherwise.
    ///
    /// This is an **O(n)** operation.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.0.iter_mut().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the key exists, swaps the old value with the new one and returns the old value.
    /// Otherwise, appends the new pair and returns `None`.
    ///
    /// This is an **O(n)** operation due to the linear search.
    pub fn insert(&mut self, k: K, mut v: V) -> Option<V> {
        if let Some(vv) = self.get_mut(&k) {
            mem::swap(&mut v, vv);
            Some(v)
        } else {
            self.0.push((k, v));
            None
        }
    }

    /// Returns a mutable reference to the value for the given key.
    ///
    /// If the key exists, returns a mutable reference to the value.
    /// Otherwise, inserts a new value created by the provided closure and returns a reference to it.
    ///
    /// This is an **O(n)** operation due to the key search.
    pub fn get_or_insert_with(&mut self, k: K, f: impl FnOnce() -> V) -> &mut V {
        let existing = self.0.iter().position(|(kk, _)| kk == &k);
        if let Some(idx) = existing {
            &mut self.0[idx].1
        } else {
            self.0.push((k, f()));
            let slot = self.0.last_mut().unwrap();
            &mut slot.1
        }
    }

    /// Returns an iterator over the values in the map.
    ///
    /// Values are yielded in insertion order.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|(_, v)| v)
    }

    /// Returns an iterator over the keys in the map.
    ///
    /// Keys are yielded in insertion order.
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.iter().map(|(k, _)| k)
    }

    /// Returns an iterator over all key-value pairs in the map.
    ///
    /// Items are yielded in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter().map(|(k, v)| (k, v))
    }
}

impl<K: Eq, V> FromIterator<(K, V)> for LinearMap<K, V> {
    /// Builds a `LinearMap` from an iterator of key-value pairs.
    ///
    /// Later duplicates overwrite earlier entries.
    ///
    /// This calls `insert` in a loop, so is O(n^2)!!
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut me = Self::default();
        for (k, v) in iter {
            me.insert(k, v);
        }
        me
    }
}

impl<K, V> IntoIterator for LinearMap<K, V> {
    type Item = (K, V);
    type IntoIter = <Vec<(K, V)> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<K: Eq, V> Index<&K> for LinearMap<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<'a, K, V> IntoIterator for &'a LinearMap<K, V> {
    type Item = &'a (K, V);
    type IntoIter = <&'a Vec<(K, V)> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        (&self.0).into_iter()
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut map = LinearMap::new();

        // Insert key=1, value="a" → should return None (new insert)
        assert_eq!(map.insert(1, "a"), None);

        // Get key=1 → should return Some("a")
        assert_eq!(map.get(&1), Some(&"a"));

        // Insert same key with new value → should return old value
        assert_eq!(map.insert(1, "b"), Some("a"));

        // After update, get should return updated value
        assert_eq!(map.get(&1), Some(&"b"));

        // Non-existent key → should return None
        assert_eq!(map.get(&2), None);
    }

    #[test]
    fn test_get_mut() {
        let mut map = LinearMap::new();
        map.insert(42, 100);

        // Mutably get the value for key=42
        if let Some(val) = map.get_mut(&42) {
            *val += 1;
        }

        // Value should now be 101
        assert_eq!(map.get(&42), Some(&101));

        // get_mut on missing key should return None
        assert_eq!(map.get_mut(&999), None);
    }

    #[test]
    fn test_get_or_insert_with() {
        let mut map = LinearMap::new();

        // First call should insert 10 with value computed as 123
        let val = map.get_or_insert_with(10, || 123);
        assert_eq!(*val, 123);

        // Second call should not invoke the closure, just return existing
        let val2 = map.get_or_insert_with(10, || panic!("should not be called"));
        assert_eq!(*val2, 123);

        // Insert another value with a different key
        let val3 = map.get_or_insert_with(20, || 777);
        assert_eq!(*val3, 777);
    }

    #[test]
    fn test_values_iterator() {
        let mut map = LinearMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        // Collect all values into a vector
        let values: Vec<_> = map.values().copied().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_keys_iterator() {
        let mut map = LinearMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        // Collect all values into a vector
        let values: Vec<_> = map.keys().copied().collect();
        assert_eq!(values, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_index() {
        let mut map = LinearMap::new();
        map.insert("a", 1);
        map.insert("b", 2);
        map.insert("c", 3);

        assert_eq!(map[&"a"], 1);
        assert_eq!(map[&"b"], 2);
        assert_eq!(map[&"c"], 3);
    }

    #[test]
    fn test_from_iterator_behavior() {
        // Use .collect() from an iterator of key-value pairs
        let map: LinearMap<_, _> = vec![(1, "a"), (2, "b"), (1, "c")].into_iter().collect();

        // Should insert (1, "a"), (2, "b"), then replace (1, "a") with (1, "c")
        assert_eq!(map.get(&1), Some(&"c"));
        assert_eq!(map.get(&2), Some(&"b"));
    }

    #[test]
    fn test_into_iterator() {
        let mut map = LinearMap::new();
        map.insert("x", 10);
        map.insert("y", 20);

        // Consume the LinearMap into an iterator
        let mut iter = map.into_iter().collect::<Vec<_>>();

        // Since it's just a Vec internally, order is preserved
        iter.sort(); // For comparison purposes
        assert_eq!(iter, vec![("x", 10), ("y", 20)]);
    }

    #[test]
    fn test_empty_map_behavior() {
        let map: LinearMap<i32, &str> = LinearMap::new();

        // Getting any key from an empty map should return None
        assert_eq!(map.get(&0), None);
        assert_eq!(map.get(&999), None);

        // values() iterator should be empty
        assert_eq!(map.values().count(), 0);
    }
}
