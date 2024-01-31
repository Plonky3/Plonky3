use alloc::vec::Vec;
use core::mem;

pub struct LinearMap<K, V>(Vec<(K, V)>);

impl<K, V> Default for LinearMap<K, V> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<K: Eq, V> LinearMap<K, V> {
    pub fn new() -> Self {
        Default::default()
    }
    pub fn get(&self, k: &K) -> Option<&V> {
        self.0.iter().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.0.iter_mut().find(|(kk, _)| kk == k).map(|(_, v)| v)
    }
    pub fn insert(&mut self, k: K, mut v: V) -> Option<V> {
        if let Some(vv) = self.get_mut(&k) {
            mem::swap(&mut v, vv);
            Some(v)
        } else {
            self.0.push((k, v));
            None
        }
    }
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|(_, v)| v)
    }
}

impl<K: Eq, V> FromIterator<(K, V)> for LinearMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut me = LinearMap::default();
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
