use core::slice::{Chunks, ChunksExact, ChunksExactMut, ChunksMut};
use std::iter::FlatMap;

pub trait IntoParallelIterator: IntoIterator {
    fn into_par_iter(self) -> Self::IntoIter;
}
impl<T: IntoIterator> IntoParallelIterator for T {
    fn into_par_iter(self) -> Self::IntoIter {
        self.into_iter()
    }
}

pub trait ParChunks<T> {
    fn par_chunks(&self, chunk_size: usize) -> Chunks<'_, T>;
    fn par_chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T>;
}
impl<T> ParChunks<T> for [T] {
    fn par_chunks(&self, chunk_size: usize) -> Chunks<'_, T> {
        self.chunks(chunk_size)
    }
    fn par_chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T> {
        self.chunks_exact(chunk_size)
    }
}

pub trait ParChunksMut<T> {
    fn par_chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<'_, T>;
    fn par_chunks_exact_mut(&mut self, chunk_size: usize) -> ChunksExactMut<'_, T>;
}
impl<T> ParChunksMut<T> for [T] {
    fn par_chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<'_, T> {
        self.chunks_mut(chunk_size)
    }
    fn par_chunks_exact_mut(&mut self, chunk_size: usize) -> ChunksExactMut<'_, T> {
        self.chunks_exact_mut(chunk_size)
    }
}

pub trait ParIterExt: Iterator {
    fn find_any<P>(self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool + Sync + Send;

    fn flat_map_iter<U, F>(self, map_op: F) -> FlatMap<Self, U, F>
    where
        Self: Sized,
        U: IntoIterator,
        F: Fn(Self::Item) -> U;
}

impl<T: Iterator> ParIterExt for T {
    fn find_any<P>(mut self, predicate: P) -> Option<Self::Item>
    where
        P: Fn(&Self::Item) -> bool + Sync + Send,
    {
        self.find(predicate)
    }

    fn flat_map_iter<U, F>(self, map_op: F) -> FlatMap<Self, U, F>
    where
        Self: Sized,
        U: IntoIterator,
        F: Fn(Self::Item) -> U,
    {
        self.flat_map(map_op)
    }
}
