//! Example provers and the harnesses they share.
//!
//! On Linux the examples allocate through jemalloc, asked to back its arenas with transparent
//! huge pages and to keep freed extents mapped. A prover of this size touches several gigabytes
//! of fresh memory per proof and releases them again, and with 4 KiB pages the page faults and
//! the unmapping cost as much as the arithmetic they surround. Other targets keep the system
//! allocator, which has no huge pages to offer and measures no slower than jemalloc there.

#[cfg(target_os = "linux")]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// jemalloc reads its configuration from this symbol before the first allocation.
///
/// Huge pages cut the fault count of a fresh multi-gigabyte buffer by five hundred times, and
/// retained extents let a buffer freed mid-proof be reused by the next one instead of being
/// unmapped and faulted in again.
#[cfg(target_os = "linux")]
#[allow(non_upper_case_globals)]
#[unsafe(export_name = "_rjem_malloc_conf")]
pub static malloc_conf: &[u8] = b"thp:always,metadata_thp:auto,retain:true\0";

pub mod airs;
pub mod binary;
pub mod dfts;
pub mod parsers;
pub mod proofs;
pub mod types;

#[cfg(test)]
mod tests;
