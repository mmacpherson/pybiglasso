//! Rust backend for pybiglasso's binomial solver: the path loop of
//! `fit_path` and every kernel it calls, in one call from Python.
//!
//! It follows the numba backend step for step, so that it matches R biglasso
//! the same way: same coordinate order, same screening, same stopping rules.
//! Two things are under our control here and are part of the contract:
//!
//! - Summation order. Every column reduction runs in a fixed order that does
//!   not depend on the thread count: sequential where the numba backend is
//!   strict, and LANES independent accumulators, combined in a fixed tree,
//!   where it uses fastmath. Threads split work across columns, and a tall
//!   column's rows into CHUNK-row chunks, whose sums are added in order, so
//!   the chunks and their order are the same at every thread count.
//! - X is read in place, a column at a time, through whatever strides numpy
//!   gives; it is never copied or converted.

#![deny(unsafe_code)]
// index loops, as in the numba backend, so the two read side by side
#![allow(clippy::needless_range_loop)]

// Each binding uses different views; keep the native strided/gathered kernels
// available for their tests even when the browser only uses contiguous f64.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
mod solver;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "wasm")]
mod wasm;

#[cfg(any(feature = "wasm", test))]
mod model_scale;
