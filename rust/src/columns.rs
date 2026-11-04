//! Element types and column access: one column of X at a time, read in
//! place through whatever layout or storage it has, never copied.

use super::*;

pub(super) const LANES: usize = 8;
// Each byte's lanes occupy one aligned cache line. Only exact +0.0/+1.0
// values are substituted; all multiplication, addition and reduction stay
// in the kernels, including their signed-zero behavior.
pub(super) static BOOL_BLOCKS: [Block; 256] = {
    let mut table = [Block([0.0; LANES]); 256];
    let mut byte = 0;
    while byte < 256 {
        let mut lane = 0;
        while lane < LANES {
            table[byte].0[lane] = ((byte >> lane) & 1) as f64;
            lane += 1;
        }
        byte += 1;
    }
    table
};

// Element types and column access

pub(crate) trait Elem: Copy + Send + Sync {
    fn f(self) -> f64;
}
macro_rules! elem {
    ($($t:ty),*) => { $(impl Elem for $t { #[inline(always)] fn f(self) -> f64 { self as f64 } })* };
}
elem!(f64, f32, i8, i16, i32, i64, u8, u16, u32, u64);
impl Elem for bool {
    #[inline(always)]
    fn f(self) -> f64 {
        if self { 1.0 } else { 0.0 }
    }
}

/// One column of X, as the rows being fit (all of them, or row_indices).
///
/// The two reads of X itself, `Contig::at` and `Strided::at`, check their
/// bounds unless CHECKED is false, the crate's only unsafe code: checking
/// every gathered or strided row costs 11-15% of a fit. Every caller indexes
/// below n(), and Gather's row_indices are range-checked against X's rows
/// before the fit, so the check guards against a bug here, not bad input; it
/// is on by default (the boundscheck setting), and changes no result.
pub(crate) trait Col: Sync {
    fn n(&self) -> usize;
    /// Row i of the fit; i < n().
    fn at(&self, i: usize) -> f64;
    /// Rows i0 .. i0 + LANES; i0 + LANES <= n().
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        std::array::from_fn(|k| self.at(i0 + k))
    }
    /// block(i0), read in one ascending pass over a column. `hint` is the
    /// pass's own position (one per kernel call, never shared), so a chunked
    /// column can step to the next chunk instead of searching for it.
    #[inline(always)]
    fn block_seq(&self, i0: usize, _hint: &mut usize) -> [f64; LANES] {
        self.block(i0)
    }
    /// at(i), read in one ascending pass, as for block_seq.
    #[inline(always)]
    fn at_seq(&self, i: usize, _hint: &mut usize) -> f64 {
        self.at(i)
    }
    /// Complete byte-aligned bool blocks, excluding any scalar tail.
    #[inline(always)]
    fn packed_bytes(&self) -> Option<&[u8]> {
        None
    }
    /// A run starting at i0, addressed from zero, and its end in this column.
    /// Except for a final partial block, the run contains whole LANES blocks.
    /// Kernels keep their accumulators across runs: storage never partitions sums.
    #[inline(always)]
    fn run(&self, i0: usize, end: usize) -> (impl Col, usize)
    where
        Self: Sized,
    {
        (Span { col: self, i0, n: end - i0 }, end)
    }
}

/// A column contiguous in memory (X in Fortran order).
pub(crate) struct Contig<'a, T, const CHECKED: bool>(pub(super) &'a [T]);
impl<T: Elem, const CHECKED: bool> Col for Contig<'_, T, CHECKED> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    #[allow(unsafe_code)]
    fn at(&self, i: usize) -> f64 {
        if CHECKED {
            return self.0[i].f();
        }
        debug_assert!(i < self.0.len());
        // SAFETY: i < n(); see Col
        unsafe { self.0.get_unchecked(i) }.f()
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        let rows: &[T; LANES] = self.0[i0..i0 + LANES].try_into().expect("LANES rows");
        // For int8 on NEON, map gives one 64-bit load, not eight byte loads.
        // On x86, indexing exposes packed sign-extension/conversion instead.
        // Same conversions and lane order; measurements: benchmarks/int8.md.
        #[cfg(target_arch = "aarch64")]
        {
            rows.map(Elem::f)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            std::array::from_fn(|k| rows[k].f())
        }
    }
}

/// A column with an arbitrary element stride (X in C order, or a view).
pub(crate) struct Strided<'a, T, const CHECKED: bool>(pub(super) ArrayView1<'a, T>);
impl<T: Elem, const CHECKED: bool> Col for Strided<'_, T, CHECKED> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.0.len()
    }
    #[inline(always)]
    #[allow(unsafe_code)]
    fn at(&self, i: usize) -> f64 {
        if CHECKED {
            return self.0[i].f();
        }
        debug_assert!(i < self.0.len());
        // SAFETY: i < n(); see Col
        unsafe { self.0.uget(i) }.f()
    }
}

/// The rows of a column picked out by row_indices, in their given order.
pub(crate) struct Gather<'a, C> {
    pub(super) col: C,
    pub(super) idx: &'a [usize],
}
impl<C: Col> Col for Gather<'_, C> {
    #[inline(always)]
    fn n(&self) -> usize {
        self.idx.len()
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self.col.at(self.idx[i])
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        let rows: &[usize; LANES] = self.idx[i0..i0 + LANES].try_into().expect("LANES rows");
        rows.map(|i| self.col.at(i))
    }
}

/// Rows i0 .. i0 + n of a column.
pub(super) struct Span<'a, C> {
    pub(super) col: &'a C,
    pub(super) i0: usize,
    pub(super) n: usize,
}
impl<C: Col> Col for Span<'_, C> {
    #[inline(always)]
    fn packed_bytes(&self) -> Option<&[u8]> {
        if !self.i0.is_multiple_of(LANES) {
            return None;
        }
        let bytes = self.col.packed_bytes()?;
        Some(&bytes[self.i0 / LANES..self.i0 / LANES + self.n / LANES])
    }
    #[inline(always)]
    fn n(&self) -> usize {
        self.n
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        self.col.at(self.i0 + i)
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        self.col.block(self.i0 + i0)
    }
    #[inline(always)]
    fn block_seq(&self, i0: usize, hint: &mut usize) -> [f64; LANES] {
        self.col.block_seq(self.i0 + i0, hint)
    }
    #[inline(always)]
    fn at_seq(&self, i: usize, hint: &mut usize) -> f64 {
        self.col.at_seq(self.i0 + i, hint)
    }
    #[inline(always)]
    fn run(&self, i0: usize, end: usize) -> (impl Col, usize) {
        let (run, stop) = self.col.run(self.i0 + i0, self.i0 + end);
        (run, stop - self.i0)
    }
}

pub(crate) trait Design: Sync {
    type C<'b>: Col
    where
        Self: 'b;
    fn col(&self, j: usize) -> Self::C<'_>;
    fn ncol(&self) -> usize;
    fn nrow(&self) -> usize;
}

/// Heterogeneous mapped columns: dispatch storage, never the reduction order.
#[derive(Clone, Copy)]
pub(crate) enum MappedCol<'a> {
    F64(&'a [f64]),
    F32(&'a [f32]),
    I8(&'a [i8]),
    U8(&'a [u8]),
    I16(&'a [i16]),
    U64(&'a [u64]),
    I64(&'a [i64]),
    U32(&'a [u32]),
    I32(&'a [i32]),
    U16(&'a [u16]),
    Bits { bytes: &'a [u8], offset: usize, len: usize },
}

impl MappedCol<'_> {
    /// Borrow exactly this run. Checking its extent once also exposes the
    /// inner loop's load bounds to the optimizer, rather than carrying an
    /// offset into a larger chunk through every block load.
    #[inline(always)]
    pub(super) fn slice(&self, start: usize, len: usize) -> Self {
        match self {
            Self::F64(v) => Self::F64(&v[start..start + len]),
            Self::F32(v) => Self::F32(&v[start..start + len]),
            Self::I8(v) => Self::I8(&v[start..start + len]),
            Self::U8(v) => Self::U8(&v[start..start + len]),
            Self::I16(v) => Self::I16(&v[start..start + len]),
            Self::U64(v) => Self::U64(&v[start..start + len]),
            Self::I64(v) => Self::I64(&v[start..start + len]),
            Self::U32(v) => Self::U32(&v[start..start + len]),
            Self::I32(v) => Self::I32(&v[start..start + len]),
            Self::U16(v) => Self::U16(&v[start..start + len]),
            Self::Bits { bytes, offset, len: total } => {
                assert!(start <= *total && len <= *total - start);
                if len == 0 {
                    return Self::Bits { bytes: &[], offset: 0, len: 0 };
                }
                let bit = offset.checked_add(start).expect("bit offset");
                let offset = bit % 8;
                let count = offset.checked_add(len).expect("bit length").div_ceil(8);
                Self::Bits { bytes: &bytes[bit / 8..bit / 8 + count], offset, len }
            }
        }
    }
}

impl Col for MappedCol<'_> {
    #[inline(always)]
    fn packed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Bits { bytes, offset, len } if offset.is_multiple_of(8) => {
                Some(&bytes[offset / 8..offset / 8 + len / LANES])
            }
            _ => None,
        }
    }
    fn n(&self) -> usize {
        match self {
            Self::F64(v) => v.len(),
            Self::F32(v) => v.len(),
            Self::I8(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I16(v) => v.len(),
            Self::U64(v) => v.len(),
            Self::I64(v) => v.len(),
            Self::U32(v) => v.len(),
            Self::I32(v) => v.len(),
            Self::U16(v) => v.len(),
            Self::Bits { len, .. } => *len,
        }
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        match self {
            Self::F64(v) => v[i],
            Self::F32(v) => v[i].f(),
            Self::I8(v) => v[i].f(),
            Self::U8(v) => v[i].f(),
            Self::I16(v) => v[i].f(),
            Self::U64(v) => v[i].f(),
            Self::I64(v) => v[i].f(),
            Self::U32(v) => v[i].f(),
            Self::I32(v) => v[i].f(),
            Self::U16(v) => v[i].f(),
            Self::Bits { bytes, offset, len } => {
                assert!(i < *len);
                let bit = offset + i;
                // at_seq delegates here too: centering and Gaussian residual
                // scans must not retain the per-bit conversion bottleneck.
                BOOL_BLOCKS[usize::from(bytes[bit / 8])].0[bit % 8]
            }
        }
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        match self {
            Self::F64(v) => Contig::<_, true>(v).block(i0),
            Self::F32(v) => Contig::<_, true>(v).block(i0),
            Self::I8(v) => Contig::<_, true>(v).block(i0),
            Self::U8(v) => Contig::<_, true>(v).block(i0),
            Self::I16(v) => Contig::<_, true>(v).block(i0),
            Self::U64(v) => Contig::<_, true>(v).block(i0),
            Self::I64(v) => Contig::<_, true>(v).block(i0),
            Self::U32(v) => Contig::<_, true>(v).block(i0),
            Self::I32(v) => Contig::<_, true>(v).block(i0),
            Self::U16(v) => Contig::<_, true>(v).block(i0),
            Self::Bits { bytes, offset, len } => {
                assert!(*len >= LANES && i0 <= *len - LANES);
                // The block's LANES bits span at most two bytes: read them
                // once rather than locating each bit.
                let bit = offset + i0;
                if bit.is_multiple_of(8) {
                    return BOOL_BLOCKS[usize::from(bytes[bit / 8])].0;
                }
                let lo = u16::from(bytes[bit / 8]);
                let hi = if bit.is_multiple_of(8) { 0 } else { u16::from(bytes[bit / 8 + 1]) };
                let bits = (lo | hi << 8) >> (bit % 8);
                std::array::from_fn(|k| ((bits >> k) & 1).f())
            }
        }
    }
}

// Chunk boundaries affect addressing only. Logical kernel blocks and reduction
// accumulators are unchanged, including when a block crosses several chunks.
#[derive(Clone)]
pub(crate) struct ChunkedColumn<'a> {
    pub(crate) chunks: Vec<MappedCol<'a>>,
    pub(crate) ends: Vec<usize>,
}
impl<'a> ChunkedColumn<'a> {
    pub(crate) fn new(chunks: Vec<MappedCol<'a>>, nrow: usize) -> Result<Self, String> {
        let mut end = 0usize;
        let mut ends = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            end = end.checked_add(chunk.n()).ok_or("Chunk lengths overflow.")?;
            ends.push(end);
        }
        if end != nrow || chunks.is_empty() {
            return Err("Chunk lengths must match X rows.".into());
        }
        Ok(Self { chunks, ends })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ChunkedCol<'a> {
    chunks: &'a [MappedCol<'a>],
    ends: &'a [usize],
}

/// An immutable column for an inner loop. Its chunk and type cannot change
/// during that loop, allowing the compiler to hoist the storage dispatch.
enum ChunkRun<'a> {
    Whole(MappedCol<'a>),
    Crossing(Span<'a, ChunkedCol<'a>>),
}
impl Col for ChunkRun<'_> {
    #[inline(always)]
    fn packed_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Whole(c) => c.packed_bytes(),
            Self::Crossing(_) => None,
        }
    }
    #[inline(always)]
    fn n(&self) -> usize {
        match self {
            Self::Whole(c) => c.n(),
            Self::Crossing(c) => c.n(),
        }
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        match self {
            Self::Whole(c) => c.at(i),
            Self::Crossing(c) => c.at(i),
        }
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        match self {
            Self::Whole(c) => c.block(i0),
            Self::Crossing(c) => c.block(i0),
        }
    }
    #[inline(always)]
    fn at_seq(&self, i: usize, hint: &mut usize) -> f64 {
        match self {
            Self::Whole(c) => c.at_seq(i, hint),
            Self::Crossing(c) => c.at_seq(i, hint),
        }
    }
}
impl ChunkedCol<'_> {
    #[inline(always)]
    fn locate(&self, i: usize) -> (usize, usize) {
        assert!(i < self.n());
        let k = if self.chunks.len() == 1 { 0 } else { self.ends.partition_point(|&end| end <= i) };
        (k, i - if k == 0 { 0 } else { self.ends[k - 1] })
    }
}
// A one-chunk column (the common case) skips locating rows entirely; the
// chunk's own indexing checks bounds, since its length is the column's.
impl Col for ChunkedCol<'_> {
    #[inline(always)]
    fn run(&self, i0: usize, end: usize) -> (impl Col, usize) {
        assert!(i0 < end && end <= self.n());
        let (k, local) = self.locate(i0);
        let whole = (self.ends[k].min(end) - i0) / LANES * LANES;
        if whole != 0 {
            (ChunkRun::Whole(self.chunks[k].slice(local, whole)), i0 + whole)
        } else {
            // Never realign a logical block to a storage boundary. Read the
            // crossing block (or final scalar tail) through the existing path.
            let n = LANES.min(end - i0);
            (ChunkRun::Crossing(Span { col: self, i0, n }), i0 + n)
        }
    }
    #[inline(always)]
    fn n(&self) -> usize {
        *self.ends.last().expect("nonempty chunks")
    }
    #[inline(always)]
    fn at(&self, i: usize) -> f64 {
        if let [only] = self.chunks {
            return only.at(i);
        }
        let (k, local) = self.locate(i);
        self.chunks[k].at(local)
    }
    #[inline(always)]
    fn block(&self, i0: usize) -> [f64; LANES] {
        if let [only] = self.chunks {
            return only.block(i0);
        }
        let (k, local) = self.locate(i0);
        if self.chunks[k].n() - local >= LANES {
            self.chunks[k].block(local)
        } else {
            std::array::from_fn(|lane| self.at(i0 + lane))
        }
    }
    #[inline(always)]
    fn block_seq(&self, i0: usize, hint: &mut usize) -> [f64; LANES] {
        if let [only] = self.chunks {
            return only.block(i0);
        }
        // Step past chunks ending at or before i0 (empty chunks included);
        // anything but an ascending pass falls back to searching.
        let mut k = *hint;
        while k < self.ends.len() && self.ends[k] <= i0 {
            k += 1;
        }
        let start = if k == 0 { 0 } else { self.ends[k - 1] };
        if k < self.ends.len() && start <= i0 && self.ends[k] - i0 >= LANES {
            *hint = k;
            return self.chunks[k].block(i0 - start);
        }
        self.block(i0)
    }
    #[inline(always)]
    fn at_seq(&self, i: usize, hint: &mut usize) -> f64 {
        if let [only] = self.chunks {
            return only.at(i);
        }
        let mut k = *hint;
        while k < self.ends.len() && self.ends[k] <= i {
            k += 1;
        }
        let start = if k == 0 { 0 } else { self.ends[k - 1] };
        if k < self.ends.len() && start <= i {
            *hint = k;
            return self.chunks[k].at(i - start);
        }
        self.at(i)
    }
}

pub(crate) struct MappedX<'a> {
    pub(crate) columns: Vec<ChunkedColumn<'a>>,
}
impl Design for MappedX<'_> {
    type C<'b>
        = ChunkedCol<'b>
    where
        Self: 'b;
    fn col(&self, j: usize) -> ChunkedCol<'_> {
        ChunkedCol { chunks: &self.columns[j].chunks, ends: &self.columns[j].ends }
    }
    fn ncol(&self) -> usize {
        self.columns.len()
    }
    fn nrow(&self) -> usize {
        *self.columns[0].ends.last().expect("nonempty chunks")
    }
}

pub(crate) struct ContigX<'a, T, const CHECKED: bool> {
    pub(crate) x: ArrayView2<'a, T>,
}
impl<T: Elem, const CHECKED: bool> Design for ContigX<'_, T, CHECKED> {
    type C<'b>
        = Contig<'b, T, CHECKED>
    where
        Self: 'b;
    fn col(&self, j: usize) -> Contig<'_, T, CHECKED> {
        Contig(self.x.column(j).to_slice().expect("Fortran-contiguous column"))
    }
    fn ncol(&self) -> usize {
        self.x.ncols()
    }
    fn nrow(&self) -> usize {
        self.x.nrows()
    }
}

pub(crate) struct StridedX<'a, T, const CHECKED: bool> {
    pub(crate) x: ArrayView2<'a, T>,
}
impl<T: Elem, const CHECKED: bool> Design for StridedX<'_, T, CHECKED> {
    type C<'b>
        = Strided<'b, T, CHECKED>
    where
        Self: 'b;
    fn col(&self, j: usize) -> Strided<'_, T, CHECKED> {
        Strided(self.x.column(j))
    }
    fn ncol(&self) -> usize {
        self.x.ncols()
    }
    fn nrow(&self) -> usize {
        self.x.nrows()
    }
}

pub(crate) struct GatherX<'a, D> {
    pub(crate) x: D,
    pub(crate) idx: &'a [usize],
}
impl<D: Design> Design for GatherX<'_, D> {
    type C<'b>
        = Gather<'b, D::C<'b>>
    where
        Self: 'b;
    fn col(&self, j: usize) -> Gather<'_, D::C<'_>> {
        Gather { col: self.x.col(j), idx: self.idx }
    }
    fn ncol(&self) -> usize {
        self.x.ncol()
    }
    fn nrow(&self) -> usize {
        self.idx.len()
    }
}
