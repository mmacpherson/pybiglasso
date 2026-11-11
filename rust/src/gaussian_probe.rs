//! Native-test instrumentation for the Gaussian solver: the binary trace,
//! the score cache's read counters, the scan-mode switch and fault injection.
//! The solver calls it at a few named points. Outside `cargo test` every call
//! is an empty inline function and `Probe` has no fields, so the library
//! contains none of it.

use super::*;

/// How scans use the score cache. Only native tests choose anything but
/// `Enabled`, through PYBIGLASSO_TEST_SCAN_MODE.
#[derive(Clone, Copy, PartialEq)]
pub(super) enum ScanMode {
    /// No cache: every scan reads every column it considers.
    #[cfg_attr(not(test), allow(dead_code))]
    Off,
    /// Certificates skip column reads.
    Enabled,
    /// Certificates are made but every column is still read, and each
    /// certified decision is asserted against the exact score.
    #[cfg_attr(not(test), allow(dead_code))]
    Shadow,
}

/// Where the solver records a trace frame. The numbers are the frame's kind
/// in the trace file, which gaussian_skip_tests.rs reads.
#[derive(Clone, Copy)]
pub(super) enum Point {
    /// The lambda loop is about to start, with the cache built
    CacheReady = 0,
    /// Lambda il's strong set has been chosen
    StrongSet = 1,
    StrongScan = 2,
    StrongScanDone = 3,
    FullScan = 4,
    FullScanDone = 5,
    /// Lambda il's coefficients have been recorded
    Recorded = 6,
    /// The grid is set; the cache is not yet built
    GridReady = 7,
    /// A sweep of the unpenalized prefit
    PrefitSweep = 9,
}

#[cfg(not(test))]
pub(super) use release::*;
#[cfg(not(test))]
mod release {
    use super::*;

    pub(in super::super) struct Probe;

    impl Probe {
        #[inline(always)]
        pub(in super::super) fn new(_mode: ScanMode, _ncol: usize) -> Self {
            Self
        }
        #[inline(always)]
        pub(in super::super) fn shadow(&self) -> bool {
            false
        }
        #[inline(always)]
        pub(in super::super) fn planned(&mut self, _j: usize, _plan: &Plan) {}
        #[inline(always)]
        pub(in super::super) fn exact_read(&mut self) {}
        #[inline(always)]
        pub(in super::super) fn certificate_used(&mut self) {}
        #[inline(always)]
        pub(in super::super) fn invalidate_prefix(&mut self, _hi: &mut f64) {}
        #[inline(always)]
        pub(in super::super) fn corrupt_certificate(&mut self, _anchor: &mut Anchor) {}
    }

    #[inline(always)]
    pub(in super::super) fn scan_mode() -> ScanMode {
        ScanMode::Enabled
    }

    #[inline(always)]
    pub(in super::super) fn trace(
        _path: &Path,
        _st: &GaussianState,
        _cache: Option<&ScoreCache>,
        _point: Point,
        _lambda: f64,
    ) {
    }
}

#[cfg(test)]
pub(super) use instrumented::*;
#[cfg(test)]
mod instrumented {
    use super::*;
    use std::io::Write;

    /// The cache's read counts, written to PYBIGLASSO_TEST_STATS when the
    /// cache is dropped, and the last scan's per-column plan for the trace.
    pub(in super::super) struct Probe {
        shadow: bool,
        /// PYBIGLASSO_TEST_FAULT, if set
        fault: Option<String>,
        fault_applied: bool,
        /// Columns a scan considered, which stock would have read
        baseline_reads: usize,
        /// Columns whose score was actually read
        actual_reads: usize,
        candidate_skips: usize,
        /// Skips that relied on a positive radius, not on an unchanged residual
        positive_skips: usize,
        next_certificates_used: usize,
        pub(in super::super) last_skips: Vec<u8>,
        pub(in super::super) last_radius: Vec<f64>,
    }

    impl Probe {
        pub(in super::super) fn new(mode: ScanMode, ncol: usize) -> Self {
            Self {
                shadow: mode == ScanMode::Shadow,
                fault: std::env::var("PYBIGLASSO_TEST_FAULT").ok(),
                fault_applied: false,
                baseline_reads: 0,
                actual_reads: 0,
                candidate_skips: 0,
                positive_skips: 0,
                next_certificates_used: 0,
                last_skips: vec![0; ncol],
                last_radius: vec![0.; ncol],
            }
        }

        pub(in super::super) fn shadow(&self) -> bool {
            self.shadow
        }

        pub(in super::super) fn planned(&mut self, j: usize, plan: &Plan) {
            self.last_skips[j] = u8::from(plan.skip);
            self.last_radius[j] = plan.radius;
            if !plan.selected {
                return;
            }
            self.baseline_reads += 1;
            if plan.skip {
                self.candidate_skips += 1;
                self.positive_skips += usize::from(plan.radius > 0.);
            }
        }

        pub(in super::super) fn exact_read(&mut self) {
            self.actual_reads += 1;
        }

        pub(in super::super) fn certificate_used(&mut self) {
            self.next_certificates_used += 1;
        }

        /// Fault "invalidate": once a positive-radius skip has happened,
        /// break the prefix so the cache must fall back to exact reads.
        pub(in super::super) fn invalidate_prefix(&mut self, hi: &mut f64) {
            if self.fault.as_deref() == Some("invalidate") && self.positive_skips > 0 && !self.fault_applied {
                *hi = f64::INFINITY;
                self.fault_applied = true;
            }
        }

        /// Faults "epoch", "represented", "index", "lambda" and "cutoff": once
        /// a positive-radius skip has happened, alter one identity of the
        /// first next-lambda certificate, which the strong rule must then
        /// refuse. Called only for inactive columns.
        pub(in super::super) fn corrupt_certificate(&mut self, anchor: &mut Anchor) {
            if self.fault_applied || self.positive_skips == 0 || anchor.next.is_none() {
                return;
            }
            let Some(fault) = self.fault.as_deref() else { return };
            let next = anchor.next.as_mut().unwrap();
            match fault {
                "epoch" => next.epoch ^= 1,
                "represented" => anchor.represented_epoch ^= 1,
                "index" => next.il += 1,
                "lambda" => next.lambda_bits ^= 1,
                "cutoff" => next.cutoff_bits ^= 1,
                _ => {}
            }
            self.fault_applied = fault != "invalidate";
        }
    }

    impl Drop for ScoreCache {
        fn drop(&mut self) {
            let Some(file) = std::env::var_os("PYBIGLASSO_TEST_STATS") else { return };
            let s = &self.probe;
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(file).unwrap();
            writeln!(
                f,
                "{} {} {} {} {} {} {}",
                s.baseline_reads,
                s.actual_reads,
                s.candidate_skips,
                s.positive_skips,
                s.next_certificates_used,
                self.valid as u8,
                s.fault_applied as u8
            )
            .unwrap();
        }
    }

    pub(in super::super) fn scan_mode() -> ScanMode {
        match std::env::var("PYBIGLASSO_TEST_SCAN_MODE").as_deref() {
            Ok("off") => ScanMode::Off,
            Ok("shadow") => ScanMode::Shadow,
            _ => ScanMode::Enabled,
        }
    }

    /// Appends a frame to PYBIGLASSO_TEST_TRACE: the exact trajectory, then
    /// the cache's certificate metadata (zeros without a cache). The layout
    /// is what gaussian_skip_tests.rs's `frames` reads.
    pub(in super::super) fn trace(path: &Path, st: &GaussianState, cache: Option<&ScoreCache>, point: Point, lam: f64) {
        let Some(name) = std::env::var_os("PYBIGLASSO_TEST_TRACE") else {
            return;
        };
        let mut f = std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(name).unwrap());
        let p = st.beta.len();
        let h = [point as u64, st.il as u64, st.residual.len() as u64, p as u64, path.center.len() as u64, st.version];
        f.write_all(bytemuck::cast_slice(&h)).unwrap();
        let (hi, lo, rn) = cache.map_or((0., 0., 0.), |s| (s.hi, s.lo, s.rnorm));
        f.write_all(bytemuck::cast_slice(&[lam, st.residual_sum, hi, lo, rn])).unwrap();
        for a in [&st.residual[..], &st.scores[..], &st.beta[..], &path.center[..], &path.scale[..]] {
            f.write_all(bytemuck::cast_slice(a)).unwrap();
        }
        f.write_all(bytemuck::cast_slice(&path.col_map.iter().map(|&j| j as u64).collect::<Vec<_>>())).unwrap();
        for a in [&st.active, &st.strong] {
            f.write_all(&a.iter().map(|&v| u8::from(v)).collect::<Vec<_>>()).unwrap();
        }
        let Some(s) = cache else {
            // last skips, last radii, anchor scores, five anchor u64 fields,
            // next decisions; then the cache's version, validity, previous
            // residual and per-column prefix, norm, next epoch and metadata
            f.write_all(&vec![0u8; p]).unwrap();
            f.write_all(bytemuck::cast_slice(&vec![0f64; 2 * p])).unwrap();
            for _ in 0..5 {
                f.write_all(bytemuck::cast_slice(&vec![0u64; p])).unwrap();
            }
            f.write_all(&vec![0u8; p]).unwrap();
            f.write_all(&vec![0; 16 + 8 * st.residual.len() + 80 * p]).unwrap();
            return;
        };
        f.write_all(&s.probe.last_skips).unwrap();
        f.write_all(bytemuck::cast_slice(&s.probe.last_radius)).unwrap();
        let values: Vec<f64> = s.anchors.iter().map(|a| a.score).collect();
        f.write_all(bytemuck::cast_slice(&values)).unwrap();
        for values in [
            s.anchors.iter().map(|a| a.epoch).collect::<Vec<_>>(),
            s.anchors.iter().map(|a| a.represented_epoch).collect(),
            s.anchors.iter().map(|a| a.next.map_or(u64::MAX, |v| v.il as u64)).collect(),
            s.anchors.iter().map(|a| a.next.map_or(0, |v| v.lambda_bits)).collect(),
            s.anchors.iter().map(|a| a.next.map_or(0, |v| v.cutoff_bits)).collect(),
        ] {
            f.write_all(bytemuck::cast_slice(&values)).unwrap();
        }
        f.write_all(
            &s.anchors.iter().map(|a| a.next.map_or(0, |v| if v.decision { 2 } else { 1 })).collect::<Vec<_>>(),
        )
        .unwrap();
        f.write_all(bytemuck::cast_slice(&[s.version, u64::from(s.valid)])).unwrap();
        f.write_all(bytemuck::cast_slice(&s.previous)).unwrap();
        for values in
            [s.anchors.iter().map(|a| a.prefix_lo).collect::<Vec<_>>(), s.anchors.iter().map(|a| a.rnorm).collect()]
        {
            f.write_all(bytemuck::cast_slice(&values)).unwrap();
        }
        let epochs: Vec<u64> = s.anchors.iter().map(|a| a.next.map_or(0, |v| v.epoch)).collect();
        f.write_all(bytemuck::cast_slice(&epochs)).unwrap();
        for m in &s.meta {
            let bits = m.map_or([0; 7], |m| m.test_metadata());
            f.write_all(bytemuck::cast_slice(&bits)).unwrap();
        }
    }
}
