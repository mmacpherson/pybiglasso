//! Native-only controls keep the production entry point unconditional.
use super::*;
use ndarray::{Array2, ShapeBuilder};
use std::{
    fs,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

fn append_frame(out: &mut Vec<u8>, name: &str, dtype: &str, shape: &[usize], bytes: &[u8]) {
    // Name, dtype, rank, shape and payload length prevent ambiguous concatenation.
    for text in [name, dtype] {
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(text.as_bytes());
    }
    out.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for &n in shape {
        out.extend_from_slice(&(n as u64).to_le_bytes());
    }
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

fn path_bytes(path: &Path) -> Vec<u8> {
    let mut out = vec![];
    for (name, a) in [
        ("center", &path.center),
        ("scale", &path.scale),
        ("lambda", &path.lambda),
        ("intercept", &path.intercept),
        ("deviance", &path.deviance),
    ] {
        append_frame(&mut out, name, "f64", &[a.len()], bytemuck::cast_slice(a));
    }
    for (name, a) in [("iters", &path.iters), ("n_reject", &path.n_reject)] {
        append_frame(&mut out, name, "i64", &[a.len()], bytemuck::cast_slice(a));
    }
    let map: Vec<u64> = path.col_map.iter().map(|&v| v as u64).collect();
    append_frame(&mut out, "col_map", "u64", &[map.len()], bytemuck::cast_slice(&map));
    let mut indices = vec![];
    let mut values: Vec<f64> = vec![];
    let mut indptr = vec![0u64];
    let mut present = vec![];
    for (rows, vals) in path.coef_rows.iter().zip(&path.coef_vals) {
        present.push(u8::from(rows.is_some()));
        if let (Some(rows), Some(vals)) = (rows, vals) {
            indices.extend(rows.iter().map(|&v| v as u64));
            values.extend(vals);
        }
        indptr.push(indices.len() as u64);
    }
    append_frame(&mut out, "coef.format", "utf8", &[], b"csc");
    append_frame(&mut out, "coef.present", "u8", &[present.len()], &present);
    append_frame(&mut out, "coef.data", "f64", &[values.len()], bytemuck::cast_slice(&values));
    append_frame(&mut out, "coef.indices", "u64", &[indices.len()], bytemuck::cast_slice(&indices));
    append_frame(&mut out, "coef.indptr", "u64", &[indptr.len()], bytemuck::cast_slice(&indptr));
    let shape = [path.center.len() as u64, path.lambda.len() as u64];
    append_frame(&mut out, "coef.shape", "u64", &[2], bytemuck::cast_slice(&shape));
    let events: Vec<u64> = path
        .events
        .iter()
        .flat_map(|e| match *e {
            Event::ImpreciseCenter(j) => [4, j as u64],
            Event::PrefitSaturated => [0, 0],
            Event::Dfmax(i) => [1, i as u64],
            Event::Saturated(i) => [2, i as u64],
            Event::MaxIter(i) => [3, i as u64],
        })
        .collect();
    append_frame(&mut out, "warnings", "u64", &[events.len() / 2, 2], bytemuck::cast_slice(&events));
    out
}

#[test]
fn framed_path_digest_is_unambiguous() {
    let mut a = vec![];
    let mut b = vec![];
    append_frame(&mut a, "a", "u8", &[2], &[1, 2]);
    append_frame(&mut b, "a", "u8", &[1, 2], &[1, 2]);
    assert_ne!(a, b);
    b.clear();
    append_frame(&mut b, "a", "i8", &[2], &[1, 2]);
    assert_ne!(a, b);
}

fn whole_pool_mode(flags: &str) -> Option<&str> {
    flags.strip_prefix("all-").or_else(|| flags.strip_prefix("inherit-"))
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn assert_creator_thread_policy(mode: &str) {
    let word = fp::control_word();
    if cfg!(target_os = "linux") {
        assert!(fp::flush_mode_active(mode), "Linux worker must inherit creator flags: {word:#x}");
        assert!(!bound::arithmetic_ok());
    } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        // XNU machine_thread_state_initialize sets FPCR_DEFAULT = 0.
        // Read the state before any worker-side control-register write.
        assert_eq!(word, 0, "new macOS ARM64 worker must start with default FPCR");
        assert!(bound::arithmetic_ok());
    } else {
        // Do not assume another platform's inheritance policy. Its observed
        // uniform state must be either the requested mode or supported arithmetic.
        assert!(fp::flush_mode_active(mode) || bound::arithmetic_ok(), "unexpected worker FP state: {word:#x}");
    }
}

#[test]
fn isolated_path_child() {
    let Ok(output) = std::env::var("PYBIGLASSO_TEST_OUTPUT") else {
        return;
    };
    let mut bytes = vec![];
    let mut worker_support = vec![];
    for alpha in [1., 0.5] {
        let mut first_thread = None;
        for threads in [1, 2, 4, 6] {
            let worker_bad = std::env::var("PYBIGLASSO_TEST_FP").unwrap_or_default();
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let _creator_fp = whole_pool_mode(&worker_bad).map(fp::Guard::new);
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            if let Some(mode) = worker_bad.strip_prefix("inherit-") {
                assert!(fp::flush_mode_active(mode), "creator must have the requested flags before pool creation");
            }
            let worker_bad_clone = worker_bad.clone();
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .start_handler(move |index| {
                    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                    {
                        if let Some(mode) = worker_bad_clone.strip_prefix("inherit-") {
                            // Read-only check BEFORE any worker-side mode setter.
                            assert_creator_thread_policy(mode);
                        } else if let Some(mode) = worker_bad_clone.strip_prefix("all-") {
                            fp::set_mode(mode);
                            assert!(!bound::arithmetic_ok(), "all workers must use unsupported arithmetic");
                        } else if worker_bad_clone.starts_with("worker-") && index == threads - 1 {
                            fp::set_mode(worker_bad_clone.trim_start_matches("worker-"));
                        }
                    }
                    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
                    let _ = (&worker_bad_clone, index);
                })
                .build()
                .unwrap();
            let expected_support = whole_pool_mode(&worker_bad).map(|_| {
                let observed = pool.broadcast(|_| bound::arithmetic_ok());
                assert!(observed.iter().all(|&v| v == observed[0]), "worker environment must be uniform");
                // One ordinary and one subnormal fit follow. Record the observed
                // worker support independently of the candidate cache metadata.
                worker_support.extend([u8::from(observed[0]); 2]);
                observed[0]
            });
            let x = Array2::from_shape_fn((128, 7).f(), |(i, j)| if (i >> j) & 1 == 0 { -1. } else { 1. });
            let y: Vec<f64> = (0..128).map(|i| 2. * x[(i, 0)]).collect();
            let lambdas = [1., 63. / 64., 31. / 32., 61. / 64., 15. / 16.];
            let penalties = [1.; 7];
            let params = Params {
                family: Family::Gaussian,
                alpha,
                lambdas: Some(&lambdas),
                nlambda: lambdas.len(),
                eps: 1e-12,
                lambda_min: 0.01,
                lambda_log_scale: true,
                penalty_factor: &penalties,
                dfmax: 7,
                max_iter: 100,
                stop_on_saturation: false,
                r_compatible: false,
                ybar: 0.,
                null_deviance: 512.,
                on_lambda: None,
            };
            let run = || {
                if worker_bad.starts_with("worker-") && threads > 1 {
                    assert!(bound::arithmetic_ok(), "fit thread must be normal; another worker is unsupported");
                }
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                let _guard = if worker_bad.starts_with("main-") {
                    Some(fp::Guard::new(worker_bad.trim_start_matches("main-")))
                } else {
                    None
                };
                if let Some(supported) = expected_support {
                    assert_eq!(bound::arithmetic_ok(), supported, "fit must use the observed worker environment");
                }
                let design = ContigX::<_, true> { x: x.view() };
                let fit = Fit { x: &design, y: &y, params: &params, mask: None, rows: None, n_fit: 128 };
                let (center, scale, _, _) = fit.center_scale().unwrap();
                fit.run(center, scale).unwrap()
            };
            let path = if worker_bad.starts_with("worker-") {
                // Execute the fit on worker 0; the last worker has FTZ/DAZ set.
                // This proves that checking only the fit thread is insufficient.
                pool.broadcast(|context| if context.index() == 0 { Some(run()) } else { None })
                    .into_iter()
                    .flatten()
                    .next()
                    .unwrap()
            } else {
                pool.install(run)
            };
            if alpha == 1. {
                for (k, &lambda) in lambdas.iter().enumerate() {
                    assert_eq!(path.coef_rows[k], Some(vec![0]));
                    // Walsh columns have mean 0 and norm sqrt(n). Soft thresholding
                    // gives beta0=2-lambda, all other coefficients exactly zero.
                    assert_eq!(path.coef_vals[k], Some(vec![2. - lambda]));
                    assert_eq!(path.intercept[k], 0.);
                }
            }
            let mut cell_bytes = path_bytes(&path);
            if let Some(supported) = expected_support {
                // Compare the same actual worker state in stock and enabled:
                // inherited flags on Linux, defaults on macOS ARM64.
                pool.install(|| subnormal_case(&mut cell_bytes, alpha, supported));
            }
            if worker_bad.is_empty() && std::env::var("PYBIGLASSO_TEST_FAULT").unwrap_or_default().is_empty() {
                pool.install(|| storage_cases(&mut cell_bytes, alpha));
                pool.install(|| recentered_cases(&mut cell_bytes, alpha));
            }
            if worker_bad.is_empty() {
                if let Some(first) = &first_thread {
                    assert_eq!(&cell_bytes, first, "thread invariance {threads}");
                } else {
                    first_thread = Some(cell_bytes.clone());
                }
            }
            bytes.extend(cell_bytes);
        }
    }
    fs::write(std::path::Path::new(&output).with_extension("fp"), worker_support).unwrap();
    fs::write(output, bytes).unwrap();
}

struct Cell {
    dir: std::path::PathBuf,
    result: std::process::Output,
}
impl Drop for Cell {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}
fn run(mode: &str, width: &str, fp: &str, fault: &str) -> Cell {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "gaussian-skip-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).unwrap();
    let result = Command::new(std::env::current_exe().unwrap())
        .arg("--exact")
        .arg("solver::gaussian::skip_tests::isolated_path_child")
        .arg("--nocapture")
        .env("PYBIGLASSO_TEST_SCAN_MODE", mode)
        .env("PYBIGLASSO_TEST_SIMD", width)
        .env("PYBIGLASSO_TEST_OUTPUT", dir.join("path"))
        .env("PYBIGLASSO_TEST_TRACE", dir.join("trace"))
        .env("PYBIGLASSO_TEST_STATS", dir.join("stats"))
        .env("PYBIGLASSO_TEST_FP", fp)
        .env("PYBIGLASSO_TEST_FAULT", fault)
        .output()
        .unwrap();
    Cell { dir, result }
}
fn succeeded(c: &Cell) {
    assert!(c.result.status.success(), "{}", String::from_utf8_lossy(&c.result.stderr));
    assert!(c.dir.join("path").exists(), "child test was not executed");
}
fn stats(c: &Cell) -> Vec<Vec<usize>> {
    fs::read_to_string(c.dir.join("stats"))
        .unwrap()
        .lines()
        .map(|l| l.split_whitespace().map(|n| n.parse().unwrap()).collect())
        .collect()
}

// Captures are native-endian within one test machine, never portable fixtures.
fn frames(data: &[u8]) -> Vec<Vec<Vec<u8>>> {
    let mut out = vec![];
    let mut pos = 0;
    while pos < data.len() {
        let mut take = |n| {
            let v = data[pos..pos + n].to_vec();
            pos += n;
            v
        };
        let header = take(48);
        let h: Vec<_> = header.as_chunks::<8>().0.iter().map(|v| u64::from_ne_bytes(*v) as usize).collect();
        let (n, p, q) = (h[2], h[3], h[4]);
        let mut f = vec![header, take(16), take(24)]; // core floats / prefix metadata
        for size in
            [n * 8, p * 8, p * 8, q * 8, q * 8, p * 8, p, p, p, p * 8, p * 8, p * 8, p * 8, p * 8, p * 8, p * 8, p]
        {
            f.push(take(size));
        }
        for size in [16, n * 8, p * 8, p * 8, p * 8, p * 56] {
            f.push(take(size));
        }
        out.push(f);
    }
    out
}
fn compare_traces(a: &Cell, b: &Cell, metadata: bool) {
    let a = frames(&fs::read(a.dir.join("trace")).unwrap());
    let b = frames(&fs::read(b.dir.join("trace")).unwrap());
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(&b).enumerate() {
        // f[4] holds intentionally stale inactive score storage in enabled mode.
        for k in [0, 1, 3, 5, 6, 7, 8, 9, 10] {
            assert_eq!(a[k], b[k], "trajectory boundary {i}, field {k}");
        }
        for (j, &active) in a[9].iter().enumerate() {
            if active != 0 {
                assert_eq!(a[4][8 * j..8 * j + 8], b[4][8 * j..8 * j + 8], "active score {i}/{j}");
            }
        }
        if metadata {
            // Prefix, radius, anchors, represented epochs, next lambda/cutoff bits
            // and decisions must agree explicitly, not only final coefficients.
            for k in [2, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] {
                assert_eq!(a[k], b[k], "cache metadata {i}/{k}");
            }
        }
    }
}
#[test]
fn all_widths_threads_shadow_and_enabled_match_exact() {
    let widths = vec!["scalar", "128", "256", "512", "auto"];
    #[cfg(target_arch = "x86_64")]
    let widths = {
        let mut widths = widths;
        if pulp::x86::V3::try_new().is_some() {
            widths.push("avx2");
        }
        if pulp::x86::V4::try_new().is_some() {
            widths.push("avx512");
        }
        widths
    };
    let stock = run("off", "scalar", "", "");
    succeeded(&stock);
    let expected = fs::read(stock.dir.join("path")).unwrap();
    for width in widths {
        let shadow = run("shadow", width, "", "");
        succeeded(&shadow);
        let enabled = run("enabled", width, "", "");
        succeeded(&enabled);
        assert_eq!(fs::read(shadow.dir.join("path")).unwrap(), expected);
        assert_eq!(fs::read(enabled.dir.join("path")).unwrap(), expected);
        compare_traces(&stock, &enabled, false);
        compare_traces(&shadow, &enabled, true);
        let all = stats(&enabled);
        assert!(all.iter().any(|s| s[0] > s[1] && s[3] > 0 && s[4] > 0), "no positive-radius/next-threshold skips");
        for s in all {
            assert!(s[0] >= s[1]);
        }
    }
}
#[test]
fn prefix_invalidation_falls_back_without_changing_path() {
    let stock = run("off", "auto", "", "invalidate");
    succeeded(&stock);
    let invalid = run("enabled", "auto", "", "invalidate");
    succeeded(&invalid);
    assert_eq!(fs::read(stock.dir.join("path")).unwrap(), fs::read(invalid.dir.join("path")).unwrap());
    compare_traces(&stock, &invalid, false);
    for s in stats(&invalid) {
        assert!(s[3] > 0 && s[5] == 0 && s[6] == 1);
    }
}
#[test]
fn stale_certificate_epochs_and_thresholds_fail_before_use() {
    for fault in ["epoch", "represented", "index", "lambda", "cutoff"] {
        let c = run("enabled", "auto", "", fault);
        assert!(!c.result.status.success(), "fault {fault} was not detected");
        let text = String::from_utf8_lossy(&c.result.stderr);
        assert!(
            text.contains(if ["epoch", "represented"].contains(&fault) {
                "score epoch mismatch"
            } else {
                "next threshold mismatch"
            }),
            "{fault}: {text}"
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod fp {
    // Test-only RAII modification of MXCSR in isolated native child processes.
    // Exceptions remain masked; all original control/status bits are restored.
    fn get() -> u32 {
        let mut value = 0u32;
        // SAFETY: stmxcsr writes exactly four bytes to an aligned live u32.
        unsafe {
            core::arch::asm!("stmxcsr [{}]",in(reg)&mut value,options(nostack,preserves_flags));
        }
        value
    }
    fn put(value: u32) {
        // SAFETY: value comes from MXCSR with only valid rounding/FTZ/DAZ bits
        // modified. No reserved bits or exception masks are changed.
        unsafe {
            core::arch::asm!("ldmxcsr [{}]",in(reg)&value,options(nostack,preserves_flags));
        }
    }
    pub(super) fn supports_daz() -> bool {
        #[repr(align(16))]
        struct Save([u8; 512]);
        let mut area = Save([0; 512]);
        // SAFETY: FXSAVE is available on x86-64, writes exactly 512 bytes to
        // this 16-byte-aligned live buffer and leaves the FP state unchanged.
        unsafe {
            core::arch::asm!("fxsave64 [{}]",in(reg)&mut area,options(nostack,preserves_flags));
        }
        let mask = u32::from_le_bytes(area.0[28..32].try_into().unwrap());
        // A zero mask denotes the architectural default, which excludes DAZ.
        mask & (1 << 6) != 0
    }
    pub(super) fn control_word() -> u64 {
        get() as u64
    }
    pub(super) fn flush_modes() -> Vec<&'static str> {
        let mut modes = vec!["ftz"];
        if supports_daz() {
            modes.extend(["daz", "ftz-daz"]);
        }
        modes
    }
    pub(super) fn flush_mode_active(mode: &str) -> bool {
        let expected = match mode {
            "ftz" => 1 << 15,
            "daz" => 1 << 6,
            "ftz-daz" => (1 << 15) | (1 << 6),
            _ => panic!("unknown flush mode"),
        };
        get() & ((3 << 13) | (1 << 15) | (1 << 6)) == expected
    }
    pub(super) fn set_mode(mode: &str) {
        let old = get();
        assert!(!mode.contains("daz") || supports_daz(), "DAZ unsupported by this CPU");
        let bits = match mode {
            "ftz" => 1 << 15,
            "daz" => 1 << 6,
            "ftz-daz" => (1 << 15) | (1 << 6),
            "down" => 1 << 13,
            "up" => 2 << 13,
            "zero" => 3 << 13,
            _ => panic!("unknown FP mode"),
        };
        put((old & !((3 << 13) | (1 << 15) | (1 << 6))) | bits);
    }
    pub(super) struct Guard(u32);
    impl Guard {
        pub(super) fn new(mode: &str) -> Self {
            let old = get();
            set_mode(mode);
            Self(old)
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            put(self.0);
        }
    }
}
// FPCR layout and access: Arm DDI 0502G, section 2.3.
// https://documentation-service.arm.com/static/5e907926c8052b16087615a6
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod fp {
    fn get() -> u64 {
        let value;
        // SAFETY: reading FPCR is an unprivileged AArch64 operation.
        unsafe {
            core::arch::asm!("mrs {}, fpcr",out(reg)value,options(nostack,preserves_flags));
        }
        value
    }
    fn put(value: u64) {
        // SAFETY: preserve the original FPCR except its architectural FZ and
        // rounding fields. Exceptions remain masked; no reserved bits are set.
        unsafe {
            core::arch::asm!("msr fpcr, {}", "isb",in(reg)value,options(nostack,preserves_flags));
        }
    }
    pub(super) fn control_word() -> u64 {
        get()
    }
    pub(super) fn flush_modes() -> Vec<&'static str> {
        vec!["fz"]
    }
    pub(super) fn flush_mode_active(mode: &str) -> bool {
        assert_eq!(mode, "fz");
        get() & ((3 << 22) | (1 << 24)) == (1 << 24)
    }
    pub(super) fn set_mode(mode: &str) {
        let bits = match mode {
            "fz" => 1 << 24,
            "up" => 1 << 22,
            "down" => 2 << 22,
            "zero" => 3 << 22,
            _ => panic!("unknown FP mode"),
        };
        put((get() & !((3 << 22) | (1 << 24))) | bits);
    }
    pub(super) struct Guard(u64);
    impl Guard {
        pub(super) fn new(mode: &str) -> Self {
            let old = get();
            set_mode(mode);
            Self(old)
        }
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            put(self.0);
        }
    }
}
#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn unsupported_main_and_worker_fp_environment_uses_exact_reads() {
    for mode in fp::flush_modes().into_iter().chain(["down", "up", "zero"]) {
        {
            let _guard = fp::Guard::new(mode);
            assert!(!bound::arithmetic_ok());
            assert!(bound::norm(2, |_| 1e-200).is_none());
        }
        assert!(bound::arithmetic_ok());
        // Directed rounding is rejected above. A heterogeneous rounding pool
        // need not itself produce deterministic stock signed zeros. The full
        // path comparison uses FTZ/DAZ on normal-valued dyadic inputs.
        if !fp::flush_modes().contains(&mode) {
            continue;
        }
        for scope in ["main", "worker"] {
            let fp = format!("{scope}-{mode}");
            let stock = run("off", "auto", &fp, "");
            succeeded(&stock);
            let enabled = run("enabled", "auto", &fp, "");
            succeeded(&enabled);
            assert_eq!(fs::read(stock.dir.join("path")).unwrap(), fs::read(enabled.dir.join("path")).unwrap());
            compare_traces(&stock, &enabled, false);
            for s in stats(&enabled) {
                assert_eq!(s[0], s[1]);
                assert_eq!(s[2], 0);
                assert_eq!(s[5], 0);
            }
        }
    }
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn fitting_thread_and_all_workers_with_flush_flags_match_stock() {
    assert_whole_pool_matches_stock("all");
}

#[test]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn creator_flush_flags_follow_platform_policy_without_worker_writes() {
    assert_whole_pool_matches_stock("inherit");
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn assert_whole_pool_matches_stock(prefix: &str) {
    for mode in fp::flush_modes() {
        let flags = format!("{prefix}-{mode}");
        let stock = run("off", "auto", &flags, "");
        succeeded(&stock);
        let enabled = run("enabled", "auto", &flags, "");
        succeeded(&enabled);
        assert_eq!(fs::read(stock.dir.join("path")).unwrap(), fs::read(enabled.dir.join("path")).unwrap());
        compare_traces(&stock, &enabled, false);
        let supported = fs::read(enabled.dir.join("path.fp")).unwrap();
        assert_eq!(
            supported,
            fs::read(stock.dir.join("path.fp")).unwrap(),
            "stock and candidate must observe the same worker state"
        );
        let rows = stats(&enabled);
        assert_eq!(rows.len(), 16); // normal and subnormal paths, two alphas x four thread counts
        assert_eq!(supported.len(), rows.len());
        if supported.contains(&1) {
            assert!(
                rows.iter().any(|s| s[0] > s[1] && s[3] > 0 && s[4] > 0),
                "supported worker state must exercise positive-radius/next-threshold skips"
            );
        }
        for (s, &support) in rows.iter().zip(&supported) {
            assert!(s[0] > 0, "must exercise inactive scans");
            assert_eq!(s[5], support as usize, "cache validity must match the observed worker environment");
            if support == 0 {
                assert_eq!(s[0], s[1], "unsupported arithmetic must read every selected column");
                assert_eq!(s[2], 0, "no certificates, including same-epoch shortcuts");
                assert_eq!(s[4], 0, "no next-strong certificates");
            } else {
                assert!(s[1] <= s[0]);
            }
        }
    }
}

fn subnormal_case(out: &mut Vec<u8>, alpha: f64, supported: bool) {
    assert_eq!(bound::arithmetic_ok(), supported);
    let x = Array2::from_shape_fn((128, 7).f(), |(i, j)| if (i >> j) & 1 == 0 { -1. } else { 1. });
    // Construct bit patterns directly, so fixture creation cannot flush them.
    let y: Vec<f64> = (0..128).map(|i| f64::from_bits((1 << 51) | if i & 1 == 0 { 1 << 63 } else { 0 })).collect();
    assert!(y.iter().all(|v| v.to_bits() & 0x7fff_ffff_ffff_ffff == 1 << 51));
    let lambdas = [f64::from_bits(1 << 50), f64::from_bits(1 << 49)];
    let penalties = [1.; 7];
    let (ybar, null_deviance) = Family::Gaussian.null_model(&y, None).unwrap();
    let params = Params {
        family: Family::Gaussian,
        alpha,
        lambdas: Some(&lambdas),
        nlambda: 2,
        eps: 1e-7,
        lambda_min: 0.01,
        lambda_log_scale: true,
        penalty_factor: &penalties,
        dfmax: 7,
        max_iter: 100,
        stop_on_saturation: false,
        r_compatible: false,
        ybar,
        null_deviance,
        on_lambda: None,
    };
    add_case(out, &ContigX::<_, true> { x: x.view() }, &y, &params, None, None);
}

fn add_case<D: Design>(
    out: &mut Vec<u8>,
    x: &D,
    y: &[f64],
    params: &Params,
    mask: Option<&[f64]>,
    rows: Option<&[usize]>,
) {
    let fit = Fit { x, y, params, mask, rows, n_fit: rows.map_or(y.len(), |r| r.len()) };
    let (center, scale, safe, _) = fit.center_scale().unwrap();
    assert!(safe);
    let path = fit.run(center, scale).unwrap();
    out.extend(path_bytes(&path));
}

fn recentered_cases(out: &mut Vec<u8>, alpha: f64) {
    let n = 4096;
    let lambdas = [1., 63. / 64., 31. / 32., 61. / 64., 15. / 16.];
    let penalties = [1.; 7];
    // Column 1 drives residual updates, so the repaired column 0 can
    // exercise inactive-score certificates as well as cache initialization.
    let y: Vec<_> = (0..n).map(|i| if (i >> 1) & 1 == 0 { -2. } else { 2. }).collect();
    let (ybar, null_deviance) = Family::Gaussian.null_model(&y, None).unwrap();
    let params = Params {
        family: Family::Gaussian,
        alpha,
        lambdas: Some(&lambdas),
        nlambda: lambdas.len(),
        eps: 1e-12,
        lambda_min: 0.01,
        lambda_log_scale: true,
        penalty_factor: &penalties,
        dfmax: 7,
        max_iter: 1000,
        stop_on_saturation: false,
        r_compatible: false,
        ybar,
        null_deviance,
        on_lambda: None,
    };
    for unrepresentable in [false, true] {
        let a = Array2::from_shape_fn((n, 7).f(), |(i, j)| {
            if j == 0 {
                if unrepresentable { 2f64.powi(54) + 4. * (i % 4) as f64 } else { 2f64.powi(50) + 2. * (i % 2) as f64 }
            } else if (i >> j) & 1 == 0 {
                -1.
            } else {
                1.
            }
        });
        let design = ContigX::<_, true> { x: a.view() };
        let fit = Fit { x: &design, y: &y, params: &params, mask: None, rows: None, n_fit: n };
        let (center, scale, _, warnings) = fit.center_scale().unwrap();
        assert_eq!(!warnings.is_empty(), unrepresentable);
        if !unrepresentable {
            assert_eq!((center[0], scale[0]), (2f64.powi(50) + 1.0, 1.0));
        }
        out.extend(path_bytes(&fit.run(center, scale).unwrap()));
    }
}

fn storage_cases(out: &mut Vec<u8>, alpha: f64) {
    let n = 97;
    let y: Vec<f64> = (0..n).map(|i| ((i * 23) % 29) as f64 / 4. - 3.).collect();
    let lambdas = [1., 0.99, 0.98, 0.97, 0.96];
    let factors = [0., 0.25, 2., 1., 1., 1., 1., 1., 1.];
    let (ybar, null_deviance) = Family::Gaussian.null_model(&y, None).unwrap();
    let mut params = Params {
        family: Family::Gaussian,
        alpha,
        lambdas: Some(&lambdas),
        nlambda: lambdas.len(),
        eps: 1e-7,
        lambda_min: 0.01,
        lambda_log_scale: true,
        penalty_factor: &factors,
        dfmax: 9,
        max_iter: 100,
        stop_on_saturation: false,
        r_compatible: false,
        ybar,
        null_deviance,
        on_lambda: None,
    };
    macro_rules! dtype {
        ($t:ty) => {{
            let x = Array2::from_shape_fn((n, 9).f(), |(i, j)| (((i * 17 + j * 7) % 31) as i64 - 15) as $t);
            add_case(out, &ContigX::<_, true> { x: x.view() }, &y, &params, None, None);
            add_case(out, &ContigX::<_, false> { x: x.view() }, &y, &params, None, None);
        }};
    }
    dtype!(f64);
    dtype!(f32);
    dtype!(i8);
    dtype!(u8);
    dtype!(i16);
    dtype!(u16);
    dtype!(i32);
    dtype!(u32);
    dtype!(i64);
    dtype!(u64);
    let x = Array2::from_shape_fn((n, 9).f(), |(i, j)| ((i * 17 + j * 7) % 31) as f64 / 4. - 3.);
    let strided = x.as_standard_layout();
    add_case(out, &StridedX::<_, true> { x: strided.view() }, &y, &params, None, None);
    let rows: Vec<usize> = (0..n).filter(|i| i % 3 != 0).collect();
    let mask: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0. } else { 1. }).collect();
    let masked: Vec<f64> = y.iter().zip(&mask).map(|(&y, &m)| if m == 0. { 0. } else { y }).collect();
    let selected: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
    let (mean, rss) = Family::Gaussian.null_model(&selected, None).unwrap();
    params.ybar = mean;
    params.null_deviance = rss;
    add_case(out, &ContigX::<_, true> { x: x.view() }, &masked, &params, Some(&mask), Some(&rows));
    let gather = GatherX { x: ContigX::<_, true> { x: x.view() }, idx: &rows };
    add_case(out, &gather, &selected, &params, None, None);
    let repeated: Vec<usize> = (0..121).map(|i| i * 13 % n).collect();
    let selected: Vec<f64> = repeated.iter().map(|&i| y[i]).collect();
    let (mean, rss) = Family::Gaussian.null_model(&selected, None).unwrap();
    params.ybar = mean;
    params.null_deviance = rss;
    add_case(out, &GatherX { x: ContigX::<_, true> { x: x.view() }, idx: &repeated }, &selected, &params, None, None);
    params.ybar = ybar;
    params.null_deviance = null_deviance;
    for offset in [0., 1e12] {
        let shifted = x.mapv(|v| v + offset);
        add_case(out, &ContigX::<_, true> { x: shifted.view() }, &y, &params, None, None);
    }
    // Near-collinearity and wide integers are addressing/arithmetic invariances,
    // not coefficient goldens produced by the solver.
    let correlated = Array2::from_shape_fn((n, 9).f(), |(i, j)| x[(i, 0)] + x[(i, j)] * 0.001);
    add_case(out, &ContigX::<_, true> { x: correlated.view() }, &y, &params, None, None);
    let wide = Array2::from_shape_fn((n, 9).f(), |(i, j)| (1u64 << 54) + ((i * 17 + j * 7) % 31) as u64 * 4);
    add_case(out, &ContigX::<_, true> { x: wide.view() }, &y, &params, None, None);
    let boolean = Array2::from_shape_fn((n, 9).f(), |(i, j)| (i * 17 + j * 7) % 3 == 0);
    add_case(out, &ContigX::<_, true> { x: boolean.view() }, &y, &params, None, None);
    let raw: Vec<Vec<f64>> = (0..9).map(|j| x.column(j).to_vec()).collect();
    let chunks: Vec<_> = raw
        .iter()
        .enumerate()
        .map(|(j, c)| {
            let mut chunks = vec![MappedCol::F64(&[])];
            chunks.extend(c.chunks(7 + j).map(MappedCol::F64));
            ChunkedColumn::new(chunks, n).unwrap()
        })
        .collect();
    add_case(out, &MappedX { columns: chunks }, &y, &params, None, None);
    // Each packed chunk has its own mid-byte offset and odd logical length.
    let parts: Vec<Vec<Vec<u8>>> = (0..9)
        .map(|j| {
            let vals: Vec<bool> = boolean.column(j).to_vec();
            vals.chunks(7 + 2 * j)
                .enumerate()
                .map(|(k, part)| {
                    let offset = (k + j) % 8;
                    let mut bytes = vec![0; (offset + part.len()).div_ceil(8)];
                    for (i, &v) in part.iter().enumerate() {
                        if v {
                            bytes[(offset + i) / 8] |= 1 << ((offset + i) % 8);
                        }
                    }
                    bytes
                })
                .collect()
        })
        .collect();
    let columns = parts
        .iter()
        .enumerate()
        .map(|(j, parts)| {
            let mut chunks = vec![MappedCol::Bits { bytes: &[], offset: 0, len: 0 }];
            chunks.extend(parts.iter().enumerate().map(|(k, bytes)| {
                let size = 7 + 2 * j;
                let len = (n - k * size).min(size);
                MappedCol::Bits { bytes, offset: (k + j) % 8, len }
            }));
            ChunkedColumn::new(chunks, n).unwrap()
        })
        .collect();
    add_case(out, &MappedX { columns }, &y, &params, None, None);
    params.lambdas = None; // zero-factor prefit precedes cache initialization
    add_case(out, &ContigX::<_, true> { x: x.view() }, &y, &params, None, None);
    params.lambdas = Some(&lambdas);
    for alpha in [1e-12, 0.999999] {
        params.alpha = alpha;
        add_case(out, &ContigX::<_, true> { x: x.view() }, &y, &params, None, None);
    }
    params.alpha = alpha;
    for grid in &[[1., 1., 0.5, 0.01, 0.], [1., 0.99, 0.98, 0.97, 0.96]] {
        params.lambdas = Some(grid);
        for compatible in [false, true] {
            params.r_compatible = compatible;
            for budget in [1, 2, 100] {
                params.max_iter = budget;
                add_case(out, &ContigX::<_, true> { x: x.view() }, &y, &params, None, None);
            }
        }
    }
}
