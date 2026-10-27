default: test

.PHONY: test
# across every core (pytest-xdist); the R comparison below stays in one
# process, which runs R once
test:
	uv run pytest -n auto

# the newest versions pyproject.toml allows, ignoring uv.lock: what a new
# install gets
.PHONY: test-latest
test-latest:
	UV_LOCKED=false uv run --upgrade pytest -n auto

.PHONY: lint
lint:
	uv run ruff check src tests

# its kernels against plain scalar sums, at every SIMD width this CPU runs
.PHONY: test-rust
test-rust:
	cargo test --release --manifest-path rust/Cargo.toml

.PHONY: browser test-browser
browser:
	cd web && npm ci && npm run build

test-browser: browser
	uv run python scripts/check_wasm.py
	cd web && npm test && npm run test:browser

.PHONY: update-hooks run-hooks-all-files
update-hooks:
	uv run pre-commit autoupdate

run-hooks-all-files:
	uv run pre-commit run --all-files

# R comparisons; set CONTAINER=podman to use podman instead
CONTAINER ?= docker
R_VERSION ?= latest
R_TAG = $(if $(filter latest,$(R_VERSION)),latest,r$(R_VERSION))
# the R release whose package snapshot made tests/fixtures (biglasso 1.7.2)
FIXTURES_R = 4.6.1
R_ENV = CONTAINER=$(CONTAINER) R_IMAGE=pybiglasso-r:$(R_TAG)

.PHONY: r-image
r-image:
	$(CONTAINER) build --build-arg R_VERSION=$(R_VERSION) -t pybiglasso-r:$(R_TAG) -f docker/r-biglasso.Dockerfile docker

# Optional benchmark packages have their own image; fixture dependencies stay lean.
BENCH_R_VERSION ?= 4.6.1
BENCH_R_TARGET ?= snpnet
BENCH_BUILD_JOBS ?= 2
.PHONY: r-bench-image
r-bench-image:
	$(CONTAINER) build --build-arg R_VERSION=$(BENCH_R_VERSION) --build-arg BUILD_JOBS=$(BENCH_BUILD_JOBS) --target $(BENCH_R_TARGET) -t pybiglasso-bench:r$(BENCH_R_VERSION)-$(BENCH_R_TARGET) -f docker/r-benchmarks.Dockerfile docker

.PHONY: r-versions
r-versions: r-image
	$(CONTAINER) run --rm pybiglasso-r:$(R_TAG) Rscript -e 'cat(R.version.string, "/ biglasso", format(packageVersion("biglasso")), "\n")'

.PHONY: test-r
test-r: r-image
	$(R_ENV) uv run pytest -m r

# first, since it's quick and needs no R
.PHONY: soak
soak: soak-hypothesis r-image
	$(R_ENV) uv run pytest -m soak -s

# the property tests on thousands of random examples each (make test runs a
# fixed 20); a failure is kept in .hypothesis/ and replayed first next time
.PHONY: soak-hypothesis
soak-hypothesis:
	HYPOTHESIS_PROFILE=soak uv run pytest tests/test_hypothesis.py tests/test_hypothesis_boundaries.py

.PHONY: fixtures
fixtures:
	$(MAKE) r-image R_VERSION=$(FIXTURES_R)
	CONTAINER=$(CONTAINER) R_IMAGE=pybiglasso-r:r$(FIXTURES_R) uv run python scripts/make_r_fixtures.py

# where a fit spends its time, step by step, with Python 3.15's sampling
# profiler on the numba backend; runs in an isolated environment so the
# project's .venv keeps its Python
.PHONY: profile
profile:
	uv run --python 3.15 --isolated --with-editable '.[numba]' python -m profiling.sampling \
	  run --sort tottime -l 15 benchmarks/profile_fit.py

# the README's benchmark plot, from the committed results; later files win,
# and glmnet and skglm-enet come first so they only fill in glmnet and skglm's
# elastic net; path-v020 is pybiglasso 0.2.0 (Rust)
RESULTS = $(addprefix benchmarks/results/path,-glmnet.jsonl -skglm-enet.jsonl .jsonl -allthreads.jsonl -r.jsonl -quiet.jsonl -statsmodels.jsonl -final.jsonl -v020.jsonl)

docs/benchmarks.png: benchmarks/plot.py $(RESULTS)
	uv run --group bench benchmarks/plot.py $(RESULTS) -o $@

.PHONY: build
build:
	rm -rf dist
	uv build

# the sdist release.yml publishes, and a wheel built from it: build them,
# check the metadata PyPI will render, and fit from the wheel with no extras
.PHONY: check-dist
check-dist: build
	uvx twine check --strict dist/*
	uv run --isolated --no-project --with dist/*.whl python scripts/check_wheel.py
