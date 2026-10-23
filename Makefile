default: test

.PHONY: test
test:
	uv run pytest

# -- the newest versions pyproject.toml allows, ignoring uv.lock: what a new
#    install gets
.PHONY: test-latest
test-latest:
	UV_LOCKED=false uv run --upgrade pytest

.PHONY: lint
lint:
	uv run ruff check src tests

.PHONY: update-hooks run-hooks-all-files
update-hooks:
	uv run pre-commit autoupdate

run-hooks-all-files:
	uv run pre-commit run --all-files

# -- R comparisons; set CONTAINER=podman to use podman instead
CONTAINER ?= docker
R_VERSION ?= latest
R_TAG = $(if $(filter latest,$(R_VERSION)),latest,r$(R_VERSION))
# -- the R release whose package snapshot made tests/fixtures (biglasso 1.7.2)
FIXTURES_R = 4.6.1
R_ENV = CONTAINER=$(CONTAINER) R_IMAGE=pybiglasso-r:$(R_TAG)

.PHONY: r-image
r-image:
	$(CONTAINER) build --build-arg R_VERSION=$(R_VERSION) -t pybiglasso-r:$(R_TAG) -f docker/r-biglasso.Dockerfile docker

.PHONY: r-versions
r-versions: r-image
	$(CONTAINER) run --rm pybiglasso-r:$(R_TAG) Rscript -e 'cat(R.version.string, "/ biglasso", format(packageVersion("biglasso")), "\n")'

.PHONY: test-r
test-r: r-image
	$(R_ENV) uv run pytest -m r

.PHONY: soak
soak: r-image
	$(R_ENV) uv run pytest -m soak -s

.PHONY: fixtures
fixtures:
	$(MAKE) r-image R_VERSION=$(FIXTURES_R)
	CONTAINER=$(CONTAINER) R_IMAGE=pybiglasso-r:r$(FIXTURES_R) uv run python scripts/make_r_fixtures.py

# -- where a fit spends its time, with Python 3.15's sampling profiler; runs in
#    an isolated environment so the project's .venv keeps its Python
.PHONY: profile
profile:
	uv run --python 3.15 --isolated --with-editable . python -m profiling.sampling \
	  run --sort tottime -l 15 benchmarks/profile_fit.py

# -- the README's benchmark plot, from the committed results
RESULTS = $(addprefix benchmarks/results/path,.jsonl -allthreads.jsonl -r.jsonl -quiet.jsonl -statsmodels.jsonl -final.jsonl)

docs/benchmarks.png: benchmarks/plot.py $(RESULTS)
	uv run --group bench benchmarks/plot.py $(RESULTS) -o $@

.PHONY: build
build:
	rm -rf dist
	uv build

# -- what release.yml publishes: build it, check the metadata PyPI will render,
#    and fit from the installed wheel with no extras
.PHONY: check-dist
check-dist: build
	uvx twine check --strict dist/*
	uv run --isolated --no-project --with dist/*.whl python scripts/check_wheel.py
