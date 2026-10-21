default: test

.PHONY: test
test:
	uv run pytest

.PHONY: lint
lint:
	uv run ruff check src tests

.PHONY: notebook
notebook:
	uv run --group scripts jupyter notebook --no-browser

R_IMAGE=rbox
.PHONY: r-image
r-image: Dockerfile.R-biglasso
	docker build -t $(R_IMAGE) -f $^ .

.PHONY: run-py-vs-r
run-py-vs-r: r-image
	uv run --group scripts python scripts/compare_vs_R_biglasso.py

.PHONY: build
build:
	uv build
