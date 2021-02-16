# pybiglasso


`pybiglasso` is a python implementation -- a port -- of the R
[`biglasso`](https://github.com/YaohuiZeng/biglasso) R package, which does this:

> ... extends lasso and elastic-net linear and logistic regression models for
> ultrahigh-dimensional, multi-gigabyte data sets that cannot be loaded into
> memory.

## Quick Start

Install the package:

`pip install -U pybiglasso`

In a python interpreter:

``` python
import pybiglasso as bl

test_data = bl.common.generate_test_data(num_rows=50, num_features=5)
X, y = test_data["X"], test_data["y"]
X = np.asfortranarray(X)

results = bl.biglasso(X, y, alpha=0.8, family="binomial")

```

And see [this demo notebook](./biglasso-demo.ipynb) for more, including performance benchmarks against R `biglasso`.


## Validating Against Released R `biglasso`

The repo includes code to test its own biglasso implementation against the released R version of `biglasso`, to check for agreement. Docker is required. To run the checks, do a:

    make run-py-vs-r

These checks compare the lambda array values, the fitted weights, and some other returned state from the two implementations for agreement, over a grid of inputs, which vary the random seed, number of lambda values, design matrix size, alpha, and number of monomorphic columns. The agreement check is performed using numpy's `allclose` function with its default settings, which checks that the following equation is elementwise true for all elements checked:

    absolute(a - b) <= (atol + rtol * absolute(b))

where the defaults are `atol=1e-8` and `rtol=1e-5`. ([ref](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html))

## Installation and Development

To install a released package:

```
pip install pybiglasso
```

To install from source:

```
git clone https://github.com/mmacpherson/pybiglasso
cd pybiglasso
make install
```

To install for development:

```
git clone https://github.com/mmacpherson/pybiglasso
cd pybiglasso
make develop
```

To run the tests run

```
make test  # installs and runs tox
```

If you see a `make: tox: No such file or directory` error when running `make test`, you first need to `pip install tox` (globally).


## TODO

- Implement predict() methods.
- Implement row subsetting mechanism.
- Implement alternative performance stats, eg AUC.
- Use sparse weights matrix during training to save memory.
- Investigate "overuse" of compute cycles by `pybiglasso` relative to R `biglasso`.
