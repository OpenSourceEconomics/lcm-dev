# LCM Development Repository

[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/lcm-dev/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/lcm-dev/main)
[![image](https://codecov.io/gh/OpenSourceEconomics/lcm-dev/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenSourceEconomics/lcm-dev)

This repository aims to facilitate the development of the package
[lcm](https://github.com/OpenSourceEconomics/lcm). In particular, here we

- Generate data that is used in the testing process of `lcm`
- Publish the developer documentation, which is designed to explain the inner workings
  of the `lcm` code base
- Create graphical comparisons between `lcm` and analytical solutions

## Getting Started

Get started by installing the conda environment. The conda environment will contain a
local installation of `lcm`. Therefore, we require the following directory structure:

```console
parent_folder
 |-lcm-dev
 |-lcm
```

Then you can install the environment using

```console
cd /path/to/lcm-dev
mamba env create
```

and generate all data and figures by running

```console
pytask
```
