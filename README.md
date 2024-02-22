# a_dcf
a-DCF: an architecture agnostic metric

## Installation
- We support package installation via PyPi. For Python users, `pip install a_dcf` will install the package.
- Alternatively, installation of this repo is also available via `python -m pip install -e .`

## Usage
- With default a-DCF configuration (a-DCF1 in the paper)
```
from a_dcf import calculate_a_dcf
results = calculate_a_dcf(YOUR_SCORE_FILE_DIR)
```
