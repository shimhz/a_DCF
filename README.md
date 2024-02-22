# a_dcf
a-DCF: an architecture agnostic metric

## Installation
- We support package installation via PyPi. For Python users, `pip install a_dcf` will install the package.
- Alternatively, installation of this repo is also available via `python -m pip install -e .`

## Usage
- With default a-DCF configuration (a-DCF1 in the paper)
```
from a_dcf import a_dcf
results = a_dcf.calculate_a_dcf(YOUR_SCORE_FILE_DIR)
```
### Score file format
- The score file format should adhere to SASV protocol.
  - Four columns required: (i) speaker model, (ii) test utterance, (iii) score, and (iv) trial type
    - Trial type should comprise three types: target, nontarget, and spoof
- Partial example of score file
```
LA_0015 LA_E_1103494 6.960134565830231 target
LA_0007 LA_E_5013670 6.150891035795212 nontarget
LA_0007 LA_E_7417804 -2.306972861289978 spoof
```
- Evaluation protocol (i.e., set of trials) used in SASV2022 Challenge (from ASVspoof2019 LA corpora)
  - https://github.com/sasv-challenge/SASVC2022_Baseline/blob/main/protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt
