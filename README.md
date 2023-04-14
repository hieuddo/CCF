# Continual Collaborative Filtering

## How to run
* For MF:
  * Install dependencies:
    `pip install Cython numpy scipy`
  * Build inplace `cornac` library:  
    `python setup.py build_ext --inplace`
  * Config dataset and seed then run scripts file: `scripts/batch_ua.sh`
* For NCF:
  * Run `ua_rep.py` with optional details in `config.py`
