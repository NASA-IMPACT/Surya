# Surya
Implementation of the Surya Foundation Model and Downstream Tasks for Heliophysics

## Getting started

Clone and install the environment (requires [uv](https://docs.astral.sh/uv/) package manager)

```sh
git clone git@github.com:NASA-IMPACT/Surya.git
cd Surya

mkdir ./uvcache

UV_CACHE_DIR=./.uvcache TMPDIR=./.uvcache uv sync
```

Run an end to end test via `python -m pytest -s -o log_cli=true tests/test_surya.py`. The script will run the Surya to generate two-hour ahead forecasts for 2014-01-07 and generate a file `surya_model_validation.png` which should look as below. Moreover, it will test the performance of the model against threshold values as below:
```
============================= test session starts ==============================
plugins: asdf-3.4.0
collected 1 item

tests/test_surya.py::test_surya_20140107 
-------------------------------- live log call ---------------------------------
INFO     test_surya:test_surya.py:188 GPU detected. Running the test on device 0.
INFO     test_surya:test_surya.py:195 Surya FM: 366.19 M total parameters.
INFO     test_surya:test_surya.py:199 Loaded weights.
INFO     test_surya:test_surya.py:201 Starting inference run.
INFO     test_surya:test_surya.py:215 Completed validation run. Local loss 0.31665. Reference loss 0.31665. Deviation 0.0.
INFO     test_surya:test_surya.py:219 Preparing visualization
INFO     test_surya:test_surya.py:247 Saved visualization at surya_model_validation.png.
PASSED

=================== 1 passed, 1 warning in 130.12s (0:02:10) ===================
```
![Sample output of surya for 2014-01-07.](assets/surya_model_validation.png)
