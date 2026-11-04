# Common tasks. Run `make setup` once, then `make serve`.
PY := .venv/bin/python

setup:            ## create the virtualenv and install dependencies (Python 3.11)
	python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest

smoke:            ## check the API key against every Meta model VeriCell uses
	$(PY) scripts/smoke_test.py

test:             ## run the unit tests (no API calls)
	$(PY) -m pytest -q tests

data:             ## download the LIVECell sample and rebuild the MorphoOrgaVQA image set
	$(PY) scripts/get_livecell.py --per-type 3 && $(PY) scripts/get_morphoorga.py

benchmark:        ## LIVECell verifier results + the full MorphoOrgaVQA run
	$(PY) scripts/run_candidates.py && $(PY) scripts/build_results.py && $(PY) scripts/run_morphoorga.py

models:           ## train the deployable verifiers and calibrate the answer ranges
	$(PY) scripts/train_verifiers.py && $(PY) scripts/calibrate.py

serve:            ## start the web app on http://localhost:8765
	.venv/bin/uvicorn vericell.web.app:app --port 8765

serve-open:       ## same, with every reasoning step on Muse Spark Open (Fireworks)
	VERICELL_LLM=open .venv/bin/uvicorn vericell.web.app:app --port 8765

.PHONY: setup smoke test data benchmark models serve serve-open
