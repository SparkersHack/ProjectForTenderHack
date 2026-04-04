PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
UVICORN ?= $(PYTHON) -m uvicorn
PYTHONPATH_EXPORT = PYTHONPATH=src:.

.PHONY: install install-frontend frontend-build normalize-datasets bootstrap bootstrap-lite backend frontend backend-prod prod-build test

install:
	$(PIP) install -r requirements.txt

install-frontend:
	npm --prefix frontend install

frontend-build:
	npm --prefix frontend run build

normalize-datasets:
	$(PYTHON) scripts/normalize_datasets.py --copy

bootstrap:
	$(PYTHON) scripts/bootstrap_local.py --train-personalization

bootstrap-lite:
	$(PYTHON) scripts/bootstrap_local.py --skip-fasttext

backend:
	$(PYTHONPATH_EXPORT) $(UVICORN) backend.main:create_app --factory --host 0.0.0.0 --port 8000

backend-prod:
	$(PYTHONPATH_EXPORT) $(UVICORN) backend.main:create_app --factory --host 0.0.0.0 --port 8000

frontend:
	npm --prefix frontend run dev -- --host 0.0.0.0 --port 5173

prod-build: frontend-build

test:
	$(PYTHONPATH_EXPORT) $(PYTHON) -m unittest discover -s tests -v
