venv:
	python -m venv venv

install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	pip install --upgrade pip
	pip install -r requirements.txt

.PHONY: venv install setup