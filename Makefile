venv:
	python -m venv venv

activate:
	source venv/bin/activate

install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	source venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt

envdelete:
	rm -r venv

reset:
	rm -r tmp
	rm -r local-worker

.PHONY: venv install setup