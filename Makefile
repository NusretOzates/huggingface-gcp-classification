all: isort reformat lint mypy

isort:
	isort .

reformat:
	black .

lint:
	pylint custom_training_docker training.py

mypy:
	mypy .
