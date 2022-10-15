all: isort reformat lint mypy
format: isort reformat

isort:
	isort .

reformat:
	black .

lint:
	pylint custom_training_docker training.py

mypy:
	mypy .
