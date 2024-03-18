SHELL := /bin/bash

.PHONY = init clean

LABS_LAUNCH_DIRECTORY?=$$(pwd)

#############################################################################################################
# Artifact cleanup
#############################################################################################################
clean-files: # Very basic clean functionality, limited to just files
	rm -rf dist/ build/ *.egg-info

	find ./ -name "*.pyc" -and -type f -and -not -path ".//.git/*" -delete
	find ./ -name "test.log" -and -type f -and  -not -path ".//.git/*" -delete
	find ./ -name "__pycache__" -and -type d -and -not -path ".//.git/*" -delete

clean-predictive-reports:
	@[ -f ./predictive_modeling.html ] && rm predictive_modeling.html || true
	@[ -f ./predictive_modeling.ipynb ] && rm predictive_modeling.ipynb || true

clean-campaign-analysis-reports:
	@[ -f ./campaign_analysis.html ] && rm campaign_analysis.html || true
	@[ -f ./campaign_analysis.ipynb ] && rm campaign_analysis.ipynb || true

clean-lite:: clean-files ## Clean files plus clean virtualenv (without recreating it)
	pipenv clean

clean:: clean-files ## Very basic clean functionality
	# This will totally remove the virtual environment, you will need to run  init  after
	# A less invasive alternative could be to just use  pipenv --clean  (which uninstalls removed packages)
	-pipenv --rm

	# git gc is really just minor tidying - https://git-scm.com/docs/git-gc
	git gc --aggressive

init-pipenv:
	pipenv install --dev

init: clean-files init-pipenv

format: ## Autoformat the code.
	pipenv run isort --atomic . $(EXTRA_FLAGS)
	pipenv run black --safe . $(EXTRA_FLAGS)

run-jupyterlab:
	LABS_LAUNCH_DIRECTORY=$(LABS_LAUNCH_DIRECTORY) pipenv run jupyter lab

generate-campaign-analysis-report: clean-campaign-analysis-reports
	pipenv run jupytext --to ipynb campaign_analysis.py
	pipenv run jupyter nbconvert --to html --no-input --execute campaign_analysis.ipynb campaign_analysis.html

generate-predictive-modeling-report:clean-predictive-reports
	pipenv run jupytext --to ipynb predictive_modeling.py
	pipenv run jupyter nbconvert --to html --no-input --execute predictive_modeling.ipynb predictive_modeling.html

