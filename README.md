# Tasks:

## Campaign Performance Analysis:
* Analyze campaign Performance by Key attributes to derive actionable insights.

## Predict MQL volume:
* Data Exploration and Cleaning:
    * Distribution of key performance indicators and campaign attributes
    * Outlier treatment
    * Missing value analysis
* Predictive Modeling:
   * Problem formulation
   * Feature Engineering
   * Model Design
   * Evaluation
   * Comments and Next Steps
 

## Setup:

### Pre-requisite:
* Python 3.11 <br>
  * I highly recommend using Pyenv to manage python version on local machine. More info: [here](https://pypi.org/project/pyenv/) <br>
```pyenv install -v 3.11```
* Install Pipenv to manage virtul environment <br>
```pip install pipenv```
* Initialize Virtual Environment <br>
```make clean init```

## Pregenerated Reports:
 Download and view in a browser.  
* [Campaign Performance Analysis](https://github.com/jadhavpritish/meta_assessment/blob/main/campaign_analysis.html)
* [Predictive Modelling Analysis](https://github.com/jadhavpritish/meta_assessment/blob/main/predictive_modeling.html)


## (Re)Generating Reports

* (Re)Generate Campaign Performance Analysis Report:
```make generate-campaign-analysis-report```
* (Re)Generate Predictive Modelling Report:
```make generate-predictive-modeling-report```