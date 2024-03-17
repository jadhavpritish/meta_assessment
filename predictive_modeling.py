# ## Task:
#
# * Data Exploration and Cleaning:
#     * Distribution of key performance indicators and campaign attributes
#     * Outlier treatment
#     * Missing value analysis
# * Predictive Modeling:
#    * Problem formulation
#    * Feature Engineering
#    * Model Design
#    * Evaluation
#    * Comments and Next Steps
#
# ## About the Data:
#
# Dataset contains campaign performance data (called experiment), including metrics such as impressions, clicks, cost, etc. The dataset also include campaign metadata such as channel, creative type, offer, etc.
#

# +
import warnings
from datetime import datetime
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown
from itables import init_notebook_mode, show
from pivottablejs import pivot_ui
from scipy.stats import gmean
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from tabulate import tabulate

warnings.simplefilter(action="ignore")

pio.renderers.default = "iframe"

init_notebook_mode(all_interactive=True)
# -

data_schema = pd.DataFrame(
    {
        "parameter name": [
            "experiment_id",
            "created_date",
            "spent",
            "impressions",
            "leads",
            "mqls",
            "daily_budget",
            "experiment_goal",
            "wiz_campaign_channel",
            "offer_library_type",
            "ad_library_type",
        ],
        "Description": [
            "Unique Identifier for campaign",
            "Date for performance record",
            "Total Spent per campaign and date",
            "Total Impressions per campaign and date",
            "Total Leads per campaign and date",
            "Total mqls per campaign and date",
            "Assigned daily budget per campaign and date",
            "CPL: Lead Generation Campaign \nCTR: Brand Awareness Campaign",
            "Ad Channel where campaign was running",
            "LG: Lead Gen Form \nLP: Landing Page",
            "Ad Type assigned to campaign",
        ],
    }
)
print(tabulate(data_schema, headers="keys", tablefmt="psql"))

# ##  Getting to Know the Data:
#
# * The motivation is to setup a scrappy way to slice and dice the data to derive insights.
# * **Assumption**: All the spends are assumed to be in USD.
#
# The conversion rate for the experiments is defined as:
# $$\text{conversion\_rate} = \frac{MQLs}{leads}$$
#
# The cost per lead for experiments is defined as:
# $$\text{cost\_per\_lead} = \frac{spent}{leads}$$
#
# The cost per MQL for experiments is defined as:
# $$\text{cost\_per\_MQL} = \frac{spent}{MQL}$$
#
#

# +
# helpers

RAW_KPIS: list[str] = ["impressions", "leads", "mqls", "spent", "cvr"]

AGGREGATION_COLS: dict[str, tuple[str, str]] = {
    "impressions": ("impressions", "sum"),
    "leads": ("leads", "sum"),
    "mqls": ("mqls", "sum"),
    "spent": ("spent", "sum"),
}


def compute_derived_kpis(df: pd.DataFrame) -> pd.DataFrame:

    df["cpi"] = df["spent"] / df["impressions"]
    df["cpl"] = df["spent"] / df["leads"]
    df["cpmql"] = df["spent"] / df["mqls"]
    df["cvr"] = df["mqls"] / df["leads"]
    df["inv_cpmql"] = 1 / df["cpmql"]
    df["inv_cpl"] = 1 / df["cpl"]

    df[["cpi", "cpl", "cpmql", "cvr", "inv_cpmql"]] = df[
        ["cpi", "cpl", "cpmql", "cvr", "inv_cpmql"]
    ].fillna(0)
    df[["cpi", "cpl", "cpmql", "cvr", "inv_cpmql", "inv_cpl"]] = df[
        ["cpi", "cpl", "cpmql", "cvr", "inv_cpmql", "inv_cpl"]
    ].replace([-np.inf, np.inf], 0)

    df["cpm"] = df["cpi"] * 1000

    return df


# -

raw_campaign_df = pd.read_csv("./data/experiment_data.csv")
raw_campaign_df["created_date"] = pd.to_datetime(
    raw_campaign_df["created_date"], format="%m/%d/%y"
)
raw_campaign_df["day_of_week"] = raw_campaign_df["created_date"].dt.day_name()
raw_campaign_df["day_of_month"] = raw_campaign_df["created_date"].dt.strftime("%d")

# ## Exploratory Analysis
#
#
# |Numeric variables| Categorical Variables |
# |-----------------| ----------------------|
# |impressions|wiz_campaign_channel|
# |leads|ad_library_type|
# |mqls|experiment_goal|
# |spent|offer_library_type|
# |daily_budget|day_of_week|
# ||day_of_month|
#
#
#

# ## Missing Values Analysis:
#
# Missing values can cause unexpected failures while training a predictive model. Efficiently handling them is imperative for training a robust model.

# +
columns_with_na_values = raw_campaign_df.columns[raw_campaign_df.isna().any()].values

print(f"Length of data BEFORE dropping NA rows: {len(raw_campaign_df)}")

print(f"Columns with NA/missing values: {columns_with_na_values}")

na_rows_df = raw_campaign_df[raw_campaign_df.isna().any(axis=1)]
print(f"{len(na_rows_df)} rows founds with NA values")

if len(na_rows_df) > 10:
    display(na_rows_df.sample(10))
else:
    display(na_rows_df)

# Defensive assert to prevent missing values handling from being invalid if data changes
assert len(na_rows_df) == 2
display(
    Markdown(
        f"Since there are only 2 rows with NULL values with zero spend, we can safely drop such rows"
    )
)


valid_campaign_df = raw_campaign_df[~raw_campaign_df.isna().any(axis=1)]

print(f"Length of data after dropping NA rows: {len(valid_campaign_df)}")
# -

# ## Feature Engineering, Data Aggregation:
#
# * The daily MQL data is zero inflated. This poses a significant challenge in training a robuist predictive model.
# * Therefore, in addition to the categorical features like channel , ad-type, experiment goal, etc, we shall also leverage date related features.
# * Aggregate the data by potential input features before proceeding to data transformation and model training.

# +
agg_df = (
    raw_campaign_df.groupby(
        [
            "wiz_campaign_channel",
            "ad_library_type",
            "experiment_goal",
            "offer_library_type",
            "day_of_week",
            "day_of_month",
        ]
    )
    .agg(**AGGREGATION_COLS)
    .pipe(compute_derived_kpis)
    .reset_index()
)

print(f"{len(agg_df)=}")

# +
display(Markdown("## Performance based data cleaning"))

display(
    Markdown(
        """
* Filter out rows with impressions = 0 
* Filter out rows with insignificant spend, say, < \$1. 
    * The threshold **\$1** is pullled out of the hat (arbitrary)
    * Ideally, we would want to leverage click through rate with confidence intervals to identify the number of impressions needed to drive __at least__ one clicks. 
    * Unfortunately, clicks are not reported on the report. 
"""
    )
)

print(f"Length of data before performance based cleaning: {len(agg_df)}")

filtered_campaign_df = agg_df.loc[((agg_df.impressions > 0) & (agg_df.spent > 1))]

print(f"Length of data after performance based cleaning: {len(filtered_campaign_df)}")
print(
    f"Dropped {len(agg_df) - len(filtered_campaign_df)} rows based on insignificant performance"
)
# -

# ## Feature Distribution and Transformations

# +
fig = go.Figure()

for metric in RAW_KPIS:
    figx = go.Histogram(x=filtered_campaign_df[metric], visible=False)
    fig.add_trace(figx)

fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "label": metric,
                    "method": "update",
                    "args": [
                        {"visible": [k == metric for k in RAW_KPIS]},
                    ],
                }
                for metric in RAW_KPIS
            ],
            "y": 1.2,
            "x": 0.4,
        }
    ],
    title=f"RAW KPIS Historgram",
).update_traces(visible=True, selector=0)
# -

# ## Insights:
#
# * The distribution for impressions and spent is right skewed (long tailed)
# * The distribution of leads and MQLs is zero inflated.
#
# ### Impact of long tailed distribution:
#
# * Model Sensitivity to Outliers:
#   * In case of long tailed distributions, outliers can disproportionately influence the model's coefficients, leading to biased parameter estimates.
# * Violation of Normality Assumption:
#   * Linear regression models often assumes that the residuals (the differences between the actual and predicted values) are normally distributed. Long-tailed distributions can result in residuals that violate this assumption, affecting the reliability of the model.
#
# ### Solution:
#
# * Applying Non-Linear transformations like log or square root to long tailed distribution compresses the range of extreme values in the long tail.
# * Data Normalization: Long-tailed distributions often exhibit positive skewness, where the majority of observations cluster towards lower values while a few observations have extremely high values. Logarithmic transformation tends to normalize the skewed data by spreading out the lower values and compressing the higher values.
#
# ### Transformed features and dependent variable

# +
for metric in ["impressions", "spent", "leads", "mqls", "cvr"]:
    if metric == "cvr":
        filtered_campaign_df[f"trans_{metric}"] = np.sqrt(filtered_campaign_df[metric])
    else:
        filtered_campaign_df[f"trans_{metric}"] = np.log1p(filtered_campaign_df[metric])

fig = go.Figure()

for metric in [
    "trans_impressions",
    "trans_spent",
    "trans_mqls",
    "trans_leads",
    "trans_cvr",
]:
    figx = go.Histogram(x=filtered_campaign_df[metric], visible=False)
    fig.add_trace(figx)

fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "label": metric,
                    "method": "update",
                    "args": [
                        {
                            "visible": [
                                k == metric
                                for k in [
                                    "trans_impressions",
                                    "trans_spent",
                                    "trans_mqls",
                                    "trans_leads",
                                    "trans_cvr",
                                ]
                            ]
                        },
                    ],
                }
                for metric in [
                    "trans_impressions",
                    "trans_spent",
                    "trans_mqls",
                    "trans_leads",
                    "trans_cvr",
                ]
            ],
            "y": 1.2,
            "x": 0.4,
        }
    ],
    title=f"Post Transfromation Distribution",
).update_traces(visible=True, selector=0)
# -

# ## Training:
#
#
# ### Model Choice:
#
#
# ### Evaluation Metric:
#
# ### Loss Function:
#
#
#
#
#

# +
FEATURES = [
    "wiz_campaign_channel",
    "ad_library_type",
    "offer_library_type",
    "experiment_goal",
    "day_of_month",
    "day_of_week",
    "trans_impressions",
    "trans_spent",
    "trans_leads",
]
LABEL = "trans_cvr"


class trainOut(NamedTuple):
    cv_summary: pd.DataFrame
    model_pipeline: Pipeline
    hold_out_pred_df: pd.DataFrame


def split_data(all_data: pd.DataFrame) -> tuple[pd.DataFrame, ...]:

    train_val_df, test_df = train_test_split(
        all_data,
        test_size=0.2,
        random_state=32,
    )
    return train_val_df, test_df


def train_model(train_val_df: pd.DataFrame, k_fold: int = 4) -> Pipeline:

    categorical_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    preprocessor = make_column_transformer(
        (categorical_encoder, make_column_selector(dtype_include=object)),
        remainder="passthrough",
    )

    regressor = RandomForestRegressor(random_state=51)

    param_grid = {
        "n_estimators": [5, 10, 12],
        "max_depth": [3, 5, 7, 9],
        "max_features": [0.2, 0.3, 0.5, 0.6],
    }

    grid_regressor = GridSearchCV(
        regressor, param_grid, cv=k_fold, refit=True, scoring="r2"
    )

    tree = make_pipeline(preprocessor, grid_regressor).fit(
        train_val_df[FEATURES], train_val_df[LABEL]
    )

    return tree


# -


def run(perf_data: pd.DataFrame) -> trainOut:

    train_val_df, test_df = split_data(perf_data)

    trained_pipeline = train_model(train_val_df=train_val_df)
    grid_search_res = trained_pipeline.named_steps["gridsearchcv"]

    cv_summary_df = pd.DataFrame(grid_search_res.cv_results_).sort_values(
        "rank_test_score"
    )

    print(grid_search_res.best_estimator_)

    print("Cross Validation Summary")
    show(cv_summary_df)

    train_pred = round(
        np.power(trained_pipeline.predict(train_val_df), 2) * train_val_df["leads"]
    )
    test_pred = round(np.power(trained_pipeline.predict(test_df), 2) * test_df["leads"])

    train_r2_score = r2_score(train_pred, train_val_df["mqls"])
    test_r2_score = r2_score(test_pred, test_df["mqls"])

    test_df["predicted_mqls"] = test_pred

    print(f"{train_r2_score=}")
    print(f"{test_r2_score=}")

    return trainOut(
        cv_summary=cv_summary_df,
        model_pipeline=trained_pipeline,
        hold_out_pred_df=test_df,
    )


# +
# Train RandomForest Regressor using K fold cross validation

res = run(perf_data=filtered_campaign_df)
# -

# ## Hold out data precictions Comparison

# +
out_cols = [
    "wiz_campaign_channel",
    "ad_library_type",
    "experiment_goal",
    "offer_library_type",
    "day_of_week",
    "day_of_month",
    "impressions",
    "leads",
    "spent",
    "mqls",
    "predicted_mqls",
]

comparison_df = res.hold_out_pred_df[out_cols].reset_index(drop=True)
comparison_df["diff"] = comparison_df["mqls"] - comparison_df["predicted_mqls"]
display(comparison_df)
# -

# ## Closing Comments:
#
# * Transforming features with long tailed distribution minimizes the probability of model being sensitive to outliers.
# * Predicting a bounded variable, conversion_rate, yields a more robust model than directly predicting volume of mqls.
# * A tree based model facilitates training a non-linear model. A Rnadom Forest was used to train the model for predicing conversion rates and hence volume of mqls.
#   * By randomly selecting the features as well as the datapoints used for training each tree, Random Forest are less likely to overfit as compared to Decision Trees.
#
# * The trained model achieved a R2 score **72%** on the training data and score of **66.5%** on the hold data.
#     * Given that the difference between R2 scores for training data and test (hold-out) data is NOT significant, the chances of model overfitting the data is **low**. In other words, based on datapoints above, the model is generalizable.
#
#
# ## Future Work:
#
# * A thorough analysis of training vs test R2 score is needed to ensure that the model is truly generalizable.
# * Proposed Approach:
#   * Set aside a subset of test data as out of the bag samples (OOB). DO NOT touch this samples.
#   * Repeatedly train the model and measure the train and test (in the bag) R2 Scores.
#   * Perform statistical test to ensure that the there is no statistically significant difference between the train and test R2 scores.
#   * Ensure that the model's R2 score on the oob samples is in line with the training R2 score.
#
# * If the model is indeed overfitting, add more training data.
#
# * Leverage MLFlow for tracking experiments and enforcing version control on trained models.
#
# * Supervised learning models assumes stationarity. In real world, the data is seldom stationary. User behavior coupled with changing auction dynamic on advertising plaforms increases the probability of model diverging. Monitor model's performance in production and retrain as the model diverges.
#
