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
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import Markdown
from itables import init_notebook_mode, show
from pivottablejs import pivot_ui
from scipy.stats import gmean
from sklearn.model_selection import train_test_split
from tabulate import tabulate

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

RAW_KPIS: list[str] = ["impressions", "leads", "mqls", "spent"]

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

# ## Data Aggregation:

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
    * The threshold **\$0.5** is pullled out of the hat (arbitrary)
    * Ideally, we would want to leverage click through rate with confidence intervals to identify the number of impressions needed to drive __at least__ one clicks. 
    * Unfortunately, clicks are not reported on the report. 
"""
    )
)

filtered_campaign_df = agg_df.loc[((agg_df.impressions > 0) & (agg_df.spent > 1))]

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

# +
for metric in ["impressions", "spent"]:
    filtered_campaign_df[f"log_{metric}"] = np.log1p(filtered_campaign_df[metric])

fig = go.Figure()

for metric in ["log_impressions", "log_spent"]:
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
                                k == metric for k in ["log_impressions", "log_spent"]
                            ]
                        },
                    ],
                }
                for metric in ["log_impressions", "log_spent"]
            ],
            "y": 1.2,
            "x": 0.4,
        }
    ],
    title=f"Post Transfromation Distribution",
).update_traces(visible=True, selector=0)

# +
CATEGORICAL_FEATURES = [
    "wiz_campaign_channel",
    "ad_library_type",
    "offer_library_type",
    "experiment_goal",
    "day_of_month",
    "day_of_week",
]
filtered_campaign_df[CATEGORICAL_FEATURES] = filtered_campaign_df[
    CATEGORICAL_FEATURES
].astype("category")


X = filtered_campaign_df[
    [
        "wiz_campaign_channel",
        "ad_library_type",
        "offer_library_type",
        "experiment_goal",
        "day_of_month",
        "day_of_week",
        "log_impressions",
        "log_spent",
        "leads",
    ]
]

y = filtered_campaign_df["mqls"]

X_train, X_other, y_train, y_other = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_other, y_other, test_size=0.4, random_state=56
)
# -
