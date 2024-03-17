# ## Task:
#
# Analyze campaign Performance by Key attributes to derive actionable insights.
# ## About the Data:
#
# Dataset contains campaign performance data (called experiment), including metrics such as impressions, clicks, cost, etc. The dataset also include campaign metadata such as channel, creative type, offer, etc.
#

# +
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from itables import init_notebook_mode, show
from pivottablejs import pivot_ui
from scipy.stats import gmean
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
# * The conversion rate for the experiments is defined as:
# $$\text{conversion\_rate} = \frac{MQLs}{leads}$$
#
# * The cost per lead for experiments is defined as:
# $$\text{cost\_per\_lead} = \frac{spent}{leads}$$
#
#
# * The cost per MQL for experiments is defined as:
# $$\text{cost\_per\_MQL} = \frac{spent}{MQL}$$
#
#

# +
raw_campaign_df = pd.read_csv("./data/experiment_data.csv")
raw_campaign_df["created_date"] = pd.to_datetime(
    raw_campaign_df["created_date"], format="%m/%d/%y"
)
raw_campaign_df["day_of_week"] = raw_campaign_df["created_date"].dt.day_name()

display(pivot_ui(
    raw_campaign_df,
    rows=[
        "wiz_campaign_channel",
    ],
    cols=[
        "ad_library_type",
    ],
    vals=["impressions", "leads", "mqls", "daily_budget", "spent"],
    aggregatorName="Sum",
    rendererName="Heatmap",
    hiddenFromAggregators=[
        "experiment_id",
        "experiment_goal",
        "offer_library_type",
        "ad_library_type",
        "wiz_campaign_channel",
    ],
    hiddenFromDragDrop=["impressions", "leads", "mqls", "daily_budget", "spent"],
))
# -

# # Executive Summary 
#
# **Assumptions:**
# * Efficiency is defined as the ability of a channel/ad_type/campaign to drive efficient MQLs.
# * Simply sorting by cost per MQL can be misleading since it does NOT account for volume of MQLs.
# * As a result, efficiency is determined by computing a weighted geometric mean of a leads and conversion_rate.
#     * Geometric Mean allows us to compare attributes with varying properties and ranges, for instance, leads and conversion rate.
#     * By design, with Geometric mean, changes in the smaller measures have just as dramatic an effect on the result as changing the big ones.
#  
#
#
# | Questions? | Insights | Actionables |
# |:--------------------|:----------|:------------|
# | What is the most efficient Channel? | - LinkedIn is the most efficient channel accounting for **63%** of the leads and **76%** of the MQLs. <br>- LinkedIn has the lowest CMQL of **\$543** and the highest conversion rate of **36.9%** | - Double click into LinkedIn to identify growth opportunities. |
# | What is the most inefficient Channel? | - Google is the least efficient channel with **\$12k** in spend and **zero** MQLs <br> - Instagram is the second most inefficient Channel with conversion rate of **15.6%** and a CMQL of **\$1203** <br>- Facebook has a weak impressions to lead ratio. Facebook accounts for **63.4%** of the total impressions while driving only **16%** of the MQLs| - Consider Pausing Google <br> - Investigate opportunities on Instagram to improve conversion rate and thereby the overall efficiency. <br> - Analyze Facebook targeting to impove upper funnel metrics| 
# |What are the most efficient ad types?| - Image and Convo are the most efficient Ad Types with **647** MQLs,accounting for **96%** of the total MQLs <br> CONVO ads have a conversion rate of 88%, the highest amongst all ad types, while driving **37%** of the total MQLs. <br> - CONVO ads have the highest CPM of **\$1026** thereby indicating limited ad inventory and high competition. <br> - IMAGE ads have the lowest CPM of **\$28.5**. | - Allocate more maketing budget to `CONVO` ads to drive incremental and efficient mqls. <br> - Leverage IMAGE Ads for driving brand awareness. |
# |What are the most inefficient channels?| - VIDEO ads have a high conversion rate of **76%** but a high cost per lead of **$901**. <br> CAROUSEL ads have the worst cost per mql of **\$4922.95**.| - Identify opportunities to improve impression to lead ratio for VIDEO ads. <br> - Consider pausing CAROUSEL ads. |
# |Facebook Upper Funnel Analytics|- Brand Awareness campaigns 55% of the total impressions attributed to campaigns on Facebook with ad type IMAGE. <br> - Excluding brandareness campaigns, FACEBOOK IMAGE ads optimized for CPL has the second best conversion rate of 33% and cost per MQL of **\$331**. <br> - Facebook VIDEO campaigns have the lowest CPM of **\$7.42** while FACEBOOK IMAGE campaigns have the second lowest CPM of **\$11.7**|- Double Down on Facebook Campaigns optimzied for MQLs <br> - Facebook campaigns are the cheapest way to drive brand awareness|
# |What are the best days and worst days to advertise?|- Tuesday, Wednesday, Thursday and Friday drive more than **70%** of the MQLS at a conversion rate of **33.2%** <br> - Weekend has the lowest conversion rate of **23%**| - Reallocate budgets from weekends to weekdays|
# |Concluding thoughts| - LinkedIn drives the lion's share of MQLS. <br> - Weekdays are the best time to advertise the product| - Based on the observed trends, the product being marketed is a **B2B product**. <br> - Generalize the day of week trends from Lead generation campaigns to Brand Awareness Campaigns. |

# +
# Helpers


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


AGGREGATION_COLS: dict[str, tuple[str, str]] = {
    "impressions": ("impressions", "sum"),
    "leads": ("leads", "sum"),
    "mqls": ("mqls", "sum"),
    "spent": ("spent", "sum"),
}

PERCENTAGE_COLS: list[str] = ["impressions", "leads", "mqls", "spent"]
RATIO_COLS: list[str] = ["cpm", "cpl", "cpmql", "cvr", "efficiency"]

DISPLAY_COLS: list[str] = PERCENTAGE_COLS + RATIO_COLS
# -

# # Campaign Analysis Insights:

# ### By `wiz_campaign_channel`:


# +
channel_agg_df = (
    raw_campaign_df.groupby("wiz_campaign_channel")
    .agg(**AGGREGATION_COLS)
    .pipe(compute_derived_kpis)
    .sort_values("spent", ascending=False)
    .reset_index()
)
channel_agg_df["efficiency"] = gmean(
    channel_agg_df[["leads", "cvr", "inv_cpmql"]],
    axis=1,
    weights=[3, 5, 2],
    nan_policy="propagate",
)


display(show(channel_agg_df[["wiz_campaign_channel"] + DISPLAY_COLS]))


channel_agg_df[[f"percent_{col}" for col in PERCENTAGE_COLS]] = (
    100 * channel_agg_df[PERCENTAGE_COLS] / channel_agg_df[PERCENTAGE_COLS].sum()
)
fig = px.bar(
    channel_agg_df,
    x="wiz_campaign_channel",
    y=[f"percent_{col}" for col in PERCENTAGE_COLS],
    title="Campaign Performance by Channel",
)
fig.update_layout(yaxis_title="Percentage Contribution")

fig.show()
# -

# * `Facebook` ads have garnered 9.5M impressions accounting for 63.4% of the total impressions while driving only 16% of the MQLs.
# * `Linkedin` has driven 1386 leads accounting for 63.8% of the total leads.
# * 76% of the total MQLs are attributed to Linkedin. 
# * `Linkedin` is the most efficient channel with 37% conversion rate and a cost per MQL of \$543.09.
# * `Google Ads` have the highest CPL of \$1022.53 with no MQLs
#
# **Potential Actionables for driving incremental leads/mqls:**
# * Double click into `Linkedin` campaigns to identify growth opportunities.
# * Double clicks into `Facebook` campaigns to reduce wasted spend. 

#
# ### By `ad_library_type`:

# +
ad_type_agg_df = (
    raw_campaign_df.groupby("ad_library_type")
    .agg(**AGGREGATION_COLS)
    .pipe(compute_derived_kpis)
    .sort_values("spent", ascending=False)
    .reset_index()
)
ad_type_agg_df["efficiency"] = gmean(
    ad_type_agg_df[["leads", "cvr", "inv_cpmql"]],
    axis=1,
    weights=[3, 5, 2],
    nan_policy="propagate",
)


display(
    show(
        ad_type_agg_df[["ad_library_type"] + DISPLAY_COLS].sort_values(
            "efficiency", ascending=False
        )
    )
)


ad_type_agg_df[[f"percent_{col}" for col in PERCENTAGE_COLS]] = (
    100 * ad_type_agg_df[PERCENTAGE_COLS] / ad_type_agg_df[PERCENTAGE_COLS].sum()
)
fig = px.bar(
    ad_type_agg_df,
    x="ad_library_type",
    y=[f"percent_{col}" for col in PERCENTAGE_COLS],
    title="Campaign Performance by Ad Type",
)
fig.update_layout(yaxis_title="Percentage Contribution")

fig.show()
# -

# * `Image` ads account for 93% of the total impressions and 75% of the total leads across all marketing campaigns
# * `Image` and `Convo` ads have garnered 647 MQLs, accounting for 96.5% of the total MQLs.
# * `Convo` ads have a conversion rate of 88%, the highest amongst all ad types. On the other hand, `CAROUSEL` is the worst perfoming ad type with a conversion rate of 2%.
# * `VIDEO` ads have a high conversion rate of 76% but a high cost per lead of $901.
# * `CAROUSEL` ads have the worst cost per mql of \$4922.95.
#
#
# **Potential Actionables for driving incremental leads/mqls:**
# * Allocate more maketing budget to `CONVO` ads to drive efficient mqls.
# * Double click into IMAGE ads to identify opportunities for fine tuning audience targeting. 
# * Fine tune the audience targeting for `VIDEO` ads to lower the cost per lead. 
# * Consider Pausing `CAROUSEL` ads. 


# ### By `ad_library_type`, `wiz_campaign_channel` and `experiment_goal`:

# +
agg_df = (
    raw_campaign_df.groupby(
        ["wiz_campaign_channel", "ad_library_type", "experiment_goal"]
    )
    .agg(**AGGREGATION_COLS)
    .pipe(compute_derived_kpis)
    .sort_values("spent", ascending=False)
    .reset_index()
)
agg_df["efficiency"] = gmean(
    agg_df[["leads", "cvr", "inv_cpmql"]],
    axis=1,
    weights=[3, 5, 2],
    nan_policy="propagate",
)

pivot_df = pd.pivot(
    agg_df[
        ["wiz_campaign_channel", "ad_library_type", "experiment_goal"] + DISPLAY_COLS
    ],
    index=["wiz_campaign_channel", "ad_library_type"],
    columns=["experiment_goal"],
    values=DISPLAY_COLS,
).fillna(0)


for experiment_goal, group_df in agg_df.groupby("experiment_goal"):

    group_df[[f"percent_{col}" for col in PERCENTAGE_COLS]] = (
        100 * group_df[PERCENTAGE_COLS] / group_df[PERCENTAGE_COLS].sum()
    )
    group_df["x_label"] = (
        group_df["wiz_campaign_channel"] + "_" + group_df["ad_library_type"]
    )
    fig = px.bar(
        group_df,
        x="x_label",
        y=[f"percent_{col}" for col in PERCENTAGE_COLS],
        title=f"Campaign Performance by Channel + Ad Type for Experiment Goal: {experiment_goal}",
    )
    fig.update_layout(
        yaxis_title="Percentage Contribution", xaxis_title="Channel + Adtype"
    )

    display(fig.show())


show(pivot_df)
# -

#
# ### Insights:
#
# * `ad_library_type` is a function of `wiz_campaign_channel`.
#   * All ad types are NOT supported/leveraged across campaign channels.
#   * For instance, `CAROUSEL` ads are only supported/leveraged on `Facebook` and `Linkedin`. Similarly `TEXT` ads are only supported on `Google`.
#   * `IMAGE` ads have the highest coverage with presence across 3 channels- `Facebook`, `Instagram` and `Linkedin`.
# * `FACEBOOK - IMAGE` ads are responsible for **61%** of the impressions while driving only **15.4%** of the total MQLs.
#   * However, 55% of the total impressions for Facebook IMAGE campaigns are attributed to campaigns being optimized for awareness with no lead/MQL measurement.
#   * Excluding brandareness campaigns, `FACEBOOK IMAGE` ads optimized for CPL has the second best conversion rate of **30%** and cost per MQL of **\$331**.
# * `CONVO` ads on `Linkedin` have the highest conversion rate of 88% resulting in a cost per MQL of **\$147**.
# * `LinkedIn Image` ads optimized for CPL have a high conversion rate of 25% accompanied by a high cost per lead of **\$193**.
# * `Instagram Video` ads optimized for CPL are probably an **anomaly** with a 100% conversion rate. 
#
# **Potential Actionables for driving incremental leads/mqls:**
# * Double Down on Lead generation `Facebook Image` and `LinkedIn CONVO` cmapigns to drive quality leads that maximize the probability of driving MQLs.
# * Boost the overall efficiency of `LinkedIn Image` ads optimized for lead generation by reducing wasted ad spends. 


# ### Analysis by Day of Week 
#

# +
dow_agg_df = (
    raw_campaign_df.groupby(["day_of_week"])
    .agg(**AGGREGATION_COLS)
    .pipe(compute_derived_kpis)
    .sort_values("spent", ascending=False)
    .reset_index()
)
dow_agg_df["efficiency"] = gmean(
    dow_agg_df[["leads", "cvr", "inv_cpmql"]],
    axis=1,
    weights=[3, 5, 2],
    nan_policy="propagate",
)

display(
    show(
        dow_agg_df[["day_of_week"] + DISPLAY_COLS].sort_values(
            "efficiency", ascending=False
        )
    )
)


dow_agg_df[[f"percent_{col}" for col in PERCENTAGE_COLS]] = (
    100 * dow_agg_df[PERCENTAGE_COLS] / dow_agg_df[PERCENTAGE_COLS].sum()
)
fig = px.bar(
    dow_agg_df,
    x="day_of_week",
    y=[f"percent_{col}" for col in PERCENTAGE_COLS],
    title="Campaign Performance by Day of Week",
)
fig.update_layout(yaxis_title="Percentage Contribution")
fig.update_xaxes(
    categoryorder="array",
    categoryarray=[
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
)
fig.show()
# -

# ### Insights:
# * Tuesday, Wednesday, Thursday and Friday drive more than 70% of the MQLS at a conversion rate of 33.2%
# * Weekend has the lowest conversion rate of 23%
# ### Actionables:
# * Reallocate budgets from weekends to weekdays
