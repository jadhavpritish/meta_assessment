import os
import tempfile
from typing import Any, Optional

import pandas as pd
from humanfriendly import module as humanfriendly
from IPython.display import HTML, Markdown


def display_pivot_ui(df: pd.DataFrame, **kwargs: Any) -> None:
    """
    Helper wrapper around pivottablejs which adds support for nbconvert.

    Unless you explicitly pass show_pivot_table_help=False, help markdown is output on the first run.

    See https://github.com/nicolaskruchten/pivottable/wiki/Parameters#options-object-for-pivotui

    A common set of invocation args might be:

        display_pivot_ui(
            df,
            rows=["campaign_name"],
            cols=["match_type"],
            aggregatorName="Sum",
            vals=["spend"],
            rendererName="Heatmap",
            hiddenFromDragDrop=list(tmp_df.columns_glob(*standard_perf_columns, *derived_perf_columns)),
            hiddenFromAggregators=list(set(tmp_df.columns) - set(standard_perf_columns)),
        )
    """
    from pivottablejs import pivot_ui

    with tempfile.NamedTemporaryFile("rt") as output_file:
        pivot_ui_result = pivot_ui(df, outfile_path=output_file.name, **kwargs)
        print(
            f"Pivot table HTML size: {humanfriendly.format_size(os.path.getsize(output_file.name))}"
        )

        pivot_html = output_file.read()

    # We need to embedd the generated html file directly in the output to work with nbconvert, wrap in an iframe for correct styling
    pivot_html = pivot_html.replace('"', "&quot;")
    display(
        HTML(
            pivot_ui_result._repr_html_().replace(
                f'src="{output_file.name}"', f'srcdoc="{pivot_html}"'
            )
        )
    )
