# -*- coding: utf-8 -*-
import pandas as pd


def style_table(sdf, bar_color="forestgreen", max_width=None, bar_subset=None):

    scores_subset = pd.IndexSlice[:, ("Scores", slice(None))]

    if bar_subset is None:
        bar_subset = scores_subset

    styler = (
        sdf.style
        .format("{:.0%}", subset=scores_subset)
        .format("{:.0f}", subset=pd.IndexSlice[:, ("Count", slice(None))])
        .bar(bar_subset, vmin=0, vmax=1, color=bar_color, height=80)

        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("background", "#d3d3d3"),
                    ("color", "#333333"),
                    ("font-weight", "400"),
                    ("text-align", "center"),
                    ("padding", "8px"),
                ],
            },

            {
                "selector": "thead tr:first-child th[colspan]:not([colspan='1'])",
                "props": [("position", "relative")],
            },
            {
                "selector": "thead tr:first-child th[colspan]:not([colspan='1'])::after",
                "props": [
                    ("content", "''"),
                    ("position", "absolute"),
                    ("left", "5%"),
                    ("right", "5%"),
                    ("bottom", "1px"),
                    ("border-bottom", "1px solid #b0b0b0"),
                ],
            },

            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "8px"),
                    ("border-bottom", "1px solid #ededed"),
                ],
            },

            {
                "selector": "thead th:first-child",
                "props": [
                    ("text-align", "left"),
                    ("padding-left", "12px"),
                ],
            },
            {
                "selector": "tbody td:first-child",
                "props": [
                    ("text-align", "left"),
                    ("padding-left", "12px"),
                ],
            },
        ])

        .set_properties(scores_subset, color="white")
        .hide(axis="index")
    )

    style_attr = "margin-left:auto;margin-right:auto;"
    if max_width:
        style_attr += f"max-width:{max_width};"

    return styler.set_table_attributes(f'style="{style_attr}"')
