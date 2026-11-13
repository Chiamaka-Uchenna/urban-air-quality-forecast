"""Shared helpers for the Streamlit app: data loading, model scoring and chart styling."""
import json
import math
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#77766f"


@st.cache_data
def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def margin(model, x):
    """Sum the leaf values of the gradient boosted trees exported by the notebook.

    XGBoost compares 32-bit floats, so inputs are rounded the same way here.
    A missing value (None or NaN) follows the tree's default direction.
    """
    total = model["base"]
    for tree in model["trees"]:
        i = 0
        while tree["f"][i] != -1:
            value = x[tree["f"][i]]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                i = tree["m"][i]
            else:
                i = tree["y"][i] if float(np.float32(value)) < tree["t"][i] else tree["n"][i]
        total += tree["v"][i]
    return total


def predict(model, x):
    z = margin(model, x)
    return 1 / (1 + math.exp(-z)) if model.get("objective") == "logistic" else z


def vector(model, values):
    """Feature vector in the model's column order. Features not supplied default to 0."""
    return [values.get(name, 0) for name in model["features"]]


def style(fig, height=340, **layout):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12, x=0),
                      colorway=[BLUE, ORANGE, AQUA, "#eda100"], **layout)
    return fig


def bar(x, y, title_x="", title_y="", horizontal=False, reference=None, color=BLUE, height=340):
    fig = go.Figure(go.Bar(x=y if horizontal else x, y=x if horizontal else y, orientation="h" if horizontal else "v", marker_color=color))
    if reference is not None:
        if horizontal:
            fig.add_vline(x=reference, line_dash="dot", line_color=GREY)
        else:
            fig.add_hline(y=reference, line_dash="dot", line_color=GREY)
    fig.update_xaxes(title=title_x, type=None if horizontal else "category")
    fig.update_yaxes(title=title_y)
    return style(fig, height)


def lines(series, title_x="", title_y="", height=340, diagonal=None):
    """series: list of (name, x, y, color, dash)."""
    fig = go.Figure()
    if diagonal is not None:
        fig.add_trace(go.Scatter(x=[0, diagonal], y=[0, diagonal], mode="lines", name="Reference", line=dict(color=GREY, dash="dot")))
    for name, x, y, color, dash in series:
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, dash=dash, width=2)))
    fig.update_xaxes(title=title_x)
    fig.update_yaxes(title=title_y)
    return style(fig, height)


def show(fig):
    st.plotly_chart(fig, width="stretch")
