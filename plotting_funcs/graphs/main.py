import plotly.graph_objects as go
from typing import Union
from numpy import ndarray
from torch import Tensor


# Maybe set the default custom size for all plots and then individual sizes can be given to each lot

def list_plot(x_data: Union[ndarray, Tensor], y_data: Union[ndarray, Tensor], name="") -> go.Scatter:
    return go.Scatter(y=y_data, x=x_data, mode="markers", marker=dict(size=3), name=name)


def line_plot(x_data: Union[ndarray, Tensor], y_data: Union[ndarray, Tensor], name="") -> go.Scatter:
    return go.Scatter(y=y_data, x=x_data, mode="lines", marker=dict(size=3), name=name)


def show(graphs: [go.Scatter], layout: go.Layout = None, title: str = "", height=700, width=1_000):
    go.Figure(
        data=graphs,
        layout=go.Layout(title=dict(text=title, x=0.5), height=height, width=width) if (layout is None) else layout
    ).show()

# Plot of changing loss function during training
