"""Helper functions to generate ROC-style plots of various metrics."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List, Union

import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from torch import Tensor


def plot_figure(
    x_vals: Union[Tensor, List[Tensor]],
    y_vals: Union[Tensor, List[Tensor]],
    auc: Tensor,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    xlabel: str,
    ylabel: str,
    loc: str,
    title: Union[str, List[str]],
    sample_points: int = 1000,
) -> Tuple[Figure, Axis]:
    """Generate a simple, ROC-style plot, where x_vals is plotted against y_vals.

    Note that a subsampling is applied if > sample_points are present in x/y, as matplotlib plotting draws
    every single plot which takes very long, especially for high-resolution segmentations.

    Args:
        x_vals (Tensor): x values to plot
        y_vals (Tensor): y values to plot
        auc (Tensor): normalized area under the curve spanned by x_vals, y_vals
        xlim (Tuple[float, float]): displayed range for x-axis
        ylim (Tuple[float, float]): displayed range for y-axis
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        loc (str): string-based legend location, for details see
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): title of the plot
        sample_points (int): number of sampling points to subsample x_vals/y_vals with

    Returns:
        Tuple[Figure, Axis]: Figure and the contained Axis
    """
    num_curves = len(x_vals) if isinstance(x_vals, list) else 1
    fig, axis = plt.subplots()

    # put all variables in lists if necessary
    aucs = [auc] if len(auc.shape) == 0 else auc
    x_vals = [x_vals] if not isinstance(x_vals, list) else x_vals
    y_vals = [y_vals] if not isinstance(y_vals, list) else y_vals

    for x, y, auc in zip(x_vals, y_vals, aucs):
        x = x.detach().cpu()
        y = y.detach().cpu()

        if sample_points < x.size(0):
            possible_idx = range(x.size(0))
            interval = len(possible_idx) // sample_points

            idx = [0]  # make sure to start at first point
            idx.extend(possible_idx[::interval])
            idx.append(possible_idx[-1])  # also include last point

            idx = torch.tensor(
                idx,
                device=x.device,
            )
            x = torch.index_select(x, 0, idx)
            y = torch.index_select(y, 0, idx)

        axis.plot(
            x,
            y,
            figure=fig,
            lw=2,
            label=f"AUC: {auc.detach().cpu():0.2f}",
        )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc=loc)
    axis.set_title(title)

    return fig, axis
