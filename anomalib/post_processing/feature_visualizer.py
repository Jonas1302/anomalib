import colorsys
import pickle
from itertools import zip_longest

import math
from typing import Union, Dict, Tuple, List, Optional

import numpy as np
import torch
import umap
import umap.plot
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# for umap tutorial see https://umap-learn.readthedocs.io/en/latest/basic_usage.html

class FeatureVisualizer:
    def __init__(self, *reducers):
        self.reducers = reducers
        self.reducer_uptodate = True
        self.features: Dict[str, List[np.ndarray]] = {}
        self.values: Dict[str, List[np.ndarray]] = {}

    def add_features(self, features: Union[np.ndarray, torch.Tensor], label: Union[str, List[str], np.ndarray] = 0,
                     values: Optional[Union[float, np.ndarray]] = None):
        """Add features to be visualized. Features with the same `label` will have the same color."""
        assert self.values is None or values is not None, "either all features must have values or none of them"

        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if len(features.shape) == 1:  # make sure all features have two dimensions because we'll concatenate them
            features = [features]


        if isinstance(label, str):
            self.features.setdefault(label, []).append(features)
            if values is not None:
                if isinstance(values, (int, float)):
                    values = [values]
                self.values.setdefault(label, []).append(values)
            self.reducer_uptodate = False
        else:  # recursive
            assert len(label.shape) == 1, "category must either be `int` or an one-dimensional array"
            assert len(values) == len(features), f"{len(values)} != {len(features)}"

            for i in range(len(label)):
                self.add_features(features[i], label[i], values[i] if isinstance(values, (list, np.ndarray)) else values)

    def _sorted_flattened_coordinates_and_labels(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        # the value entries are currently list of two-dimensional arrays
        # => concatenate them so that we have one numpy array per label
        concatenated_features = {label: np.concatenate(features_list) for label, features_list in self.features.items()}
        # sort coordinates and labels such that labels with more points are drawn first
        # (and thus rare labels are shown on top of them in case they overlap)
        if self.values:
            concatenated_values = {label: np.concatenate(values_list) for label, values_list in self.values.items()}
            tuples = [(l, f, v) for (l, f), v in zip(concatenated_features.items(), concatenated_values.values())]
            labels, sorted_features, values = zip(*sorted(tuples, key=lambda x: len(x[1]), reverse=True))
        else:
            labels, sorted_features = zip(*sorted(concatenated_features.items(), key=lambda x: len(x[1]), reverse=True))
            values = []

        # create a single two-dimensional array of all features
        flattened_features = np.concatenate(sorted_features)
        # same for values
        if values:
            values = np.concatenate(values)
        # create a verbose array of labels with one entry per coordinate (like [1, 1, 1, 0, 0, 2] for 6 coordinates and 3 labels)
        labels = np.concatenate([np.full(len(concatenated_features[label]), label) for label in labels])

        # apply dimensionality reductions
        coordinates = flattened_features
        for reducer in self.reducers:
            coordinates = reducer.fit_transform(coordinates)
        result = {}
        for c, l, v in zip_longest(coordinates, labels, values):
            result.setdefault(l, []).append((c, v))
        for l in result:
            features, values = zip(*result[l])
            result[l] = (np.stack(features), np.stack(values))
        return result

    def get_pyplot(self, **pyplot_args) -> plt.Figure:
        """Process all features and returns a pyplot figure."""
        coordinates = self._sorted_flattened_coordinates_and_labels()

        # generate point colors
        color_mapper = ColorMapper(list(self.features.keys()))  # use `self.features` instead of `coordinates` to keep the order of the labels
        #print(color_mapper.labels)
        #print(color_mapper.nested_labels)
        #print({l: colorsys.rgb_to_hsv(*color_mapper.get_color(l)) for l in color_mapper.labels})
        #print({l: color_mapper.get_color(l) for l in color_mapper.labels})
        #categories = {k: LinearSegmentedColormap.from_list("", ["white", v]) for k, v in categories.items()}

        # create the plot
        max_marker_size = 1
        fig = plt.Figure(figsize=(19.20 / 2, 10.80 / 2), dpi=800)
        axes = fig.gca()
        # plot points
        # TODO maybe use axes.contour? (https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contour.html)
        for label in coordinates:
            c, values = coordinates[label]
            if values[0] is None:
                pyplot_args['color'] = color_mapper.get_color(label)
                pyplot_args['s'] = max_marker_size / max(math.log(len(c) / 10, 100), 1)
            else:
                pyplot_args['c'] = values
                pyplot_args['cmap'] = color_mapper.get_colormap(label)
                pyplot_args['s'] = max_marker_size / max(math.log(sum(values) / 10, 100), 1)
            print(len(c), max_marker_size / max(math.log(len(c) / 10, 100), 1), pyplot_args['s'])

            axes.scatter(
                c[:, 0],
                c[:, 1],
                linewidths=0,
                label=label,
                **pyplot_args)

        # create legend
        legend = axes.legend(fontsize='xx-small', markerscale=5/max_marker_size)
        for lh in legend.legendHandles:  # from https://stackoverflow.com/a/42403471
            lh.set_alpha(1)

        # other stuff
        axes.set_aspect('equal', 'datalim')
        axes.axis('off')
        return fig


class FeatureVisualizerUMAP(FeatureVisualizer):
    def __init__(self):
        super().__init__(umap.UMAP())

    def get_pyplot(self, **pyplot_args) -> plt.Figure:
        coordinates, labels = self._sorted_flattened_coordinates_and_labels()

        fig = plt.Figure(figsize=(5, 5), dpi=800)
        ax = umap.plot.points(self.reducers[-1],
                              points=coordinates,
                              labels=labels,
                              theme="inferno",
                              ax=plt.Axes(fig, (0., 0., 1., 1.)))
        fig.add_axes(ax)
        return fig


class FeatureVisualizerTSNE(FeatureVisualizer):
    def __init__(self):
        super().__init__(PCA(n_components=50), TSNE())


class ColorMapper:
    def __init__(self, labels: List[str]):
        self.labels = labels

        # create a dict with categories as keys and sets of anomaly types as values
        # this allows us to easily count the number of categories as well as anomaly types per category
        self.nested_labels = {}
        for label in labels:
            if "." in label:
                category, label = label.split(".")
                self.nested_labels.setdefault(category, dict())[label] = None  # use a dict as ordered set (see https://stackoverflow.com/q/1653970)
            else:
                self.nested_labels.setdefault(label, None)

    def _get_hsv_color(self, label: str) -> Tuple[float, float, float]:
        # check if we have more than two categories and more than one anomaly type per category
        # if so, ensure that different anomaly types of the same category have more similar colors
        if len(self.nested_labels) > 2 and max(map(len, self.nested_labels.values())) > 100:
            if "." in label:
                category, label = label.split(".")
            else:
                category, label = label, None
            category_index = list(self.nested_labels.keys()).index(category)
            # divides the hue into equidistant hue values
            hue_stepsize = 1. / len(self.nested_labels)
            hue = hue_stepsize * category_index
            # alternate value if there are many labels (and therefore close hue values)
            value = 1. - 0.3 * (category_index % 2) if len(self.nested_labels) > 8 else 1.

            if label:
                num_category_labels = len(self.nested_labels[category])
                if num_category_labels > 1:
                    label_index = list(self.nested_labels[category].keys()).index(label)
                    hue += (hue_stepsize * label_index / num_category_labels - hue_stepsize / 2) / 1.2
                    hue %= 1
                    value -= 0.2 * (label_index % 2)
            return hue, 1., value
        else:
            index = self.labels.index(label)
            # divides the hue into equidistant hue values
            hue = 1. / len(self.labels) * index
            # alternate value (1.0 and 0.6) if there are many labels (and therefore close hue values)
            value = 1. - 0.2 * (index % 2) if len(self.labels) > 6 else 1.
            return hue, 1., value

    def get_color(self, label: str) -> Tuple[float, float, float]:
        """Returns a color for the given label based on the labels provided during initialization.

        Labels can either be strings without dots or a category and anomaly type seperated by a dot.
        The colors are usually equidistant. If there are multiple categories and anomaly types,
        the color of anomaly types of the same category are more similar.
        """
        return colorsys.hsv_to_rgb(*self._get_hsv_color(label))

    def get_colormap(self, label: str):
        """Return a colormap using `get_color(label)` with an increasing alpha-value."""
        color = self.get_color(label)
        return LinearSegmentedColormap.from_list("", [(*color, 0), (*color, 1)])
