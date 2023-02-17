import colorsys
import math
from typing import Union, Dict, Tuple, List, Any

import numpy as np
import torch
import umap
import umap.plot
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# for umap tutorial see https://umap-learn.readthedocs.io/en/latest/basic_usage.html

class VisualizerEntry:
    def __init__(self, labels: Union[str, List[str]], features: np.ndarray, values: Union[int, float, np.ndarray]):
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if len(features.shape) == 1:  # make sure all features have two dimensions because we'll concatenate them
            features = np.expand_dims(features, 0)
        assert isinstance(labels, str) or len(labels) == len(features)
        assert isinstance(values, (int, float)) or len(values) == len(features)

        self.labels = labels
        self.features = features
        self.values = values

    def content(self):
        return self.labels, self.features, self.values

    @property
    def expanded_labels(self):
        return self.labels if isinstance(self.labels, (list, np.ndarray)) else np.full(len(self.features), self.labels)

    @property
    def expanded_values(self):
        return self.values if isinstance(self.values, np.ndarray) else np.full(len(self.features), self.values)

    @staticmethod
    def merge(*instances):
        labels, features, values = zip(*map(VisualizerEntry.content, instances))
        expand_labels = any([isinstance(l, list) or l != labels[0] for l in labels])
        expand_values = any([isinstance(v, np.ndarray) or v != values[0] for v in values])
        new_labels = np.concatenate([e.expanded_labels for e in instances]) if expand_labels else labels[0]
        new_values = np.concatenate([e.expanded_values for e in instances]) if expand_values else values[0]
        new_features = np.concatenate(features)

        assert isinstance(new_labels, str) or len(new_labels) == len(new_features)
        assert isinstance(new_values, (int, float)) or len(new_values) == len(new_features)
        return VisualizerEntry(new_labels, new_features, new_values)

    @classmethod
    def create_by_label(cls, labels: List[str], features: np.ndarray, values: np.ndarray) -> "Dict[Any, VisualizerEntry]":
        assert len(labels) == len(features) == len(values)
        result = {}
        grouped_features = {}
        grouped_values = {}
        for l, f, v in zip(labels, features, values):
            grouped_features.setdefault(l, []).append(f)
            grouped_values.setdefault(l, []).append(v)
        for l in grouped_features:
            result[l] = VisualizerEntry(l, np.stack(grouped_features[l]), np.stack(grouped_values[l]))
        return result


class FeatureVisualizer:
    def __init__(self, *reducers):
        self.reducers = reducers
        self.reducer_uptodate = True
        self.entries: List[VisualizerEntry] = []

    def add_features(self, features: Union[np.ndarray, torch.Tensor], label: Union[str, List[str]] = 0,
                     values: Union[float, np.ndarray] = 1):
        """Add features to be visualized. Features with the same `label` will have the same color."""
        self.entries.append(VisualizerEntry(label, features, values))
        self.reducer_uptodate = False

    def _sorted_flattened_coordinates_and_labels(self) -> Dict[str, VisualizerEntry]:
        # the value entries are currently list of two-dimensional arrays
        # => concatenate them so that we have one numpy array per label
        entries_by_label = VisualizerEntry.create_by_label(*VisualizerEntry.merge(*self.entries).content())
        # sort coordinates and labels such that labels with more points are drawn first
        # (and thus rare labels are shown on top of them in case they overlap)
        entries_by_label = {l: e for l, e in sorted(entries_by_label.items(), key=lambda i: sum(i[1].expanded_values), reverse=True)}

        # apply dimensionality reductions
        concatenated_entries = VisualizerEntry.merge(*entries_by_label.values())
        coordinates = concatenated_entries.features
        for reducer in self.reducers:
            coordinates = reducer.fit_transform(coordinates)

        return VisualizerEntry.create_by_label(concatenated_entries.labels, coordinates, concatenated_entries.values)

    def get_pyplot(self, **pyplot_args) -> plt.Figure:
        """Process all features and returns a pyplot figure."""
        coordinates = self._sorted_flattened_coordinates_and_labels()

        # generate point colors
        color_mapper = ColorMapper(list(sorted(coordinates.keys())))

        # create the plot
        max_marker_size = 2
        fig = plt.Figure(figsize=(19.20 / 2, 10.80 / 2), dpi=800)
        axes = fig.gca()
        # plot points
        # TODO maybe use axes.contour? (https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.contour.html)
        for label, entry in coordinates.items():
            print(label, len(entry.expanded_values), sum(entry.expanded_values), max_marker_size / max(math.log(sum(entry.expanded_values), 100), 1))
            axes.scatter(
                entry.features[:, 0],
                entry.features[:, 1],
                linewidths=0,
                label=label,
                c=entry.expanded_values,
                cmap=color_mapper.get_colormap(label),
                s=max_marker_size / max(math.log(sum(entry.expanded_values), 100), 1),
                vmin=0,
                vmax=1,
                **pyplot_args)

        # create legend
        handles, labels = axes.get_legend_handles_labels()
        handles, labels = list(zip(*sorted(zip(handles, labels), key=lambda t: t[1])))
        legend = axes.legend(handles, labels, fontsize='xx-small', markerscale=6/max_marker_size)
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
        all_entries = VisualizerEntry.merge(*self._sorted_flattened_coordinates_and_labels().values())

        fig = plt.Figure(figsize=(19.20, 10.80), dpi=400)
        ax = umap.plot.points(self.reducers[-1],
                              points=all_entries.features,
                              labels=all_entries.expanded_labels,
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
