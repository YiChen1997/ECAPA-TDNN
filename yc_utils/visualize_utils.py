import os

import umap
import torch
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial import ConvexHull
from sklearn.decomposition import TruncatedSVD


def to_numpy(arr):
    """
    Convert the given array to the numpy format
    """
    if isinstance(arr, np.ndarray):
        return arr
    elif isinstance(arr, list):
        return np.array(arr)
    elif isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return None


def reduce(embeddings, n_components=2, reduction_method="umap", seed=42):
    """
    Applies the selected dimensionality reduction technique
    to the given input data
    """
    reducer = None
    assert reduction_method in ("svd", "tsne", "umap"), "Unsupported reduction method"
    if reduction_method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=seed)
    elif reduction_method == "tsne":
        reducer = TSNE(n_components=n_components, metric="cosine", random_state=seed)
    elif reduction_method == "umap":
        reducer = umap.UMAP(
            n_components=n_components, metric="cosine", random_state=seed
        )
    return reducer.fit_transform(embeddings)


def visualize_embeddings(
        embeddings,
        labels,
        labels_mapping=None,
        reduction_method="umap",
        remove_outliers=False,
        only_centroids=True,
        convex_hull=False,
        figsize=(12, 10),
        legend=False,
        show=True,
        save=None,
):
    """
    Plot the given embedding vectors, after reducing them to 2D
    """
    # Convert embeddings and labels to numpy
    embeddings, labels = to_numpy(embeddings), to_numpy(labels)

    # Check inputs
    assert (
            len(embeddings.shape) == 2 and embeddings.shape[1] > 1
    ), "Wrong embeddings format/dimension"
    assert (
            len(labels.shape) == 1 and labels.shape[0] == embeddings.shape[0]
    ), "Wrong labels format/dimension"
    assert not (
            only_centroids and convex_hull
    ), "Cannot compute convex hull when only centroids are displayed"

    # Compute dimesionality reduction to 2D
    if embeddings.shape[1] > 2:
        embeddings = reduce(
            embeddings, n_components=2, reduction_method=reduction_method
        )

    # Store embeddings in a dataframe and compute cluster colors
    embeddings_df = pd.DataFrame(embeddings, columns=["x", "y"], dtype=np.float32)
    embeddings_df["l"] = np.expand_dims(labels, axis=-1)
    cluster_colors = {l: np.random.random(3) for l in np.unique(labels)}
    embeddings_df["c"] = embeddings_df.l.map(
        {l: tuple(c) for l, c in cluster_colors.items()}
    )

    # Plot embeddings and centroids
    fig, ax = plt.subplots(figsize=figsize)
    for l, c in cluster_colors.items():
        to_plot = embeddings_df[embeddings_df.l == l]
        label = labels_mapping[l] if labels_mapping is not None else l
        ax.scatter(
            to_plot.x.mean(),
            to_plot.y.mean(),
            color=c,
            label=f"{label} (C)",
            marker="^",
            s=250,
        )
        if not only_centroids:
            ax.scatter(to_plot.x, to_plot.y, color=c, label=f"{label}")

    # Do not represent outliers
    if remove_outliers:
        xmin_quantile = np.quantile(embeddings[:, 0], q=0.01)
        xmax_quantile = np.quantile(embeddings[:, 0], q=0.99)
        ymin_quantile = np.quantile(embeddings[:, 1], q=0.01)
        ymax_quantile = np.quantile(embeddings[:, 1], q=0.99)
        ax.set_xlim(xmin_quantile, xmax_quantile)
        ax.set_ylim(ymin_quantile, ymax_quantile)

    # Plot a shaded polygon around each cluster
    if convex_hull:
        for l, c in cluster_colors.items():
            try:
                # Get the convex hull
                points = embeddings_df[embeddings_df.l == l][["x", "y"]].values
                hull = ConvexHull(points)
                x_hull = np.append(
                    points[hull.vertices, 0], points[hull.vertices, 0][0]
                )
                y_hull = np.append(
                    points[hull.vertices, 1], points[hull.vertices, 1][0]
                )

                # Interpolate to get a smoother figure
                dist = np.sqrt(
                    (x_hull[:-1] - x_hull[1:]) ** 2 + (y_hull[:-1] - y_hull[1:]) ** 2
                )
                dist_along = np.concatenate(([0], dist.cumsum()))
                spline, _ = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0)
                interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
                interp_x, interp_y = interpolate.splev(interp_d, spline)

                # Plot the smooth polygon
                ax.fill(interp_x, interp_y, "--", color=c, alpha=0.2)
            except:
                continue

    # Spawn the plot
    if legend:
        plt.legend()
    if save is not None:
        os.makedirs(os.path.dirname(save), exist_ok=True)
        plt.savefig(save)
    if show:
        plt.show()
    else:
        plt.close(fig)