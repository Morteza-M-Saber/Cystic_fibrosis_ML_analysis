from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial import distance


def clustermap_plot(
    genetic_infile: pd.DataFrame,
    metadata_infile: pd.DataFrame,
    outfile: Path,
    method: str = "average",
    metric: str = "cityblock",
    cmap: str = "viridis",
    figsize: List(float) = (12 / 2.54, 16 / 2.54),
) -> None:
    """Plot clustermap

    Args:
        genetic_infile (pd.DataFrame): path to genetic data
        metadata_infile (pd.DataFrame): path to metadata
        outfile (Path): path to outfile
        method (str, optional): Defaults to "average".
        metric (str, optional): Defaults to "cityblock".
        cmap (str, optional):  Defaults to "viridis".
        figsize (List, optional):  Defaults to (12 / 2.54, 16 / 2.54).
    """
    plt.figure(figsize=figsize)
    # heatmap
    genetic_data_table = pd.read_csv(genetic_infile)
    metadata_table = pd.read_csv(metadata_infile)
    my_palette1 = {"PES": "r", "Unique": "b"}
    row_colors1 = metadata_infile.PFGE_typing.map(my_palette1)

    # FEVp
    my_palette2 = {"Severe": "g", "Mild": "y"}
    row_colors2 = metadata_infile.FEVp_binary.astype(int).replace(0, "Mild").replace(1, "Severe").map(my_palette2)

    # lung decline
    my_palette3 = {"Rapid": "m", "Non-rapid": "orange"}
    row_colors3 = (
        metadata_infile.Relative_Rate_of_lung_function_decline.astype(int)
        .replace(0, "Non-rapid")
        .replace(1, "Rapid")
        .map(my_palette3)
    )

    my_palette1.update(my_palette2)
    my_palette1.update(my_palette3)
    row_colors = pd.concat([row_colors1, row_colors2, row_colors3], axis=1)

    # calculate correlations
    correlations_array = (
        genetic_data_table.corr()
    )  # np.asarray(), skip converting to array because we want column names
    # matrix = np.triu(correlations_array)
    row_linkage = hierarchy.linkage(distance.pdist(correlations_array), method=method)

    col_linkage = hierarchy.linkage(distance.pdist(correlations_array.T), method=method)
    g = sns.clustermap(
        correlations_array,
        row_colors=row_colors,
        row_linkage=row_linkage,
        col_linkage=col_linkage,
        method=method,
        metric=metric,
        figsize=figsize,
        cbar_pos=(0.025, 0.91, 0.025, 0.12),  # mask=correlations_array < 0.5,
        cmap=cmap,
        xticklabels=False,
        yticklabels=False,
    )
    ax = g.ax_heatmap
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax1 = g.ax_row_colors
    # ax1.set_xticklabels('')
    # ax.set_xticklabels("")
    # ax.set_yticklabels("")

    # adding labels for the phenotypes
    for label in ["PES", "Unique", "Severe", "Mild", "Rapid", "Non-rapid"]:
        g.ax_col_dendrogram.bar(0, 0, color=my_palette1[label], label=label, linewidth=0)

    g.ax_col_dendrogram.legend(
        ncol=3,
        frameon=False,
        bbox_to_anchor=(1.1, 1.7),
        fancybox=True,
        shadow=True,
        fontsize=9,
    )
    plt.tight_layout()
    g.savefig(outfile, dpi=600)
