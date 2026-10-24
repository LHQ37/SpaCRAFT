"""Shared, deterministic spatial-domain palette for Figure 2 panels a, c and d."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree


# Medium-depth qualitative colours drawn from Tableau-style scientific palettes.
# The pale companion colours in tab20 are deliberately excluded because they lose
# separation when the spatial maps are reduced to manuscript size.
DOMAIN_COLORS = (
    "#4E79A7",  # blue
    "#E15759",  # red
    "#59A14F",  # green
    "#F28E2B",  # orange
    "#B07AA1",  # purple
    "#76B7B2",  # teal
    "#9C755F",  # brown
    "#D37295",  # rose
    "#79706E",  # grey
    "#B6992D",  # olive
    "#2F8AC4",  # azure
    "#8B5FBF",  # violet
    "#3A9D78",  # blue-green
    "#C86B32",  # burnt orange
)


def spatial_domain_colors(labels, spatial_coordinates, k=6):
    """Assign distinct colours while separating domains that touch in space.

    Labels and coordinates are never changed.  Spatial adjacency is used only to
    choose a deterministic permutation of the fixed qualitative palette.
    """
    labels = np.asarray(labels).astype(str)
    coordinates = np.asarray(spatial_coordinates, dtype=float)[:, :2]
    levels = pd.Categorical(labels).categories.astype(str).tolist()
    if len(levels) > len(DOMAIN_COLORS):
        raise ValueError(
            f"Figure 2 palette supports {len(DOMAIN_COLORS)} domains; "
            f"received {len(levels)}"
        )

    level_index = {level: i for i, level in enumerate(levels)}
    encoded = np.asarray([level_index[value] for value in labels], dtype=int)
    neighbour_count = min(k + 1, len(coordinates))
    neighbours = cKDTree(coordinates).query(coordinates, k=neighbour_count)[1]
    if neighbours.ndim == 1:
        neighbours = neighbours[:, None]

    adjacency = np.zeros((len(levels), len(levels)), dtype=float)
    for spot, row in enumerate(neighbours):
        for neighbour in np.atleast_1d(row):
            if int(neighbour) == spot:
                continue
            left, right = encoded[spot], encoded[int(neighbour)]
            if left != right:
                adjacency[left, right] += 1.0
                adjacency[right, left] += 1.0

    palette_rgb = np.asarray([to_rgb(value) for value in DOMAIN_COLORS])
    colour_distance = np.linalg.norm(
        palette_rgb[:, None, :] - palette_rgb[None, :, :], axis=2
    )
    assigned = {}
    unused = list(range(len(DOMAIN_COLORS)))
    first = int(np.argmax(adjacency.sum(axis=1)))
    assigned[first] = unused.pop(0)

    while len(assigned) < len(levels):
        remaining = [i for i in range(len(levels)) if i not in assigned]
        node = max(
            remaining,
            key=lambda i: (
                adjacency[i, list(assigned)].sum(),
                adjacency[i].sum(),
                -i,
            ),
        )

        def score(colour):
            touching = [j for j in assigned if adjacency[node, j] > 0]
            if touching:
                weighted = [
                    adjacency[node, j] * colour_distance[colour, assigned[j]]
                    for j in touching
                ]
                return (min(weighted), sum(weighted), -colour)
            used = list(assigned.values())
            return (min(colour_distance[colour, used]), 0.0, -colour)

        chosen = max(unused, key=score)
        assigned[node] = chosen
        unused.remove(chosen)

    return {
        level: DOMAIN_COLORS[assigned[level_index[level]]]
        for level in levels
    }
