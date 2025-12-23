from collections import defaultdict

import networkx as nx
import numpy as np
import plotly.graph_objects as go

# Optional imports
try:
    from sklearn.neighbors import BallTree
except ImportError:
    BallTree = None

try:
    import faiss
except ImportError:
    faiss = None


# =====================================================
# Landmark Computation
# =====================================================


def computeLandmarks(
    X: np.ndarray,
    eps: float,
    method: Literal["ballTree", "faiss"] = "ballTree",
    metric: str = "euclidean",
    leafSize: int = 40,
) -> Tuple[List[int], List[np.ndarray]]:
    """
    Compute landmarks and their coverage sets from data points.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (n_samples, n_features) representing the data points.
    eps : float
        Radius used to define the neighborhood of each landmark.
    method : str, default="ballTree"
        Backend method for nearest neighbor search. Options are:
        - "ballTree": uses scikit-learn's BallTree.
        - "faiss": uses FAISS library for faster searches on large datasets.
    metric : str, default="minkowski"
        Distance metric for BallTree. Ignored if method="faiss".

    Returns
    -------
    landmarks : list[int]
        Indices of selected landmark points.
    cover : list[np.ndarray]
        List of arrays, each containing the indices of points covered by the corresponding landmark.

    Raises
    ------
    ValueError
        If `method` is not one of {"ballTree", "faiss"}.
    ImportError
        If required libaissrary (scikit-learn or faiss) is not installed.

    Example
    -------
    >>> X = np.random.rand(100, 3)
    >>> landmarks, cover = computeLandmarks(X, eps=0.2, method="ballTree")
    """
    method = method.lower()

    if method == "balltree" or method == "ballTree".lower():
        return _computeLandmarksBallTree(X, eps, metric, leafSize)

    elif method == "faiss":
        return _computeLandmarksFAISS(X, eps)

    else:
        raise ValueError("method must be 'ballTree' or 'faiss'.")


def _computeLandmarksBallTree(X, eps, metric, leafSize):
    """
    Internal helper: compute landmarks using scikit-learn BallTree.
    """
    if BallTree is None:
        raise ImportError("scikit-learn is required for BallTree method.")

    n = X.shape[0]
    tree = BallTree(X, metric=metric, leaf_size=leafSize)

    uncovered = np.ones(n, dtype=bool)
    landmarks, cover = [], []

    while np.any(uncovered):
        i = np.argmax(uncovered)
        landmarks.append(i)
        idx = tree.query_radius(X[i : i + 1], eps)[0]
        cover.append(idx)
        uncovered[idx] = False

    return landmarks, cover


def _computeLandmarksFAISS(X, eps):
    """
    Internal helper: compute landmarks using FAISS.
    """
    if faiss is None:
        raise ImportError("FAISS is required for method='faiss'.")

    points = X.astype(np.float32)
    index = faiss.IndexFlatL2(points.shape[1])
    index.add(points)

    covered = np.zeros(len(points), dtype=bool)
    landmarks, cover = [], []

    for i in range(len(points)):
        if not covered[i]:
            landmarks.append(i)
            lims, _, I = index.range_search(points[i].reshape(1, -1), eps**2)
            pts = I[lims[0] : lims[1]]
            cover.append(pts)
            covered[pts] = True

    return landmarks, cover


# Include farthest point sampling
def computeLandmarkFPS(X, eps, start_index=None, use_faiss=False):
    """
    Deterministic farthest point sampling AND epsilon-ball cover.

    Parameters
    ----------
    X : array (n_samples, n_features)
    eps : float
        Radius for Ball Mapper cover.
    start_index : int or None
        Deterministic starting point (default = lexicographically smallest).
    use_faiss : bool
        If True, use FAISS for radius queries (requires faiss).

    Returns
    -------
    landmarks : list[int]
        Indices of landmark points.
    cover : list[np.ndarray]
        List where cover[i] contains indices of all points within eps of landmarks[i].
    """

    n = X.shape[0]

    # --------------------------------------------------------------
    # 1) FPS landmark selection (same optimized algorithm as before)
    # --------------------------------------------------------------

    if start_index is None:
        start_index = np.lexsort(X.T)[0]

    norms = np.sum(X * X, axis=1)

    def sqdist_to(i):
        return norms + norms[i] - 2 * (X @ X[i])

    dists = sqdist_to(start_index)
    landmarks = [start_index]
    eps2 = eps * eps

    while True:
        next_index = np.argmax(dists)
        max_dist = dists[next_index]

        if max_dist <= eps2:
            break

        landmarks.append(next_index)
        dists = np.minimum(dists, sqdist_to(next_index))

    # --------------------------------------------------------------
    # 2) Build cover (epsilon neighborhoods) using BallTree or FAISS
    # --------------------------------------------------------------

    cover = []

    if not use_faiss:
        # --- BallTree path (recommended for most cases) ---
        tree = BallTree(X)
        for idx in landmarks:
            inds = tree.query_radius(X[idx].reshape(1, -1), r=eps)[0]
            cover.append(inds)

    else:
        # --- FAISS path (GPU or CPU exact L2) ---
        import faiss

        X32 = X.astype(np.float32)
        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X32)

        for idx in landmarks:
            D, I = index.range_search(X32[idx].reshape(1, -1), eps2)
            cover.append(I[0])

    return landmarks, cover


# =====================================================
# Ball Mapper Graph Construction
# =====================================================


def buildMapper(cover):
    """
    Build a Ball Mapper graph using an inverted index for faster edge computation.

    Parameters
    ----------
    cover : list[np.ndarray]
        List of arrays containing indices of points covered by each landmark.

    Returns
    -------
    G : networkx.Graph
        Graph object with:
        - Nodes representing landmarks
        - Edges connecting landmarks that share at least one point.

    Notes
    -----
    This function is faster than buildMapper for large datasets, as it avoids
    repeated set intersections.

    Example
    -------
    >>> landmarks, cover = computeLandmarks(X, eps=0.2)
    >>> G = buildMapperFast(cover)
    """
    G = nx.Graph()
    n = len(cover)
    G.add_nodes_from(range(n))
    pointToCovers = defaultdict(list)

    for coverId, points in enumerate(cover):
        for p in points:
            pointToCovers[p].append(coverId)

    for covers in pointToCovers.values():
        for i in range(len(covers)):
            for j in range(i + 1, len(covers)):
                G.add_edge(covers[i], covers[j])

    return G


# =====================================================
# Ball Mapper Coloring
# =====================================================


def colorByFunction(X, cover, func=np.mean):
    """
    Color Ball Mapper nodes using a function applied to covered points.

    Parameters
    ----------
    X : np.ndarray
        Data array (n_samples, n_features) or scalar values (n_samples,).
    cover : list[np.ndarray]
        Cover sets from landmarks.
    func : callable, default=np.mean
        Function applied to X[cover[i]].

    Returns
    -------
    colors : np.ndarray
        One value per landmark (node).
    """
    colors = np.zeros(len(cover))

    for i, pts in enumerate(cover):
        if len(pts) > 0:
            colors[i] = func(X[pts])
        else:
            colors[i] = np.nan

    return colors


from collections import Counter


def colorByMode(y, cover):
    """
    Assign each ball the most frequent label among covered points.
    """
    colors = np.zeros(len(cover), dtype=int)

    for i, pts in enumerate(cover):
        if len(pts) > 0:
            colors[i] = Counter(y[pts]).most_common(1)[0][0]
        else:
            colors[i] = -1

    return colors


def colorByEntropy(y, cover):
    """
    Color balls by label entropy (heterogeneity).
    """

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return -np.sum(p * np.log2(p + 1e-12))

    colors = np.zeros(len(cover))

    for i, pts in enumerate(cover):
        if len(pts) > 0:
            colors[i] = entropy(y[pts])
        else:
            colors[i] = 0.0

    return colors


def colorBySize(cover):
    """
    Color balls by number of covered points.
    """
    return np.array([len(c) for c in cover])


def colorByDensity(cover):
    sizes = np.array([len(c) for c in cover])
    return sizes / sizes.max()


# =====================================================
# Ball Mapper with Colors
# =====================================================


def drawBallMapper(
    G,
    colors=None,
    sizes=None,
    layout="spring",
    cmap="viridis",
    with_labels=True,
    node_scale=300,
    ax=None,
):
    """
    Draw a Ball Mapper graph.

    Parameters
    ----------
    G : networkx.Graph
    colors : array-like or None
        Optional node colors.
    sizes : array-like or None
        Optional node sizes.
    layout : str
    cmap : str
    with_labels : bool
    node_scale : float
    ax : matplotlib.axes.Axes or None

    Returns
    -------
    pos : dict
    nodes : matplotlib.collections.PathCollection
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G)
    else:
        raise ValueError("Unknown layout")

    # Sizes
    if sizes is None:
        sizes = np.ones(len(G))
    sizes = node_scale * (np.asarray(sizes) / np.max(sizes))

    # Colors
    if colors is None:
        node_kwargs = dict(node_color="lightgray")
    else:
        node_kwargs = dict(node_color=colors, cmap=cmap)

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=sizes,
        ax=ax,
        **node_kwargs,
    )

    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    ax.set_axis_off()
    return pos, nodes


def addColorbar(nodes, ax, label=None):
    """
    Add a colorbar if node colors are present.
    """
    if not hasattr(nodes, "cmap") or nodes.get_array() is None:
        return  # No colors → no colorbar

    sm = plt.cm.ScalarMappable(cmap=nodes.cmap, norm=nodes.norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    if label:
        cbar.set_label(label)


def computeEdgeOverlaps(cover, G):
    """
    Compute overlap size for each edge in the Ball Mapper graph.
    """
    overlaps = {}
    for u, v in G.edges():
        overlaps[(u, v)] = len(np.intersect1d(cover[u], cover[v]))
    return overlaps


def drawBallMapperPlotly(
    G,
    cover,
    colorings=None,
    sizes=None,
    layout="spring",
    node_scale=20,
    export_html=None,
):
    """
    Full-featured interactive Ball Mapper visualization.

    Parameters
    ----------
    G : networkx.Graph
    cover : list[np.ndarray]
        Cover sets for hover info and overlap computation.
    colorings : dict or None
        {name: array-like} for dropdown coloring selection.
    sizes : array-like or None
        Node sizes (e.g., ball sizes).
    layout : str
    node_scale : float
    export_html : str or None
        Path to save interactive HTML.
    """

    # -----------------------
    # Layout
    # -----------------------
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Unknown layout")

    # -----------------------
    # Edge overlaps → thickness
    # -----------------------
    overlaps = computeEdgeOverlaps(cover, G)
    max_overlap = max(overlaps.values()) if overlaps else 1

    edge_traces = []
    for (u, v), w in overlaps.items():
        edge_traces.append(
            go.Scatter(
                x=[pos[u][0], pos[v][0]],
                y=[pos[u][1], pos[v][1]],
                mode="lines",
                line=dict(width=1 + 4 * w / max_overlap, color="gray"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # -----------------------
    # Node sizes
    # -----------------------
    if sizes is None:
        sizes = np.array([len(c) for c in cover])
    sizes = node_scale * sizes / sizes.max()

    # -----------------------
    # Hover text
    # -----------------------
    hover_text = [f"Ball {i}<br>Points: {len(cover[i])}" for i in range(len(cover))]

    node_x = [pos[i][0] for i in G.nodes()]
    node_y = [pos[i][1] for i in G.nodes()]

    # -----------------------
    # Colorings
    # -----------------------
    if colorings is None:
        colorings = {"None": None}

    traces = []
    buttons = []

    for i, (name, values) in enumerate(colorings.items()):
        marker = dict(
            size=sizes,
            line=dict(width=1, color="black"),
        )

        if values is None:
            marker["color"] = "lightgray"
            marker["showscale"] = False
        else:
            marker["color"] = values
            marker["colorscale"] = "Viridis"
            marker["showscale"] = True
            marker["colorbar"] = dict(title=name)

        trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=marker,
            hoverinfo="text",
            text=hover_text,
            visible=(i == 0),
            showlegend=False,
        )

        traces.append(trace)

        buttons.append(
            dict(
                label=name,
                method="update",
                args=[
                    {
                        "visible": [True] * len(edge_traces)
                        + [j == i for j in range(len(traces))]
                    },
                ],
            )
        )

    # -----------------------
    # Figure
    # -----------------------
    fig = go.Figure(
        data=edge_traces + traces,
        layout=go.Layout(
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    x=0.02,
                    y=0.98,
                )
            ],
            hovermode="closest",
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False,
        ),
    )

    if export_html:
        fig.write_html(export_html)

    fig.show()
