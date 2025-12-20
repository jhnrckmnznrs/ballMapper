from collections import defaultdict

import networkx as nx
import numpy as np

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


def computeLandmarks(X, eps, method="ballTree", metric="minkowski"):
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
        If required library (scikit-learn or faiss) is not installed.

    Example
    -------
    >>> X = np.random.rand(100, 3)
    >>> landmarks, cover = computeLandmarks(X, eps=0.2, method="ballTree")
    """
    method = method.lower()

    if method == "balltree" or method == "ballTree".lower():
        return _computeLandmarksBallTree(X, eps, metric)

    elif method == "faiss":
        return _computeLandmarksFaiss(X, eps)

    else:
        raise ValueError("method must be 'ballTree' or 'faiss'.")


def _computeLandmarksBallTree(X, eps, metric):
    """
    Internal helper: compute landmarks using scikit-learn BallTree.
    """
    if BallTree is None:
        raise ImportError("scikit-learn is required for BallTree method.")

    n = X.shape[0]
    tree = BallTree(X, metric=metric)

    uncovered = np.ones(n, dtype=bool)
    landmarks, cover = [], []

    while np.any(uncovered):
        i = np.argmax(uncovered)
        landmarks.append(i)
        idx = tree.query_radius(X[i : i + 1], eps)[0]
        cover.append(idx)
        uncovered[idx] = False

    return landmarks, cover


def _computeLandmarksFaiss(X, eps):
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
