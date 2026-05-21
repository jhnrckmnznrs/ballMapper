# ballMapper

`ballMapper` is a Python implementation of the **Ball Mapper** algorithm for building graph-based summaries of high-dimensional point cloud data.

This implementation focuses on efficient cover construction using:

- `scikit-learn`'s `BallTree`
- Optional `FAISS` support for faster nearest-neighbor searches on larger datasets
- Greedy landmark selection
- Deterministic farthest point sampling
- NetworkX graph construction
- Matplotlib and Plotly visualization utilities
- Node coloring by size, density, labels, entropy, or custom functions

---

## What is Ball Mapper?

Ball Mapper is a topological data analysis method that summarizes the shape of a dataset as a graph.

Given a point cloud and a radius `eps`, the algorithm:

1. Selects representative points called **landmarks**
2. Builds an `eps`-ball around each landmark
3. Creates a graph where each node represents a ball
4. Connects two nodes if their balls share at least one data point

The resulting graph can reveal clusters, branches, loops, transitions, and other large-scale geometric structure in the data.

---

## Features

- Compute Ball Mapper landmarks using a greedy covering strategy
- Compute landmarks using deterministic farthest point sampling
- Build Ball Mapper graphs using `networkx`
- Use `BallTree` for efficient radius queries
- Optionally use `FAISS` for Euclidean or cosine search
- Color graph nodes by:
  - Custom function values
  - Most common class label
  - Label entropy
  - Ball size
  - Ball density
- Visualize graphs with:
  - Matplotlib
  - Interactive Plotly figures
- Export interactive visualizations to HTML

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ballMapper.git
cd ballMapper
```

Install the required dependencies:

```bash
pip install numpy networkx matplotlib plotly scikit-learn
```

Optional FAISS support:

```bash
pip install faiss-cpu
```

For GPU-enabled FAISS, follow the official FAISS installation instructions for your platform.

---

## Dependencies

Required:

```text
numpy
networkx
matplotlib
plotly
scikit-learn
```

Optional:

```text
faiss-cpu
```

`FAISS` is only required when using `method="faiss"`.

---

## Basic Usage

### 1. Generate or load data

```python
import numpy as np

np.random.seed(42)

# Example point cloud with 500 samples and 2 features
X = np.random.rand(500, 2)
```

`X` should be a NumPy array with shape:

```text
(n_samples, n_features)
```

---

## Compute Landmarks and Covers

The package provides two main landmark construction methods:

1. `computeLandmarks`
2. `computeLandmarksFPS`

---

## Method 1: Greedy Landmark Selection

The greedy method iterates through the dataset and selects the first uncovered point as a new landmark. It then marks all points within distance `eps` as covered.

```python
from ballMapper import computeLandmarks

eps = 0.1

landmarks, cover = computeLandmarks(
    X,
    eps=eps,
    method="ballTree",
    metric="euclidean",
    leafSize=40,
)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `X` | `np.ndarray` | required | Input data of shape `(n_samples, n_features)` |
| `eps` | `float` | required | Radius of each Ball Mapper ball |
| `method` | `"ballTree"` or `"faiss"` | `"ballTree"` | Backend used for radius queries |
| `metric` | `str` | `"euclidean"` | Distance metric for BallTree or FAISS |
| `leafSize` | `int` | `40` | Leaf size used by BallTree |

### Return values

```python
landmarks, cover
```

where:

- `landmarks` is a list of landmark point indices
- `cover` is a list of NumPy arrays
- each `cover[i]` contains the indices of data points inside the ball around `landmarks[i]`

---

## Method 2: Farthest Point Sampling

Farthest point sampling chooses landmarks by repeatedly selecting the point farthest from the current set of landmarks.

```python
from ballMapper import computeLandmarksFPS

landmarks, cover = computeLandmarksFPS(
    X,
    eps=eps,
    start_index=None,
    method="ballTree",
    metric="euclidean",
    leafSize=40,
)
```

When `start_index=None`, the algorithm starts from the lexicographically smallest point, giving deterministic behavior.

### Parameters

| Parameter | Type | Default | Description |
|---|---:|---:|---|
| `X` | `np.ndarray` | required | Input data |
| `eps` | `float` | required | Radius used for the cover |
| `start_index` | `int` or `None` | `None` | Optional starting point for FPS |
| `method` | `"ballTree"` or `"faiss"` | `"ballTree"` | Backend used to build the final cover |
| `metric` | `str` | `"euclidean"` | Distance metric |
| `leafSize` | `int` | `40` | BallTree leaf size |

---

## Using FAISS

FAISS can be used instead of BallTree for faster similarity search on larger datasets.

### Euclidean FAISS

```python
landmarks, cover = computeLandmarks(
    X,
    eps=0.1,
    method="faiss",
    metric="euclidean",
)
```

For Euclidean FAISS, `eps` is interpreted as a Euclidean distance radius.

---

### Cosine FAISS

```python
landmarks, cover = computeLandmarks(
    X,
    eps=0.1,
    method="faiss",
    metric="cosine",
)
```

For cosine FAISS, the data is internally normalized and neighbors are selected using:

```text
cosine_similarity >= 1 - eps
```

So `eps` behaves like a cosine-distance radius.

---

## Build the Ball Mapper Graph

Once you have a cover, build the Ball Mapper graph:

```python
from ballMapper import buildMapper

G = buildMapper(cover)
```

The graph is a `networkx.Graph`.

- Each node represents one landmark ball
- An edge connects two balls if they contain at least one shared data point

```python
print(G.number_of_nodes())
print(G.number_of_edges())
```

---

## Complete Minimal Example

```python
import numpy as np
from ballMapper import computeLandmarks, buildMapper

np.random.seed(42)
X = np.random.rand(500, 2)

eps = 0.1

landmarks, cover = computeLandmarks(
    X,
    eps=eps,
    method="ballTree",
)

G = buildMapper(cover)

print(f"Number of landmarks: {len(landmarks)}")
print(f"Number of graph nodes: {G.number_of_nodes()}")
print(f"Number of graph edges: {G.number_of_edges()}")
```

---

## Node Coloring

The package includes several helper functions for assigning values to Ball Mapper nodes.

---

### Color by Ball Size

```python
from ballMapper import colorBySize

sizes = colorBySize(cover)
```

This returns the number of data points contained in each ball.

---

### Color by Density

```python
from ballMapper import colorByDensity

density = colorByDensity(cover)
```

This normalizes ball sizes by the largest ball size.

---

### Color by a Custom Function

```python
from ballMapper import colorByFunction

colors = colorByFunction(X, cover, func=np.mean)
```

This applies a function to the points contained in each ball.

For example, to color by the average first coordinate:

```python
colors = colorByFunction(X[:, 0], cover, func=np.mean)
```

---

### Color by Most Common Label

```python
from ballMapper import colorByMode

y = np.random.randint(0, 3, size=X.shape[0])

colors = colorByMode(y, cover)
```

This assigns each node the most frequent label among the points in its ball.

---

### Color by Label Entropy

```python
from ballMapper import colorByEntropy

entropy = colorByEntropy(y, cover)
```

This measures how mixed the labels are inside each ball.

Low entropy means a ball mostly contains one class.

High entropy means a ball contains a mixture of classes.

---

## Matplotlib Visualization

Use `drawBallMapper` for a static NetworkX/Matplotlib visualization.

```python
import matplotlib.pyplot as plt
from ballMapper import drawBallMapper, addColorbar, colorBySize

sizes = colorBySize(cover)

fig, ax = plt.subplots(figsize=(8, 6))

pos, nodes = drawBallMapper(
    G,
    colors=sizes,
    sizes=sizes,
    layout="spring",
    cmap="viridis",
    with_labels=True,
    node_scale=500,
    ax=ax,
)

addColorbar(nodes, ax, label="Ball size")

plt.show()
```

### Available layouts

```python
layout="spring"
layout="kamada_kawai"
layout="spectral"
```

---

## Interactive Plotly Visualization

Use `drawBallMapperPlotly` for interactive visualizations with hover text and optional dropdown colorings.

```python
from ballMapper import (
    drawBallMapperPlotly,
    colorBySize,
    colorByDensity,
    colorByFunction,
)

size_coloring = colorBySize(cover)
density_coloring = colorByDensity(cover)
mean_x_coloring = colorByFunction(X[:, 0], cover, func=np.mean)

colorings = {
    "Ball size": size_coloring,
    "Density": density_coloring,
    "Mean x-coordinate": mean_x_coloring,
}

drawBallMapperPlotly(
    G,
    cover,
    colorings=colorings,
    sizes=size_coloring,
    layout="spring",
    node_scale=25,
)
```

---

## Export Plotly Graph to HTML

```python
drawBallMapperPlotly(
    G,
    cover,
    colorings=colorings,
    sizes=size_coloring,
    export_html="ball_mapper_graph.html",
)
```

This creates a standalone interactive HTML file.

---

## Edge Overlaps

You can compute how many data points are shared between connected balls:

```python
from ballMapper import computeEdgeOverlaps

overlaps = computeEdgeOverlaps(cover, G)

for edge, overlap_size in overlaps.items():
    print(edge, overlap_size)
```

This can be useful for measuring how strongly two Ball Mapper nodes are connected.

---

## Full Example with Coloring and Plotting

```python
import numpy as np
import matplotlib.pyplot as plt

from ballMapper import (
    computeLandmarks,
    buildMapper,
    colorBySize,
    colorByDensity,
    colorByFunction,
    drawBallMapper,
    addColorbar,
    drawBallMapperPlotly,
)

# Generate data
np.random.seed(42)
X = np.random.rand(500, 2)

# Build Ball Mapper cover
eps = 0.1

landmarks, cover = computeLandmarks(
    X,
    eps=eps,
    method="ballTree",
    metric="euclidean",
)

# Build graph
G = buildMapper(cover)

# Compute node colorings
sizes = colorBySize(cover)
density = colorByDensity(cover)
mean_x = colorByFunction(X[:, 0], cover, func=np.mean)

# Static visualization
fig, ax = plt.subplots(figsize=(8, 6))

pos, nodes = drawBallMapper(
    G,
    colors=mean_x,
    sizes=sizes,
    layout="spring",
    cmap="viridis",
    with_labels=True,
    node_scale=500,
    ax=ax,
)

addColorbar(nodes, ax, label="Mean x-coordinate")

plt.show()

# Interactive visualization
colorings = {
    "Mean x-coordinate": mean_x,
    "Ball size": sizes,
    "Density": density,
}

drawBallMapperPlotly(
    G,
    cover,
    colorings=colorings,
    sizes=sizes,
    layout="spring",
    node_scale=25,
    export_html="ball_mapper.html",
)
```

---

## Choosing `eps`

The parameter `eps` controls the size of the balls.

Smaller `eps` values usually produce:

- More landmarks
- More graph nodes
- Finer local structure
- Potentially fragmented graphs

Larger `eps` values usually produce:

- Fewer landmarks
- Fewer graph nodes
- More global structure
- More overlap between balls

Example:

```python
for eps in [0.05, 0.1, 0.2]:
    landmarks, cover = computeLandmarks(X, eps=eps)
    G = buildMapper(cover)

    print(
        f"eps={eps}: "
        f"{G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges"
    )
```

---

## BallTree vs FAISS

### Use BallTree when:

- Your dataset is small to medium sized
- You want support for scikit-learn distance metrics
- You do not want to install FAISS
- You need a simple CPU-based backend

### Use FAISS when:

- Your dataset is large
- You want faster Euclidean or cosine similarity search
- You are working with high-dimensional vector embeddings
- You already use FAISS in your workflow

---

## API Reference

### `computeLandmarks`

```python
computeLandmarks(
    X,
    eps,
    method="ballTree",
    metric="euclidean",
    leafSize=40,
)
```

Computes landmarks and their cover sets using a greedy strategy.

Returns:

```python
landmarks, cover
```

---

### `computeLandmarksFPS`

```python
computeLandmarksFPS(
    X,
    eps,
    start_index=None,
    method="ballTree",
    metric="euclidean",
    leafSize=40,
)
```

Computes landmarks using deterministic farthest point sampling, then builds the cover.

Returns:

```python
landmarks, cover
```

---

### `buildMapper`

```python
buildMapper(cover)
```

Builds a NetworkX graph from the cover.

Returns:

```python
G
```

---

### `colorByFunction`

```python
colorByFunction(X, cover, func=np.mean)
```

Colors nodes by applying a function to the points inside each ball.

---

### `colorByMode`

```python
colorByMode(y, cover)
```

Colors nodes by the most common label in each ball.

---

### `colorByEntropy`

```python
colorByEntropy(y, cover)
```

Colors nodes by label entropy.

---

### `colorBySize`

```python
colorBySize(cover)
```

Colors nodes by number of covered points.

---

### `colorByDensity`

```python
colorByDensity(cover)
```

Colors nodes by normalized ball size.

---

### `drawBallMapper`

```python
drawBallMapper(
    G,
    colors=None,
    sizes=None,
    layout="spring",
    cmap="viridis",
    with_labels=True,
    node_scale=300,
    ax=None,
)
```

Draws a static Ball Mapper graph using Matplotlib.

Returns:

```python
pos, nodes
```

---

### `addColorbar`

```python
addColorbar(nodes, ax, label=None)
```

Adds a colorbar to a Matplotlib Ball Mapper plot.

---

### `computeEdgeOverlaps`

```python
computeEdgeOverlaps(cover, G)
```

Computes the number of shared points for each edge.

---

### `drawBallMapperPlotly`

```python
drawBallMapperPlotly(
    G,
    cover,
    colorings=None,
    sizes=None,
    layout="spring",
    node_scale=20,
    export_html=None,
)
```

Creates an interactive Plotly Ball Mapper visualization.

---

## Common Errors

### `ImportError: scikit-learn is required for BallTree method`

Install scikit-learn:

```bash
pip install scikit-learn
```

---

### `ImportError: FAISS is required for method='faiss'`

Install FAISS:

```bash
pip install faiss-cpu
```

---

### `ValueError: Method must be 'ballTree' or 'faiss'`

Use one of the supported backends:

```python
method="ballTree"
```

or:

```python
method="faiss"
```

When using FAISS, the supported metrics are:

```python
metric="euclidean"
metric="cosine"
```

---

## Notes

- `cover[i]` contains the indices of the data points covered by node `i`.
- Graph node IDs correspond to positions in the `cover` list, not directly to original data indices.
- To retrieve the landmark point for graph node `i`, use:

```python
landmark_index = landmarks[i]
landmark_point = X[landmark_index]
```

- FAISS internally converts data to `float32`.
- Cosine FAISS normalizes the input vectors before indexing.
- Plotly visualizations call `fig.show()` and can optionally export to HTML.

---

## Example Project Structure

```text
ballMapper/
├── ballMapper.py
├── README.md
├── examples/
│   ├── basic_usage.py
│   ├── fps_usage.py
│   └── plotly_visualization.py
└── requirements.txt
```

Example `requirements.txt`:

```text
numpy
networkx
matplotlib
plotly
scikit-learn
```

Optional FAISS dependency:

```text
faiss-cpu
```

---

## License

```text
MIT License
```

---

## Citation

Please cite the original Ball Mapper paper.

---

## Contributing

Contributions are welcome.

Suggested improvements include:

- More visualization options
- Benchmark examples
- Unit tests
- Documentation examples
- GPU FAISS support examples

---

## Acknowledgments

This project is inspired by the Ball Mapper algorithm from topological data analysis and uses open-source Python tools including NumPy, NetworkX, scikit-learn, FAISS, Matplotlib, and Plotly.
