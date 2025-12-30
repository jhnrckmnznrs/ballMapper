# ballMapper

This is an alternative Ball Mapper Python implementation that uses the Ball Tree data structure and Facebook AI Similarity Search to speedup the cover construction.

## Example Usage

This example demonstrates how to build and visualize a **Ball Mapper graph** from a point cloud.

---

### 1. Prepare the data

```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(500, 2)   # (n_samples, n_features)
```

---

### 2. Compute landmarks and covers

You can use either a greedy method or farthest point sampling.

#### Greedy Method (Ball Tree)

```python
from ballMapper import computeLandmarks

eps = 0.1
landmarks, cover = computeLandmarks(
    X,
    eps=eps,
    method="ballTree",   # or "faiss" if FAISS is installed
)
```

#### Farthest Point Sampling (Ball Tree)

```python
from ballMapper import computeLandmarkFPS

landmarks, cover = computeLandmarkFPS(
    X,
    eps=eps,
    start_index=None,    # deterministic default
    use_faiss=False
)
```

---

### 3. Build the Ball Mapper graph

```python
from ballMapper import buildMapper

G = buildMapper(cover)
```
