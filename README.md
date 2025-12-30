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
