import numpy as np

def compute_stats(values):
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95))
    }