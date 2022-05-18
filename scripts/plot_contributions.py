import numpy as np
import matplotlib.pyplot as plt

def reverse_windowing(scores: np.ndarray, window_size: int, reduction: str = "mean") -> np.ndarray:
    unwindowed_length = (window_size - 1) + len(scores)
    mapped = np.zeros((unwindowed_length, window_size))
    for p, s in enumerate(scores):
        assignment = np.eye(window_size) * s
        mapped[p:p + window_size] += assignment
    if reduction == "mean":
        # transform non-score zeros to NaNs
        h = np.tril([1.] * window_size, 0)
        h[h == 0] = np.nan
        h2 = np.triu([1.] * window_size, 0)
        h2[h2 == 0] = np.nan
        mapped[:window_size] *= h
        mapped[-window_size:] *= h2
        mapped = np.nanmean(mapped, axis=1)
    elif reduction == "sum":
        mapped = mapped.sum(axis=1)
    return mapped

s = np.genfromtxt("anomaly-contribution.csv", delimiter=",")
a = np.genfromtxt("anomaly_scores.ts", delimiter=",")
scores = []
for d in range(s.shape[1]):
    scores.append(reverse_windowing(s[:, d], window_size=10000 - len(s[:, d]) + 1))
for d, s in enumerate(scores):
    plt.plot(s, label=f"{d}")
plt.plot(a, label="anomaly score")
plt.legend()
plt.show()
