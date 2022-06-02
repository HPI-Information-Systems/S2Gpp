from .s2gpp import s2gpp_local_array, s2gpp_local_file, s2gpp_distributed_main, s2gpp_distributed_sub
from sklearn.base import BaseEstimator
from typing import Optional
from multiprocessing import cpu_count
from enum import Enum
import numpy as np
from pathlib import Path


class Clustering(Enum):
    KDE = "kde"
    MeanShift = "meanshift"


class Series2GraphPP(BaseEstimator):
    def __init__(self,
                 pattern_length: int,
                 latent: Optional[int] = None,
                 rate: int = 100,
                 query_length: Optional[int] = None,
                 n_threads: int = -1,
                 clustering: Clustering = Clustering.KDE,
                 # explainability: bool = False,
                 self_correction: bool = False,
                 local_host="127.0.0.1:1992"
                 ):
        self.pattern_length = pattern_length
        self.latent = latent or int(self.pattern_length / 3)
        self.rate = rate
        self.query_length = query_length or self.pattern_length
        self.n_threads = n_threads if n_threads > 0 else min(cpu_count() - 1, 1)
        self.clustering = clustering
        self.self_correction = self_correction
        self.local_host = local_host

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)

        return s2gpp_local_array(
            X,
            self.pattern_length,
            self.latent,
            self.query_length,
            self.rate,
            self.n_threads,
            self.clustering.value,
            self.self_correction
        )


class DistributedRole(Enum):
    Main = "main"
    Sub = "sub"


class DistributedSeries2GraphPP(Series2GraphPP):
    def __init__(self, _role: DistributedRole, n_cluster_nodes: int, mainhost: Optional[str] = None, output_path: Path = Path("./anomaly_scores.ts"), column_start_idx: int = 0, column_end_idx: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._role: DistributedRole = _role
        self.output_path: Path = output_path
        self.column_start: int = column_start_idx
        self.column_end: int = column_end_idx
        self.n_cluster_nodes: int = n_cluster_nodes
        self.mainhost: str = mainhost

    def fit_predict(self, X: Optional[Path] = None):
        if self._role == DistributedRole.Main:
            s2gpp_distributed_main(
                str(X),
                self.pattern_length,
                self.latent,
                self.query_length,
                self.rate,
                self.n_threads,
                str(self.output_path),
                self.column_start,
                self.column_end,
                self.clustering.value,
                self.self_correction,
                self.local_host,
                self.n_cluster_nodes
            )
        else:  # self._role == DistributedRole.Sub:
            s2gpp_distributed_sub(
                self.pattern_length,
                self.latent,
                self.query_length,
                self.rate,
                self.n_threads,
                str(self.output_path),
                self.column_start,
                self.column_end,
                self.clustering.value,
                self.self_correction,
                self.local_host,
                self.n_cluster_nodes,
                self.mainhost
            )

    @staticmethod
    def main(*args, **kwargs) -> 'DistributedSeries2GraphPP':
        return DistributedSeries2GraphPP(
            *args,
            _role=DistributedRole.Main,
            **kwargs
        )

    @staticmethod
    def sub(*args, **kwargs) -> 'DistributedSeries2GraphPP':
        return DistributedSeries2GraphPP(
            *args,
            _role=DistributedRole.Sub,
            **kwargs
        )
