"""
Aggregated Constrained Mutual Subspace Method (AggregatedCMSM)

各クラスの学習データを複数クラスタに分割し、各クラスタごとに部分空間を算出した後、
それらをPCAなどで集約して代表的な部分空間を得るアプローチです。
最終的にはGDS射影（_gds_projection）を適用して、従来のCMSM同様にクラス間の差が最大となるようにします。
"""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as _normalize

from cvt_custom.models.cmsm import ConstrainedMSM
from cvt_custom.utils import subspace_bases
from lib import configure_logger
from lib.cluster.kmeans import CachedKMeans
from lib.cluster.spherical_kmeans import CachedSphericalKMeans, SphericalKMeans
from lib.timer import timer

log = configure_logger(__name__)


class AggregatedCMSM_SphericalKMeans(ConstrainedMSM):
    """
    Aggregated Constrained Mutual Subspace Method (AggregatedCMSM)
    Spherical K-means clustering is used for clustering.

    Args:
        n_subdims (int): 部分空間の次元数。
        n_gds_dims (int): GDS射影後の次元数。
        n_clusters (int): 各クラス内で分けるクラスタ数。1の場合、従来のCMSMと同等。
        normalize (bool): 特徴ベクトルを正規化するかどうか。
        cache_per_class (bool): 各クラスごとの部分空間をキャッシュとして保存するかどうか。
        cache_dir (Path or None): キャッシュ用のディレクトリ。cache_per_classがTrueの場合は指定必須。
        clustering_cache_dir (Path or None): クラスタリング結果のキャッシュディレクトリ。
        use_cached_clustering (bool): 事前計算されたクラスタリング結果を使用するかどうか。

    """

    def __init__(
        self,
        n_subdims,
        n_gds_dims,
        n_clusters=2,
        normalize=False,
        cache_per_class=False,
        cache_dir: Optional[Path] = None,
        clustering_cache_dir: Optional[Path] = None,
        use_cached_clustering: bool = False,
    ):
        super(AggregatedCMSM_SphericalKMeans, self).__init__(
            n_subdims, n_gds_dims, normalize
        )
        self.n_clusters = n_clusters
        self.cache_per_class = cache_per_class
        self.cache_dir = cache_dir
        self.clustering_cache_dir = clustering_cache_dir
        self.use_cached_clustering = use_cached_clustering

        self._dic_dict = {}  # 各クラス毎の代表基底を一時的に保持

    def aggregate_subspaces(self, subspaces):
        """
        複数の部分空間をPCAによって集約し、代表的な部分空間基底を生成する。

        Args:
            subspaces (list of np.ndarray): クラスタごとに算出された部分空間のリスト

        Returns:
            agg_basis (np.ndarray): 集約後の代表的部分空間基底
        """
        concatenated = np.concatenate(subspaces, axis=1)
        agg_basis = subspace_bases(concatenated, self.n_subdims)
        return agg_basis

    def fit_1_class(self, X_c, y_c):
        """
        1クラスの学習データに対して、クラスタリングを行い各クラスタの部分空間を算出、
        その後それらを集約して代表的な部分空間を得るメソッドです。
        また、cache_per_classがTrueの場合はキャッシュファイルへ保存します。

        Args:
            X_c (np.ndarray): クラス y_c の特徴行列。通常は (n_dims, n_samples) であることを想定しているが、
                (n_files, n_samples, n_dims) の場合はフラット化して (n_dims, n_files*n_samples) に変換する。
            y_c (int): クラスラベル。
        """
        assert X_c.ndim == 3, f"Invalid shape of input data: {X_c.shape}"
        init_X_c_shape = X_c.shape

        # (n_files, n_sample_times, n_dims) -> (n_dims, n_files*n_sample_times)
        X_c = X_c.transpose(2, 0, 1).reshape(X_c.shape[2], -1)

        if self.n_clusters <= 1 or X_c.shape[1] < self.n_clusters:
            basis = subspace_bases(X_c, self.n_subdims)
        else:
            # 特徴ベクトルを正規化
            X_c_normalized = _normalize(X_c.T)  # (n_samples, n_dims)

            spherical_kmeans = SphericalKMeans(
                n_clusters=self.n_clusters, random_state=0
            )

            # キャッシュを使用するSphericalKMeansを初期化
            if self.use_cached_clustering and self.clustering_cache_dir is not None:
                cached_spherical_kmeans = CachedSphericalKMeans(
                    X_c_shape=init_X_c_shape,
                    spherical_kmeans=spherical_kmeans,
                    cache_dir=self.clustering_cache_dir,
                )
                # クラスIDを含むキャッシュプレフィックスを生成
                cache_prefix = f"class_{y_c}"
                with timer(timer_name=f"fit_predict_with_cache: class {y_c}") as t:
                    labels = cached_spherical_kmeans.fit_predict_with_cache(
                        X_c_normalized, cache_prefix=cache_prefix
                    )
                log.info(f"fit_predict_with_cache: class {y_c} {t.elapsed:.3f}s")
            else:
                # キャッシュを使用しない通常のSphericalKMeans
                with timer(timer_name=f"fit_predict: class {y_c}") as t:
                    labels = spherical_kmeans.fit_predict(X_c_normalized)
                log.info(f"fit_predict: class {y_c} {t.elapsed:.3f}s")

            subspaces = []
            for cluster in np.unique(labels):
                cluster_indices = labels == cluster
                X_cluster = X_c[:, cluster_indices]
                # サンプル数が部分空間次元に満たない場合はスキップ
                if X_cluster.shape[1] >= self.n_subdims:
                    sub_basis = subspace_bases(X_cluster, self.n_subdims)
                    subspaces.append(sub_basis)
            if len(subspaces) == 0:
                basis = subspace_bases(X_c, self.n_subdims)
            else:
                basis = self.aggregate_subspaces(subspaces)

        # キャッシュが有効ならばファイルに保存
        if self.cache_per_class and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{y_c}.npy", basis)
        else:
            # 内部辞書に保存（キャッシュする場合はメモリに保存しない）
            self._dic_dict[y_c] = basis

    def fit_cache(self, y_list):
        """
        cache_per_classがTrueの場合、各クラスのキャッシュファイルから部分空間を読み込み、
        内部辞書 _dic_dict に格納する。

        Args:
            y_list (list): 学習に利用した各クラスのラベルのリスト
        """
        if not self.cache_per_class or self.cache_dir is None:
            return

        for y in y_list:
            file_path = self.cache_dir / f"{y}.npy"
            if file_path.exists():
                basis = np.load(file_path)
                self._dic_dict[y] = basis

    def finalize_fit(self):
        """
        すべてのクラスに対する学習後、内部辞書 _dic_dict から self.dic への変換を行う。
        self.dic は後続の _get_gramians などで利用される (n_classes, n_dims, n_subdims) の形状となる。
        """
        keys = sorted(self._dic_dict.keys())
        dic_list = [self._dic_dict[k] for k in keys]
        self.dic = np.stack(dic_list, axis=0)
        self.label_encoder = {i: key for i, key in enumerate(keys)}

    # 注意: _get_gramians は親クラス ConstrainedMSM から継承されるため、
    # self.dic に集約済みの代表部分空間が格納されていればそのまま GDS 射影を適用します。


class AggregatedCMSM_KMeans(ConstrainedMSM):
    """
    Aggregated Constrained Mutual Subspace Method (AggregatedCMSM)
    K-means clustering is used for clustering.

    Args:
        n_subdims (int): 部分空間の次元数。
        n_gds_dims (int): GDS射影後の次元数。
        n_clusters (int): 各クラス内で分けるクラスタ数。1の場合、従来のCMSMと同等。
        normalize (bool): 特徴ベクトルを正規化するかどうか。
        cache_per_class (bool): 各クラスごとの部分空間をキャッシュとして保存するかどうか。
        cache_dir (Path or None): キャッシュ用のディレクトリ。cache_per_classがTrueの場合は指定必須。
        clustering_cache_dir (Path or None): クラスタリング結果のキャッシュディレクトリ。
        use_cached_clustering (bool): 事前計算されたクラスタリング結果を使用するかどうか。

    """

    def __init__(
        self,
        n_subdims,
        n_gds_dims,
        n_clusters=2,
        normalize=False,
        cache_per_class=False,
        cache_dir: Optional[Path] = None,
        clustering_cache_dir: Optional[Path] = None,
        use_cached_clustering: bool = False,
    ):
        super(AggregatedCMSM_KMeans, self).__init__(n_subdims, n_gds_dims, normalize)
        self.n_clusters = n_clusters
        self.cache_per_class = cache_per_class
        self.cache_dir = cache_dir
        self.clustering_cache_dir = clustering_cache_dir
        self.use_cached_clustering = use_cached_clustering

        self._dic_dict = {}  # 各クラス毎の代表基底を一時的に保持

    def aggregate_subspaces(self, subspaces):
        """
        複数の部分空間をPCAによって集約し、代表的な部分空間基底を生成する。

        Args:
            subspaces (list of np.ndarray): クラスタごとに算出された部分空間のリスト

        Returns:
            agg_basis (np.ndarray): 集約後の代表的部分空間基底
        """
        concatenated = np.concatenate(subspaces, axis=1)
        agg_basis = subspace_bases(concatenated, self.n_subdims)
        return agg_basis

    def fit_1_class(self, X_c, y_c):
        """
        1クラスの学習データに対して、クラスタリングを行い各クラスタの部分空間を算出、
        その後それらを集約して代表的な部分空間を得るメソッドです。
        また、cache_per_classがTrueの場合はキャッシュファイルへ保存します。

        Args:
            X_c (np.ndarray): クラス y_c の特徴行列。通常は (n_dims, n_samples) であることを想定しているが、
                (n_files, n_samples, n_dims) の場合はフラット化して (n_dims, n_files*n_samples) に変換する。
            y_c (int): クラスラベル。
        """
        assert X_c.ndim == 3, f"Invalid shape of input data: {X_c.shape}"
        init_X_c_shape = X_c.shape

        # (n_files, n_sample_times, n_dims) -> (n_dims, n_files*n_sample_times)
        X_c = X_c.transpose(2, 0, 1).reshape(X_c.shape[2], -1)

        if self.n_clusters <= 1 or X_c.shape[1] < self.n_clusters:
            basis = subspace_bases(X_c, self.n_subdims)
        else:
            # 特徴ベクトルを正規化
            X_c_normalized = _normalize(X_c.T)  # (n_samples, n_dims)
            assert isinstance(X_c_normalized, np.ndarray)

            kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)

            # キャッシュを使用するSphericalKMeansを初期化
            if self.use_cached_clustering and self.clustering_cache_dir is not None:
                cached_kmeans = CachedKMeans(
                    X_c_shape=init_X_c_shape,
                    kmeans=kmeans,
                    cache_dir=self.clustering_cache_dir,
                )
                # クラスIDを含むキャッシュプレフィックスを生成
                cache_prefix = f"class_{y_c}"
                with timer(timer_name=f"fit_predict_with_cache: class {y_c}") as t:
                    labels = cached_kmeans.fit_predict_with_cache(
                        X_c_normalized, cache_prefix=cache_prefix
                    )
                log.info(f"fit_predict_with_cache: class {y_c} {t.elapsed:.3f}s")
            else:
                # キャッシュを使用しない通常のSphericalKMeans
                with timer(timer_name=f"fit_predict: class {y_c}") as t:
                    labels = kmeans.fit_predict(X_c_normalized)
                log.info(f"fit_predict: class {y_c} {t.elapsed:.3f}s")

            subspaces = []
            for cluster in np.unique(labels):
                cluster_indices = labels == cluster
                X_cluster = X_c[:, cluster_indices]
                # サンプル数が部分空間次元に満たない場合はスキップ
                if X_cluster.shape[1] >= self.n_subdims:
                    sub_basis = subspace_bases(X_cluster, self.n_subdims)
                    subspaces.append(sub_basis)
            if len(subspaces) == 0:
                basis = subspace_bases(X_c, self.n_subdims)
            else:
                basis = self.aggregate_subspaces(subspaces)

        # キャッシュが有効ならばファイルに保存
        if self.cache_per_class and self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{y_c}.npy", basis)
        else:
            # 内部辞書に保存（キャッシュする場合はメモリに保存しない）
            self._dic_dict[y_c] = basis

    def fit_cache(self, y_list):
        """
        cache_per_classがTrueの場合、各クラスのキャッシュファイルから部分空間を読み込み、
        内部辞書 _dic_dict に格納する。

        Args:
            y_list (list): 学習に利用した各クラスのラベルのリスト
        """
        if not self.cache_per_class or self.cache_dir is None:
            return

        for y in y_list:
            file_path = self.cache_dir / f"{y}.npy"
            if file_path.exists():
                basis = np.load(file_path)
                self._dic_dict[y] = basis

    def finalize_fit(self):
        """
        すべてのクラスに対する学習後、内部辞書 _dic_dict から self.dic への変換を行う。
        self.dic は後続の _get_gramians などで利用される (n_classes, n_dims, n_subdims) の形状となる。
        """
        keys = sorted(self._dic_dict.keys())
        dic_list = [self._dic_dict[k] for k in keys]
        self.dic = np.stack(dic_list, axis=0)
        self.label_encoder = {i: key for i, key in enumerate(keys)}

    # 注意: _get_gramians は親クラス ConstrainedMSM から継承されるため、
    # self.dic に集約済みの代表部分空間が格納されていればそのまま GDS 射影を適用します。
