"""
Constrained Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize as _normalize

from ..utils import mean_square_singular_values, subspace_bases
from ..utils.typing import PredResult
from .base_class import ConstrainedSMBase, MSMInterface


class ConstrainedMSM(MSMInterface, ConstrainedSMBase):
    """
    Constrained Mutual Subspace Method
    """

    def __init__(
        self,
        n_subdims: int,
        n_gds_dims: int,
        normalize: bool = False,
        cache_per_class: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.
        n_gds_dims : int
            The dimension of GDS.
        normalize : boolean, optional (default=False)
            If this is True, all vectors are normalized as |v| = 1.
        cache_per_class : boolean, optional (default=False)
            If this is True, cache the computation results for each class.
        cache_dir : Path or str, optional (default=None)
            Directory to store cache files.
        """
        super().__init__(
            n_subdims=n_subdims,
            n_gds_dims=n_gds_dims,
            normalize=normalize,
        )
        self.cache_per_class = cache_per_class
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def fit_1_class(self, X_c: np.ndarray, y_c: int) -> None:
        """
        Parameters
        ----------
        X_c: list[np.ndarray], (n_files, n_samples, n_dims)
        y_c: int
        """
        if self.normalize:
            X_c = np.array([_normalize(x) for x in X_c])
        X_c = np.array([x.T for x in X_c])

        # (n_files, n_dims, n_samples) -> (n_dims, n_files * n_samples)
        X_c = X_c.transpose(1, 0, 2).reshape(X_c.shape[1], -1)
        dic_c = subspace_bases(X_c, self.n_subdims)

        if self.cache_per_class and self.cache_dir is not None:
            cache_file = self.cache_dir / f"class_{y_c}.npy"
            np.save(cache_file, dic_c)
        else:
            if self.dic is None:
                self.dic = []
            self.dic.append(dic_c)

    def fit_cache(self, y_list: list[int]):
        if self.cache_dir is None:
            raise ValueError("cache_dir is not set")

        self.dic = []
        for y_c in y_list:
            cache_file = self.cache_dir / f"class_{y_c}.npy"
            dic_c = np.load(cache_file)
            print(dic_c.shape)
            self.dic.append(dic_c)

    def fit_y(self, y: list[int]):
        self._prepare_y(y)

    def project_bases(self):
        dic = np.array(self.dic)
        all_bases = np.hstack(dic)  # type: ignore

        # n_gds_dims
        if 0.0 < self.n_gds_dims <= 1.0:
            n_gds_dims = int(all_bases.shape[1] * self.n_gds_dims)
        else:
            n_gds_dims = self.n_gds_dims

        # gds, (n_dims, n_gds_dims)
        self.gds = subspace_bases(all_bases, n_gds_dims, higher=False)

        dic = self._gds_projection(dic)
        self.dic = dic

    def get_predictions(self, X: np.ndarray) -> PredResult:
        if self.normalize:
            X = _normalize(X)  # type: ignore
        X = X.T  # transpose (n_dims, n_samples)

        gramians = self._get_gramians(X)

        # i_th singular value of grammian of subspace bases is
        # square root of cosine of i_th cannonical angles
        # average of square of them is caonnonical angle between subspaces
        c = [mean_square_singular_values(g) for g in gramians]
        probs = np.array(c)
        return PredResult(label=int(np.argmax(c)), probs=probs)

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)

        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """
        # bases, (n_dims, n_subdims)
        bases = subspace_bases(X, self.n_subdims)
        # bases, (n_gds_dims, n_subdims)
        bases = self._gds_projection(bases)

        # gramians, (n_classes, n_subdims, n_subdims)
        gramians = np.dot(self.dic.transpose(0, 2, 1), bases)

        return gramians

    def plot_3d_subspaces(self):
        """Plot 3D subspaces for all classes"""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        colors = plt.cm.get_cmap("jet", len(self.dic))  # クラスごとに異なる色を設定

        # GDSのベクトルを黒色で描画
        if self.gds is not None:
            for i, vector in enumerate(self.gds.T):
                ax.quiver(
                    0,
                    0,
                    0,
                    vector[0] * 10,
                    vector[1] * 10,
                    vector[2] * 10,
                    color=colors(i),
                    label=f"GDS {i}",
                )

        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))  # type: ignore

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.set_title("3D Subspaces for GDS (ConstrainedMSM)")

        plt.legend()
        plt.savefig("3d_plot_constrained_msm_gds.png")
        plt.close()


class rffCMSM(MSMInterface, ConstrainedSMBase):
    """
    Mutual Subspace Method
    """

    def __init__(
        self,
        n_subdims,
        n_gds_dims,
        m_rand_samples,
        normalize=False,
        sigma=None,
        faster_mode=False,
        test_n_subdims=None,
        n_approx=1,
    ):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1
        """
        self.n_subdims = n_subdims
        self.n_gds_dims = n_gds_dims
        self.normalize = normalize
        self.faster_mode = faster_mode
        self.le = LabelEncoder()
        self.dic = None
        self.labels = None
        self.n_classes = None
        self._test_n_subdims = test_n_subdims
        self.params = ()

        self.m = m_rand_samples
        self.n_approx = n_approx
        self.sigma = sigma

    def z(self, X, w, b, m):
        return np.sqrt(2 / m) * np.cos((w @ X).T + b).T

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """
        n_dims = X[0].shape[0]
        if self.sigma is None:
            self.sigma = np.sqrt(n_dims / 2)

        w = []
        b = []
        for n in range(self.n_approx):
            w.append(np.random.randn(self.m, n_dims) / self.sigma)
            b.append(np.random.rand(self.m) * 2 * np.pi)
        self.w = np.stack(w)
        self.b = np.stack(b)

        newX = []
        for _X in X:
            _newX = []
            for n in range(self.n_approx):
                _newX.append(self.z(_X, self.w[n], self.b[n], self.m))
            _newX = np.stack(_newX, axis=0).mean(axis=0)
            newX.append(_newX)

        dic = [subspace_bases(_X, self.n_subdims) for _X in newX]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        # all_bases, (n_dims, n_classes * n_subdims)
        all_bases = np.hstack(dic)

        # n_gds_dims
        if 0.0 < self.n_gds_dims <= 1.0:
            n_gds_dims = int(all_bases.shape[1] * self.n_gds_dims)
        else:
            n_gds_dims = self.n_gds_dims

        # gds, (n_dims, n_gds_dims)
        self.gds = subspace_bases(all_bases, n_gds_dims, higher=False)

        dic = self._gds_projection(dic)
        self.dic = dic

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)

        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """
        newX = []
        for n in range(self.n_approx):
            newX.append(self.z(X, self.w[n], self.b[n], self.m))
        newX = np.stack(newX, axis=0).mean(axis=0)

        # bases, (n_dims, n_subdims)
        bases = subspace_bases(newX, self.n_subdims)
        # bases, (n_gds_dims, n_subdims)
        bases = self._gds_projection(bases)

        # gramians, (n_classes, n_subdims, n_subdims)
        gramians = np.dot(self.dic.transpose(0, 2, 1), bases)

        return gramians
