"""
Mutual Subspace Method
"""

# Authors: Junki Ishikawa
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize as _normalize

from ..utils import subspace_bases
from ..utils.typing import PredResult
from .base_class import MSMInterface, SMBase, mean_square_singular_values


class MutualSubspaceMethod(MSMInterface, SMBase):
    """
    Mutual Subspace Method
    """

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
        bases = subspace_bases(X, self.test_n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, : self.n_subdims]
        gramians = np.dot(dic.transpose(0, 2, 1), bases)

        return gramians


class MultiMSM(MutualSubspaceMethod):
    """Multi Mutual Subspace Method"""

    def __init__(
        self,
        n_subdims: int,
        n_classes: int,
        n_data_per_subspace: int = 1,
        normalize: bool = False,
        faster_mode: bool = False,
        test_n_subdims: Optional[int] = None,
        *,
        cache_per_class: bool = False,
        cache_dir: Optional[Path] = None,
    ):
        super().__init__(
            n_subdims=n_subdims,
            normalize=normalize,
            faster_mode=faster_mode,
            test_n_subdims=test_n_subdims,
        )

        self.n_classes = n_classes
        self.n_data_per_subspace = n_data_per_subspace
        self.cache_per_class = cache_per_class
        self.cache_dir = (
            cache_dir if cache_dir is not None else Path("_cache_multi_msm")
        )

        self.dic: dict[int, list] = {}

    def _process_chunk(self, chunk, n_subdims):
        x = chunk.transpose(1, 0, 2).reshape(chunk.shape[1], -1)
        # x = np.stack(chunk, axis=1).reshape(512, -1)  # type: ignore
        basis = subspace_bases(x, n_subdims)
        return basis

    def fit_1_class(self, X_c: np.ndarray, y_c: int):
        """
        Parameters
        ----------
        X_c: array, (n_samples, n_dims)
        y_c: int
        """

        if self.normalize:
            X_c = np.array([_normalize(x) for x in X_c])
        X_c = np.array([x.T for x in X_c])

        chunks = [
            X_c[i : i + self.n_data_per_subspace]  # type: ignore
            for i in range(0, X_c.shape[0], self.n_data_per_subspace)
        ]
        results = Parallel(n_jobs=-1, return_as="list")(
            delayed(self._process_chunk)(chunk, self.n_subdims) for chunk in chunks
        )
        assert isinstance(results, list)

        if self.cache_per_class:
            x = np.array(results)
            np.save(self.cache_dir / f"{y_c}.npy", x)
        else:
            self.dic[y_c] = results

    def fit_cache(self):
        for i in range(self.n_classes):
            self.dic[i] = np.load(self.cache_dir / f"{i}.npy", allow_pickle=False)

    def plot_3d_subspaces(self):
        from tqdm import tqdm

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        colors = plt.cm.get_cmap("tab10", len(self.dic))  # クラスごとに異なる色を設定

        for y_c, bases_list in tqdm(self.dic.items()):
            for basis in bases_list:
                for vector in basis.T:
                    ax.quiver(
                        0, 0, 0, vector[0], vector[1], vector[2], color=colors(y_c)
                    )

        ax.set_xlim((-1, 1))
        ax.set_ylim((-1, 1))
        ax.set_zlim((-1, 1))  # type: ignore

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.set_title("3D Subspaces for All Classes")

        # plt.show()
        plt.savefig("3d_plot.png")
        plt.close()

    def _get_gramians(self, X: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)

        Returns
        -------
        G: dict
            gramian matricies of references of each class
        """
        bases = subspace_bases(X, self.test_n_subdims)

        gramians = []
        for y_c, dic_list in self.dic.items():
            for dic in dic_list:
                gramians.append((y_c, np.dot(dic.T, bases)))
        return gramians

    def get_predictions(self, X: np.ndarray, k_neighbors: int = 1) -> PredResult:
        """
        Predict class probabilities with k-nearest neighbors.

        Parameters:
        -----------
        X: array, (n_samples, n_dims)
            List of input vector sets.

        Returns:
        --------
        pred: int
            Predicted class.
        """
        if self.normalize:
            X = _normalize(X)
        X = X.T  # transpose (n_dims, n_samples)

        gramians = self._get_gramians(X)
        preds = [(y_c, mean_square_singular_values(g)) for y_c, g in gramians]

        # k-nearest neighbors and select top-k
        top_preds = sorted(preds, key=lambda x: x[1], reverse=True)
        top_preds = top_preds[:k_neighbors]

        votes = np.zeros(self.n_classes)
        for y_c, _ in top_preds:
            votes[y_c] += 1

        # sorted_votes = sorted(
        #     [(i, int(v)) for i, v in enumerate(votes) if v > 0],
        #     key=lambda x: x[1],
        #     reverse=True,
        # )
        # vote_output = ", ".join([f"{i}({v})" for i, v in sorted_votes])
        # print(f"[Votes] {vote_output}")

        max_cnt = int(np.max(votes, axis=0))
        candidates = np.where(votes == max_cnt)[0]
        if len(candidates) > 1:
            pred_label = top_preds[0][0]
        else:
            pred_label = int(candidates[0])

        probs = np.zeros((X.shape[1], self.n_classes))
        for y_c, g in preds:
            probs[:, y_c] = g
        return PredResult(label=pred_label, probs=probs)


class rffMSM(MSMInterface, SMBase):
    """
    Mutual Subspace Method
    """

    def __init__(
        self,
        n_subdims,
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

        # bases, (n_dims, n_subdims)
        newX = []
        for n in range(self.n_approx):
            newX.append(self.z(X, self.w[n], self.b[n], self.m))
        newX = np.stack(newX, axis=0).mean(axis=0)

        bases = subspace_bases(newX, self.test_n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims or greater)
        dic = self.dic[:, :, : self.n_subdims]
        gramians = np.dot(dic.transpose(0, 2, 1), bases)

        return gramians
