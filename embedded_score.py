import os
import time
import pickle
# import umap
import umap.umap_ as umap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections.abc import Sequence
from matplotlib.colors import LinearSegmentedColormap
import plotly.express as px
from scipy.special import softmax
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans, MiniBatchKMeans
# from k_means_constrained import KMeansConstrained
from sklearn.decomposition import PCA

from topic_score import Topic_SCORE
from utils import *

from tqdm import tqdm
from scipy.linalg import svd
import warnings
warnings.filterwarnings("ignore", message=".*force_all_finite.*")


class UMAP_Fast:
    def __init__(
        self,
        new_dim: int,
        n_neighbors: int = 10,
        min_dist: float = 0.1,
        fit_prop: float = 0.1,
        points: np.ndarray | None = None,
        npy_path: str | None = None,
        tick: str = "AP",
        pre_pca_reduction: bool = False,
        pca_components: int = 50,
        pca_save_dir: str | None = None,
        pca_data_save_dir: str | None = None,
        reduce_dim: int = 0,
        random_seed: int | None = None,
        **kwargs
    ) -> None:
        self.kwargs = kwargs
        self.new_dim = new_dim
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.tick = tick
        if points is not None:
            self.points = points
        else:
            self.points = np.load(npy_path)[reduce_dim:]
            if pre_pca_reduction:
                print(f"pts before pca have shape:{self.points.shape}")
                self.pca = PCA(n_components=pca_components)
                self.points = self.pca.fit_transform(self.points)[reduce_dim:]
                print(f"finishing pca with acc: {np.cumsum(self.pca.explained_variance_ratio_)[pca_components-1]}")
                plt.plot(np.arange(pca_components), np.cumsum(self.pca.explained_variance_ratio_))
                plt.show()
                # save pca model
                os.makedirs(pca_save_dir, exist_ok=True)
                file_name = f"PCA_n_{pca_components}.pkl"
                save_path = os.path.join(pca_save_dir, file_name)
                with open(save_path, "wb") as file:
                    pickle.dump(self.pca, file)
                # save pca transformed data
                os.makedirs(pca_data_save_dir, exist_ok=True)
                save_file_name = f"word_PCA_n_{pca_components}.npy"
                pca_data_save_path = os.path.join(pca_data_save_dir, save_file_name)
                np.save(pca_data_save_path, self.points)

        self.full_size = self.points.shape[0]
        self.fit_prop = fit_prop
        self.fit_size = int(self.full_size * self.fit_prop)
        self.original_dim = self.points.shape[1]
        self.reduce_dim = reduce_dim
        self.pre_pca_reduction = pre_pca_reduction
        self.pca_components = pca_components
        self.umap_model = None
        self.random_seed = random_seed

    def fit_model(
        self,
        load_dir: str = "./pkl",
        load_path: str = "",
        dump_pkl: bool = True,
        dump_dir: str = "./pkl",
        save_dir: str = "./npz",
    ) -> None:
        file_name = f"{self.tick}_umap_{self.new_dim}_nn_{self.n_neighbors}_mdist_{self.min_dist}_sample_{self.fit_prop}"
        if self.pre_pca_reduction:
            file_name += f"_wPCA_{self.pca_components}"
        if self.reduce_dim > 0:
            file_name += f"_reduce_dim_{self.reduce_dim}"
        file_name += f"_seed_{self.random_seed}.pkl"
        if not load_path:
            os.makedirs(load_dir, exist_ok=True)
            load_path = os.path.join(load_dir, file_name)
        if os.path.exists(load_path):
            print("umap model exists, retrieving data...")
            dump_pkl = False
            with open(load_path, "rb") as file:
                umap_model = pickle.load(file)
        else:
            inds = np.arange(self.full_size)
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            np.random.shuffle(inds)
            inds = inds[: self.fit_size]
            self.umap_sample_inds = inds  # save the sampling pts
            sample_inds_file_name = f"{self.tick}_umap_sample_inds_{self.new_dim}_nn_{self.n_neighbors}_mdist_{self.min_dist}_sample_{self.fit_prop}"
            if self.pre_pca_reduction:
                sample_inds_file_name += f"_wPCA_{self.pca_components}"
            if self.reduce_dim > 0:
                sample_inds_file_name += f"_reduce_dim_{self.reduce_dim}"
            sample_inds_file_name += f"_seed_{self.random_seed}.npy"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, sample_inds_file_name)
            if self.kwargs.get("save_umap_sample_inds", False):
                np.save(save_path, self.umap_sample_inds)
            fit_points = self.points[inds]
            umap_model = umap.UMAP(
                n_neighbors=self.n_neighbors,
                min_dist=self.min_dist,
                n_components=self.new_dim,
                random_state=self.random_seed,
            )
            print(f"pts shape for fitting umap is:{fit_points.shape}")
            umap_model.fit(fit_points)
            print("finishing fitting umap")
        if dump_pkl:
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, file_name)
            with open(dump_path, "wb") as file:
                pickle.dump(umap_model, file)
        self.umap_model = umap_model

    def transform_all_points(
        self,
        load_dir: str = "./pkl",
        load_path: str = "",
        dump_pkl: bool = True,
        dump_dir: str = "./pkl",
        save_npy: bool = True,
        save_dir: str = "./npz",
        batch_size: int = 10000,
    ) -> np.ndarray:
        if self.umap_model is None:
            self.fit_model(
                load_dir=load_dir,
                load_path=load_path,
                dump_pkl=dump_pkl,
                dump_dir=dump_dir,
                save_dir=save_dir,
            )
        self.new_points = []
        for i in tqdm(range(0, self.points.shape[0], batch_size), desc="Transforming in mini-batches"):
            batch = self.points[i : i + batch_size]
            transformed_batch = self.umap_model.transform(batch)
            self.new_points.append(transformed_batch)
        self.new_points = np.vstack(self.new_points)
        print("finishing transform word_repr")

        if save_npy:
            os.makedirs(save_dir, exist_ok=True)
            save_file_name = f"{self.tick}_word_umap_{self.new_dim}_nn_{self.n_neighbors}_mdist_{self.min_dist}_sample_{self.fit_prop}"
            if self.pre_pca_reduction:
                save_file_name += f"_wPCA_{self.pca_components}"
            if self.reduce_dim > 0:
                save_file_name += f"_reduce_dim_{self.reduce_dim}"
            save_file_name += f"_seed_{self.random_seed}.npy"
            save_path = os.path.join(save_dir, save_file_name)
            np.save(save_path, self.new_points)

        return self.new_points

    def get_scatter_plot(self, dim1, dim2):
        plt.scatter(self.new_points[:, dim1], self.new_points[:, dim2])


class Embedded_SCORE_C:
    def __init__(
        self,
        num_doc: int,
        doc_lens: Sequence[int] | np.ndarray,
        # new vocab argument:
        vocab: Sequence[str] | None = None,
        num_topic: int | None = 3,
        num_hyperword: int | None = 800,
        word_repr: np.ndarray | None = None,
        npy_path: str | None = None,
        dim_repr: int = 10,
        bandwidth: float | None = 0.1,
        tick: str = "AP",
        random_seed: int | None = None,
        clustering_method: str = "mini-batch kmeans",
        vertex_hunting_method: str = "pp_SPA",
        remove_outlier: bool = True,
        cluster_first: bool = False,
        **kwargs,
    ):
        # store your vocabulary
        self.vocab = vocab
        if word_repr is None:
            word_repr = np.load(npy_path)
        self.num_doc = num_doc
        self.doc_lens = np.array(doc_lens)
        self.num_repr = word_repr.shape[0]
        self.tick = tick
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.umap_kwargs = {}
        if dim_repr is None or dim_repr >= word_repr.shape[1]:
            self.dim_repr = word_repr.shape[1]
            self.word_repr = word_repr
        else:
            self.dim_repr = dim_repr
            for key in ["n_neighbors", "min_dist", "fit_prop"]:
                if key in kwargs:
                    self.umap_kwargs[key] = kwargs.pop(key)
            umap_get = UMAP_Fast(
                new_dim=dim_repr,
                **self.umap_kwargs,
                points=word_repr,
                tick=tick,
                random_seed=self.random_seed,
            )
            self.word_repr = umap_get.transform_all_points() + 0.0
            del umap_get
        self.bandwidth = bandwidth
        self.num_hyperword = num_hyperword
        self.num_topic = num_topic
        self.clustering_method = clustering_method
        self.vertex_hunting_method = vertex_hunting_method
        self.remove_outlier = remove_outlier
        self.cluster_first = cluster_first
        self.kwargs = kwargs

    def get_hyperword(
        self,
        num_hyperword: int | None = None,
        clustering_method: str | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> np.ndarray:
        if num_hyperword is not None:
            self.num_hyperword = num_hyperword
        if clustering_method is not None:
            self.clustering_method = clustering_method
        if random_seed is None:
            random_seed = self.random_seed
        for key in kwargs:
            self.kwargs[key] = kwargs[key]
        print(f"number of hyperwords is {self.num_hyperword}")
        print(f"using {self.clustering_method} to get hyperwords...")
        if self.clustering_method == "mini-batch kmeans":
            batch_size = self.kwargs.get("batch_size", int(self.num_repr / 10))
            num_init = self.kwargs.get("num_init", 1)
            cluster_algo = MiniBatchKMeans(
                n_clusters=self.num_hyperword,
                random_state=random_seed,
                batch_size=batch_size,
                n_init=num_init,
                init='k-means++',
            )
        
            self.word_labels = cluster_algo.fit_predict(self.word_repr)
            self.hyperword_repr = cluster_algo.cluster_centers_
            print("shape of cluster = ", self.hyperword_repr.shape)
            #print("hyperword shape = ", self.hyperword_repr.shape)
        elif self.clustering_method == "kmeans":
            num_init = self.kwargs.get("num_init", 1)
            cluster_algo = KMeans(
                n_clusters=self.num_hyperword,
                random_state=random_seed,
                n_init=num_init,
                init='random',
            )
            self.word_labels = cluster_algo.fit_predict(self.word_repr)
            self.hyperword_repr = cluster_algo.cluster_centers_
        elif self.clustering_method == "hierarchical":
            Z = linkage(self.word_repr, method="average")
            self.word_labels = fcluster(Z, t=self.num_hyperword, criterion="maxclust") - 1
            centroids = np.zeros((self.num_hyperword, self.dim_repr))
            for label in range(self.num_hyperword):
                centroids[label, :] = self.word_repr[self.word_labels == label].mean(axis=0)
            self.hyperword_repr = centroids
        else:
            raise NotImplementedError
        return self.hyperword_repr

    def prepare_matrix(self) -> None:
        self.hyperword_freq = np.zeros((self.num_hyperword, self.num_doc))
        doc_lens_cumsum = np.cumsum(self.doc_lens)
        for doc_k, (doc_len, cumsum) in enumerate(zip(self.doc_lens, doc_lens_cumsum)):
            unique_elements, counts = np.unique(
                self.word_labels[cumsum - doc_len : cumsum], return_counts=True
            )
            self.hyperword_freq[unique_elements, doc_k] = counts
        self.hyperword_freq /= self.hyperword_freq.sum(0)
        

    def fit_topic_score(
        self,
        num_topic: int | None = None,
        vertex_hunting_method: str | None = None,
        remove_outlier: bool | None = None,
        cluster_first: bool | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        if num_topic is not None:
            self.num_topic = num_topic
        if vertex_hunting_method is not None:
            self.vertex_hunting_method = vertex_hunting_method
        if remove_outlier is not None:
            self.remove_outlier = remove_outlier
        if cluster_first is not None:
            self.cluster_first = cluster_first
        if random_seed is None:
            random_seed = self.random_seed
        for key in kwargs:
            self.kwargs[key] = kwargs[key]
        print("In fit_topic_score:")
        self.model = Topic_SCORE(
            num_doc=self.num_doc,
            num_word=self.num_hyperword,
            word_freq=self.hyperword_freq,
            num_topic=self.num_topic,
            vertex_hunting_method=self.vertex_hunting_method,
            remove_outlier=self.remove_outlier,
            cluster_first=self.cluster_first,
            random_seed=random_seed,
            **self.kwargs,
        )
        self.model.fit_all()
        self.hyperword_density = self.model.A_hat
        print("finish fitting TS")
        if self.num_topic > 2:
            self.model.plot_vertex_hunting(dim1=0, dim2=1, label1="Hyperwords", show_centroids=True)
        self.hyperword_mixed_member = self.hyperword_density / self.hyperword_density.sum(1, keepdims=True)

    
    def get_word_density(
    self,
    bandwidth: float | None = None,
    down_sample: float | None = None,
    minibatch: bool = True,
):
        if bandwidth is not None:
            self.bandwidth = bandwidth
        print(f"bandwidth is {self.bandwidth}")

        if minibatch:
            # make more, smaller batches to cut peak memory (tune this up)
            num_batches = self.kwargs.get("num_batches_cal_density", 32)

            idx_batches = np.array_split(np.arange(self.word_repr.shape[0]), num_batches)
            # ensure float32 to halve memory
            self.word_density     = np.zeros((self.word_repr.shape[0], self.num_topic), dtype=np.float32)
            self.word_mixed_member = np.zeros((self.word_repr.shape[0], self.num_topic), dtype=np.float32)

            inv_bw2   = np.float32(1.0 / (self.bandwidth * self.bandwidth))
            norm_term = np.float32(np.log(self.bandwidth * np.sqrt(np.pi)) * self.dim_repr)

            for batch in tqdm(idx_batches, desc="Calculating word densities."):
                # Distances ^2 (already squared), keep as float32
                d2 = euc_dist_sq(self.word_repr[batch], self.hyperword_repr).astype(np.float32)

                # ---- Gaussian kernel path (for density) ----
                # exp( - d2 / bw^2 - log(bw*sqrt(pi))*dim )
                tmp = (-d2 * inv_bw2) - norm_term
                np.exp(tmp, out=tmp)                         # tmp is the gaussian kernel now
                # tmp @ A_hat
                self.word_density[batch] = tmp @ self.hyperword_density.astype(np.float32, copy=False)
                del tmp  # free memory

                # ---- Softmax kernel path (for mixture weights) ----
                # softmax( -d2 / bw^2 ) row-wise, stabilized
                sm = -d2 * inv_bw2
                del d2
                sm -= sm.max(axis=1, keepdims=True)          # stabilize
                np.exp(sm, out=sm)
                sm_sum = sm.sum(axis=1, keepdims=True)
                sm /= sm_sum
                # sm @ hyperword_mixed_member
                self.word_mixed_member[batch] = sm @ self.hyperword_mixed_member.astype(np.float32, copy=False)
                del sm, sm_sum

                # optional tiny cleanup to keep peak low
                import gc; gc.collect()

            self.down_sample_word_index = np.arange(self.word_repr.shape[0], dtype=int)
            return self.word_density, self.word_mixed_member, self.down_sample_word_index


    def fit_all(
        self,
        num_topic: int | None = None,
        num_hyperword: int | None = None,
        bandwidth: float | None = None,
        clustering_method: str | None = None,
        vertex_hunting_method: str | None = None,
        remove_outlier: bool | None = None,
        cluster_first: bool | None = None,
        **kwargs,
    ):
        if num_topic is not None:
            self.num_topic = num_topic
        if num_hyperword is not None:
            self.num_hyperword = num_hyperword
        if bandwidth is not None:
            self.bandwidth = bandwidth
        if clustering_method is not None:
            self.clustering_method = clustering_method
        if vertex_hunting_method is not None:
            self.vertex_hunting_method = vertex_hunting_method
        if remove_outlier is not None:
            self.remove_outlier = remove_outlier
        if cluster_first is not None:
            self.cluster_first = cluster_first
        for key in kwargs:
            self.kwargs[key] = kwargs[key]
        self.get_hyperword()
        self.prepare_matrix()
        self.plot_scree()
        self.fit_topic_score()
        self.get_word_density()

        # --- NEW: automatically print top words per topic if vocab was supplied ---
        if self.vocab is not None:
            print("\n📝 Discovered Topics:")
            top_k = kwargs.get("top_k_words", 8)
            for k in range(self.num_topic):
                idxs = np.argsort(-self.word_mixed_member[:, k])[:top_k]
                words_k = [self.vocab[i] for i in idxs]
                print(f"Topic {k}: {', '.join(words_k)}")

    def plot_word_vector(self, dim1, dim2):
        pass

    def interactive_word_vector(self):
        pass

    def plot_doc_vector(self, dim1: int = 0, dim2: int = 1):
        pass

    def interactive_doc_vector(self, dim: int = 0, dim2: int = 1):
        pass

    # def get_top_k_words(
    #     self,
    #     min_occ: int = 5,
    #     k: int = 50,
    #     word_arr: list | np.ndarray | None = None,
    # ):
    #     lst_elem = []
    #     lst_indices = []
    #     for topic in range(self.num_topic):
    #         arr = word_arr[self.down_sample_word_index][np.argsort(self.word_mixed_member[:, topic])[::-1]]
    #         unique_elements, first_indices = first_k_reoccurring_elements(arr, k=k, min_occ=min_occ)
    #         lst_elem.append(unique_elements.tolist())
    #         lst_indices.append(first_indices)
    #     return lst_elem, lst_indices

    def get_topic_words(
    self,
    top_k: int = 30,
    min_occ: int = 5,
    q: float = 0.5,
    stopwords: set[str] | None = None,
    denylist: set[str] | None = None,
    allowlist: set[str] | None = None,
) -> dict[int, list[str]]:
        """
        Robust topic-word extraction.

        - Aggregates over repeated tokens with the same surface form in self.vocab
        - Requires at least `min_occ` occurrences to be eligible
        - Scores a word w for topic k by the mean of the top-q fraction of
        membership scores among occurrences of w (stable vs. outliers)
        - Filters out stopwords and denylist tokens (unless in allowlist)

        Returns:
            dict: {topic_index: [word1, word2, ...]} with exactly top_k words per topic
        """
        if self.vocab is None:
            raise RuntimeError("Embedded_SCORE_C.vocab is None. Pass vocab when constructing the model.")

        # Build index lists per unique surface form (word -> list of positions)
        # self.vocab is length = number of word occurrences (same as self.word_mixed_member rows)
        word_to_indices: dict[str, list[int]] = {}
        for idx, w in enumerate(self.vocab):
            if w is None:
                continue
            w_norm = w.strip()
            if not w_norm:
                continue
            word_to_indices.setdefault(w_norm, []).append(idx)

        # Default stopwords (optional, lightweight)
        if stopwords is None:
            stopwords = set()
        if denylist is None:
            denylist = set()
        if allowlist is None:
            allowlist = set()

        # Helper: decide if word is allowed
        def allowed(w: str) -> bool:
            lw = w.lower()
            if lw in allowlist:
                return True
            if lw in denylist:
                return False
            if lw in stopwords:
                return False
            # filter very short tokens
            if len(lw) <= 2:
                return False
            return True

        # Precompute per-word eligible indices (respecting min_occ & filters)
        eligible: dict[str, np.ndarray] = {}
        for w, idxs in word_to_indices.items():
            if len(idxs) >= min_occ and allowed(w):
                eligible[w] = np.asarray(idxs, dtype=int)

        if not eligible:
            # Fallback: return empty lists rather than crashing
            return {k: [] for k in range(self.num_topic)}

        topic_words: dict[int, list[str]] = {}

        # For each topic, compute a robust score for each word and take top_k
        for k in range(self.num_topic):
            scores = []
            for w, idxs in eligible.items():
                vals = self.word_mixed_member[idxs, k]
                if vals.size == 0:
                    continue
                # take the top-q quantile mass to reduce sensitivity to tails
                m = max(1, int(np.ceil(q * vals.size)))
                # partial sort for efficiency on large arrays
                # (descending) — use argpartition then take top m
                part = np.argpartition(-vals, m - 1)[:m]
                robust_mean = float(vals[part].mean())
                scores.append((robust_mean, w))

            if not scores:
                topic_words[k] = []
                continue

            # Sort by score descending and take the first top_k unique words
            scores.sort(key=lambda x: x[0], reverse=True)
            words_k = [w for _, w in scores[:top_k]]
            topic_words[k] = words_k

        return topic_words


    def plot_scree(self, max_k=50, title="Scree Plot of Hyperword Matrix"):
        """
        Performs SVD on the Hyperword Frequency Matrix to help determine optimal n_topics.
        
        Args:
            max_k: Max number of singular values to display (x-axis limit).
        """
        if self.hyperword_freq is None:
            raise ValueError("Matrix not ready! Run .fit() or .prepare_matrix() first.")

        print(f"📉 Computing SVD on Hyperword Matrix {self.hyperword_freq.shape}...")

        # 1. Decompose the Matrix
        # We use full SVD because n_hyperwords (e.g., 500) is small enough to be fast.
        # self.hyperword_freq is (n_hyperwords, n_docs)
        U, S, Vt = svd(self.hyperword_freq, full_matrices=False)

        # 2. Prepare Plot Data
        # S contains the singular values in descending order
        singular_values = S[:max_k]
        ranks = np.arange(1, len(singular_values) + 1)

        # 3. Plotting
        plt.figure(figsize=(10, 6))
        
        # Plot the Singular Values (The Scree)
        plt.plot(ranks, singular_values, 'o-', linewidth=2, color='blue', label='Singular Values')
        
        # Optional: Plot the "Explained Variance" style logic (Cumulative)
        # total_variance = np.sum(S**2)
        # explained_variance = (singular_values**2) / total_variance
        # plt.plot(ranks, explained_variance * max(singular_values), 'r--', alpha=0.5, label='Scaled Expl. Variance')

        plt.title(title, fontsize=14)
        plt.xlabel("Topic Rank (k)", fontsize=12)
        plt.ylabel("Singular Value Magnitude", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(np.arange(0, max_k+1, 5))
        plt.legend()
        
        # Annotate the "Elbow" visually
        plt.annotate('Potential Elbow?', 
                    xy=(10, singular_values[9]), 
                    xytext=(15, singular_values[5]),
                    arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig("scree_plo.png")
        plt.show()

    # Monkey-patch it into your class if you don't want to edit the file directly:
    # PPTMTopicModel.plot_scree = plot_scree

    def get_top_k_docs(
        self,
        k: int = 50,
        doc_indices: list | np.ndarray | None = None,
    ):
        if doc_indices is None:
            doc_indices = np.arange(self.num_doc)
        anchor_doc_info = []
        for topic in range(self.num_topic):
            top_k_ind = doc_indices[np.argsort(self.model.W_hat[topic, :])[::-1][:k]]
            anchor_doc_info.append(top_k_ind)
        return anchor_doc_info

    def plot_density_regions(
        self,
        new_bandwidth: int = 0.5,
        seed: int | None = None,
        load_path_umap2: str = "",
        dump_dir_umap2: str = "./pkl",
        load_path: str = "",
        save_path: str = "",
        region_coords_list1: list | None = None,
        region_coords_list2: list | None = None,
        titles: list | None = None,
        titles_ac: list | None = None,
        save_plot_dir: str = "",
        save_plot_tick: str = "AP",
        epsilon: int = 1e-2,
        batch_size: int | None = None,
    ):
        
      if os.path.exists(load_path):
        with np.load(load_path,) as data:
          x_grid = data['x_grid']
          y_grid = data['y_grid']
          z_grid = data['z_grid']
          w_grid = data['w_grid']
          z_word = data['z_word']
          w_word = data['w_word']
          new_word_repr = data['new_word_repr']
      else:
        if seed is None:
          seed = self.seed
        new_umap_get = UMAP_Fast(
          new_dim=2,
          points=self.word_repr,
          tick=self.tick,
          random_seed=seed * 2,
          **self.umap_kwargs,
        )
        new_umap_get.fit_model(load_path=load_path_umap2, dump_dir=dump_dir_umap2)
        
        if batch_size is not None:
          # transform all pts
          print(f"Transforming all pts with batch size: {batch_size}...")
          new_word_repr = []
          z_word = []
          w_word = []
          for i in tqdm(range(0, self.word_repr.shape[0], batch_size), desc="Transforming in mini-batches"):
              batch = self.word_repr[i:i + batch_size]
              transformed_batch = new_umap_get.umap_model.transform(batch)
              kernel_batch = euc_dist_sq(transformed_batch, new_hyperword_repr)
              kernel_batch = np.exp( 
                -(kernel_batch**2) / new_bandwidth**2 - np.log(new_bandwidth * np.sqrt(np.pi)) * 2
              )
              z_word_batch = kernel_batch @ self.hyperword_density
              w_word_batch = z_word_batch / (epsilon + z_word_batch.sum(-1, keepdims=True))
              z_word.append(z_word_batch)
              w_word.append(w_word_batch)
              new_word_repr.append(transformed_batch)
          z_word = np.vstack(z_word)
          w_word = np.vstack(w_word)
          new_word_repr = np.vstack(new_word_repr)
        else:
          # transform all pts
          print("Transforming all pts...")
          new_hyperword_repr = new_umap_get.umap_model.transform(self.hyperword_repr)
          new_word_repr = new_umap_get.umap_model.transform(self.word_repr) # ~30s
          kernel = euc_dist_sq(new_word_repr, new_hyperword_repr) # ~30s
          kernel = np.exp(
            -(kernel**2) / new_bandwidth**2 - np.log(new_bandwidth * np.sqrt(np.pi)) * 2
          )
          z_word = kernel @ self.hyperword_density
          w_word = z_word / (epsilon + z_word.sum(-1, keepdims=True))
        del new_umap_get
        
        # create plot grids
        print("Creating plot grids...")
        xlow, ylow = np.min(new_hyperword_repr, axis=0)
        xhigh, yhigh = np.max(new_hyperword_repr, axis=0)
        delta = 0.05
        x_seq = np.arange(xlow, xhigh, delta)
        y_seq = np.arange(ylow, yhigh, delta)
        x_grid, y_grid = np.meshgrid(x_seq, y_seq)
        dist_grid = euc_dist_sq(np.dstack([x_grid, y_grid]), new_hyperword_repr)
        gaussian_kernel = np.exp(
          -(dist_grid**2) / new_bandwidth**2 - np.log(new_bandwidth * np.sqrt(np.pi)) * 2
        )
        del dist_grid
        z_grid = gaussian_kernel @ self.hyperword_density
        w_grid = z_grid / (epsilon + z_grid.sum(-1, keepdims=True))
        del gaussian_kernel

        # save unnorm_density (z_word)/ norm_density (w_word)
        if save_path:
          np.savez(
            save_path, 
            z_word=z_word, 
            w_word=w_word, 
            new_word_repr=new_word_repr,
            z_grid=z_grid,
            w_grid=w_grid,
            x_grid=x_grid,
            y_grid=y_grid
          )
      
      viridis = plt.get_cmap('viridis')
      n_colors = 256
      viridis_colors = viridis(np.linspace(0, 1, n_colors))
      new_viridis_colors = viridis(np.linspace(0, 1, n_colors))
      for i in range(0, 256):
        # Reduce the saturation of yellow hues by making it more grayish
        new_viridis_colors[i] = viridis_colors[255-i]*np.array([1, 0.6+0.4*i/256, 0.7+0.3*i/256, 0.85+0.15*i/256])  # Slightly desaturate the RGB values
        new_viridis_colors[i] = viridis_colors[255-i]*np.array([1, 0.9+0.1*(255-i)/256, 0.9+0.1*(255-i)/256, 0.9+0.1*(255-i)/256])  
      custom_cmap = LinearSegmentedColormap.from_list("custom_viridis", new_viridis_colors, N=n_colors)
      cmap_ = mcolors.LinearSegmentedColormap.from_list("CustomGreen", ["white", "green"])

      for topic in range(self.num_topic):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        w_grid_masked = np.ma.masked_where(w_grid == 0, w_grid)
        contours = ax.contourf(x_grid, y_grid, w_grid_masked[:, :, topic], cmap=cmap_, zorder=1, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        contour_lines = ax.contour(x_grid, y_grid, w_grid[:, :, topic], colors='black', linewidths=0.1, zorder=2, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        
        handles = [
          plt.Line2D([0], [0], color=contours.cmap(contours.norm(level)), lw=2)
          for level in contours.levels
        ]
        labels = [f"{level:.2g}" for level in contours.levels]
        
        if region_coords_list1 is not None and len(region_coords_list1[topic]):
          # Add additional points
          ax.scatter(region_coords_list1[topic][:,0], region_coords_list1[topic][:,1],c='gold',s=125,marker='*',edgecolors='black',linewidths=0.2,zorder=3)
          for idx in range(len(region_coords_list1[topic])):
            ax.text(
                region_coords_list1[topic][idx,0] + 1,
                region_coords_list1[topic][idx,1] - 1,
                region_coords_list2[topic][idx],
                fontsize=16,
                c="magenta",
                bbox=dict(
                  boxstyle="round,pad=0.2", facecolor="white", edgecolor="magenta", alpha=0.8
                ),
                ha='left',
                va='top',
                zorder=4
              )
        if titles is not None:
          ax.set_title(titles[topic], fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save_plot_dir and save_plot_tick:
          os.makedirs(save_plot_dir, exist_ok=True)
          fig.savefig(
            os.path.join(save_plot_dir, save_plot_tick+f"_contour_normalized_{titles_ac[topic]}.pdf"), format="pdf", bbox_inches="tight"
          )
        
        self.z_word = z_word
        self.w_word = w_word
        self.new_word_repr = new_word_repr