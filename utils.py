import numpy as np
from scipy.linalg import det
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections.abc import Sequence
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.stats import norm
import choix
from scipy.special import expit  # sigmoid

def euc_dist_sq(word_repr1, word_repr2):
  return (
    (word_repr1**2).sum(-1, keepdims=True)
    + (word_repr2**2).sum(1)
    - 2 * word_repr1 @ word_repr2.T
  )


def PCA_transform(M, n_components):
  # M as shape (N x p)
  # scikit.learn expect (N x p)
  pca = PCA(n_components=n_components)
  return pca.fit_transform(M)


def NFINDR(data, q: int, transform):
  """
  N-FINDR endmembers induction algorithm.

  Parameters:
    data: `numpy array`
      Column data matrix [nvariables x nsamples].

    q: `int`
      Number of endmembers to be induced.

    transform: `numpy array [default None]`
      The transformed 'data' matrix by MNF (N x components). In this
      case the number of components must == q-1. If None, the built-in
      call to PCA is used to transform the data.

    maxit: `int [default None]`
      Maximum number of iterations. Default = 3*q.


  Returns: `tuple: numpy array, numpy array, int`
    * Set of induced endmembers (N x p)
    * Set of transformed induced endmembers (N x p)
    * Array of indices into the array data corresponding to the
      induced endmembers
    * The number of iterations.

  References:
    Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral
    end-member determination in hyperspectral data", presented at the Imaging
    Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pgs. 266-275.
  """
  # data size
  nsamples, nvariables = data.shape

  if maxit is None:
    maxit = 3 * q

  if transform is None:
    # transform as shape (N x p)
    transform = data
    transform = PCA_transform(data, q - 1)
  else:
    transform = transform

  # Initialization
  # TestMatrix is a square matrix, the first row is set to 1
  TestMatrix = np.zeros((q, q), dtype=np.float32, order="F")
  TestMatrix[0, :] = 1
  IDX = np.zeros((q), dtype=np.int64)
  for i in range(q):
    idx = np.random.randint(nsamples)
    TestMatrix[1:q, i] = transform[idx]
    IDX[i] = idx

  actualVolume = 0
  it = 0
  v1 = -1.0
  v2 = actualVolume

  while it <= maxit and v2 > v1:
    for k in range(q):
      for i in range(nsamples):
        TestMatrix[1:q, k] = transform[i]
        volume = np.abs(det(TestMatrix))
        if volume > actualVolume:
          actualVolume = volume
          IDX[k] = i
      TestMatrix[1:q, k] = transform[IDX[k]]
    it = it + 1
    v1 = v2
    v2 = actualVolume

  E = np.zeros((len(IDX), nvariables), dtype=np.float32)
  Et = np.zeros((len(IDX), q - 1), dtype=np.float32)
  for j in range(len(IDX)):
    E[j] = data[IDX[j]]
    Et[j] = transform[IDX[j]]

  return E, Et, IDX, it


def SPA(R, K):
  n = R.shape[0]

  Y = np.hstack((np.ones((n, 1)), R))
  indexSet = []

  while len(indexSet) < K:
    l2Norms = np.linalg.norm(Y, axis=1)
    index = np.argmax(l2Norms)
    if index not in indexSet:
      indexSet.append(index)
      u = Y[index, :] / np.linalg.norm(Y[index, :])
      Y = Y - np.outer(Y @ u, u)

  return R[indexSet, :], indexSet  # Adjust for 1-based indexing

def SVS(R, K, quantile=0.9, percentage=0.05):
  # first use knn with 0.05% maximal 
  n = R.shape[0]
  euc_dist = cdist(R, R, metric='euclidean')
  dist_thres = percentage * np.quantile(euc_dist.flatten(), quantile)
  R_knn = np.zeros_like(R)
  for i in range(R.shape[0]):
      neighbors = np.where(euc_dist[i] <= dist_thres)[0]
      R_knn[i] = R[neighbors].mean(axis=0) 

  return SPA(R_knn, K)
  

def get_nearest_point(point_cloud, point):
  return np.argmin(np.linalg.norm(point_cloud - point, axis=1))





def first_k_unique_elements(lst: np.ndarray, k: int):
  # Convert the list to a NumPy array
  arr = np.array(lst)
  # Preallocate arrays for storing unique elements and their indices
  unique_elements = np.empty(k, dtype=arr.dtype)
  indices = np.empty(k, dtype=int)
  count = 0  # Track the number of unique elements found
  # Iterate through the array
  for idx, elem in enumerate(arr):
    if elem not in unique_elements[:count]:
      unique_elements[count] = elem
      indices[count] = idx
      count += 1
    # Stop once we have found k unique elements
    if count == k:
      break
  if count > k:
    print('fewer than k unique elements were found')
  # Slice the result in case fewer than k unique elements were found
  return unique_elements[:count], indices[:count]


def first_k_reoccurring_elements(lst: np.ndarray, k: int, min_occ: int=5):
  # Convert the list to a NumPy array
  arr = np.array(lst)
  # Preallocate arrays for storing unique elements and their indices
  unique_elements = np.empty(len(lst), dtype=arr.dtype)
  reoccurring_elements  = np.empty(k, dtype=arr.dtype)
  occurrence = np.empty(len(lst), dtype=int)
  indices = np.empty(k, dtype=int)
  count_0 = 0  # Track the number of unique elements found
  count_1 = 0
  # Iterate through the array
  for idx, elem in enumerate(arr):
    if elem not in unique_elements[:count_0]:
      unique_elements[count_0] = elem
      occurrence[count_0]=1
      count_0 += 1
    else:
      idx_new = np.argwhere(unique_elements[:count_0]==elem)[0]
      occurrence[idx_new]+=1
      if occurrence[idx_new]== min_occ:
        indices[count_1] = idx
        reoccurring_elements[count_1] = elem
        count_1+=1
    # Stop once we have found k unique elements
    if count_1 == k:
      break
  if count_1 < k:
    print("fewer than k unique elements were found")
  # Slice the result in case fewer than k unique elements were found
  return reoccurring_elements[:count_1], indices[:count_1]


def trace_ind_to_doc(
  indices: np.ndarray, doc_lens: np.ndarray
):
  indices, doc_lens = np.array(indices), np.array(doc_lens)
  doc_lens_cumsum = np.cumsum(doc_lens)
  combined = np.hstack((indices, doc_lens_cumsum))
  sorted_indices = np.argsort(combined)
  index_map = np.zeros_like(sorted_indices)
  index_map[sorted_indices] = np.arange(len(combined))
  doc_indices = index_map[: len(indices)] - np.arange(len(indices))
  return doc_indices


def top_k_docs_info_per_topic(es_model, W_hat, doc_indices, k):
  anchor_doc_info = []
  for topic in range(es_model.num_topic):
    top_k_ind = doc_indices[np.argsort(W_hat[topic,:])[::-1][:k]]
    anchor_doc_info.append(top_k_ind)
  return anchor_doc_info


def fit_classic_btl_llmarena(dataset, alpha_ci=0.05, print_counts=True, top_k=None):
    """
    Fit standard BTL and return point estimates + confidence intervals.

    Parameters
    ----------
    print_counts : bool
        If True, prints number of comparisons per journal/model.
    top_k : int or None
        If set, prints only top_k and bottom_k by comparisons (useful if many models).

    Returns
    -------
    theta : (M,) point estimates
    ci_lower : (M,) lower bounds
    ci_upper : (M,) upper bounds
    se : (M,) standard errors
    """

    M = len(dataset.journal_names)

    # --- Print comparison counts per journal/model
    if print_counts:
        a_ids = np.asarray(dataset.model_a_ids, dtype=int)
        b_ids = np.asarray(dataset.model_b_ids, dtype=int)

        counts = np.bincount(np.concatenate([a_ids, b_ids]), minlength=M)

        # sort by most comparisons
        order = np.argsort(-counts)

        print("\n[BTL] Number of comparisons per model:")
        if top_k is None:
            for i in order:
                print(f"  {dataset.journal_names[i]:<35s} {counts[i]}")
        else:
            k = min(top_k, M)
            print(f"  (top {k} most compared)")
            for i in order[:k]:
                print(f"  {dataset.journal_names[i]:<35s} {counts[i]}")
            print(f"  (bottom {k} least compared)")
            for i in order[-k:][::-1]:
                print(f"  {dataset.journal_names[i]:<35s} {counts[i]}")

    # 1. Fit point estimates
    pairs = []
    for a, b, w in zip(dataset.model_a_ids, dataset.model_b_ids, dataset.winners):
        if w == 0:
            pairs.append((a, b))  # A won
        else:
            pairs.append((b, a))  # B won

    theta = choix.ilsr_pairwise(M, pairs, alpha=0.01)
    theta -= theta.mean()
    # counts = np.zeros(M, dtype=int)
    # wins = np.zeros(M, dtype=int)
    # losses = np.zeros(M, dtype=int)

    # for w, l in pairs:
    #     counts[w] += 1
    #     counts[l] += 1
    #     wins[w] += 1
    #     losses[l] += 1

    # print("\n[MADStat BTL] Comparisons per journal (top/bottom 10):")
    # order_counts = np.argsort(-counts)
    # topk = min(10, M)
    # for i in order_counts[:topk]:
    #     print(f"  {dataset.journal_names[i]:<10s}  comps={counts[i]:6d}  wins={wins[i]:6d}  losses={losses[i]:6d}")
    # print("  ...")
    # for i in order_counts[-topk:][::-1]:
    #     print(f"  {dataset.journal_names[i]:<10s}  comps={counts[i]:6d}  wins={wins[i]:6d}  losses={losses[i]:6d}")

    # # ---- 2) Win rate sanity (catch near-separation)
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     win_rate = wins / np.maximum(1, wins + losses)

    # extreme = np.where((wins + losses) > 0)[0]
    # extreme = extreme[(win_rate[extreme] < 0.05) | (win_rate[extreme] > 0.95)]
    # if len(extreme) > 0:
    #     print("\n[MADStat BTL] Journals with extreme win rate (<5% or >95%):")
    #     for i in extreme:
    #         print(f"  {dataset.journal_names[i]:<10s}  win_rate={win_rate[i]:.3f}  (wins={wins[i]}, losses={losses[i]})")


    # 2. Compute Fisher information matrix
    Fisher = np.zeros((M, M))

    for (winner, loser) in pairs:
        delta = theta[winner] - theta[loser]
        p = expit(delta)
        info = p * (1.0 - p)

        Fisher[winner, winner] += info
        Fisher[loser, loser]   += info
        Fisher[winner, loser]  -= info
        Fisher[loser, winner]  -= info

    # 3. Ridge to address singularity
    Fisher += (1.0 / M) * np.ones((M, M))

    # 4. Covariance + SE
    Cov = np.linalg.inv(Fisher)
    se = np.sqrt(np.diag(Cov))

    # 5. Confidence intervals
    z = norm.ppf(1 - alpha_ci / 2)
    ci_lower = theta - z * se
    ci_upper = theta + z * se

    return theta, ci_lower, ci_upper, se


def fit_classic_btl_madstat(dataset, alpha_ci=0.05):
    """
    Fit standard BTL on MADStat citation network with confidence intervals.
    
    Each citation (paper i cites paper j) is treated as a battle
    where journal(j) beats journal(i).
    
    Returns
    -------
    theta : (M,) point estimates
    ci_lower : (M,) lower bounds
    ci_upper : (M,) upper bounds
    se : (M,) standard errors
    """
    M = len(dataset.journal_names)
    
    # 1. Build pairs: (winner=cited journal, loser=citing journal)
    C = dataset.citation_matrix.tocoo()
    pairs = []
    for i, j in zip(C.row, C.col):
        ji = dataset.journal_ids[i]
        jj = dataset.journal_ids[j]
        if ji != jj:
            pairs.append((jj, ji))  # cited beats citing
    
    print(f"Number of citation pairs: {len(pairs)}")

    # 2. Fit point estimates
    theta = choix.ilsr_pairwise(M, pairs)
    theta -= theta.mean()
    
    # 3. Compute Fisher information matrix
    Fisher = np.zeros((M, M))
    
    for (winner, loser) in pairs:
        delta = theta[winner] - theta[loser]
        p = expit(delta)
        info = p * (1.0 - p)
        
        Fisher[winner, winner] += info
        Fisher[loser, loser]   += info
        Fisher[winner, loser]  -= info
        Fisher[loser, winner]  -= info
    
    # 4. Regularize (Fisher is singular due to identifiability)
    # Fisher += 0.01 * np.eye(M)
    Fisher += (1.0 / M) * np.ones((M, M))
    
    # 5. Invert to get covariance
    Cov = np.linalg.inv(Fisher)
    se = np.sqrt(np.diag(Cov))
    
    # 6. Confidence intervals
    z = norm.ppf(1 - alpha_ci / 2)
    ci_lower = theta - z * se
    ci_upper = theta + z * se

    # # ---- 3) Widest confidence intervals / largest SE
    # ci_width = ci_upper - ci_lower
    # order_se = np.argsort(-se)

    # print("\n[MADStat BTL] Largest SE / widest CI (top 10):")
    # topk = min(10, M)
    # for i in order_se[:topk]:
    #     print(f"  {dataset.journal_names[i]:<10s}  se={se[i]:.3f}  CI=[{ci_lower[i]:.2f}, {ci_upper[i]:.2f}]  width={ci_width[i]:.2f}  comps={counts[i]}")
  
    # # ---- 4) Fisher conditioning / near-singularity
    # eigvals = np.linalg.eigvalsh(Fisher)
    # print("\n[MADStat BTL] Fisher eigenvalues (min/median/max): "
    #       f"{eigvals.min():.3e} / {np.median(eigvals):.3e} / {eigvals.max():.3e}")

    # # condition number (can be huge if ill-conditioned)
    # cond = eigvals.max() / max(eigvals.min(), 1e-12)
    # print(f"[MADStat BTL] Fisher condition number (approx): {cond:.3e}")
    return theta, ci_lower, ci_upper, se

# ======================================================================
# PHASE 1: Precompute V_i and g_i for all documents
# ======================================================================


def _compute_h_vector_fast(X_i, N_i, b_i):
    """
    Compute h_r = sum_{u,v} b_u b_v kappa3_hat_{uvr} for all r.

    Fully vectorized — O(R^2) NumPy ops, NO Python loop over r.

    Parameters
    ----------
    X_i : (R,) float — raw word counts
    N_i : int — document length
    b_i : (R,) float — A @ G^{-1} @ s

    Returns
    -------
    h : (R,) float
    tau : float — b^T h = sum_{u,v,r} b_u b_v b_r kappa3_hat_{uvr}
    """
    R = len(X_i)
    if N_i < 3:
        return np.zeros(R), 0.0

    X = X_i.astype(np.float64)
    b = b_i
    N = float(N_i)

    d3 = N * (N - 1.0) * (N - 2.0)
    d2 = N * (N - 1.0)
    m = X / N

    # Aggregates
    bX = b * X                          # (R,)
    S1 = bX.sum()                       # scalar
    Sb = (b * m).sum()                  # scalar
    b2X = b * b * X                     # (R,)
    b2X_sum = b2X.sum()                 # scalar

    # ==================================================================
    # Piece 5:  2 m_r (sum_u b_u m_u)^2
    # ==================================================================
    P5 = 2.0 * m * (Sb * Sb)

    # ==================================================================
    # Piece 2:  m_r * Q_mf   where Q_mf = (S1^2 - sum(b^2 X)) / d2
    # ==================================================================
    Q_mf = (S1 * S1 - b2X_sum) / d2
    P2 = m * Q_mf

    # ==================================================================
    # Pieces 3,4:  Sb * F_r   where F_r = (S1 X_r - b_r X_r) / d2
    # ==================================================================
    F = (S1 * X - b * X) / d2
    P3 = Sb * F
    # P4 = P3 by symmetry

    # ==================================================================
    # Piece 1:  sum_{u,v} b_u b_v c_uvr   (VECTORIZED over r)
    #
    # For each r:
    #   S1r = S1 - b[r] X[r]           (sum_{u!=r} b_u X_u)
    #   b2Xr = b2X_sum - b[r]^2 X[r]   (sum_{u!=r} b_u^2 X_u)
    #
    #   part1 = (S1r^2 - b2Xr) * X[r] / d3
    #   part2 = 2 b[r] X[r] (X[r]-1) S1r / d3
    #   part3 = b[r]^2 X[r] (X[r]-1) (X[r]-2) / d3
    # ==================================================================
    S1r = S1 - b * X                       # (R,) vectorized
    b2Xr = b2X_sum - b * b * X             # (R,) vectorized

    Xm1 = X - 1.0
    Xm2 = X - 2.0

    part1 = (S1r * S1r - b2Xr) * X / d3
    part2 = 2.0 * b * X * Xm1 * S1r / d3
    part3 = b * b * X * Xm1 * Xm2 / d3

    P1 = part1 + part2 + part3

    # ==================================================================
    # Combine:  h = P1 - P2 - 2*P3 + P5
    # ==================================================================
    h = P1 - P2 - 2.0 * P3 + P5

    tau = b @ h

    return h, tau


def precompute_C1_document_quantities_fast(
    S,                  # (M, K)
    journal_ids,        # (n_docs,)
    Sigma_delta_w,      # (n_docs, K, K)
    A,                  # (R, K)
    G_inv,              # (K, K)
    X_counts,           # (R, n_docs) raw word counts
    doc_lens,           # (n_docs,)
):
    """
    Precompute V_i and g_i for all documents using vectorized h computation.

    Returns
    -------
    V_all : (n_docs,) float
    g_all : (n_docs, K) float
    """
    n_docs = len(journal_ids)
    K = S.shape[1]

    AG_inv = A @ G_inv          # (R, K)
    L = G_inv @ A.T             # (K, R)

    V_all = np.zeros(n_docs)
    g_all = np.zeros((n_docs, K))

    for i in range(n_docs):
        N_i = int(doc_lens[i])
        if N_i < 3:
            continue

        ji = int(journal_ids[i])
        s_i = S[ji]                             # (K,)

        # V_i = s_i^T Sigma_i s_i
        Sig_s = Sigma_delta_w[i] @ s_i          # (K,)
        V_all[i] = s_i @ Sig_s

        # b_i = A G^{-1} s_i
        b_i = AG_inv @ s_i                      # (R,)

        # h_i via FAST vectorized kappa3 contraction
        X_i = X_counts[:, i]
        h_i, _ = _compute_h_vector_fast(X_i, N_i, b_i)

        # g_i = L @ h_i
        g_all[i] = L @ h_i                     # (K,)

    return V_all, g_all


def precompute_C1_fully_vectorized(
    S,                  # (M, K)
    journal_ids,        # (n_docs,)
    Sigma_delta_w,      # (n_docs, K, K)
    A,                  # (R, K)
    G_inv,              # (K, K)
    X_counts,           # (R, n_docs) raw counts
    doc_lens,           # (n_docs,)
):
    """
    Fully vectorized V_i and g_i computation — no Python loops at all.

    Treats all documents as a batch. O(n_docs * R) memory.

    Returns
    -------
    V_all : (n_docs,) float
    g_all : (n_docs, K) float
    """
    n_docs = len(journal_ids)
    K = S.shape[1]
    R = A.shape[0]

    AG_inv = A @ G_inv          # (R, K)
    L = G_inv @ A.T             # (K, R)

    # ------------------------------------------------------------------
    # V_i = s_i^T Sigma_i s_i  (vectorized)
    # ------------------------------------------------------------------
    ji = journal_ids.astype(int)
    s_all = S[ji]                                           # (n_docs, K)
    Sig_s = np.einsum('ijk,ik->ij', Sigma_delta_w, s_all)  # (n_docs, K)
    V_all = np.einsum('ij,ij->i', s_all, Sig_s)            # (n_docs,)

    # ------------------------------------------------------------------
    # b_all = AG_inv @ s_i  for all docs  (vectorized)
    # ------------------------------------------------------------------
    b_all = s_all @ AG_inv.T                                # (n_docs, R)

    # ------------------------------------------------------------------
    # h_all via vectorized kappa3 contraction (all docs at once)
    # ------------------------------------------------------------------
    X = X_counts.T.astype(np.float64)   # (n_docs, R)
    N = doc_lens.astype(np.float64)     # (n_docs,)

    # Mask out short documents
    valid = N >= 3
    N_safe = np.where(valid, N, 3.0)    # avoid division by zero

    d3 = N_safe * (N_safe - 1.0) * (N_safe - 2.0)    # (n_docs,)
    d2 = N_safe * (N_safe - 1.0)                       # (n_docs,)
    m = X / N_safe[:, None]                             # (n_docs, R)

    b = b_all                                           # (n_docs, R)
    bX = b * X                                          # (n_docs, R)
    S1 = bX.sum(axis=1)                                 # (n_docs,)
    Sb = (b * m).sum(axis=1)                            # (n_docs,)
    b2X = b * b * X                                     # (n_docs, R)
    b2X_sum = b2X.sum(axis=1)                           # (n_docs,)

    Xm1 = X - 1.0
    Xm2 = X - 2.0

    # Piece 5: 2 m_r Sb^2
    P5 = 2.0 * m * (Sb * Sb)[:, None]                  # (n_docs, R)

    # Piece 2: m_r * (S1^2 - b2X_sum) / d2
    Q_mf = (S1 * S1 - b2X_sum) / d2                    # (n_docs,)
    P2 = m * Q_mf[:, None]                              # (n_docs, R)

    # Pieces 3+4: 2 * Sb * F_r  where F_r = (S1 X_r - b_r X_r) / d2
    F = (S1[:, None] * X - b * X) / d2[:, None]        # (n_docs, R)
    P34 = 2.0 * Sb[:, None] * F                         # (n_docs, R)

    # Piece 1: vectorized over both docs and r
    S1r = S1[:, None] - b * X                           # (n_docs, R)
    b2Xr = b2X_sum[:, None] - b * b * X                 # (n_docs, R)

    part1 = (S1r * S1r - b2Xr) * X / d3[:, None]       # (n_docs, R)
    part2 = 2.0 * b * X * Xm1 * S1r / d3[:, None]      # (n_docs, R)
    part3 = b * b * X * Xm1 * Xm2 / d3[:, None]        # (n_docs, R)

    P1 = part1 + part2 + part3                           # (n_docs, R)

    # h_all = P1 - P2 - P34 + P5
    h_all = P1 - P2 - P34 + P5                          # (n_docs, R)

    # Zero out invalid documents
    h_all[~valid] = 0.0
    V_all[~valid] = 0.0

    # g_all = h_all @ L.T = h_all @ A @ G_inv^T
    g_all = h_all @ L.T                                 # (n_docs, K)

    return V_all, g_all