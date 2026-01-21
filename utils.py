import numpy as np
from scipy.linalg import det
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections.abc import Sequence
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist
from scipy.spatial.distance import cdist


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


def pp_SPA(R, K, num_clusters=None):
    """
    Pseudo-point SPA tailored for your pipeline.
    Returns: list of K integer indices valid for R.
    """
    print("computing shape of R")
    n = R.shape[0]
    print("shape = ", n)
    
    # 1. Heuristic: How many pseudo-points? (Paper suggests L > K)
    # Usually 10*K to 20*K is good, but must be less than n.
    if num_clusters is None:
        num_clusters = min(n, max(K * 20, 50))
    
    # 2. Generating Pseudo-points (Denoising)
    # We use MiniBatchKMeans because it's fast and consistent with your PPTM
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, 
        batch_size=256,
        n_init='auto', 
        random_state=42
    )
    kmeans.fit(R)
    print("fitted k means")
    
    # These are the 'M' matrix in the paper (The centroids)
    pseudo_points = kmeans.cluster_centers_
    
    # 3. Run YOUR SPA on the pseudo-points
    # NOTE: These indices refer to the 'pseudo_points' array, NOT R!
    # Returns e.g., [5, 12, 0, ...] (indices 0 to num_clusters-1)
    _, pseudo_indices = SPA(pseudo_points, K)
    print("done doing SPA")
    
    # Extract the actual vectors of the chosen vertices
    chosen_centroids = pseudo_points[pseudo_indices]
    print("so far so good")
    
    # 4. Map Back: Find the original rows in R closest to these centroids
    # This is necessary because your Topic_SCORE needs indices [0...n]
    # Calculate distance between the K chosen centroids and all N original points
    # Shape: (K, n)
    dists = cdist(chosen_centroids, R, metric='euclidean')
    print("cmputing dist")
    
    # For each chosen centroid, find the index of the closest original point
    # Returns shape (K,)
    final_indices = np.argmin(dists, axis=1)
    print("done with final indices")
    print("final_indices = ", final_indices)
    print("type of indices = ", type(final_indices))
    return final_indices.tolist()





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