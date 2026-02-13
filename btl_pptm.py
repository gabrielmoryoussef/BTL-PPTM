"""
btl_pptm.py
Core Topic-Specific Journal Ranking Model

This module contains only the core BTL-PPTM model components:
1. BERT Embedding Extraction
2. PPTM Topic Modeling (using original GitHub code)
3. Novel Topic-Specific BTL Extension
4. Main BTLPPTMAnalyzer class for fitting

Analysis and visualization functions are in btl_analysis.py
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import IncrementalPCA
from transformers import AutoTokenizer, AutoModel
import gc
import warnings
import spacy
warnings.filterwarnings('ignore')

# Import standardized format
from data_format import AcademicDataset, validate_dataset_format

# Import PPTM modules (from GitHub repo)
from utils import *
from topic_score import Topic_SCORE
from embedded_score import Embedded_SCORE_C

# Import transformer libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ Warning: transformers not available. Install with: pip install torch transformers")

from typing import List
import numpy as np
from itertools import compress
import gc
from collections import Counter
from bert_embeddings import *
from vocab_cleaning import clean_vocabulary, drop_short_documents



class PPTMTopicModel:
    """
    PPTM Topic Model using Embedded_SCORE_C
    """
    def __init__(self,
                 n_topics: int = 10,
                 n_hyperwords: int = 500,
                 embed_dim: int = 30,
                 bandwidth: float = 0.1,
                 random_seed: int = 42,
                 vocab: List[str] = None,
                 **kwargs):
        self.n_topics = n_topics
        self.n_hyperwords = n_hyperwords
        self.embed_dim = embed_dim
        self.bandwidth = bandwidth
        self.random_seed = random_seed
        self.vocab = vocab
        self.kwargs = kwargs
        self.model: Embedded_SCORE_C = None
        self.topic_weights: np.ndarray = None

    def fit(self, embeddings: np.ndarray, doc_lengths: np.ndarray, verbose: bool = True) -> np.ndarray:
        if verbose:
            print(f"Fitting PPTM model: {self.n_topics} topics, {self.n_hyperwords} hyperwords...")
        #print("doc lengths = ",doc_lengths)
        n_docs = len(doc_lengths)
        self.model = Embedded_SCORE_C(
            num_doc=n_docs,
            doc_lens=doc_lengths,
            vocab=self.vocab,
            num_topic=self.n_topics,
            num_hyperword=self.n_hyperwords,
            word_repr=embeddings,
            dim_repr=self.embed_dim,
            bandwidth=self.bandwidth,
            random_seed=self.random_seed,
            clustering_method="mini-batch kmeans",
            remove_outlier=True,
            vertex_hunting_method="SPA",
            **self.kwargs
        )
        self.model.fit_all()
        if hasattr(self.model.model, 'W_hat'):
            self.topic_weights = self.model.model.W_hat.T
        else:
            self.model.model.get_W_hat(regression_type="ridge", lambda_reg=0.1)
            self.topic_weights = self.model.model.W_hat.T
        if verbose:
            print(f"✓ PPTM topic weights shape: {self.topic_weights.shape}")
        return self.topic_weights
    
    def get_topic_assignments(self) -> np.ndarray:
        """
        For each paper, return the index of its highest‐weight topic.
        """
        # self.topic_weights is shape (n_papers, n_topics)
        return np.argmax(self.topic_weights, axis=1)


class TopicSpecificBTL:
    """
    Topic-Specific Bradley–Terry–Luce model

    Paper score:
        μ_i = w_iᵀ S π_i  (π_i one-hot journal indicator)

    If use_debiased=False:
        → MLE via negative log-likelihood (L-BFGS)

    If use_debiased=True:
        → Debiased projected gradient descent solving
          ∇ℓ̂*_n(S) = 0, where we currently implement only C_data^[0]
          (we will add C^[1] later).
    """

    def __init__(
        self,
        n_topics: int,
        n_journals: int,
        reg_lambda: float = 0,
        use_debiased: bool = False,
        step_size: float = 50,
        max_iter: int = 20000,
        tol: float = 1e-6,
    ):
        self.K = int(n_topics)
        self.M = int(n_journals)
        self.reg_lambda = float(reg_lambda)

        self.use_debiased = bool(use_debiased)
        self.step_size = float(step_size)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.S: Optional[np.ndarray] = None  # (M × K)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _project(self, S: np.ndarray) -> np.ndarray:
        """Project onto identifiability subspace V: mean(S)=0."""
        return S - S.mean()
    
    def _project_llmarena(self, S: np.ndarray) -> np.ndarray:
        """Project onto identifiability subspace V: mean(S_i)=0 for all rows."""
        #print("so far shape of S is ", S.shape)
        return S - S.mean(axis=0, keepdims=True)


    @staticmethod
    def _sigmoid(x):
        # stable enough for typical scales; adjust if you see overflow
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_prime(self, x):
        s = self._sigmoid(x)
        return s * (1.0 - s)

    # ------------------------------------------------------------------
    # Scores
    # ------------------------------------------------------------------

    def _compute_paper_scores(
        self,
        S: np.ndarray,
        topic_weights: np.ndarray,
        journal_ids: np.ndarray
    ) -> np.ndarray:
        """
        μ_i = w_iᵀ S[journal_i]
        topic_weights: (n, K)
        S: (M, K)
        journal_ids: (n,)
        """
        n = topic_weights.shape[0]
        mu = np.zeros(n, dtype=float)
        for i in range(n):
            j = int(journal_ids[i])
            if 0 <= j < self.M:
                mu[i] = float(topic_weights[i] @ S[j])
        return mu


    def _negative_log_likelihood(
        self,
        S_flat: np.ndarray,
        i_indices: np.ndarray, 
        j_indices: np.ndarray,
        topic_weights: np.ndarray,
        journal_ids: np.ndarray
    ) -> float:
        S = S_flat.reshape(self.M, self.K)
        n_edges = len(i_indices)  # Get N_e

        # 1. Vectorized Mu
        mu = np.sum(topic_weights * S[journal_ids], axis=1)
        
        # 2. Vectorized Likelihood (Sum)
        mu_i = mu[i_indices]
        mu_j = mu[j_indices]
        delta = mu_j - mu_i
        
        # Sum of Log-Likelihoods
        # log(sigmoid(x)) = -softplus(-x)
        total_nll = np.sum(np.logaddexp(0, -delta))
        
        # 3. SCALED Loss (Mean NLL)
        # We divide the data term by n_edges
        # mean_nll = total_nll / n_edges
        mean_nll = total_nll

        # 4. Regularization (Unscaled or Standard Scale)
        # We keep this independent of n_edges so lambda=1e-3 means the same strength 
        # regardless of dataset size.
        reg = self.reg_lambda * np.sum(S**2)
        
        return mean_nll + reg
    

    def _mle_gradient(
        self,
        S_flat: np.ndarray,
        i_indices: np.ndarray, 
        j_indices: np.ndarray,
        topic_weights: np.ndarray,
        journal_ids: np.ndarray
    ) -> np.ndarray:
        """
        Computes the gradient of the Negative Log Likelihood w.r.t S.
        
        Optimized:
        1. Vectorized (no loops over citations).
        2. Scaled by 1/n_edges so gradients are ~O(1) instead of O(N).
        3. Uses np.add.at for fast aggregation.
        """
        # Reconstruct matrix S from the optimizer's flat vector
        S = S_flat.reshape(self.M, self.K)
        n_edges = len(i_indices)
        
        grad = np.zeros_like(S, dtype=float)
        
        # ------------------------------------------------------------
        # 1. Compute Scores (Vectorized)
        # ------------------------------------------------------------
        # mu[k] = dot(w_k, S[journal_ids[k]])
        # shape: (n_docs,)
        mu = np.sum(topic_weights * S[journal_ids], axis=1)
        
        # ------------------------------------------------------------
        # 2. Compute "Error" for every citation (Vectorized)
        # ------------------------------------------------------------
        # For gradient, we use (Loser - Winner)
        # p = sigmoid(mu_i - mu_j) = 1 / (1 + exp(-(mu_i - mu_j)))
        # This 'p' is exactly the gradient magnitude for BTL.
        delta = mu[i_indices] - mu[j_indices]
        p = self._sigmoid(delta)  # shape: (n_edges,)
        
        # ------------------------------------------------------------
        # 3. Accumulate Gradients (The "Push")
        # ------------------------------------------------------------
        # We need to update S for the journals involved.
        # The update is proportional to the topic weights of the papers.
        
        # Prepare the weighted pushes
        # Shape becomes (n_edges, K) via broadcasting
        weighted_p_i = p[:, None] * topic_weights[i_indices]
        weighted_p_j = p[:, None] * topic_weights[j_indices]
        
        # Aggregate into the gradient matrix
        # Subtract from Citing (Loser) -> Penalize
        np.add.at(grad, journal_ids[i_indices], -weighted_p_i)
        
        # Add to Cited (Winner) -> Reward
        np.add.at(grad, journal_ids[j_indices], weighted_p_j)

        # ------------------------------------------------------------
        # 4. Scaling and Regularization
        # ------------------------------------------------------------
        # A. Normalize the data gradient by N_e so it represents 
        #    "Average Gradient per Citation". This prevents explosion.
        grad /= n_edges
        
        # B. Add L2 Regularization Gradient
        # Derivative of (lambda * sum(S^2)) is (2 * lambda * S)
        # We DO NOT divide this by n_edges, to keep regularization strong.
        # (Since we are minimizing NLL, we *add* the gradient of the penalty,
        #  but since we return -grad at the end, we subtract it here).
        
        # NOTE: The objective is J = - (LogLikelihood/N) + Lambda * |S|^2
        # Gradient of LogLikelihood part is 'grad' (positive ascent direction).
        # Gradient of Penalty is 2*lambda*S.
        # We want to MINIMIZE J.
        # Grad_J = - (grad_LL) + 2*lambda*S
        
        # Since 'grad' currently holds the ascent direction (dLL/dS), 
        # the descent direction is -grad.
        # So we can just construct the total gradient for minimization directly:
        
        total_grad = -grad + (2.0 * self.reg_lambda * S)
        
        return total_grad.flatten()
    # In TopicSpecificBTL class...
    def _mle_gradient_battles(self, S_flat, model_A_ids, model_B_ids, winners, topic_weights):
        S = S_flat.reshape(self.M, self.K)
        n_battles = len(winners)
        grad = np.zeros_like(S)

        # 1. Scores
        mu_A = np.sum(topic_weights * S[model_A_ids], axis=1)
        mu_B = np.sum(topic_weights * S[model_B_ids], axis=1)

        # 2. Probability A beats B
        delta = mu_A - mu_B
        prob_A = self._sigmoid(delta)

        # 3. Error (Target - Prediction)
        # If A won (winners=0), Target=1. Error = 1 - prob_A
        # If B won (winners=1), Target=0. Error = 0 - prob_A
        error = (1.0 - winners) - prob_A

        # 4. Update
        weighted_push = error[:, None] * topic_weights
        np.add.at(grad, model_A_ids, weighted_push)  # Push A
        np.add.at(grad, model_B_ids, -weighted_push) # Pull B

        # 5. Regularization
        grad /= n_battles
        print("grad = ", grad)
        grad -= 2.0 * self.reg_lambda * S
        return -grad.flatten()
    # ------------------------------------------------------------------
    # Debiased gradient (ONLY C_data^[0] implemented)
    # ------------------------------------------------------------------

    # def _debiased_gradient_C0(
    #     self,
    #     S: np.ndarray,                        # (M, K)
    #     citation_pairs,                       # list of (i, j)
    #     topic_weights: np.ndarray,            # (n, K)
    #     journal_ids: np.ndarray,              # (n,)
    #     Sigma_delta_w: np.ndarray,            # (n, K, K)  <-- CHANGED TYPE
    # ) -> np.ndarray:
    #     """
    #     Implements C_data^[0] using full covariance matrices.
    #     """
    #     grad = np.zeros_like(S, dtype=float)
    #     C0   = np.zeros_like(S, dtype=float)

    #     mu = self._compute_paper_scores(S, topic_weights, journal_ids)

    #     for i, j in citation_pairs:
    #         ji = int(journal_ids[i])
    #         jj = int(journal_ids[j])

    #         if ji < 0 or jj < 0 or ji >= self.M or jj >= self.M:
    #             continue

    #         delta = mu[i] - mu[j]
    #         p     = self._sigmoid(delta)
    #         alpha = self._sigmoid_prime(delta)

    #         # Standard Gradient
    #         # Penalize the citing journal (i)
    #         grad[ji] -= p * topic_weights[i]
            
    #         # Reward the cited journal (j)
    #         grad[jj] += p * topic_weights[j]

    #         # --------------------------------------------------------
    #         # C_data^[0] correction
    #         #
    #         # Σ_{δw,i} S(π_i π_iᵀ) simplifies to:
    #         # Vector = Σ_{δw,i} @ S[ji]
    #         # (Matrix-Vector multiplication: (K,K) @ (K,) -> (K,))
    #         # --------------------------------------------------------
            
    #         # Correction for i
    #         correction_i = Sigma_delta_w[i] @ S[ji]
    #         C0[ji] += alpha * correction_i
    #         #C0[ji] -= alpha * correction_i

    #         # Correction for j
    #         correction_j = Sigma_delta_w[j] @ S[jj]
    #         C0[jj] += alpha * correction_j
    #         #C0[jj] -= alpha * correction_j

    #     #grad -= 2.0 * self.reg_lambda * S
    #     grad_star = grad - C0

    #     return grad_star


    def _debiased_gradient_C0(
        self,
        S,                    # (M, K) numpy
        citation_pairs,       # list of (i, j)
        topic_weights,        # (n_docs, K) numpy
        journal_ids,          # (n_docs,) numpy
        Sigma_delta_w,        # (n_docs, K, K) numpy
        device='cuda',
    ):
        """
        GPU-accelerated C0 debiased gradient for journal citations.
        Falls back to CPU if CUDA not available.
        """
        
        if not torch.cuda.is_available() and device == 'cuda':
            device = 'cpu'
        
        # Convert to tensors
        S_t = torch.tensor(S, dtype=torch.float64, device=device)
        tw_t = torch.tensor(topic_weights, dtype=torch.float64, device=device)
        jids_t = torch.tensor(journal_ids, dtype=torch.long, device=device)
        Sig_t = torch.tensor(Sigma_delta_w, dtype=torch.float64, device=device)
        
        pairs = np.array(citation_pairs, dtype=int)
        i_idx_t = torch.tensor(pairs[:, 0], dtype=torch.long, device=device)
        j_idx_t = torch.tensor(pairs[:, 1], dtype=torch.long, device=device)
        
        n_edges = len(pairs)
        M, K = S.shape
        
        ji_t = jids_t[i_idx_t]  # (n_edges,)
        jj_t = jids_t[j_idx_t]  # (n_edges,)
        
        # ------------------------------------------
        # 1. Scores and sigmoid
        # ------------------------------------------
        mu = (tw_t * S_t[jids_t]).sum(dim=1)     # (n_docs,)
        delta = mu[i_idx_t] - mu[j_idx_t]        # (n_edges,)
        p = torch.sigmoid(delta)                   # (n_edges,)
        alpha = p * (1.0 - p)                      # (n_edges,)
        
        # ------------------------------------------
        # 2. MLE gradient
        # ------------------------------------------
        grad = torch.zeros(M, K, dtype=torch.float64, device=device)
        
        wp_i = p.unsqueeze(1) * tw_t[i_idx_t]    # (n_edges, K)
        wp_j = p.unsqueeze(1) * tw_t[j_idx_t]    # (n_edges, K)
        
        grad.scatter_add_(0, ji_t.unsqueeze(1).expand_as(wp_i), -wp_i)
        grad.scatter_add_(0, jj_t.unsqueeze(1).expand_as(wp_j),  wp_j)
        
        # ------------------------------------------
        # 3. C0 correction
        # ------------------------------------------
        C0 = torch.zeros(M, K, dtype=torch.float64, device=device)
        
        # Sigma[i] @ S[ji] for each edge
        S_ji = S_t[ji_t].unsqueeze(-1)                        # (n_edges, K, 1)
        corr_i = (Sig_t[i_idx_t] @ S_ji).squeeze(-1)          # (n_edges, K)
        
        S_jj = S_t[jj_t].unsqueeze(-1)                        # (n_edges, K, 1)
        corr_j = (Sig_t[j_idx_t] @ S_jj).squeeze(-1)          # (n_edges, K)
        
        wc_i = alpha.unsqueeze(1) * corr_i
        wc_j = alpha.unsqueeze(1) * corr_j
        
        C0.scatter_add_(0, ji_t.unsqueeze(1).expand_as(wc_i), wc_i)
        C0.scatter_add_(0, jj_t.unsqueeze(1).expand_as(wc_j), wc_j)
        
        # ------------------------------------------
        # 4. Combine and return as numpy
        # ------------------------------------------
        grad_star = grad - C0
        return grad_star.cpu().numpy()

    # def _debiased_gradient_C0_VEC(
    #     self,
    #     S,                    # (M, K)
    #     citation_pairs,       # list of (i, j) tuples
    #     topic_weights,        # (n_docs, K)
    #     journal_ids,          # (n_docs,)
    #     Sigma_delta_w,        # (n_docs, K, K)
    # ):
    #     """
    #     Vectorized C0 debiased gradient for journal citations.
    #     Replaces the slow per-citation Python loop.
    #     """
    #     # Convert to arrays
    #     pairs = np.array(citation_pairs, dtype=int)
    #     i_idx = pairs[:, 0]
    #     j_idx = pairs[:, 1]
    #     n_edges = len(i_idx)
        
    #     ji = journal_ids[i_idx]  # (n_edges,) journal of citing paper
    #     jj = journal_ids[j_idx]  # (n_edges,) journal of cited paper
        
    #     # ------------------------------------------
    #     # 1. Compute scores and sigmoid
    #     # ------------------------------------------
    #     mu = np.sum(topic_weights * S[journal_ids], axis=1)  # (n_docs,)
    #     delta = mu[i_idx] - mu[j_idx]                         # (n_edges,)
    #     p = self._sigmoid(delta)                               # (n_edges,)
    #     alpha = p * (1.0 - p)                                  # (n_edges,)
        
    #     # ------------------------------------------
    #     # 2. Standard MLE gradient (ascent direction)
    #     # ------------------------------------------
    #     grad = np.zeros_like(S)
        
    #     weighted_p_i = p[:, None] * topic_weights[i_idx]  # (n_edges, K)
    #     weighted_p_j = p[:, None] * topic_weights[j_idx]  # (n_edges, K)
        
    #     np.add.at(grad, ji, -weighted_p_i)  # citing (loser)
    #     np.add.at(grad, jj,  weighted_p_j)  # cited (winner)
        
    #     # ------------------------------------------
    #     # 3. C0 correction (vectorized)
    #     # ------------------------------------------
    #     C0 = np.zeros_like(S)
        
    #     # correction_i = Sigma[i] @ S[ji]  for each edge
    #     # Sigma[i_idx] has shape (n_edges, K, K)
    #     # S[ji] has shape (n_edges, K)
    #     S_ji = S[ji][:, :, None]                                    # (n_edges, K, 1)
    #     correction_i = (Sigma_delta_w[i_idx] @ S_ji).squeeze(-1)    # (n_edges, K)
        
    #     S_jj = S[jj][:, :, None]                                    # (n_edges, K, 1)
    #     correction_j = (Sigma_delta_w[j_idx] @ S_jj).squeeze(-1)    # (n_edges, K)
        
    #     weighted_corr_i = alpha[:, None] * correction_i  # (n_edges, K)
    #     weighted_corr_j = alpha[:, None] * correction_j  # (n_edges, K)
        
    #     np.add.at(C0, ji, weighted_corr_i)
    #     np.add.at(C0, jj, weighted_corr_j)
        
    #     # ------------------------------------------
    #     # 4. Combine
    #     # ------------------------------------------
    #     grad_star = grad - C0
        
    #     return grad_star


    # ------------------------------------------------------------------
    # Main fit
    # ------------------------------------------------------------------

    def fit(
        self,
        citation_matrix: sp.csr_matrix,
        topic_weights: np.ndarray,          # W_hat, shape (n_docs, K)
        journal_ids: np.ndarray,             # shape (n_docs,)
        Sigma_delta_w: np.ndarray = None,  # shape (N, K, K)
        verbose: bool = True,
    ):
        """
        Fit Topic-Specific BTL model.

        If use_debiased = False:
            Standard MLE via L-BFGS.

        If use_debiased = True:
            Projected Gradient Descent solving
                ∇ℓ̂*_n(S) = 0
            where
                ∇ℓ̂*_n(S) = ∇ℓ̂_n(S) − C_data^{[0]}(S)
        """

        # ------------------------------------------------------------
        # Extract citation edges E = {(i,j)}
        # ------------------------------------------------------------
        citation_pairs = list(zip(*citation_matrix.nonzero()))
        print("len citation pairs = ", len(citation_pairs))
        print("len entry citation = ", len(citation_pairs[0]))
        print(citation_pairs[0:5])
        n_edges = len(citation_pairs)

        if verbose:
            print("Fitting TopicSpecificBTL")
            print(f"  • journals = {self.M}, topics = {self.K}")
            print(f"  • citations = {n_edges:,}")
            print(f"  • mode = {'DEBIASED PGD' if self.use_debiased else 'MLE (L-BFGS)'}")

        # ============================================================
        # Case 1 — Classical MLE (unchanged)
        # ============================================================
        # if not self.use_debiased:
        #     S0 = np.random.normal(0, 0.1, (self.M, self.K))
        #     S0 = self._project(S0)
        #     # Prepare arrays outside the minimization loop
        #     citation_pairs_arr = np.array(citation_pairs)
        #     i_indices = citation_pairs_arr[:, 0]
        #     j_indices = citation_pairs_arr[:, 1]
            

        #     result = minimize(
        #         self._negative_log_likelihood,
        #         S0.flatten(),
        #         # Pass the arrays, not the list of tuples
        #         args=(i_indices, j_indices, topic_weights, journal_ids),
        #         method="L-BFGS-B",
        #         jac=self._mle_gradient,  # <--- PASS THE GRADIENT HERE!
        #         options={"maxiter": self.max_iter, "ftol": 1e-9},
        #     )
            
        #     self.S = self._project(result.x.reshape(self.M, self.K))

        #     if verbose:
        #         print("✓ MLE optimization completed")
        #         print(f"  success = {result.success}")
        #         print(f"  final NLL = {result.fun:.4f}")

        #     return self



        # --- Momentum Projected Gradient Descent (MLE case) ---
        
                # ============================================================
        # Case 1 — Classical MLE (Iterative GD + Projection)
        # ============================================================
        if not self.use_debiased:
            #     # Extract citation edges into arrays (vectorized ops use these)
                citation_pairs_arr = np.array(citation_pairs, dtype=int)
                i_indices = citation_pairs_arr[:, 0]
                j_indices = citation_pairs_arr[:, 1]
                n_edges = len(i_indices)

                # Initialize S
                S = np.random.normal(0, 0.1, (self.M, self.K))
                S = self._project(S)   # identifiability constraint (global mean 0)

            #     beta = 0.9          # momentum (0.8–0.95 typical)
            #     lr = self.step_size # base learning rate

            #     V = np.zeros((self.M, self.K), dtype=float)

            #     for t in range(self.max_iter):
            #         grad = self._mle_gradient(
            #             S.flatten(),
            #             i_indices,
            #             j_indices,
            #             topic_weights,
            #             journal_ids
            #         ).reshape(self.M, self.K)   # grad of NLL (minimization)

            #         # Momentum update
            #         V = beta * V + grad

            #         # Step + projection
            #         S_new = self._project(S - lr * V)

            #         diff_norm = np.linalg.norm(S_new - S)

            #         if verbose and t % 10 == 0:
            #             nll = self._negative_log_likelihood(
            #                 S_new.flatten(),
            #                 i_indices,
            #                 j_indices,
            #                 topic_weights,
            #                 journal_ids
            #             )
            #             grad_norm = np.linalg.norm(grad)
            #             print(f"  iter {t:4d} | diff={diff_norm:.2e} | grad={grad_norm:.2e} | NLL={nll/n_edges:.6f}")

            #         if diff_norm < self.tol:
            #             if verbose:
            #                 print(f"✓ Converged at iter {t} (Diff: {diff_norm:.2e})")
            #             break

            #         S = S_new

            #     self.S = S
            #     if verbose:
            #         print("✓ MLE (momentum PGD) completed")
            #     return self


                for t in range(self.max_iter):
                    # 1) Gradient of objective for MINIMIZATION (same convention as _mle_gradient)
                    grad = self._mle_gradient(
                        S.flatten(),
                        i_indices,
                        j_indices,
                        topic_weights,
                        journal_ids
                    ).reshape(self.M, self.K)

                    # 2) Gradient step + project
                    S_new = self._project(S - self.step_size * grad)

                    # 3) Convergence
                    diff_norm = np.linalg.norm(S_new - S)

                    if verbose and t % 10 == 0:
                        nll = self._negative_log_likelihood(
                            S_new.flatten(),
                            i_indices,
                            j_indices,
                            topic_weights,
                            journal_ids
                        )
                        print(f"  iter {t:4d} | diff={diff_norm:.2e} | NLL={nll:.6f}")
                        print("grad_norm", np.linalg.norm(grad))


                    if diff_norm < self.tol:
                        if verbose:
                            print(f"✓ Converged at iter {t} (Diff: {diff_norm:.2e})")
                        break

                    S = S_new

                self.S = S
                if verbose:
                    print("✓ MLE (iterative) completed")
                return self

        

        # ============================================================
        # Case 2 — Debiased Projected Gradient Descent
        # ============================================================

        # ---------------------------
        # Sanity checks
        # ---------------------------
        if Sigma_delta_w is None:
            raise ValueError(
                "Sigma_delta_w must be provided for debiased fitting "
                "(implements C_data^{[0]})"
            )

        if Sigma_delta_w.ndim != 3 or Sigma_delta_w.shape[0] != len(journal_ids):
                raise ValueError(f"Sigma_delta_w must be (n_docs, K, K). Got {Sigma_delta_w.shape}")

        # ---------------------------
        # Initialization of S
        # ---------------------------
        S = np.random.normal(0, 0.1, (self.M, self.K))
        S = self._project(S)

        # ============================================================
        # Main PGD loop
        # ============================================================
        # ============================================================
        # Main PGD loop
        # ============================================================
        for t in range(self.max_iter):
            # 1. Compute debiased gradient (The "Score")
            # grad = self._debiased_gradient_C0(
            #     S=S,
            #     citation_pairs=citation_pairs,
            #     topic_weights=topic_weights,
            #     journal_ids=journal_ids,
            #     Sigma_delta_w=Sigma_delta_w,
            # )

            # # 2. Update Step
            # # Note: We divide by n_edges to keep the scale consistent 
            # # regardless of dataset size, similar to a mean gradient.
            # step = self.step_size * grad / n_edges
            # S_new = self._project(S + step)
            # Before the loop
            velocity = np.zeros_like(S)
            beta = 0.9  # Common momentum value (0.9 means 90% of speed is kept)

            # Inside the loop
            grad = self._debiased_gradient_C0(
                S=S,
                citation_pairs=citation_pairs,
                topic_weights=topic_weights,
                journal_ids=journal_ids,
                Sigma_delta_w=Sigma_delta_w,
            )
            # Update velocity
            velocity = beta * velocity + (1 - beta) * (grad / n_edges)
            # Use velocity instead of raw grad
            S_new = self._project(S + self.step_size * velocity)

            # 3. Calculate convergence metrics
            diff_norm = np.linalg.norm(S_new - S)  # How much did S change?
            grad_norm = np.linalg.norm(grad)       # How close to 0 is the gradient?

            # 4. Print status
            if verbose and t % 10 == 0:
                print(
                    f"  iter {t:4d} | "
                    f"Update Diff: {diff_norm:.2e} | "
                    f"Grad Norm: {grad_norm:.2e}"
                )

            # 5. Check Convergence
            if diff_norm < self.tol:
                if verbose:
                    print(f"✓ Converged at iter {t} (Diff: {diff_norm:.2e})")
                break

            S = S_new

        self.S = S
        if verbose: print("✓ Debiased PGD completed")
        return self
    


    def _negative_log_likelihood_battles(self, S_flat, model_A_ids, model_B_ids, winners, topic_weights):
            """
            NLL for Battles: Sum of log(sigmoid(Winner_Score - Loser_Score))
            """
            S = S_flat.reshape(self.M, self.K)
            n_battles = len(winners)
            
            # 1. Scores
            # shape: (n_battles,)
            mu_A = np.sum(topic_weights * S[model_A_ids], axis=1)
            mu_B = np.sum(topic_weights * S[model_B_ids], axis=1)
            
            # 2. Probability A beats B
            delta = mu_A - mu_B
            
            # 3. Log Likelihood
            # If A won (0), we want log(sigma(delta))
            # If B won (1), we want log(sigma(-delta))
            
            # sign = +1 if A wins, -1 if B wins
            sign = 1.0 - 2.0 * winners
            
            # aligned_delta is (Winner - Loser)
            aligned_delta = sign * delta
            
            # NLL = - sum( log( sigmoid( aligned_delta ) ) )
            # log(sigmoid(x)) = -softplus(-x)
            total_nll = np.sum(np.logaddexp(0, -aligned_delta))
            
            # Mean NLL + Regularization
            reg = self.reg_lambda * np.sum(S**2)
            return (total_nll / n_battles) + reg

    def _mle_gradient_battles(self, S_flat, model_A_ids, model_B_ids, winners, topic_weights):
        """
        Computes the gradient of NLL w.r.t S for Battles.
        Returns: Flattened NEGATIVE gradient (for minimization).
        """
        S = S_flat.reshape(self.M, self.K)
        n_battles = len(winners)
        grad = np.zeros_like(S)
        self.reg_lambda = 0

        # 1. Scores
        mu_A = np.sum(topic_weights * S[model_A_ids], axis=1)
        mu_B = np.sum(topic_weights * S[model_B_ids], axis=1)

        # 2. Probability A beats B
        delta = mu_A - mu_B
        prob_A = self._sigmoid(delta)

        # 3. Error (Target - Prediction)
        # If A won (winners=0), Target=1. Error = 1 - prob_A
        # If B won (winners=1), Target=0. Error = 0 - prob_A
        error = (1.0 - winners) - prob_A

        # 4. Update (Ascent Direction)
        weighted_push = error[:, None] * topic_weights
        
        np.add.at(grad, model_A_ids, weighted_push)  # Push A
        np.add.at(grad, model_B_ids, -weighted_push) # Pull B

        # 5. Scaling & Regularization
        grad /= n_battles
        
        # Combine: -Gradient(Likelihood) + Gradient(Penalty)
        # Gradient(Penalty) = 2 * lambda * S
        # Since we return the gradient for MINIMIZATION, we flip the sign of Likelihood grad.
        total_grad = -grad + (2.0 * self.reg_lambda * S)
        
        return total_grad.flatten()

    # ------------------------------------------------------------------
    # 2. DEBIASED GRADIENTS (Battles)
    # ------------------------------------------------------------------

    def _debiased_gradient_battles(
        self,
        S: np.ndarray,
        model_A_ids: np.ndarray,
        model_B_ids: np.ndarray,
        winners: np.ndarray,
        topic_weights: np.ndarray,
        Sigma_delta_w: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the Debiased Gradient for PGD (Ascent Direction).
        Formula: g(S) = grad_MLE(S) - C0(S)
        """
        n_battles = len(winners)

        # ------------------------------------------
        # 1. Standard MLE Gradient (Ascent Direction)
        # ------------------------------------------
        # Reuse the existing function, but recover the pure ascent gradient
        # (remove the sign flip and regularization added for scipy.minimize)
        
        # Recalculate manually for clarity to avoid sign confusion:
        grad_mle = np.zeros_like(S)
        
        mu_A = np.sum(topic_weights * S[model_A_ids], axis=1)
        mu_B = np.sum(topic_weights * S[model_B_ids], axis=1)
        delta = mu_A - mu_B
        p = self._sigmoid(delta) # Prob A wins
        
        # Error = (Target - Prediction)
        error = (1.0 - winners) - p
        
        weighted_push = error[:, None] * topic_weights
        np.add.at(grad_mle, model_A_ids, weighted_push)
        np.add.at(grad_mle, model_B_ids, -weighted_push)
        
        # Note: We do NOT normalize by n_battles yet, we do it at the very end
        # to match the scale of C0 calculation.

        # ------------------------------------------
        # 2. Bias Correction Term (C0)
        # ------------------------------------------
        C0 = np.zeros_like(S)
        
        # alpha = p * (1-p)
        alpha = p * (1.0 - p)
        
        # Correction Vector = Sigma_i @ S[Model]
        # Shapes: Sigma=(N,K,K), S[IDs]=(N,K) -> Need (N,K,1)
        
        S_A_vecs = S[model_A_ids][:, :, None]
        correction_A = (Sigma_delta_w @ S_A_vecs).squeeze(-1) # (N, K)
        
        S_B_vecs = S[model_B_ids][:, :, None]
        correction_B = (Sigma_delta_w @ S_B_vecs).squeeze(-1) # (N, K)
        
        # Weighted accumulation
        # C0[Model] += alpha * Correction[Model]
        
        weighted_corr_A = alpha[:, None] * correction_A
        weighted_corr_B = alpha[:, None] * correction_B
        
        np.add.at(C0, model_A_ids, weighted_corr_A)
        np.add.at(C0, model_B_ids, weighted_corr_B)
        
        # ------------------------------------------
        # 3. Combine
        # ------------------------------------------
        # Gradient = (Grad_MLE - C0) / N_battles
        # (Regularization is handled in the update step of the loop, or here)
        
        grad_star = (grad_mle - C0) / n_battles
        
        return grad_star
    
    
    def fit_battles(
        self, 
        model_A_ids, 
        model_B_ids, 
        winners, 
        topic_weights, 
        Sigma_delta_w=None, 
        verbose=True
    ):
        """
        Fits the BTL model using Battle List data.
        Handles both Standard MLE and Debiased PGD.
        """
        
        if verbose:
            print(f"   Fitting Battle Model ({self.M} models x {self.K} topics)")
            print(f"   Mode: {'DEBIASED PGD' if self.use_debiased else 'STANDARD MLE'}")
            
        # ============================================================
        # Case 1: Standard MLE (L-BFGS-B)
        # ============================================================
        # if not self.use_debiased:
        #     #S0 = np.ones((self.M, self.K))
        #     S0 = np.random.normal(0, 0.1, (self.M, self.K))
        #     S0 = self._project_llmarena(S0)

        #     result = minimize(
        #         fun=self._negative_log_likelihood_battles,
        #         x0=S0.flatten(),
        #         args=(model_A_ids, model_B_ids, winners, topic_weights),
        #         method="L-BFGS-B",
        #         jac=self._mle_gradient_battles,
        #         options={"maxiter": self.max_iter, "ftol": self.tol}
        #     )
        
        #     self.S = self._project_llmarena(result.x.reshape(self.M, self.K))

    
        #     if verbose:
        #         print(f"   ✓ Optimization Converged: {result.success}, Loss: {result.fun:.4f}")
        #     return self
        if not self.use_debiased:
            S = np.random.normal(0, 0.1, (self.M, self.K))
            S = self._project_llmarena(S)
            
            for t in range(self.max_iter):
                # 1. Compute gradient (minimization direction)
                grad = self._mle_gradient_battles(
                    S.flatten(), model_A_ids, model_B_ids, winners, topic_weights
                ).reshape(self.M, self.K)
                
                # 2. Gradient step + project
                S_new = self._project_llmarena(S - self.step_size * grad)
                
                # 3. Convergence check
                diff = np.linalg.norm(S_new - S)

                if t < 5:
                    print(f"  iter {t} col sums: {S_new.sum(axis=0)}")
                
                if verbose and t % 10 == 0:
                    nll = self._negative_log_likelihood_battles(
                        S_new.flatten(), model_A_ids, model_B_ids, winners, topic_weights
                    )
                    print(f"  iter {t:4d} | diff={diff:.2e} | NLL={nll:.6f}")
                
                if diff < self.tol:
                    if verbose: print(f"  ✓ Converged at iter {t}")
                    break
                
                S = S_new
                   
            self.S = S
            print(self.S.shape)
            print("sum of columns of S = ", self.S.sum(axis=0))
            return self

        # ============================================================
        # Case 2: Debiased PGD
        # ============================================================
        
        # Sanity Checks
        if Sigma_delta_w is None:
            raise ValueError("Debiased mode requires Sigma_delta_w")
            
        n_battles = len(winners)
        
        # Initialize S
        S = np.random.normal(0, 0.1, (self.M, self.K))
        S = self._project_llmarena(S)
        #S = self._project(S)
        
        # PGD Loop
        for t in range(self.max_iter):
            # 1. Compute Gradient (Ascent Direction)
            grad = self._debiased_gradient_battles(
                S, model_A_ids, model_B_ids, winners, topic_weights, Sigma_delta_w
            )
            
            # 2. Add Regularization Gradient (Penalty = Descent)
            # Total Gradient = Data_Grad - 2*lambda*S
            total_grad = grad - (2.0 * self.reg_lambda * S)
            
            # 3. Update Step (Ascent)
            S_new = self._project_llmarena(S + self.step_size * total_grad)
            #S_new = self._project(S + self.step_size * total_grad)
            
            # 4. Convergence Check
            diff_norm = np.linalg.norm(S_new - S)
            grad_norm = np.linalg.norm(total_grad)
            
            if verbose and t % 10 == 0:
                print(f"   iter {t:4d} | Diff: {diff_norm:.2e} | Grad: {grad_norm:.2e}")
                
            if diff_norm < self.tol:
                if verbose: print(f"   ✓ Converged at iter {t}")
                break
                
            S = S_new
            
        self.S = S
        if verbose: print("   ✓ Debiased PGD completed")
        return self

    def get_journal_rankings(self, topic_distribution: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get journal rankings for specific topic distribution
        
        Parameters:
        - topic_distribution: topic weights (K,). If None, uses uniform distribution
        
        Returns:
        - rankings: journal indices sorted by score (best to worst)
        - scores: journal scores for this topic distribution
        """
        if self.S is None:
            raise ValueError("Model must be fitted first")
            
        if topic_distribution is None:
            topic_distribution = np.ones(self.K) / self.K  # Uniform
            
        # Compute journal scores: S @ topic_distribution
        journal_scores = self.S @ topic_distribution
        
        # Get rankings (descending order)
        rankings = np.argsort(-journal_scores)
        
        return rankings, journal_scores
    
    def get_topic_specific_rankings(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Get rankings for each individual topic
        
        Returns:
        - rankings_dict: rankings for each topic
        - scores_dict: scores for each topic
        """
        if self.S is None:
            raise ValueError("Model must be fitted first")
            
        rankings_dict = {}
        scores_dict = {}
        
        for k in range(self.K):
            # Create one-hot topic vector
            topic_vec = np.zeros(self.K)
            topic_vec[k] = 1
            
            rankings, scores = self.get_journal_rankings(topic_vec)
            rankings_dict[f'Topic_{k}'] = rankings
            scores_dict[f'Topic_{k}'] = scores
            
        return rankings_dict, scores_dict


class BTLPPTMAnalyzer:
    def __init__(self, debiased=False, n_topics=10, n_hyperwords=500, embed_dim=30,
                 model_name='distilbert-base-uncased', reg_lambda=0.0,
                 random_seed=42):
        self.n_topics = n_topics
        self.n_hyperwords = n_hyperwords
        self.embed_dim = embed_dim
        self.reg_lambda = reg_lambda
        self.random_seed = random_seed
        self.embedding_extractor = BERTEmbeddingExtractor(model_name, embed_dim=embed_dim)
        self.pptm_model: PPTMTopicModel = None
        self.btl_model = None
        self.dataset = None
        self.embeddings = None
        self.doc_lengths = None
        self.vocab: List[str] = None
        self.topic_weights = None
        self.debiased = debiased



    def fit(self, dataset, batch_size=16, max_length=512, deduplicate=True, verbose=True):
        if verbose:
            print("🚀 FITTING BTL-PPTM SYSTEM")
            if hasattr(dataset, "print_summary"):
                dataset.print_summary()

        self.dataset = dataset
        
        # -----------------------------------------------------------
        # CHECK: Is this MadStat (Citation) or LLMArena (Battle)?
        # -----------------------------------------------------------
        # We detect "Battle Mode" by checking for the 'winners' array
        is_battle_mode = hasattr(dataset, "winners")
        
        if is_battle_mode:
            print("\n⚔️ BATTLE MODE DETECTED (LLM Arena)")
            raw_texts = dataset.prompts  # Use prompts for topic modeling
        else:
            print("\n📚 CITATION MODE DETECTED (MadStat)")
            raw_texts = dataset.abstracts # Use abstracts for topic modeling


        if deduplicate:
            if verbose: print(f"\nStep 0: Deduplicating {len(raw_texts)} texts...")
            
            # 1. Identify unique texts and their indices
            # We use a dict to keep the FIRST occurrence of every text
            unique_text_map = {}
            keep_indices = []
            
            for i, text in enumerate(raw_texts):
                # Normalize slightly to catch "Hello" vs "Hello "
                clean_text = text.strip() 
                if clean_text not in unique_text_map:
                    unique_text_map[clean_text] = i
                    keep_indices.append(i)
            
            # 2. Filter the raw_texts list
            n_orig = len(raw_texts)
            n_new = len(keep_indices)
            raw_texts = [raw_texts[i] for i in keep_indices]
            
            if verbose: print(f"✓ Removed {n_orig - n_new} duplicates. New size: {n_new}")

            # 3. UPDATE THE DATASET OBJECT 
            # We must update the dataset arrays to match this new shorter list
            # otherwise your winners/citations will be misaligned.
            keep_indices = np.array(keep_indices)
            
            if is_battle_mode:
                self.dataset.prompts = raw_texts # Already filtered
                self.dataset.model_a_ids = self.dataset.model_a_ids[keep_indices]
                self.dataset.model_b_ids = self.dataset.model_b_ids[keep_indices]
                self.dataset.winners = self.dataset.winners[keep_indices]
            else:
                self.dataset.abstracts = raw_texts # Already filtered
                self.dataset.journal_ids = np.asarray(self.dataset.journal_ids)[keep_indices]
                
                # Handle Citation Matrix
                C = self.dataset.citation_matrix
                if C.shape[0] == C.shape[1]: # Square matrix (Paper x Paper)
                    if verbose: print("⚠ Warning: Deduplicating a citation network graph is risky. Slicing matrix...")
                    self.dataset.citation_matrix = C[keep_indices][:, keep_indices]
                else:
                    self.dataset.citation_matrix = C[keep_indices, :]

        # -----------------------------------------------------------
        # Step 1: Extract Embeddings
        # -----------------------------------------------------------
        if verbose: print(f"\nStep 1: Extracting embeddings from {len(raw_texts)} texts...")
        
        sub_embs, sub_lens, sub_tokens = self.embedding_extractor.extract_embeddings(
            raw_texts,
            batch_size=batch_size,
            max_length=max_length,
            verbose=verbose
        )

        if verbose: print("✓ Subtoken embeddings extracted")

        emb, lens, vocab = merge_subtokens(sub_tokens, sub_embs, sub_lens)
        if verbose: print(f"✓ Merged into {len(vocab)} words")

        # ============================================================
        # STEP 2: ADVANCED VOCABULARY CLEANING & FILTERING
        # ============================================================
        if verbose: print("\nStep 2: Advanced Vocabulary Cleaning...")

        # Step 2: Clean vocabulary
        try:
            if not hasattr(self, 'nlp') or self.nlp is None:
                self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        except Exception as e:
            self.nlp = None
            print(f"⚠️ Spacy load failed ({e}).")

        vocab, emb, lens = clean_vocabulary(
            vocab, emb, lens, nlp=self.nlp, 
            min_freq=10, max_df_ratio=0.5, verbose=verbose
        )

        # Step 3: Drop short documents
        vocab, emb, lens, self.dataset = drop_short_documents(
            vocab, emb, lens, self.dataset, is_battle_mode,
            min_tokens=5, verbose=verbose
        )

        # -----------------------------------------------------------
        # Step 3: Run PPTM Topic Modeling
        # -----------------------------------------------------------
        if verbose: print("\nStep 3: Running PPTM Topic Model...")
        
        self.pptm_model = PPTMTopicModel(
            n_topics=self.n_topics,
            n_hyperwords=self.n_hyperwords,
            embed_dim=self.embed_dim,
            random_seed=self.random_seed,
            vocab=vocab
        )
        self.topic_weights = self.pptm_model.fit(emb, lens, verbose=verbose)
        
        if verbose:
            print(f"✓ Topic Weights Computed: {self.topic_weights.shape}")

        # -----------------------------------------------------------
        # Step 4: Run BTL (Split Logic)
        # -----------------------------------------------------------
        
        # 1. DECIDE GLOBALLY if we want debiasing
        # You can make this an argument to .fit() or hardcode it here
        ENABLE_DEBIASING = self.debiased  
        
        Sigma_delta_w = None
        if ENABLE_DEBIASING:
            if verbose: print("Computing Sigma covariance for debiasing...")
            Sigma_delta_w = self._compute_Sigma_delta_w()

        if is_battle_mode:
            # === LLM ARENA (Battle List) ===
            n_models = len(self.dataset.journal_names) # Use model_names from dataset
            
            print(f"\nStep 4: Running Battle-Aware BTL for {n_models} models...")
            
            self.btl_model = TopicSpecificBTL(
                n_topics=self.n_topics,
                n_journals=n_models,
                use_debiased=ENABLE_DEBIASING,  # Pass the flag here
                reg_lambda=self.reg_lambda
            )
            
            # Pass Sigma to fit_battles
            self.btl_model.fit_battles(
                model_A_ids=self.dataset.model_a_ids,
                model_B_ids=self.dataset.model_b_ids,
                winners=self.dataset.winners,
                topic_weights=self.topic_weights,
                Sigma_delta_w=Sigma_delta_w,
                verbose=verbose
            )

        else:
            # === MADSTAT (Citation Network) ===
            n_journals = self.dataset.n_journals
            print(f"\nStep 4: Running Standard Citation BTL for {n_journals} journals...")
            
            self.btl_model = TopicSpecificBTL(
                n_topics=self.n_topics,
                n_journals=n_journals,
                use_debiased=ENABLE_DEBIASING, 
                reg_lambda=self.reg_lambda
            )
            
            self.btl_model.fit(
                citation_matrix=self.dataset.citation_matrix,
                topic_weights=self.topic_weights,
                journal_ids=self.dataset.journal_ids,
                Sigma_delta_w=Sigma_delta_w,
                verbose=verbose
            )

        if verbose: print("✅ BTL-PPTM FIT COMPLETE")
        return self
    
    def save_results(self, filepath: str):
        """
        Save model results to file
        
        Parameters:
        - filepath: path to save results
        """
        if self.btl_model is None:
            raise ValueError("Model must be fitted first")
        
        n_journals = len(self.dataset.journal_names)
        results = {
            'S_matrix': self.btl_model.S,
            'topic_weights': self.topic_weights,
            'journal_names': self.dataset.journal_names,
            'n_topics': self.n_topics,
            'n_journals': n_journals,
            'dataset_name': self.dataset.dataset_name
        }
        
        np.savez(filepath, **results)
        print(f"✓ Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load model results from file
        
        Parameters:
        - filepath: path to load results from
        """
        results = np.load(filepath, allow_pickle=True)
        
        # Reconstruct BTL model
        self.btl_model = TopicSpecificBTL(
            n_topics=int(results['n_topics']),
            n_journals=int(results['n_journals']),
            reg_lambda=self.reg_lambda
        )
        self.btl_model.S = results['S_matrix']
        
        # Store other results
        self.topic_weights = results['topic_weights']
        self.n_topics = int(results['n_topics'])
        
        print(f"✓ Results loaded from {filepath}")


    def _compute_Sigma_delta_w(self):
        """
        Compute the full covariance matrix Σ̂_{δw,i} for all documents i.

        Formula using FREQUENCIES (X_freq) instead of COUNTS:
            Σ̂_{δw,i} = (1 / (N_i - 1)) * B^T [ Diag(X_freq) - X_freq X_freq^T ] B

        Returns
        -------
        Sigma_delta_w : np.ndarray
            Shape (n_docs, K, K)
        """
        # ------------------------------------------------------------
        # Objects from TopicScore / PPTM
        # ------------------------------------------------------------
        topic_score = self.pptm_model.model.model
        A = topic_score.A_hat                       # (V, K)
        X = self.pptm_model.model.hyperword_freq    # (V, n_docs) <-- These are FREQUENCIES
        doc_lens = self.pptm_model.model.doc_lens   # (n_docs,)

        n_docs = X.shape[1]
        K = A.shape[1]
        V = A.shape[0]

        # ------------------------------------------------------------
        # Precompute B = A G^{-1}  (Size: V x K)
        # ------------------------------------------------------------
        G = A.T @ A
        G_inv = np.linalg.inv(G)
        B = A @ G_inv

        # ------------------------------------------------------------
        # Allocate output (N, K, K)
        # ------------------------------------------------------------
        Sigma_delta_w = np.zeros((n_docs, K, K))

        # ------------------------------------------------------------
        # Loop over documents
        # ------------------------------------------------------------
        for i in range(n_docs):
            N_i = doc_lens[i]
            if N_i <= 1:
                continue

            X_i = X[:, i]  # (V,) This is the FREQUENCY vector (sums to 1)

            # --------------------------------------------------------
            # Term 1: B^T Diag(X_i) B
            # --------------------------------------------------------
            term1 = (B.T * X_i) @ B

            # --------------------------------------------------------
            # Term 2: (B^T X_i) (B^T X_i)^T
            #
            # Note: Since we are working with frequencies, the (1/N_i)
            # factor from the standard formula cancels out with the
            # N_i^2 factor inherent in the outer product of counts.
            # --------------------------------------------------------
            u = B.T @ X_i
            term2 = np.outer(u, u)

            # --------------------------------------------------------
            # Combine
            # We only divide by (N_i - 1). The 1/N_i from the paper's
            # definition of Sigma canceled out with the N_i from converting
            # frequencies to counts in the numerator.
            # --------------------------------------------------------
            Sigma_delta_w[i] = (term1 - term2) / (N_i - 1)

        return Sigma_delta_w


if __name__ == "__main__":
    # Simple test of core functionality
    from data_format import example_dataset
    
    print("Testing BTL-PPTM core functionality...")
    
    dataset = example_dataset()
    analyzer = BTLPPTMAnalyzer(n_topics=3, n_hyperwords=50, embed_dim=10)
    analyzer.fit(dataset, batch_size=2, verbose=True)
    
    print("✓ Core functionality test passed!")































