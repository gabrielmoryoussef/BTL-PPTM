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



def merge_subtokens(tokens: List[str], 
                    embeddings: np.ndarray, 
                    doc_lengths: np.ndarray):
    """
    Collapse subtokens back into full words by averaging their embeddings.
    Auto-detects BERT (##) vs RoBERTa (Ġ) tokenization styles.
    """
    merged_embs = []
    merged_lens = []
    merged_vocab = []

    idx = 0
    
    # Check the first few tokens to guess the style
    # RoBERTa uses 'Ġ' (U+0120) to represent spaces
    is_roberta = any('Ġ' in t for t in tokens[:100])
    
    print(f"  Merge logic detected: {'RoBERTa (Prefix Space)' if is_roberta else 'BERT (Suffix ##)'}")

    for i, length in enumerate(doc_lengths):
        end = idx + length
        
        if length == 0:
            merged_lens.append(0)
            idx = end
            continue
            
        doc_toks = tokens[idx:end]
        doc_embs = embeddings[idx:end]
        idx = end

        curr_embs = []
        curr_pieces = []
        doc_merged_embs = []
        doc_merged_words = []

        for j, (tok, emb) in enumerate(zip(doc_toks, doc_embs)):
            
            # -------------------------------------------------
            # LOGIC: Should we merge this token to the previous?
            # -------------------------------------------------
            should_merge = False
            
            if is_roberta:
                # RoBERTa: Merge if it does NOT start with 'Ġ' (and isn't the first token)
                # Note: The very first token of a doc often doesn't have 'Ġ' either, so we don't merge j=0
                if j > 0 and not tok.startswith('Ġ'):
                    should_merge = True
            else:
                # BERT: Merge if it STARTS with '##'
                if tok.startswith("##"):
                    should_merge = True

            # -------------------------------------------------
            # EXECUTE MERGE
            # -------------------------------------------------
            if should_merge:
                # It is a suffix -> Add to buffer
                curr_embs.append(emb)
                
                # Clean text for vocab
                clean_tok = tok.replace("##", "") # BERT clean
                curr_pieces.append(clean_tok)
                
            else:
                # It is a NEW word -> Flush buffer & Start new
                if curr_embs:
                    # Average previous word embeddings
                    avg_emb = np.mean(curr_embs, axis=0)
                    doc_merged_embs.append(avg_emb)
                    doc_merged_words.append("".join(curr_pieces))
                
                # Start new buffer
                curr_embs = [emb]
                
                # Clean text for vocab (Remove RoBERTa's G)
                clean_tok = tok.replace("Ġ", "").replace("##", "")
                curr_pieces = [clean_tok]
        
        # Flush the very last word of the document
        if curr_embs:
            avg_emb = np.mean(curr_embs, axis=0)
            doc_merged_embs.append(avg_emb)
            doc_merged_words.append("".join(curr_pieces))

        # -------------------------------------------------
        # STORE RESULT
        # -------------------------------------------------
        if len(doc_merged_embs) > 0:
            stacked = np.vstack(doc_merged_embs)
            merged_embs.append(stacked)
            merged_lens.append(len(stacked))
            merged_vocab.extend(doc_merged_words)
        else:
            merged_lens.append(0)

    # ========================================================
    # MEMORY-EFFICIENT FINAL STACK
    # ========================================================
    if not merged_embs:
        print("Warning: No valid embeddings found.")
        return np.empty((0,0)), np.array([]), []

    # Calculate dimensions
    total_words = sum(len(emb) for emb in merged_embs)
    embed_dim = merged_embs[0].shape[1]
    
    # Pre-allocate
    flat_embs = np.empty((total_words, embed_dim), dtype=np.float32)
    
    current_idx = 0
    for emb_array in merged_embs:
        L = len(emb_array)
        flat_embs[current_idx:current_idx + L] = emb_array
        current_idx += L
    
    # Cleanup
    del merged_embs
    import gc
    gc.collect()
    
    flat_lens = np.array(merged_lens, dtype=np.int32)
    
    return flat_embs, flat_lens, merged_vocab


class BERTEmbeddingExtractor:
    """
    Extracts RAW contextualized subtoken embeddings and performs PCA.
    Does NOT merge subtokens; delegates that to the external merge_subtokens function.
    Compatible with both BERT and RoBERTa.
    """
    
    def __init__(self, model_name='roberta-base', embed_dim=20, device='auto'):
        self.model_name = model_name
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.embed_dim = embed_dim
        
    def load_model(self):
        """Load Model and Tokenizer"""
        print(f"Loading {self.model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def extract_embeddings(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: int = 512,
        verbose: bool = True,
        pca_fit_batches: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns:
        - embeddings: (N_subtokens, embed_dim) -> Reduced dimension vectors
        - doc_lengths: (n_docs,) -> Number of subtokens per document
        - tokens: List[str] -> The raw subtokens (e.g. 'Ġword', 'ing', '##ing')
        """

        if self.model is None:
            self.load_model()

        # Temporary storage
        all_embeddings: List[np.ndarray] = [] # Stores 768-d vectors temporarily
        doc_lengths: List[int] = []
        all_tokens: List[str] = []

        n_batches = (len(texts) + batch_size - 1) // batch_size

        if verbose:
            print(f"Extracting raw subtoken embeddings from {len(texts)} documents...")

        # ------------------------------------------------------------
        # PASS 1: Forward Pass & Filter Special Tokens
        # ------------------------------------------------------------
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(texts))
            
            if verbose and batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx + 1}/{n_batches}")

            batch_texts = texts[start_idx:end_idx]

            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs_device = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs_device)
                hidden = outputs.last_hidden_state  # (B, L, 768)

            # Move to CPU for processing
            hidden_cpu = hidden.cpu().numpy()
            input_ids_cpu = inputs["input_ids"].cpu().numpy()
            
            # Process each document in batch
            for j in range(len(batch_texts)):
                ids = input_ids_cpu[j]
                
                # Create mask for valid tokens (ignore padding & special tokens like CLS/SEP)
                # get_special_tokens_mask returns 1 for special tokens, 0 for real words
                special_mask = self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                valid_indices = [i for i, is_special in enumerate(special_mask) if is_special == 0]
                
                # Apply mask - only keep real text subtokens
                if valid_indices:
                    # 1. Get Embeddings
                    seq_emb = hidden_cpu[j, valid_indices, :] # (n_valid, 768)
                    
                    # 2. Get Tokens (Strings)
                    # convert_ids_to_tokens gives us ['ĠHello', 'world'] or ['play', '##ing']
                    valid_ids = ids[valid_indices]
                    seq_toks = self.tokenizer.convert_ids_to_tokens(valid_ids)
                    
                    all_embeddings.append(seq_emb)
                    doc_lengths.append(len(seq_emb))
                    all_tokens.extend(seq_toks)
                else:
                    doc_lengths.append(0)

            # Cleanup GPU
            del inputs, inputs_device, outputs, hidden
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # ------------------------------------------------------------
        # PASS 2: PCA Fitting (on 768-d subtokens)
        # ------------------------------------------------------------
        if verbose:
            print(f"Fitting IncrementalPCA to reduce {all_embeddings[0].shape[1]} -> {self.embed_dim}...")

        ipca = IncrementalPCA(n_components=self.embed_dim)
        
        fit_batches = min(pca_fit_batches, len(all_embeddings))
        for i in range(fit_batches):
            ipca.partial_fit(all_embeddings[i])

        if verbose:
            print(f"  PCA explained variance: {ipca.explained_variance_ratio_.sum():.3f}")

        # ------------------------------------------------------------
        # PASS 3: Transform & Flatten
        # ------------------------------------------------------------
        total_subtokens = sum(len(e) for e in all_embeddings)
        final_embeddings = np.empty((total_subtokens, self.embed_dim), dtype=np.float32)

        cur = 0
        for i, emb in enumerate(all_embeddings):
            reduced = ipca.transform(emb)
            L = reduced.shape[0]
            final_embeddings[cur : cur + L] = reduced
            cur += L

        # Final cleanup
        del all_embeddings
        gc.collect()

        doc_lengths = np.array(doc_lengths, dtype=np.int32)
        
        if verbose:
            print(f"✓ Extraction complete. {total_subtokens:,} raw subtokens ready for merging.")

        return final_embeddings, doc_lengths, all_tokens

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
        reg_lambda: float = 1e-3,
        use_debiased: bool = False,
        step_size: float = 2,
        max_iter: int = 200,
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
        mean_nll = total_nll / n_edges

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
        grad -= 2.0 * self.reg_lambda * S
        return -grad.flatten()
    # ------------------------------------------------------------------
    # Debiased gradient (ONLY C_data^[0] implemented)
    # ------------------------------------------------------------------

    def _debiased_gradient_C0(
        self,
        S: np.ndarray,                        # (M, K)
        citation_pairs,                       # list of (i, j)
        topic_weights: np.ndarray,            # (n, K)
        journal_ids: np.ndarray,              # (n,)
        Sigma_delta_w: np.ndarray,            # (n, K, K)  <-- CHANGED TYPE
    ) -> np.ndarray:
        """
        Implements C_data^[0] using full covariance matrices.
        """
        grad = np.zeros_like(S, dtype=float)
        C0   = np.zeros_like(S, dtype=float)

        mu = self._compute_paper_scores(S, topic_weights, journal_ids)

        for i, j in citation_pairs:
            ji = int(journal_ids[i])
            jj = int(journal_ids[j])

            if ji < 0 or jj < 0 or ji >= self.M or jj >= self.M:
                continue

            delta = mu[i] - mu[j]
            p     = self._sigmoid(delta)
            alpha = self._sigmoid_prime(delta)

            # Standard Gradient
            # Penalize the citing journal (i)
            grad[ji] -= p * topic_weights[i]
            
            # Reward the cited journal (j)
            grad[jj] += p * topic_weights[j]

            # --------------------------------------------------------
            # C_data^[0] correction
            #
            # Σ_{δw,i} S(π_i π_iᵀ) simplifies to:
            # Vector = Σ_{δw,i} @ S[ji]
            # (Matrix-Vector multiplication: (K,K) @ (K,) -> (K,))
            # --------------------------------------------------------
            
            # Correction for i
            correction_i = Sigma_delta_w[i] @ S[ji]
            C0[ji] += alpha * correction_i

            # Correction for j
            correction_j = Sigma_delta_w[j] @ S[jj]
            C0[jj] += alpha * correction_j

        #grad -= 2.0 * self.reg_lambda * S
        grad_star = grad - C0

        return grad_star


    # ------------------------------------------------------------------
    # Main fit entry point
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
        if not self.use_debiased:
            S0 = np.random.normal(0, 0.1, (self.M, self.K))
            S0 = self._project(S0)
            # Prepare arrays outside the minimization loop
            citation_pairs_arr = np.array(citation_pairs)
            i_indices = citation_pairs_arr[:, 0]
            j_indices = citation_pairs_arr[:, 1]
            

            result = minimize(
                self._negative_log_likelihood,
                S0.flatten(),
                # Pass the arrays, not the list of tuples
                args=(i_indices, j_indices, topic_weights, journal_ids),
                method="L-BFGS-B",
                jac=self._mle_gradient,  # <--- PASS THE GRADIENT HERE!
                options={"maxiter": self.max_iter, "ftol": 1e-9},
            )

            # result = minimize(
            #     self._negative_log_likelihood,
            #     S0.flatten(),
            #     args=(citation_pairs, topic_weights, journal_ids),
            #     method="L-BFGS-B",
            #     options={"maxiter": self.max_iter, "ftol": 1e-9},
            # )

            self.S = self._project(result.x.reshape(self.M, self.K))

            if verbose:
                print("✓ MLE optimization completed")
                print(f"  success = {result.success}")
                print(f"  final NLL = {result.fun:.4f}")

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
            grad = self._debiased_gradient_C0(
                S=S,
                citation_pairs=citation_pairs,
                topic_weights=topic_weights,
                journal_ids=journal_ids,
                Sigma_delta_w=Sigma_delta_w,
            )

            # 2. Update Step
            # Note: We divide by n_edges to keep the scale consistent 
            # regardless of dataset size, similar to a mean gradient.
            step = self.step_size * grad / n_edges
            S_new = self._project(S + step)

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
    


    # def fit_battles(self, model_A_ids, model_B_ids, winners, topic_weights, verbose=True):
    #     """
    #     Specialized fit method for LLM Arena (Battle List format).
    #     """
    #     from scipy.optimize import minimize
        
    #     if verbose:
    #         print(f"   Optimizing S matrix ({self.M} models x {self.K} topics)...")
            
    #     S0 = np.zeros((self.M, self.K))
        
    #     # Define arguments tuple
    #     args = (model_A_ids, model_B_ids, winners, topic_weights)
        
    #     # Run L-BFGS-B
    #     result = minimize(
    #         fun=self._negative_log_likelihood_battles, # You need to implement this (see below)
    #         x0=S0.flatten(),
    #         args=args,
    #         method="L-BFGS-B",
    #         jac=self._mle_gradient_battles,            # You implemented this earlier!
    #         options={"maxiter": 100, "ftol": 1e-9}
    #     )
        
    #     self.S = result.x.reshape(self.M, self.K)
        
    #     if verbose:
    #         print(f"   ✓ Optimization Converged: Success={result.success}")
    #         print(f"   ✓ Final Loss: {result.fun:.4f}")

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
        if not self.use_debiased:
            S0 = np.zeros((self.M, self.K))
            
            result = minimize(
                fun=self._negative_log_likelihood_battles,
                x0=S0.flatten(),
                args=(model_A_ids, model_B_ids, winners, topic_weights),
                method="L-BFGS-B",
                jac=self._mle_gradient_battles,
                options={"maxiter": self.max_iter, "ftol": self.tol}
            )
            
            self.S = self._project(result.x.reshape(self.M, self.K))
            
            if verbose:
                print(f"   ✓ Optimization Converged: {result.success}, Loss: {result.fun:.4f}")
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
        S = self._project(S)
        
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
            S_new = self._project(S + self.step_size * total_grad)
            
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
    def __init__(self, n_topics=10, n_hyperwords=500, embed_dim=30,
                 model_name='distilbert-base-uncased', reg_lambda=1e-3,
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

            # 3. SYNC THE DATASET OBJECT (Crucial!)
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
                    # We usually can't simply slice a citation graph like this without 
                    # breaking links, but for MadStat (which is Papers->Papers), 
                    # removing a paper usually just removes its node.
                    # Ideally, citation networks shouldn't be deduped by text unless
                    # it's literally the same paper recorded twice.
                    # For safety in Citation Mode, you might want to SKIP dedup 
                    # or handle it very carefully.
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

        # print("embedings example = ", sub_embs[:10])
        # print("example of sub lens = ", sub_lens[:10])
        # print("example of sub tokens = ", sub_tokens[:10])
        if verbose: print("✓ Subtoken embeddings extracted")

        # -----------------------------------------------------------
        # Step 2: Merge Subtokens & Clean Vocab
        # -----------------------------------------------------------
        emb, lens, vocab = merge_subtokens(sub_tokens, sub_embs, sub_lens)
        if verbose: print(f"✓ Merged into {len(vocab)} words")
        # print("example vocab", vocab[:10])
        # print(emb.shape)
        # print(len(vocab))




        # ============================================================
        # STEP 2: ADVANCED VOCABULARY CLEANING & FILTERING
        # ============================================================
        if verbose: print("\nStep 2: Advanced Vocabulary Cleaning...")

        # 1. SETUP NLP FOR POS TAGGING
        #    We need this to differentiate "Apple" (Noun) from "Ate" (Verb)
        try:
            if not hasattr(self, 'nlp') or self.nlp is None:
                self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        except Exception as e:
            self.nlp = None
            print(f"⚠️ Spacy load failed ({e}). Topics will contain verbs/junk.")

        # 2. DEFINE AGGRESSIVE STOPWORDS
        #    Combine Standard + Riddles + LLM Boilerplate
        STOPWORDS = set(ENGLISH_STOP_WORDS)
        
        riddle_words = [
            "brother", "sister", "sally", "apple", "feather", "brick", "pound", "kg", "lb", 
            "weighs", "heavier", "lighter", "yesterday", "today", "tomorrow", "woodchuck"
        ]
        
        chatbot_fluff = [
            "example", "question", "answer", "response", "assistant", "user", "prompt",
            "chat", "bot", "explain", "describe", "story", "write", "tell", "say",
            "create", "make", "provide", "following", "using", "based", "according",
            "please", "thanks", "help", "input", "output", "given", "select",
            "task", "context", "instruction", "step", "steps", "list", "generate",
            "function", "code", "return", "true", "false", "value", "variable", "string", "print"
        ]

        common_names = [
            "tom", "jill", "matt", "paul", "steve", "jesus", "smith", "peter", 
            "maria", "john", "david", "anna", "george", "adam", "bob"
        ]
        
        # 2. Generic Adjectives (Topic 12 cleanup)
        generic_adjectives = [
            "vital", "important", "essential", "crucial", "interesting", 
            "great", "helpful", "good", "bad", "new", "main", "biggest"
        ]
        
        # 3. Multilingual Particles (Topic 2 cleanup)
        foreign_noise = [
            "ist", "mir", "der", "die", "und", "que", "est", "le", "la", "des"
        ]
        
        # 4. Unit Artifacts (Topic 14 cleanup - optional, keep if you want a "Math" topic)
        units = ["pounds", "lbs", "liters", "degrees", "cloud", "cut"]
        
        STOPWORDS.update(riddle_words)
        STOPWORDS.update(chatbot_fluff)
        STOPWORDS.update(common_names)
        STOPWORDS.update(generic_adjectives)
        STOPWORDS.update(foreign_noise)
        STOPWORDS.update(units)

        # Regex Helpers
        VALID_WORD = re.compile(r"^[a-zA-Z][a-zA-Z\-']*[a-zA-Z]$")  
        REPEAT_JUNK = re.compile(r"(.)\1{2,}")  # e.g., "aaaaa"
        ascii_ratio = lambda s: sum(ord(c) < 128 for c in s) / max(len(s), 1)

        # 3. PRE-COMPUTE POS TAGS (Optimization: Tag unique words only)
        allowed_pos_words = set()
        if self.nlp:
            if verbose: print("  Performing POS tagging (keeping Nouns/Adjectives only)...")
            unique_vocab = list(set(vocab))
            
            # Batch process for speed (chunks of 2000)
            chunk_size = 2000
            for i in range(0, len(unique_vocab), chunk_size):
                chunk = unique_words_chunk = unique_vocab[i:i+chunk_size]
                # Join with spaces to allow context approximation, or tag individually
                # Fast version: tag as a list
                doc = self.nlp.pipe(unique_words_chunk, batch_size=50)
                
                # for token_doc in doc:
                #     # Spacy pipe returns docs, we grab the first token since we fed single words
                #     token = token_doc[0] 
                #     if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                #         allowed_pos_words.add(token.text)
                for token_doc in doc:
                    # --- SAFETY CHECK ---
                    # Only access [0] if the doc actually has words
                    if len(token_doc) > 0:
                        token = token_doc[0] 
                        if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                            allowed_pos_words.add(token.text)
        else:
            # Fallback: Allow everything if spacy fails
            allowed_pos_words = set(vocab)

        # 4. COMPUTE TF AND DF
        counts = Counter()      # Total Term Frequency
        doc_counts = Counter()  # Document Frequency
        
        current_idx = 0
        total_docs = len(lens)
        
        # Loop through docs to count
        for doc_len in lens:
            doc_slice = vocab[current_idx : current_idx + doc_len]
            
            # Update TF
            counts.update(doc_slice)
            
            # Update DF (use set for uniqueness per doc)
            doc_counts.update(set(doc_slice))
            
            current_idx += doc_len

        # 5. DEFINE THE MASTER FILTER
        min_freq = 10       # Word must appear 10 times total
        max_df_ratio = 0.5  # Word must NOT appear in > 50% of docs (removes "the", "use")

        def is_clean_token(w):
            w_lower = w.strip().lower()
            
            # A. Structure Checks
            if not VALID_WORD.match(w): return False
            if len(w) < 3 or len(w) > 18: return False
            if REPEAT_JUNK.fullmatch(w_lower): return False
            if ascii_ratio(w) < 0.85: return False
            
            # B. Stopword Checks
            if w_lower in STOPWORDS: return False
            
            # C. Statistics Checks
            if counts[w] < min_freq: return False
            if (doc_counts[w] / total_docs) > max_df_ratio: return False
            
            # D. POS Check (The Semantic Filter)
            if self.nlp and w not in allowed_pos_words: return False
            
            return True

        if verbose: print(f"  Applying filters: MinFreq={min_freq}, MaxDF={max_df_ratio}, POS=Noun/Adj")

        # 6. APPLY FILTER TO CREATE MASK
        # This mask aligns 1-to-1 with 'vocab' and 'emb'
        final_mask = np.array([is_clean_token(w) for w in vocab], dtype=bool)

        # 7. UPDATE DATA STRUCTURES
        # Update Vocab
        vocab = list(compress(vocab, final_mask))
        print("vocab = ", vocab[:50])
        
        # Update Embeddings
        emb = emb[final_mask]

        # Update Document Lengths
        # We must sum the mask chunks corresponding to each document
        filtered_lens = []
        current_idx = 0
        for doc_len in lens:
            doc_mask = final_mask[current_idx : current_idx + doc_len]
            filtered_lens.append(doc_mask.sum())
            current_idx += doc_len
        
        lens = np.array(filtered_lens, dtype=np.int32)
        print("mean lenght = ", lens.mean())

        # Cleanup
        del final_mask, counts, doc_counts, allowed_pos_words
        gc.collect()

        if verbose: print(f"✓ Vocabulary cleaned: {len(vocab)} semantic tokens remaining.")


        # -----------------------------------------------------------
        # STEP 3: HANDLE SHORT/ZERO-LENGTH DOCUMENTS
        # -----------------------------------------------------------
        # We define "short" as having fewer than 5 valid tokens
        zero_mask = (lens < 5)
        
        if zero_mask.any():
            num_drop = int(zero_mask.sum())
            if verbose: print(f"⚠ Dropping {num_drop} documents with < 5 tokens.")
            
            # --- CRITICAL FIX START: SYNC VOCAB & EMBEDDINGS ---
            
            # 1. Create a mask for the TOKENS (flat list)
            # We must identify which specific tokens belong to the bad docs
            token_keep_mask = np.ones(len(emb), dtype=bool)
            
            current_idx = 0
            for doc_idx, doc_len in enumerate(lens):
                if zero_mask[doc_idx]: # If this doc is marked for deletion
                    # Mark its tokens as False
                    token_keep_mask[current_idx : current_idx + doc_len] = False
                current_idx += doc_len
            
            # 2. Update the Flat Data Structures
            emb = emb[token_keep_mask]
            vocab = list(compress(vocab, token_keep_mask))
            
            # --- CRITICAL FIX END ---

            # 3. Update the Document Lengths
            keep_docs = ~zero_mask
            lens = lens[keep_docs]
            
            # 4. Update Dataset Arrays (Battle vs Citation Mode)
            if is_battle_mode:
                self.dataset.prompts = [p for p, k in zip(self.dataset.prompts, keep_docs) if k]
                self.dataset.model_a_ids = self.dataset.model_a_ids[keep_docs]
                self.dataset.model_b_ids = self.dataset.model_b_ids[keep_docs]
                self.dataset.winners = self.dataset.winners[keep_docs]
            else:
                self.dataset.abstracts = [a for a, k in zip(self.dataset.abstracts, keep_docs) if k]
                self.dataset.journal_ids = np.asarray(self.dataset.journal_ids)[keep_docs]
                
                # Handle Citation Matrix
                C = self.dataset.citation_matrix
                if C.shape[0] == C.shape[1] and C.shape[0] == len(keep_docs):
                     # If square and matches original size
                    self.dataset.citation_matrix = C[keep_docs][:, keep_docs]
                elif C.shape[0] == len(zero_mask): 
                    # If rectangular or matching rows
                    self.dataset.citation_matrix = C[keep_docs, :]

            if verbose: print(f"✓ Drop complete. Final Dataset Size: {len(lens)} docs")

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
        ENABLE_DEBIASING = False  
        
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































