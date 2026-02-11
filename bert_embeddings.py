"""
bert_embeddings.py
This script contains the key classes for computing the contextualized embeddings used in PPTM.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import IncrementalPCA
import gc
import warnings
import spacy

from utils import *
from embedded_score import Embedded_SCORE_C

# Import transformer libraries
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠ Warning: transformers not available. Install with: pip install torch transformers")

warnings.filterwarnings('ignore')


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