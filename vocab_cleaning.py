"""
vocab_cleaning.py

Vocabulary cleaning and document filtering for BTL-PPTM pipeline.
Extracted from the main fit() method for readability.
"""

import re
import gc
import numpy as np
import spacy
from collections import Counter
from itertools import compress
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ==============================================================================
# STOPWORD DEFINITIONS
# ==============================================================================

RIDDLE_WORDS = [
    "brother", "sister", "sally", "apple", "feather", "brick", "pound", "kg", "lb",
    "weighs", "heavier", "lighter", "yesterday", "today", "tomorrow", "woodchuck"
]

CHATBOT_FLUFF = [
    "example", "question", "answer", "response", "assistant", "user", "prompt",
    "chat", "bot", "explain", "describe", "story", "write", "tell", "say",
    "create", "make", "provide", "following", "using", "based", "according",
    "please", "thanks", "help", "input", "output", "given", "select",
    "task", "context", "instruction", "step", "steps", "list", "generate",
    "function", "code", "return", "true", "false", "value", "variable", "string", "print"
]

COMMON_NAMES = [
    "tom", "jill", "matt", "paul", "steve", "jesus", "smith", "peter",
    "maria", "john", "david", "anna", "george", "adam", "bob"
]

GENERIC_ADJECTIVES = [
    "vital", "important", "essential", "crucial", "interesting",
    "great", "helpful", "good", "bad", "new", "main", "biggest"
]

FOREIGN_NOISE = [
    "ist", "mir", "der", "die", "und", "que", "est", "le", "la", "des"
]

UNITS = ["pounds", "lbs", "liters", "degrees", "cloud", "cut"]

# Regex patterns
VALID_WORD = re.compile(r"^[a-zA-Z][a-zA-Z\-']*[a-zA-Z]$")
REPEAT_JUNK = re.compile(r"(.)\1{2,}")


def _ascii_ratio(s):
    return sum(ord(c) < 128 for c in s) / max(len(s), 1)


def _build_stopwords():
    """Build the combined stopword set."""
    stopwords = set(ENGLISH_STOP_WORDS)
    stopwords.update(RIDDLE_WORDS)
    stopwords.update(CHATBOT_FLUFF)
    stopwords.update(COMMON_NAMES)
    stopwords.update(GENERIC_ADJECTIVES)
    stopwords.update(FOREIGN_NOISE)
    stopwords.update(UNITS)
    return stopwords


def _get_allowed_pos_words(vocab, nlp, verbose=False):
    """POS-tag unique words and return set of nouns/adjectives."""
    if nlp is None:
        return set(vocab)

    if verbose:
        print("  Performing POS tagging (keeping Nouns/Adjectives only)...")

    allowed = set()
    unique_vocab = list(set(vocab))
    chunk_size = 2000

    for i in range(0, len(unique_vocab), chunk_size):
        chunk = unique_vocab[i:i + chunk_size]
        for token_doc in nlp.pipe(chunk, batch_size=50):
            if len(token_doc) > 0:
                token = token_doc[0]
                if token.pos_ in ['NOUN', 'PROPN', 'ADJ']:
                    allowed.add(token.text)

    return allowed


def _compute_term_stats(vocab, lens):
    """Compute term frequency and document frequency."""
    counts = Counter()
    doc_counts = Counter()
    current_idx = 0

    for doc_len in lens:
        doc_slice = vocab[current_idx:current_idx + doc_len]
        counts.update(doc_slice)
        doc_counts.update(set(doc_slice))
        current_idx += doc_len

    return counts, doc_counts


# ==============================================================================
# MAIN CLEANING FUNCTION
# ==============================================================================

def clean_vocabulary(vocab, emb, lens, nlp=None, min_freq=10, max_df_ratio=0.5, verbose=False):
    """
    Apply vocabulary cleaning and filtering.

    Parameters
    ----------
    vocab : list of str
        Flat list of tokens (concatenated across all documents).
    emb : np.ndarray
        Embedding matrix, shape (n_tokens, embed_dim), aligned with vocab.
    lens : np.ndarray
        Document lengths (number of tokens per document).
    nlp : spacy model or None
        Spacy model for POS tagging. If None, POS filtering is skipped.
    min_freq : int
        Minimum total term frequency to keep a word.
    max_df_ratio : float
        Maximum document frequency ratio (removes words in > this fraction of docs).
    verbose : bool

    Returns
    -------
    vocab : list of str
        Filtered vocabulary.
    emb : np.ndarray
        Filtered embeddings.
    lens : np.ndarray
        Updated document lengths.
    """
    if verbose:
        print("\nVocabulary Cleaning...")

    # Build resources
    stopwords = _build_stopwords()
    allowed_pos_words = _get_allowed_pos_words(vocab, nlp, verbose=verbose)
    counts, doc_counts = _compute_term_stats(vocab, lens)
    total_docs = len(lens)

    # Define filter
    def is_clean_token(w):
        w_lower = w.strip().lower()

        # Structure checks
        if not VALID_WORD.match(w):
            return False
        if len(w) < 3 or len(w) > 18:
            return False
        if REPEAT_JUNK.fullmatch(w_lower):
            return False
        if _ascii_ratio(w) < 0.85:
            return False

        # Stopword check
        if w_lower in stopwords:
            return False

        # Statistics checks
        if counts[w] < min_freq:
            return False
        if (doc_counts[w] / total_docs) > max_df_ratio:
            return False

        # POS check
        if nlp is not None and w not in allowed_pos_words:
            return False

        return True

    if verbose:
        print(f"  Applying filters: MinFreq={min_freq}, MaxDF={max_df_ratio}, POS=Noun/Adj")

    # Apply filter
    final_mask = np.array([is_clean_token(w) for w in vocab], dtype=bool)

    # Update vocab and embeddings
    vocab = list(compress(vocab, final_mask))
    emb = emb[final_mask]

    # Update document lengths
    filtered_lens = []
    current_idx = 0
    for doc_len in lens:
        doc_mask = final_mask[current_idx:current_idx + doc_len]
        filtered_lens.append(doc_mask.sum())
        current_idx += doc_len
    lens = np.array(filtered_lens, dtype=np.int32)

    if verbose:
        print(f"  Vocab: {vocab[:30]}")
        print(f"  Mean doc length: {lens.mean():.1f}")
        print(f"✓ Vocabulary cleaned: {len(vocab)} semantic tokens remaining.")

    # Cleanup
    del final_mask, counts, doc_counts, allowed_pos_words
    gc.collect()

    return vocab, emb, lens


# ==============================================================================
# DROP SHORT DOCUMENTS
# ==============================================================================

def drop_short_documents(vocab, emb, lens, dataset, is_battle_mode, min_tokens=5, verbose=False):
    """
    Drop documents with fewer than min_tokens valid tokens.

    Parameters
    ----------
    vocab : list of str
    emb : np.ndarray
    lens : np.ndarray
    dataset : object
        The dataset object (with attributes like prompts, model_a_ids, etc.)
    is_battle_mode : bool
    min_tokens : int
    verbose : bool

    Returns
    -------
    vocab, emb, lens, dataset : updated objects
    """
    zero_mask = (lens < min_tokens)

    if not zero_mask.any():
        return vocab, emb, lens, dataset

    num_drop = int(zero_mask.sum())
    if verbose:
        print(f"⚠ Dropping {num_drop} documents with < {min_tokens} tokens.")

    # 1. Create token-level mask
    token_keep_mask = np.ones(len(emb), dtype=bool)
    current_idx = 0
    for doc_idx, doc_len in enumerate(lens):
        if zero_mask[doc_idx]:
            token_keep_mask[current_idx:current_idx + doc_len] = False
        current_idx += doc_len

    # 2. Filter tokens
    emb = emb[token_keep_mask]
    vocab = list(compress(vocab, token_keep_mask))

    # 3. Filter document-level arrays
    keep_docs = ~zero_mask
    lens = lens[keep_docs]

    # 4. Update dataset
    if is_battle_mode:
        dataset.prompts = [p for p, k in zip(dataset.prompts, keep_docs) if k]
        dataset.model_a_ids = dataset.model_a_ids[keep_docs]
        dataset.model_b_ids = dataset.model_b_ids[keep_docs]
        dataset.winners = dataset.winners[keep_docs]
    else:
        dataset.abstracts = [a for a, k in zip(dataset.abstracts, keep_docs) if k]
        dataset.journal_ids = np.asarray(dataset.journal_ids)[keep_docs]

        C = dataset.citation_matrix
        if C.shape[0] == C.shape[1] and C.shape[0] == len(zero_mask):
            dataset.citation_matrix = C[keep_docs][:, keep_docs]
        elif C.shape[0] == len(zero_mask):
            dataset.citation_matrix = C[keep_docs, :]

    if verbose:
        print(f"✓ Drop complete. Final dataset size: {len(lens)} docs")

    return vocab, emb, lens, dataset