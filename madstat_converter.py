# madstat_converter.py
#
# Minimal MADStat → AcademicDataset converter
# Purpose: prepare data for PPTM / BTL-PPTM
#
# Requires:
#   pip install rpy2 numpy pandas scipy scikit-learn
#

from __future__ import annotations

import os
import re
import unicodedata
from typing import Optional, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from data_format import AcademicDataset

# --- rpy2 ---
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


# -----------------------
# Text cleaning utilities
# -----------------------

_TOKEN_RE = re.compile(r"[a-zA-Z]+")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_URL_EMAIL_RE = re.compile(r"(https?://\S+|\bwww\.\S+|\S+@\S+)", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")

FRENCH_COMMON_WORDS = {
    "des","aux","les","dans","sur","avec","pour","une","est","sont",
    "cela","ainsi","sans","entre","par","comme","ces","dont","plus",
    "au","du","leur","leurs","afin","étant","étude","études","car",
    "ceci","cela","cet","cette","elles","il","elle","leur","leurs",
}


def clean_abstract(text: str, min_words: int = 30) -> Optional[str]:
    if text is None:
        return None

    t = str(text).strip()
    if not t or t.lower() in {"na", "nan", "none"}:
        return None

    # normalize & remove URLs / LaTeX but KEEP ALL WORDS
    t = unicodedata.normalize("NFKC", t)
    t = _URL_EMAIL_RE.sub(" ", t)
    t = _LATEX_CMD_RE.sub(" ", t)

    # extract alphabetic tokens ONLY (NO STOPWORD REMOVAL)
    tokens = [w.lower() for w in _TOKEN_RE.findall(t)]

    # require enough content
    if len(tokens) < min_words:
        return None

    # ---- French detection: reject document if many French tokens ----
    french_hits = sum(1 for w in tokens if w in FRENCH_COMMON_WORDS)
    if french_hits >= 3:     # threshold to avoid false positives
        return None
    # -----------------------------------------------------------------

    return _MULTI_SPACE_RE.sub(" ", " ".join(tokens)).strip()



def r_to_pandas(r_obj) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(r_obj)


# -----------------------
# Converter
# -----------------------

class MADStatConverter:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.paper_pd: Optional[pd.DataFrame] = None
        self.journal_pd: Optional[pd.DataFrame] = None
        self.paper_ref_r = None
        self.pappap_pd: Optional[pd.DataFrame] = None

    def load_raw(self) -> None:
        path = os.path.join(self.data_dir, "Raw-data-2019-12-version.RData")
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        ro.r["load"](path)

        self.paper_pd = r_to_pandas(ro.r["paper"])
        self.journal_pd = r_to_pandas(ro.r["journal"])
        self.paper_ref_r = ro.r["paper_ref"]

    def load_citations(self) -> None:
        path = os.path.join(self.data_dir, "AuthorPaperInfo.RData")
        if not os.path.exists(path):
            self.pappap_pd = None
            return

        ro.r["load"](path)
        if "PapPapMat" in ro.r("ls()"):
            self.pappap_pd = r_to_pandas(ro.r["PapPapMat"])
        else:
            self.pappap_pd = None

    def convert(
        self,
        min_words: int = 30,
        min_citations: int = 1,
        sample_size: Optional[int] = None,
        random_state: int = 12,
        verbose: bool = True,
    ) -> AcademicDataset:

        if self.paper_pd is None or self.journal_pd is None:
            raise RuntimeError("Call load_raw() first")

        paper_pd = self.paper_pd
        journal_pd = self.journal_pd

        # -----------------------
        # 1) Clean abstracts
        # -----------------------

        abstracts: List[str] = []
        orig_idx: List[int] = []

        for i in range(len(paper_pd)):
            a = clean_abstract(paper_pd.iloc[i].get("abstract"), min_words)
            if a is not None:
                abstracts.append(a)
                orig_idx.append(i)

        if verbose:
            print(f"[MADStat] kept {len(abstracts):,} abstracts")

        if sample_size is not None and sample_size < len(orig_idx):
            rng = np.random.default_rng(random_state)
            sel = np.sort(rng.choice(len(orig_idx), size=sample_size, replace=False))
            abstracts = [abstracts[i] for i in sel]
            orig_idx = [orig_idx[i] for i in sel]

        n = len(orig_idx)

        # -----------------------
        # 2) Journals (ISSN map)
        # -----------------------

        journal_names: List[str] = []
        journal_issn: List[str] = []

        for i in range(len(journal_pd)):
            name = str(journal_pd.iloc[i].get("shortName", f"Journal_{i}"))
            issn = str(journal_pd.iloc[i].get("issn", ""))
            journal_names.append(name)
            journal_issn.append(issn)

        issn_to_j = {issn: j for j, issn in enumerate(journal_issn) if issn}
        unknown_j = len(journal_names)
        journal_names.append("__UNKNOWN__")

        journal_ids = np.empty(n, dtype=int)
        for k, i in enumerate(orig_idx):
            issn = str(paper_pd.iloc[i].get("issn", ""))
            journal_ids[k] = issn_to_j.get(issn, unknown_j)

        # -----------------------
        # 3) Citations
        # -----------------------

        orig_to_new = {o: i for i, o in enumerate(orig_idx)}
        rows, cols = [], []

        if self.pappap_pd is not None and {"FromPap", "ToPap"}.issubset(self.pappap_pd.columns):
            df = self.pappap_pd[["FromPap", "ToPap"]].dropna()
            for a, b in zip(df["FromPap"], df["ToPap"]):
                i = orig_to_new.get(int(a) - 1)
                j = orig_to_new.get(int(b) - 1)
                if i is not None and j is not None:
                    rows.append(i)
                    cols.append(j)
        else:
            paper_ref = self.paper_ref_r
            for i_new, i_orig in enumerate(orig_idx):
                try:
                    refs = paper_ref[i_orig]
                    for r in list(refs):
                        j = orig_to_new.get(int(r) - 1)
                        if j is not None:
                            rows.append(i_new)
                            cols.append(j)
                except Exception:
                    continue

        if rows:
            data = np.ones(len(rows), dtype=np.int8)
            citation = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
            citation.data[:] = 1
        else:
            citation = sp.csr_matrix((n, n), dtype=np.int8)

        if verbose:
            print(f"[MADStat] citations: {citation.nnz:,}")

        # -----------------------
        # 4) Min-citation filter
        # -----------------------

        if min_citations > 0 and citation.nnz > 0:
            indeg = np.asarray(citation.sum(axis=0)).ravel()
            keep = indeg >= min_citations
            if keep.sum() > 0:
                abstracts = [a for a, k in zip(abstracts, keep) if k]
                journal_ids = journal_ids[keep]
                citation = citation[keep][:, keep]
                n = len(abstracts)

        # --------------------------------------------------
        # FIX DUPLICATE JOURNAL NAMES (MERGE BY NAME)
        # --------------------------------------------------
        from collections import defaultdict

        print("\n[MADStat] Deduplicating journal names (merging duplicates by name)...")

        # Normalize names for comparison (case-insensitive + strip whitespace)
        norm_names = [str(n).strip().lower() for n in journal_names]

        # Group original journal indices by normalized name
        groups = defaultdict(list)
        for j, nm in enumerate(norm_names):
            groups[nm].append(j)

        # Map original journal ids → canonical id
        canonical_map = {}
        canonical_kept_ids = []
        canonical_names = []

        for nm, ids in groups.items():
            ids = sorted(ids)

            # keep the first id as the canonical one
            keep = ids[0]
            canonical_kept_ids.append(keep)
            canonical_names.append(journal_names[keep])

            for j in ids:
                canonical_map[j] = keep

        # Remap paper journal ids to canonical ones
        journal_ids = np.array([canonical_map[j] for j in journal_ids])

        # Compress to consecutive 0..M-1 so the BTL model is happy
        unique_ids = sorted(set(journal_ids))
        id_remap = {old: new for new, old in enumerate(unique_ids)}

        journal_ids = np.array([id_remap[j] for j in journal_ids])
        journal_names = [journal_names[j] for j in unique_ids]

        print(f"[MADStat] Journals after deduplication: {len(journal_names)}")
        print("\n[MADStat] Journal merges detected:")
        for nm, ids in groups.items():
            if len(ids) > 1:
                print(f"  {journal_names[id_remap[ids[0]]]}  <= merged original ids {ids}")



        # -----------------------
        # 5) Metadata (minimal)
        # -----------------------

        paper_ids = [f"madstat_{i}" for i in range(n)]
        paper_titles = [f"paper_{i}" for i in range(n)]
        paper_years = [2000] * n

        # -----------------------
        # 6) Dataset
        # -----------------------

        ds = AcademicDataset(
            abstracts=abstracts,
            journal_ids=journal_ids,
            citation_matrix=citation,
            journal_names=journal_names,
            paper_ids=paper_ids,
            paper_titles=paper_titles,
            paper_years=paper_years,
            dataset_name="MADStat",
            description=f"MADStat: {n} papers, {len(journal_names)} journals, {citation.nnz} citations",
        )

        if verbose:
            ds.print_summary()

        return ds


def convert_madstat_dataset(
    data_dir: str,
    min_abstract_length: int = 30,
    min_citations: int = 1,
    sample_size: Optional[int] = None,
    verbose: bool = True,
) -> AcademicDataset:
    conv = MADStatConverter(data_dir)
    conv.load_raw()
    conv.load_citations()
    return conv.convert(
        min_words=min_abstract_length,
        min_citations=min_citations,
        sample_size=sample_size,
        verbose=verbose,
    )


if __name__ == "__main__":
    ds = convert_madstat_dataset(
        data_dir="madstat_data",
        min_words=30,
        min_citations=3,
        sample_size=20000,
        verbose=True,
    )
    print("Done.")
