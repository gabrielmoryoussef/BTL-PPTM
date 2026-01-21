"""
LLM Arena Converter: The "Battle List" Method (Simple & Intuitive)

This script extracts LLM battles into a clean format where each row represents ONE battle.
It produces 4 parallel arrays of length N (number of battles):
  1. prompts:      The text of the prompt (used for Topic Modeling).
  2. model_a_ids:  The ID of the first model.
  3. model_b_ids:  The ID of the second model.
  4. winners:      0 if Model A won, 1 if Model B won.
"""

from __future__ import annotations
from datasets import load_dataset
import numpy as np
import regex as re
import pickle
from dataclasses import dataclass
from typing import List

# -------------------------------------------------------
# DATA STRUCTURE
# -------------------------------------------------------
@dataclass
class BattleDataset:
    """
    A simple container for Battle Data (instead of Citation Data).
    """
    prompts: List[str]          # The text for PPTM
    model_a_ids: np.ndarray     # ID of Model A
    model_b_ids: np.ndarray     # ID of Model B
    winners: np.ndarray         # 0 = A wins, 1 = B wins
    journal_names: List[str]      # Map IDs back to names (e.g., "GPT-4")
    dataset_name: str = "LLMArena_Battles"

    def print_summary(self):
        print(f"=== {self.dataset_name} Summary ===")
        print(f"Battles (N):      {len(self.prompts):,}")
        print(f"Unique Models:    {len(self.journal_names)}")
        print(f"Win Balance:      A={np.mean(self.winners==0):.1%} / B={np.mean(self.winners==1):.1%}")
        print("==================================")


# -------------------------------------------------------
# FILTERING HELPERS (Same as before)
# -------------------------------------------------------
NONLATIN = re.compile(r"\p{Script=Arabic}|\p{Script=Cyrillic}|\p{Script=Han}")
MULTISPACE = re.compile(r"\s+")
#CODE_HINTS = ["{", "}", "()", "def ", "class ", "import ", ";", "::", "</", "<html", "```", ">>>"]

def clean_prompt(text: str) -> str:
    if text is None: return ""
    text = str(text).encode("ascii", "ignore").decode()
    return MULTISPACE.sub(" ", text).strip()

def is_good_prompt(p: str, min_words=8) -> bool:
    if not p: return False
    p = clean_prompt(p)
    if len(p.split()) < min_words: return False
    if NONLATIN.search(p): return False
    letters = sum(c.isalpha() for c in p)
    if letters / max(1, len(p)) < 0.55: return False
    return True


# -------------------------------------------------------
# MAIN CONVERTER
# -------------------------------------------------------

def convert_llmarena_dataset(
    split="train",
    min_prompt_words=8,
    sample_size=None,
    random_state=42,
    verbose=True,
) -> BattleDataset:

    if verbose: print("[LLMArena] Loading dataset...")
    
    # 1. Load Data
    ds = load_dataset("lmarena-ai/arena-human-preference-55k", split=split)
    
    # 2. Filter Ties (Strict comparisons only)
    ds = ds.filter(lambda r: r["winner_tie"] == 0)
    
    # 3. Optional Subsampling
    if sample_size and sample_size < len(ds):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(ds), size=sample_size, replace=False)
        ds = ds.select(sorted(idx))

    # 4. Build Model ID Map
    #    We need a unified list of all models that appear in either column
    unique_models = sorted(set(ds["model_a"]) | set(ds["model_b"]))
    model_to_id = {m: i for i, m in enumerate(unique_models)}
    
    if verbose:
        print(f"[LLMArena] Found {len(unique_models)} unique Models")

    # 5. Build the Lists
    prompts = []
    model_a_ids = []
    model_b_ids = []
    winners = []
    
    for r in ds:
        prompt = clean_prompt(r["prompt"])
        
        # Skip bad/short/code prompts
        if not is_good_prompt(prompt, min_words=min_prompt_words):
            continue
            
        # Store Data
        prompts.append(prompt)
        
        # Map model names to Integers (0 to M-1)
        model_a_ids.append(model_to_id[r["model_a"]])
        model_b_ids.append(model_to_id[r["model_b"]])
        
        # Determine Winner Index (0 for A, 1 for B)
        # Dataset has 'winner_model_a': 1 if A won, 0 if B won
        # Our target 'winners': 0 if A, 1 if B (Index of the winner in the pair)
        
        if r["winner_model_a"] == 1:
            winners.append(0) # A wins
        else:
            winners.append(1) # B wins

    # 6. Convert to Numpy
    dataset = BattleDataset(
        prompts=prompts,
        model_a_ids=np.array(model_a_ids, dtype=int),
        model_b_ids=np.array(model_b_ids, dtype=int),
        winners=np.array(winners, dtype=int),
        journal_names=unique_models
    )

    if verbose:
        dataset.print_summary()
        
    return dataset

if __name__ == "__main__":
    ds = convert_llmarena_dataset(sample_size=5000)
    # You can now save this object with pickle or use it directly
    # with open("llm_battles.pkl", "wb") as f:
    #     pickle.dump(ds, f)