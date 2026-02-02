"""
main.py
Main script to run BTL-PPTM on MADStat dataset

This script:
1. Loads and converts MADStat dataset from madstat_data/
2. Fits the BTL-PPTM model
3. Shows journal rankings by topic with REAL topic words

FIXED VERSION - Shows actual topic words instead of artificial names
"""

import numpy as np
import pandas as pd
from typing import Dict

# Import our modules
from madstat_converter import convert_madstat_dataset
from btl_pptm import BTLPPTMAnalyzer
from btl_analysis import BTLPPTMAnalysis, BTLPPTMVisualizer
from convert_llmarena import convert_llmarena_dataset

import argparse
from utils import fit_classic_btl_llmarena, fit_classic_btl_madstat

def get_topic_words(analyzer, top_k=30):
    m = analyzer.pptm_model.model
    if m.vocab is None:
        raise RuntimeError("You must pass vocab into Embedded_SCORE_C!")
    topic_words = {}
    for k in range(analyzer.n_topics):
        # sort descending by membership
        sorted_idxs = np.argsort(-m.word_mixed_member[:, k])
        seen = set()
        words = []
        for i in sorted_idxs:
            w = m.vocab[i]
            if w not in seen:
                seen.add(w)
                words.append(w)
            if len(words) == top_k:
                break
        topic_words[k] = words
    return topic_words


def print_topic_summary(topic_words):
    """
    Print a nice summary of discovered topics
    """
    print("\n🔍 DISCOVERED TOPICS")
    print("=" * 40)
    
    for topic_idx, words in topic_words.items():
        words_str = ", ".join(words)
        print(f"Topic {topic_idx}: {words_str}")


def run_madstat_analysis(args):
    """
    Main function to run BTL-PPTM on MADStat dataset
    """
    print("🔬RUNNING EXPERIMENT ON MADSTAT DATASET")
    print("=" * 40)
    
    # Step 1: Load and convert MADStat dataset
    print("Step 1: Loading MADStat dataset...")
    try:
        dataset = convert_madstat_dataset(
            data_dir="madstat_data",
            min_citations=3,           # Papers with at least 3 citations
            min_abstract_length=50,   # Abstracts with at least 100 characters
            sample_size=70000          # Use 2000 papers for reasonable speed
        )
        print(f"✓ Successfully loaded {dataset.n_papers} papers from {dataset.n_journals} journals")
        
    except Exception as e:
        print(f"❌ Error loading MADStat: {e}")
        print("Make sure you have:")
        print("  1. Created 'madstat_data' directory")
        print("  2. Put 'Raw-data-2019-12-version.RData' in that directory")
        print("  3. Put 'AuthorPaperInfo.RData' in that directory")
        print("  4. Installed rpy2: pip install rpy2")
        return None
    
    if args.model == "btl":
        print("\nStep 3: Fitting BTL only model...")
        theta = fit_classic_btl_madstat(dataset)
        order = np.argsort(-theta)   # indices sorted by descending theta
        ranking_df = pd.DataFrame({
            "rank": np.arange(1, len(theta) + 1),
            "model": [dataset.journal_names[i] for i in order],
            "theta": theta[order]
        })
        print("Saving ranking from simple BTL ...")
        ranking_df.to_csv("Madstat_btl_ranking.csv", index=False)
        print("Rankings saved!")

    else:

        # Step 2: Initialize BTL-PPTM model
        print("\nStep 2: Initializing BTL-PPTM model...")
        analyzer = BTLPPTMAnalyzer(
            n_topics=12,                    # 8 topics for statistics papers
            n_hyperwords=2000,              # Reasonable number of hyperwords
            embed_dim=20,                  # Embedding dimension
            model_name='distilbert-base-uncased',
            reg_lambda=1e-4,               # Regularization
            random_seed=42                 # For reproducibility
        )
        print("✓ Model initialized")
        
        # Step 3: Fit the model
        try:
            print("\nStep 3: Fitting BTL-PPTM model...")
            analyzer.fit(dataset, batch_size=16, verbose=True)
            print("✓ Model fitting completed!")
            
        except Exception as e:
            print(f"❌ Error fitting model: {e}")
            return None
            
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer    = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_mat     = vectorizer.fit_transform(dataset.abstracts)

        analyzer.vectorizer    = vectorizer
        analyzer.tfidf_matrix  = tfidf_mat

        # Step 4: Extract and display topic words
        print("\nStep 4: Extracting topic words...")
        topic_words = get_topic_words(analyzer, top_k=20)
        print_topic_summary(topic_words)
        
        # Step 5: Analyze results
        print("\nStep 5: Analyzing results...")
        analysis = BTLPPTMAnalysis(analyzer)
        
        # Show journal-topic impact matrix
        print("\n📊 JOURNAL-TOPIC IMPACT MATRIX (S)")
        print("=" * 40)
        print("Shape:", analyzer.btl_model.S.shape)
        print("Sample of impact scores:")
        
        # Show top 5 journals and their topic impacts
        for j in range(min(5, dataset.n_journals)):
            journal_name = dataset.journal_names[j][:20]  # Truncate long names
            impacts = analyzer.btl_model.S[j, :]
            print(f"{journal_name}: {impacts}")
        
        # Get topic-specific rankings
        print("\n🏆 JOURNAL RANKINGS BY DISCOVERED TOPICS")
        print("=" * 45)
        
        rankings_df = analysis.get_topic_rankings()
        
        # Show top 5 journals for each topic with real topic words
        for topic_idx in range(analyzer.n_topics):
            topic_data = rankings_df[rankings_df['topic_idx'] == topic_idx].head(20)
            
            # Get topic words for this topic
            words = topic_words.get(topic_idx, [f"word_{i}" for i in range(20)])
            topic_description = ", ".join(words[:10])  # Show top 3 words
            
            print(f"\nTopic {topic_idx} ({topic_description}) - Top 5 Journals:")
            for _, row in topic_data.iterrows():
                print(f"  {int(row['rank'])}. {row['journal'][:30]} (score: {row['score']:.3f})")
        
        # Create research areas based on discovered topics
        print("\n🔬 RESEARCH AREAS BASED ON DISCOVERED TOPICS")
        print("=" * 50)
        
        # Create topic-based research areas using the actual discovered topics
        research_areas = {}
        
        for topic_idx, words in topic_words.items():
            # Create a focus vector that emphasizes this topic
            focus_vector = np.zeros(analyzer.n_topics)
            focus_vector[topic_idx] = 0.8  # High weight on this topic
            
            # Add some weight to related topics (you could make this smarter)
            for other_idx in range(analyzer.n_topics):
                if other_idx != topic_idx:
                    focus_vector[other_idx] = 0.2 / (analyzer.n_topics - 1)
            
            # Create area name from top 2 words
            area_name = f"{words[0].title()} & {words[1].title()}" if len(words) >= 2 else f"Topic {topic_idx}"
            research_areas[area_name] = focus_vector
        
        # Show rankings for each research area
        comparison_df = analysis.compare_rankings(research_areas, top_k=8)
        
        for area_name in research_areas.keys():
            area_data = comparison_df[comparison_df['ranking_type'] == area_name].head(5)
            print(f"\nTop 5 journals for {area_name}:")
            for _, row in area_data.iterrows():
                print(f"  {int(row['rank'])}. {row['journal'][:30]}")
        
        # Step 6: Save results
        print("\nStep 6: Saving results...")
        analyzer.save_results("madstat_btl_pptm_results.npz")
        
        # Save rankings to CSV for easy viewing
        rankings_df.to_csv("madstat_journal_rankings.csv", index=False)
        comparison_df.to_csv("madstat_research_areas_comparison.csv", index=False)
        
        # Save topic words
        topic_words_df = pd.DataFrame([
            {"topic_idx": idx, "word_rank": i, "word": word}
            for idx, words in topic_words.items()
            for i, word in enumerate(words)
        ])
        topic_words_df.to_csv("madstat_topic_words.csv", index=False)
        
        print("✓ Results saved:")
        print("  - madstat_btl_pptm_results.npz (model results)")
        print("  - madstat_journal_rankings.csv (topic rankings)")
        print("  - madstat_research_areas_comparison.csv (research area comparison)")
        print("  - madstat_topic_words.csv (discovered topic words)")
        
        # Step 7: Generate plots (optional, with better error handling)
        print("\nStep 7: Generating visualizations...")
        try:
            viz = BTLPPTMVisualizer(analyzer)
            
            # Plot top journals by topic (with safer formatting)
            print("  Creating topic rankings plot...")
            viz.plot_topic_rankings(top_k=8, figsize=(15, 10))
            
            # Plot journal-topic heatmap (with safer formatting)  
            print("  Creating journal-topic heatmap...")
            topic_labels = {
                0: "Experimental Design",
                1: "Clinical Trials",
                2: "Distribution Theory",
                3: "Statistical Methodology",
                4: "Epidemiology",
                5: "Multivariate Analysis",
                6: "Time Series/Forecasting",
                7: "Math Stat Foundations",
                8: "Estimation Theory",
                9: "Stochastic Processes",
                10: "Computational Stats",
                11: "Inference & Intervals"
            }
            viz.plot_topic_journal_heatmap(figsize=(12, 8), topic_label_dict=topic_labels)
            #viz.plot_rank_heatmap()
            viz.plot_rank_stability()
            
            print("✓ Plots generated and displayed")
            
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
            print("  This is often due to formatting issues or headless environments")
            print("  Check the saved CSV files for results instead")
        
        print("\n🎉 BTL-PPTM ANALYSIS COMPLETED!")
        print("=" * 35)
    
        # Print summary with topic words
        print(f"✓ Processed {dataset.n_papers} papers")
        print(f"✓ Analyzed {dataset.n_journals} journals") 
        print(f"✓ Identified {analyzer.n_topics} topics")
        print(f"✓ Processed {dataset.n_citations:,} citations")
        
        print("\n📝 Discovered Topics Summary:")
        for topic_idx, words in topic_words.items():
            words_str = ", ".join(words[:])  # Show top 4 words
            print(f"  Topic {topic_idx}: {words_str}")
        
        return analyzer, analysis

def run_llmarena_analysis(args):
    """
    Run BTL-PPTM on LM-Arena human preference dataset
    ranking LLMs by topic-specific performance
    """

    # -------------------------------
    # Step 1 — Load dataset
    # -------------------------------
    print("Step 1: Loading LLMArena dataset...")

    try:
        dataset = convert_llmarena_dataset(
            min_prompt_words=10,
            sample_size=None,
            verbose=True
        )


        print(f"✓ Loaded {len(dataset.prompts)} prompts and {len(dataset.journal_names)} LLMs")

    except Exception as e:
        print(f"❌ Error loading LLMArena: {e}")
        return None

    # -------------------------------
    # Step 2 — Initialize PPTM + BTL
    # -------------------------------
    print("\nStep 2: Initializing BTL-PPTM model...")
    if args.model == "btl":
        print("\nStep 3: Fitting BTL only model...")
        theta = fit_classic_btl_llmarena(dataset)
        order = np.argsort(-theta)   # indices sorted by descending theta
        ranking_df = pd.DataFrame({
            "rank": np.arange(1, len(theta) + 1),
            "model": [dataset.journal_names[i] for i in order],
            "theta": theta[order]
        })
        print("Saving ranking from simple BTL ...")
        ranking_df.to_csv("Llmarena_btl_ranking.csv", index=False)
        print("Rankings saved!")
    else:
        print("🤖 Running Topic Specific BTL model")
        print("=" * 40)
        analyzer = BTLPPTMAnalyzer(
            n_topics=7,                # good default for LM prompts
            n_hyperwords=2000,
            embed_dim=30,
            model_name='distilbert-base-uncased',
            reg_lambda=1e-4,
            random_seed=42
        )

        print("✓ Model initialized")

        # -------------------------------
        # Step 3 — Fit the model
        # -------------------------------

        try:
            print("\nStep 3: Fitting model (this may take a while)...")
            analyzer.fit(dataset, batch_size=16, verbose=True)
            print("✓ Model fitting completed!")
        except Exception as e:
            print(f"❌ Model fitting error: {e}")
            return None

        # -------------------------------
        # Step 4 — Topic words
        # -------------------------------
        print("\nStep 4: Extracting topic words...")

        topic_words = get_topic_words(analyzer, top_k=25)
        print_topic_summary(topic_words)


        print("\nStep 4b: Showing example prompts for each topic...")

        W = analyzer.pptm_model.topic_weights  # shape = (num_prompts, num_topics)
        prompts = analyzer.dataset.prompts

        # normalize rows to get purity
        row_sums = W.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0   # avoid divide-by-zero
        purity = W / row_sums

        num_examples = 10


        # Open file for writing
        with open('topic_analysis_results.txt', 'w', encoding='utf-8') as f:
            for k in range(analyzer.n_topics):
                f.write("\n" + "=" * 90 + "\n")
                f.write(f"🧠 Topic {k}: {', '.join(topic_words[k][:10])}\n")
                f.write("-" * 90 + "\n")
                
                topic_scores = purity[:, k]
                top_idx = np.argsort(-topic_scores)[:num_examples]
                
                for rank, i in enumerate(top_idx, 1):
                    score = topic_scores[i]
                    text = prompts[i].strip().replace("\n", " ")
                    if len(text) > 180:
                        text = text[:180] + "..."
                    f.write(f"[{rank}] purity={score:.3f} → {text}\n")
                
                f.write("\n")  # Extra blank line between topics for readability

        print("Results saved to 'topic_analysis_results.txt'")


        # -------------------------------
        # Step 5 — Rankings
        # -------------------------------
        print("\nStep 5: Computing topic-specific LLM rankings...")

        analysis = BTLPPTMAnalysis(analyzer)
        rankings_df = analysis.get_topic_rankings()
        # Drop the dummy prompt pseudo-journal from any rankings
        rankings_df = rankings_df[rankings_df["journal"] != "__PROMPT__"].copy()


        for topic_idx in range(analyzer.n_topics):
            words = topic_words.get(topic_idx, [])
            desc = ", ".join(words[:8])

            print(f"\n🏆 Topic {topic_idx}: {desc}")

            topic_models = rankings_df[rankings_df["topic_idx"] == topic_idx].head(10)

            for _, row in topic_models.iterrows():
                print(
                    f"  {int(row['rank'])}. "
                    f"{str(row['journal'])[:40]} "
                    f"(score={float(row['score']):.3f})"
                )



        # -------------------------------
        # Step 6 — Save results
        # -------------------------------
        analyzer.save_results("llmarena_btl_pptm_results.npz")
        rankings_df.to_csv("llmarena_model_rankings.csv", index=False)

        print("\n✓ Saved:")
        print("  llmarena_btl_pptm_results.npz")
        print("  llmarena_model_rankings.csv")

        # -------------------------------
        # Final Summary
        # -------------------------------
        print("\n🎉 COMPLETED LLMARENA ANALYSIS")
        print("=" * 35)

        print("\n📝 Discovered Topics:")
        for k, words in topic_words.items():
            print(f"Topic {k}: {', '.join(words[:12])}")

        # Step 7: Generate plots (optional, with better error handling)
        print("\nStep 7: Generating visualizations...")
        try:
            viz = BTLPPTMVisualizer(analyzer)
            topic_labels = {
                0: "Software Development & Debugging",  # Bash, PHP, JSON, Security
                1: "Global Affairs & Economics",        # Climate, GDP, History, Geopolitics
                2: "Cooking & Lifestyle",               # Recipes, Meal Plans, Gardening
                3: "Logic Puzzles (Family Relations)",  # "How many sisters does her brother have?"
                4: "Math & ML Theory",                  # Proofs, Theorems, AI Metrics (F1, Recall)
                5: "Common Sense Reasoning",            # "Drying shirts" riddles & Travel planning
                6: "Creative Writing & Roleplay"        # Rap battles, Parodies, Persona adoption
            }
            
            # Plot top journals by topic (with safer formatting)
            print("  Creating topic rankings plot...")
            #viz.plot_topic_rankings(top_k=10, figsize=(15, 10))
            viz.plot_topic_journal_heatmap(topic_label_dict=topic_labels)
            viz.plot_rank_stability()

            
            print("✓ Plots generated and displayed")
            
        except Exception as e:
            print(f"⚠ Visualization error: {e}")
            print("  This is often due to formatting issues or headless environments")
            print("  Check the saved CSV files for results instead")
        
        print("\n🎉 BTL-PPTM ANALYSIS COMPLETED!")
        print("=" * 35)



        return analyzer, analysis


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["llmarena", "madstat"],
        required=True,
        help="Which dataset to run on"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["pptm", "btl"],
        default="pptm",
        help="Model type: pptm (topic-specific) or btl (classic)"
    )

    args = parser.parse_args()

    if args.dataset == "llmarena":
        run_llmarena_analysis(args)
    elif args.dataset == "madstat":
        run_madstat_analysis(args)

    