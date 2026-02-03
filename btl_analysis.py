"""
btl_analysis.py
Analysis and Visualization Tools for BTL-PPTM

This module contains all the analysis, visualization, and comparison tools
that work with the fitted BTL-PPTM models from btl_pptm.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import core model components
from btl_pptm import BTLPPTMAnalyzer


class BTLPPTMAnalysis:
    """
    Analysis toolkit for BTL-PPTM results
    """
    
    def __init__(self, analyzer: BTLPPTMAnalyzer):
        """
        Initialize analysis toolkit
        
        Parameters:
        - analyzer: fitted BTLPPTMAnalyzer instance
        """
        if analyzer.btl_model is None:
            raise ValueError("Analyzer must be fitted first")
            
        self.analyzer = analyzer
        self.dataset = analyzer.dataset
        self.btl_model = analyzer.btl_model
        self.n_topics = analyzer.n_topics
    
    def get_topic_rankings(self, topic_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get journal rankings for each topic
        
        Parameters:
        - topic_names: optional topic names for display
        
        Returns:
        - DataFrame with rankings for each topic
        """
        rankings_dict, scores_dict = self.btl_model.get_topic_specific_rankings()
        
        # Create results DataFrame
        results = []
        for topic_idx in range(self.n_topics):
            topic_key = f'Topic_{topic_idx}'
            topic_name = topic_names[topic_idx] if topic_names else topic_key
            
            rankings = rankings_dict[topic_key]
            scores = scores_dict[topic_key]
            
            for rank, journal_idx in enumerate(rankings):
                results.append({
                    'topic': topic_name,
                    'topic_idx': topic_idx,
                    'rank': rank + 1,
                    'journal': self.dataset.journal_names[journal_idx],
                    'journal_idx': journal_idx,
                    'score': scores[journal_idx]
                })
        
        return pd.DataFrame(results)
    
    def get_mixed_rankings(self, topic_distribution: np.ndarray, 
                          ranking_name: str = "Mixed") -> pd.DataFrame:
        """
        Get journal rankings for a mixed topic distribution
        
        Parameters:
        - topic_distribution: topic weights (should sum to 1)
        - ranking_name: name for this ranking
        
        Returns:
        - DataFrame with mixed topic rankings
        """
        # Normalize topic distribution
        topic_distribution = topic_distribution / np.sum(topic_distribution)
        
        rankings, scores = self.btl_model.get_journal_rankings(topic_distribution)
        
        results = []
        for rank, journal_idx in enumerate(rankings):
            results.append({
                'ranking_type': ranking_name,
                'rank': rank + 1,
                'journal': self.dataset.journal_names[journal_idx],
                'journal_idx': journal_idx,
                'score': scores[journal_idx]
            })
        
        return pd.DataFrame(results)
    
    def analyze_topic_journal_matrix(self) -> pd.DataFrame:
        """
        Analyze the estimated journal-topic impact matrix S
        
        Returns:
        - DataFrame with journal-topic impacts
        """
        S = self.btl_model.S  # Shape: (n_journals, n_topics)
        
        # Create DataFrame
        results = []
        for j in range(self.dataset.n_journals):
            for k in range(self.n_topics):
                results.append({
                    'journal': self.dataset.journal_names[j],
                    'journal_idx': j,
                    'topic': f'Topic_{k}',
                    'topic_idx': k,
                    'impact': S[j, k]
                })
        
        return pd.DataFrame(results)
    
    def compare_rankings(self, topic_distributions: Dict[str, np.ndarray], 
                        top_k: int = 10) -> pd.DataFrame:
        """
        Compare journal rankings across different topic distributions
        
        Parameters:
        - topic_distributions: dict of {name: topic_distribution}
        - top_k: number of top rankings to compare
        
        Returns:
        - DataFrame with comparative rankings
        """
        all_rankings = []
        
        for name, topic_dist in topic_distributions.items():
            ranking_df = self.get_mixed_rankings(topic_dist, name)
            top_ranking = ranking_df.head(top_k)
            all_rankings.append(top_ranking)
        
        combined_df = pd.concat(all_rankings, ignore_index=True)
        
        return combined_df
    
    def analyze_ranking_differences(self) -> pd.DataFrame:
        """
        Analyze how journal rankings differ across topics
        
        Returns:
        - DataFrame with ranking variance analysis
        """
        rankings_dict, scores_dict = self.btl_model.get_topic_specific_rankings()
        
        results = []
        for journal_idx, journal_name in enumerate(self.dataset.journal_names):
            positions = []
            scores = []
            
            for topic_idx in range(self.n_topics):
                topic_key = f'Topic_{topic_idx}'
                ranking = rankings_dict[topic_key]
                score = scores_dict[topic_key][journal_idx]
                
                # Find position of this journal in the ranking
                position = np.where(ranking == journal_idx)[0][0] + 1
                positions.append(position)
                scores.append(score)
            
            # Calculate statistics
            results.append({
                'journal': journal_name,
                'journal_idx': journal_idx,
                'mean_rank': np.mean(positions),
                'rank_variance': np.var(positions),
                'rank_std': np.std(positions),
                'min_rank': np.min(positions),
                'max_rank': np.max(positions),
                'rank_range': np.max(positions) - np.min(positions),
                'mean_score': np.mean(scores),
                'score_variance': np.var(scores),
                'positions': positions,
                'scores': scores
            })
        
        return pd.DataFrame(results)
    
    def get_topic_specialization_scores(self) -> pd.DataFrame:
        """
        Calculate topic specialization scores for each journal
        
        Returns:
        - DataFrame with specialization analysis
        """
        S = self.btl_model.S  # Shape: (n_journals, n_topics)
        
        results = []
        for j in range(self.dataset.n_journals):
            journal_scores = S[j, :]
            
            # Calculate specialization metrics
            max_score = np.max(journal_scores)
            min_score = np.min(journal_scores)
            score_range = max_score - min_score
            score_std = np.std(journal_scores)
            
            # Entropy-based specialization (lower entropy = more specialized)
            # Shift scores to be positive for entropy calculation
            shifted_scores = journal_scores - np.min(journal_scores) + 1e-8
            probs = shifted_scores / np.sum(shifted_scores)
            entropy = -np.sum(probs * np.log(probs + 1e-8))
            
            # Best topic
            best_topic_idx = np.argmax(journal_scores)
            
            results.append({
                'journal': self.dataset.journal_names[j],
                'journal_idx': j,
                'specialization_range': score_range,
                'specialization_std': score_std,
                'specialization_entropy': entropy,
                'best_topic': best_topic_idx,
                'best_topic_score': max_score,
                'worst_topic_score': min_score,
                'is_specialized': score_std > np.median([np.std(S[i, :]) for i in range(self.dataset.n_journals)])
            })
        
        return pd.DataFrame(results)


class BTLPPTMVisualizer:
    """
    Visualization toolkit for BTL-PPTM results
    """
    
    def __init__(self, analyzer: BTLPPTMAnalyzer):
        """
        Initialize visualization toolkit
        
        Parameters:
        - analyzer: fitted BTLPPTMAnalyzer instance
        """
        self.analyzer = analyzer
        self.analysis = BTLPPTMAnalysis(analyzer)
    
    def plot_topic_rankings(self, top_k: int = 10, topic_names: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 8)):
        """
        Plot top-k journal rankings for each topic
        
        Parameters:
        - top_k: number of top journals to show
        - topic_names: optional topic names
        - figsize: figure size
        """
        rankings_df = self.analysis.get_topic_rankings(topic_names)
        
        # Filter top-k for each topic
        top_rankings = rankings_df[rankings_df['rank'] <= top_k]
        
        # Create subplots
        n_cols = min(3, self.analyzer.n_topics)
        n_rows = (self.analyzer.n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for topic_idx in range(self.analyzer.n_topics):
            topic_data = top_rankings[top_rankings['topic_idx'] == topic_idx]
            topic_name = topic_names[topic_idx] if topic_names else f'Topic {topic_idx}'
            
            ax = axes[topic_idx] if self.analyzer.n_topics > 1 else axes
            
            # Create horizontal bar plot
            y_pos = np.arange(len(topic_data))
            ax.barh(y_pos, topic_data['score'], alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([j[:20] + '...' if len(j) > 20 else j 
                               for j in topic_data['journal']], fontsize=8)
            ax.set_xlabel('BTL Score')
            ax.set_title(f'{topic_name}\n(Top {top_k})', fontsize=10)
            ax.invert_yaxis()
        
        # Hide unused subplots
        for idx in range(self.analyzer.n_topics, len(axes)):
            axes[idx].set_visible(False)
            
        plt.tight_layout()
        plt.show()
    
    def plot_topic_journal_heatmap(self, topic_label_dict: Optional[Dict[int, str]] = None, 
                             figsize: Tuple[int, int] = (16, 15)): # Increased size slightly
        """
        Plot heatmap of journal-topic impact matrix with semantic labels
        """
        S = self.analyzer.btl_model.S
        
        # 1. GENERATE SEMANTIC LABELS
        if topic_label_dict:
            # Ensure they are in the correct order (0 to 11)
            topic_labels = [topic_label_dict[i] for i in range(self.analyzer.n_topics)]
        else:
            topic_labels = [f'Topic {i}' for i in range(self.analyzer.n_topics)]
        
        # Create journal labels (truncate if too long)
        journal_labels = [j[:15] + '...' if len(j) > 50 else j 
                        for j in self.analyzer.dataset.journal_names]
        
        plt.figure(figsize=figsize)
        
        # 2. CREATE HEATMAP
        sns.heatmap(S, 
                xticklabels=topic_labels,  # <--- Now uses real names
                yticklabels=journal_labels,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Journal-Topic Impact (Higher is Better)'},
                linewidths=0.5, # Optional: Adds grid lines for readability
                linecolor='gray'
                )
        
        plt.title('Journal-Topic Impact Matrix (S)', fontsize=14)
        #plt.xlabel('Research Topics', fontsize=12)
        #plt.ylabel('Journals', fontsize=12)
        
        # 3. FORMATTING (Crucial for text labels)
        plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate so they don't overlap
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.5) 
        plt.savefig('heatmap_S_labeled.png', dpi=300, bbox_inches='tight')
        plt.close()



#     def plot_topic_journal_heatmap(
#     self,
#     topic_names: Optional[List[str]] = None,
#     base_figwidth: float = 14.0,
#     row_height: float = 0.35,
#     max_figheight: float = 30.0,
# ):
#         """
#         Plot heatmap of journal-topic impact matrix S.

#         Automatically adapts figure height to number of journals
#         (critical for LLM Arena).
#         """

#         S = self.analyzer.btl_model.S  # (n_journals, n_topics)
#         n_journals, n_topics = S.shape

#         # -----------------------------
#         # Topic labels
#         # -----------------------------
#         topic_labels = (
#             topic_names
#             if topic_names is not None
#             else [f"Topic {i}" for i in range(n_topics)]
#         )

#         # -----------------------------
#         # Journal / model labels
#         # DO NOT truncate for LLM Arena
#         # -----------------------------
#         journal_labels = list(self.analyzer.dataset.journal_names)

#         # -----------------------------
#         # Dynamic figure size
#         # -----------------------------
#         fig_height = min(max(8.0, row_height * n_journals), max_figheight)

#         plt.figure(figsize=(base_figwidth, fig_height))

#         sns.heatmap(
#             S,
#             xticklabels=topic_labels,
#             yticklabels=journal_labels,
#             cmap="RdBu_r",
#             center=0.0,
#             cbar_kws={"label": "Journal–Topic Impact"},
#         )

#         # -----------------------------
#         # Formatting
#         # -----------------------------
#         plt.title("Journal–Topic Impact Matrix (S)", fontsize=16)
#         plt.xlabel("Topics", fontsize=13)
#         plt.ylabel("Models" if n_journals > 30 else "Journals", fontsize=13)

#         plt.xticks(rotation=45, ha="right", fontsize=11)
#         plt.yticks(rotation=0, fontsize=9)

#         plt.tight_layout()
#         plt.savefig("heatmap_S.png", dpi=300, bbox_inches="tight")
#         plt.close()

    
    def plot_ranking_comparison(self, topic_distributions: Dict[str, np.ndarray], 
                               top_k: int = 10, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot comparison of rankings across different topic distributions
        
        Parameters:
        - topic_distributions: dict of {name: topic_distribution}
        - top_k: number of top journals to show
        - figsize: figure size
        """
        comparison_df = self.analysis.compare_rankings(topic_distributions, top_k)
        
        # Create pivot table for heatmap
        pivot_df = comparison_df.pivot(index='journal', columns='ranking_type', values='rank')
        
        plt.figure(figsize=figsize)
        sns.heatmap(pivot_df, 
                   annot=True, 
                   fmt='d',
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Rank'})
        
        plt.title(f'Journal Ranking Comparison (Top {top_k})')
        plt.xlabel('Research Focus')
        plt.ylabel('Journals')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    def plot_specialization_analysis(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Plot journal specialization analysis
        
        Parameters:
        - figsize: figure size
        """
        spec_df = self.analysis.get_topic_specialization_scores()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Specialization distribution
        ax1.hist(spec_df['specialization_std'], bins=10, alpha=0.7, edgecolor='black')
        ax1.axvline(spec_df['specialization_std'].median(), color='red', linestyle='--', 
                   label=f'Median: {spec_df["specialization_std"].median():.3f}')
        ax1.set_xlabel('Specialization Score (Std Dev)')
        ax1.set_ylabel('Number of Journals')
        ax1.set_title('Distribution of Journal Specialization')
        ax1.legend()
        
        # Plot 2: Specialization vs Best Topic Score
        colors = ['red' if x else 'blue' for x in spec_df['is_specialized']]
        ax2.scatter(spec_df['best_topic_score'], spec_df['specialization_std'], 
                   c=colors, alpha=0.6)
        ax2.set_xlabel('Best Topic Score')
        ax2.set_ylabel('Specialization Score')
        ax2.set_title('Specialization vs Performance')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Specialized'),
                          Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='General')]
        ax2.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
        
        return spec_df
    
    def plot_ranking_variance(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot how journal rankings vary across topics
        
        Parameters:
        - figsize: figure size
        """
        var_df = self.analysis.analyze_ranking_differences()
        
        # Sort by ranking variance
        var_df_sorted = var_df.sort_values('rank_variance', ascending=False)
        
        plt.figure(figsize=figsize)
        
        # Create bar plot of ranking variance
        bars = plt.bar(range(len(var_df_sorted)), var_df_sorted['rank_variance'], 
                      alpha=0.7, edgecolor='black')
        
        # Color bars by variance level
        high_var_threshold = var_df_sorted['rank_variance'].quantile(0.75)
        for i, bar in enumerate(bars):
            if var_df_sorted.iloc[i]['rank_variance'] > high_var_threshold:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.xlabel('Journals')
        plt.ylabel('Ranking Variance Across Topics')
        plt.title('Journal Ranking Stability Analysis')
        plt.xticks(range(len(var_df_sorted)), 
                  [j[:10] + '...' if len(j) > 10 else j for j in var_df_sorted['journal']], 
                  rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', label='High Variance (Topic-Sensitive)'),
                          Patch(facecolor='blue', label='Low Variance (Stable)')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
        
        return var_df_sorted
    

    def plot_rank_heatmap(self, figsize=(14,10), annotate=False):
        """
        Heatmap of model ranking percentile across topics.
        Rows = models
        Cols = topics
        Lower = better
        """

        rankings_df = self.analysis.get_topic_rankings()
        
        # pivot → topic as column, journal as row, value = rank
        P = rankings_df.pivot(index="journal", columns="topic_idx", values="rank")

        # normalize rank to [0,1] so different topic sizes are comparable
        P_norm = (P - 1) / (P.max() - 1)

        # sort rows by mean normalized rank
        P_norm["avg"] = P_norm.mean(axis=1)
        P_norm = P_norm.sort_values("avg")
        P_norm = P_norm.drop(columns=["avg"])

        plt.figure(figsize=figsize)
        sns.heatmap(
            P_norm,
            cmap="viridis_r",
            annot=annotate,
            fmt=".2f",
            cbar_kws={"label": "Normalized Rank (0 = Best)"}
        )

        plt.title("Model Ranking Landscape Across Topics", fontsize=14)
        plt.xlabel("Topic")
        plt.ylabel("Model")
        plt.tight_layout()
        plt.show()

    def plot_rank_stability(self, figsize=(14,6)):
        rankings_df = self.analysis.get_topic_rankings()

        # 1. Calculate the TRUE global order (Mean Rank across ALL topics)
        sorted_order = rankings_df.groupby("journal")["rank"].mean().sort_values().index

        plt.figure(figsize=figsize)
        sns.violinplot(
            data=rankings_df,
            x="journal",
            y="rank",
            order=sorted_order,  # <--- Essential for correct global sorting
            inner="quartile",
            cut=0
        )

        plt.xticks(rotation=90)
        plt.xlabel("")
        plt.ylabel("Rank (Lower = Better)")
        plt.title("Rank Stability Across Topics")
        plt.tight_layout()
        plt.savefig('stability_rank.png', dpi=300, bbox_inches='tight')
        plt.close()




def create_comprehensive_report(analyzer: BTLPPTMAnalyzer, 
                              topic_names: Optional[List[str]] = None,
                              output_file: Optional[str] = None) -> Dict:
    """
    Create comprehensive analysis report
    
    Parameters:
    - analyzer: fitted BTLPPTMAnalyzer
    - topic_names: optional topic names
    - output_file: optional file to save report
    
    Returns:
    - dict with all analysis results
    """
    print("📊 GENERATING COMPREHENSIVE BTL-PPTM REPORT")
    print("=" * 50)
    
    analysis = BTLPPTMAnalysis(analyzer)
    
    # Generate all analyses
    topic_rankings = analysis.get_topic_rankings(topic_names)
    impact_matrix = analysis.analyze_topic_journal_matrix()
    ranking_differences = analysis.analyze_ranking_differences()
    specialization = analysis.get_topic_specialization_scores()
    
    # Summary statistics
    summary = {
        'dataset_info': {
            'name': analyzer.dataset.dataset_name,
            'n_papers': analyzer.dataset.n_papers,
            'n_journals': analyzer.dataset.n_journals,
            'n_topics': analyzer.n_topics,
            'n_citations': analyzer.dataset.n_citations
        },
        'model_performance': {
            'S_matrix_norm': float(np.linalg.norm(analyzer.btl_model.S)),
            'avg_topic_weight_entropy': float(np.mean([
                -np.sum(w * np.log(w + 1e-8)) 
                for w in analyzer.topic_weights
            ])),
            'journal_specialization_avg': float(specialization['specialization_std'].mean())
        },
        'ranking_insights': {
            'most_stable_journal': ranking_differences.loc[ranking_differences['rank_variance'].idxmin(), 'journal'],
            'most_variable_journal': ranking_differences.loc[ranking_differences['rank_variance'].idxmax(), 'journal'],
            'avg_ranking_variance': float(ranking_differences['rank_variance'].mean()),
            'most_specialized_journal': specialization.loc[specialization['specialization_std'].idxmax(), 'journal']
        }
    }
    
    # Compile full report
    report = {
        'summary': summary,
        'topic_rankings': topic_rankings,
        'impact_matrix': impact_matrix,
        'ranking_differences': ranking_differences,
        'specialization_analysis': specialization,
        'S_matrix': analyzer.btl_model.S
    }
    
    # Print key insights
    print("\n🔍 KEY INSIGHTS:")
    print(f"• Most stable journal: {summary['ranking_insights']['most_stable_journal']}")
    print(f"• Most topic-sensitive journal: {summary['ranking_insights']['most_variable_journal']}")
    print(f"• Most specialized journal: {summary['ranking_insights']['most_specialized_journal']}")
    print(f"• Average ranking variance: {summary['ranking_insights']['avg_ranking_variance']:.2f}")
    
    # Save if requested
    if output_file:
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(report, f)
        print(f"\n✓ Report saved to {output_file}")
    
    return report


def compare_multiple_models(analyzers: Dict[str, BTLPPTMAnalyzer], 
                          figsize: Tuple[int, int] = (15, 10)) -> pd.DataFrame:
    """
    Compare results across multiple BTL-PPTM models
    
    Parameters:
    - analyzers: dict of {model_name: fitted_analyzer}
    - figsize: figure size for plots
    
    Returns:
    - DataFrame with model comparison
    """
    print("🔄 COMPARING MULTIPLE BTL-PPTM MODELS")
    print("=" * 40)
    
    comparisons = []
    
    for model_name, analyzer in analyzers.items():
        if analyzer.btl_model is None:
            print(f"⚠ Skipping unfitted model: {model_name}")
            continue
            
        analysis = BTLPPTMAnalysis(analyzer)
        spec_df = analysis.get_topic_specialization_scores()
        var_df = analysis.analyze_ranking_differences()
        
        comparisons.append({
            'model': model_name,
            'n_topics': analyzer.n_topics,
            'n_journals': analyzer.dataset.n_journals,
            'n_papers': analyzer.dataset.n_papers,
            'S_matrix_norm': np.linalg.norm(analyzer.btl_model.S),
            'avg_specialization': spec_df['specialization_std'].mean(),
            'avg_ranking_variance': var_df['rank_variance'].mean(),
            'max_ranking_variance': var_df['rank_variance'].max(),
            'n_specialized_journals': (spec_df['is_specialized']).sum()
        })
    
    comparison_df = pd.DataFrame(comparisons)
    
    # Plot comparison
    if len(comparison_df) > 1:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Model complexity
        axes[0,0].bar(comparison_df['model'], comparison_df['n_topics'])
        axes[0,0].set_title('Number of Topics')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Specialization
        axes[0,1].bar(comparison_df['model'], comparison_df['avg_specialization'])
        axes[0,1].set_title('Average Journal Specialization')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Ranking variance
        axes[1,0].bar(comparison_df['model'], comparison_df['avg_ranking_variance'])
        axes[1,0].set_title('Average Ranking Variance')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Matrix norm
        axes[1,1].bar(comparison_df['model'], comparison_df['S_matrix_norm'])
        axes[1,1].set_title('S Matrix Norm')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    print("\n📊 Model Comparison Summary:")
    print(comparison_df.round(3))
    
    return comparison_df



def plot_btl(
    journals,
    theta_scores,
    title='',
    xlabel=r'$\hat{\theta}$ (Strength Parameter)',
    save_path=None,
    dpi=300,
    min_fontsize=10,
    show_values=False
):
    """
    Create a two-column horizontal BTL ranking plot:
    - Positive (blue) scores on the left
    - Negative (red) scores on the right
    """

    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------
    # Sort by theta (worst -> best)
    # ---------------------------
    theta_scores = np.asarray(theta_scores)
    order = np.argsort(theta_scores)
    theta_sorted = theta_scores[order]
    journals_sorted = np.array(journals)[order]

    # ---------------------------
    # Split positive / negative
    # ---------------------------
    neg_mask = theta_sorted < 0
    pos_mask = theta_sorted >= 0

    theta_neg = theta_sorted[neg_mask]
    journals_neg = journals_sorted[neg_mask]

    theta_pos = theta_sorted[pos_mask]
    journals_pos = journals_sorted[pos_mask]

    n_total = len(theta_sorted)

    # ---------------------------
    # Figure sizing
    # ---------------------------
    fig_height = max(6, 0.2 * n_total)

    fig, (ax_left, ax_right) = plt.subplots(
        ncols=2,
        sharey=True,
        figsize=(15, fig_height),
        constrained_layout=True
    )

    # ---------------------------
    # Colors (preserve semantics)
    # ---------------------------
    blues = plt.cm.Blues(np.linspace(0.4, 0.9, len(theta_pos)))
    reds = plt.cm.Reds(np.linspace(0.4, 0.9, len(theta_neg)))

    # ---------------------------
    # Left panel: positive scores
    # ---------------------------
    ax_left.barh(
        journals_pos,
        theta_pos,
        color=blues,
        edgecolor='white',
        linewidth=0.6
    )
    ax_left.invert_xaxis()
    ax_left.set_xlim(0, max(theta_pos.max(), 1e-3) * 1.05)
    ax_left.set_title('Higher skill', fontsize=12, fontweight='bold')
    ax_left.set_xlabel(xlabel, fontsize=11)

    # ---------------------------
    # Right panel: negative scores
    # ---------------------------
    ax_right.barh(
        journals_neg,
        theta_neg,
        color=reds,
        edgecolor='white',
        linewidth=0.6
    )
    ax_right.set_xlim(min(theta_neg.min(), -1e-3) * 1.05, 0)
    ax_right.set_title('Lower skill', fontsize=12, fontweight='bold')
    ax_right.set_xlabel(xlabel, fontsize=11)

    # ---------------------------
    # Zero reference lines
    # ---------------------------
    ax_left.axvline(0, color='black', linewidth=0.8, alpha=0.6)
    ax_right.axvline(0, color='black', linewidth=0.8, alpha=0.6)

    # ---------------------------
    # Tick label size
    # ---------------------------
    label_fontsize = max(min_fontsize, min(11, 350 / n_total))
    ax_left.tick_params(axis='y', labelsize=label_fontsize)
    ax_right.tick_params(axis='y', labelsize=label_fontsize)

    # ---------------------------
    # Optional value labels (not recommended for many items)
    # ---------------------------
    if show_values and n_total <= 25:
        for y, val in zip(journals_pos, theta_pos):
            ax_left.text(
                val * 0.98, y, f'{val:.2f}',
                ha='right', va='center', fontsize=9
            )
        for y, val in zip(journals_neg, theta_neg):
            ax_right.text(
                val * 0.98, y, f'{val:.2f}',
                ha='left', va='center', fontsize=9
            )

    # ---------------------------
    # Aesthetic cleanup
    # ---------------------------
    for ax in (ax_left, ax_right):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)

    # ---------------------------
    # Title
    # ---------------------------
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # ---------------------------
    # Save
    # ---------------------------
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig, (ax_left, ax_right)
