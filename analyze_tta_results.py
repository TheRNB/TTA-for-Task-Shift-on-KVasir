import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import argparse


# CODE IS OWN WORK
def load_results(results_path):
    with open(results_path, "rb") as f:
        return pickle.load(f)


def compute_statistics(all_results):
    stats_dict = {}

    for method_name, results in all_results.items():
        dice_scores = [r["dice"] for r in results]

        stats_dict[method_name] = {
            "mean": np.mean(dice_scores),
            "std": np.std(dice_scores),
            "median": np.median(dice_scores),
            "min": np.min(dice_scores),
            "max": np.max(dice_scores),
            "q25": np.percentile(dice_scores, 25),
            "q75": np.percentile(dice_scores, 75),
            "scores": dice_scores,
        }

    return stats_dict


def plot_dice_boxplot(stats_dict, save_path):
    plt.figure(figsize=(12, 6))

    methods = list(stats_dict.keys())
    scores = [stats_dict[m]["scores"] for m in methods]

    bp = plt.boxplot(scores, labels=methods, patch_artist=True)

    # Color boxes
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    for patch, color in zip(bp["boxes"], colors[: len(methods)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.ylabel("Dice Score (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Method", fontsize=12, fontweight="bold")
    plt.title(
        "Dice Score Comparison Across TTA Methods", fontsize=14, fontweight="bold"
    )
    plt.grid(axis="y", alpha=0.3)

    # Add mean markers
    means = [stats_dict[m]["mean"] for m in methods]
    plt.plot(
        range(1, len(methods) + 1),
        means,
        "D",
        color="red",
        markersize=8,
        label="Mean",
        zorder=3,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved boxplot to: {save_path}")


def plot_dice_distribution(stats_dict, save_path):
    """Create distribution plot for Dice scores"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    methods = list(stats_dict.keys())
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    for idx, method in enumerate(methods):
        if idx < len(axes):
            ax = axes[idx]
            scores = stats_dict[method]["scores"]

            # Histogram
            ax.hist(scores, bins=20, alpha=0.6, color=colors[idx], edgecolor="black")

            # Add mean line
            mean_val = stats_dict[method]["mean"]
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.2f}%",
            )

            # Add median line
            median_val = stats_dict[method]["median"]
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.2f}%",
            )

            ax.set_xlabel("Dice Score (%)", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"{method}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(methods), len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        "Distribution of Dice Scores by Method", fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved distribution plot to: {save_path}")


def plot_method_comparison_bar(stats_dict, save_path):
    """Create bar plot with error bars"""
    plt.figure(figsize=(12, 6))

    methods = list(stats_dict.keys())
    means = [stats_dict[m]["mean"] for m in methods]
    stds = [stats_dict[m]["std"] for m in methods]

    colors = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

    bars = plt.bar(
        methods,
        means,
        yerr=stds,
        capsize=5,
        color=colors[: len(methods)],
        alpha=0.7,
        edgecolor="black",
    )

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{mean:.2f}%\n±{std:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.ylabel("Dice Score (%)", fontsize=12, fontweight="bold")
    plt.xlabel("Method", fontsize=12, fontweight="bold")
    plt.title("Mean Dice Score with Standard Deviation", fontsize=14, fontweight="bold")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved bar plot to: {save_path}")


def plot_sample_wise_comparison(all_results, save_path):
    """Plot sample-wise Dice scores across methods"""
    plt.figure(figsize=(14, 6))

    methods = list(all_results.keys())
    num_samples = len(all_results[methods[0]])

    for method in methods:
        dice_scores = [r["dice"] for r in all_results[method]]
        plt.plot(range(num_samples), dice_scores, marker="o", label=method, linewidth=2)

    plt.xlabel("Sample Index", fontsize=12, fontweight="bold")
    plt.ylabel("Dice Score (%)", fontsize=12, fontweight="bold")
    plt.title("Sample-wise Dice Score Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved sample-wise comparison to: {save_path}")


def statistical_significance_test(stats_dict):
    """Perform statistical significance tests between methods"""
    print()
    print("Statistical Significance Tests (Paired t-test)")
    print()

    methods = list(stats_dict.keys())

    # Compare each method with baseline
    baseline_scores = stats_dict["Baseline"]["scores"]

    print()
    print(f"Comparing all methods against Baseline (n={len(baseline_scores)}):")
    print()

    for method in methods:
        if method == "Baseline":
            continue

        method_scores = stats_dict[method]["scores"]
        t_stat, p_value = stats.ttest_rel(baseline_scores, method_scores)

        mean_diff = np.mean(method_scores) - np.mean(baseline_scores)
        significance = (
            "***"
            if p_value < 0.001
            else "**"
            if p_value < 0.01
            else "*"
            if p_value < 0.05
            else "ns"
        )

        print(
            f"{method:15s} vs Baseline: Δ={mean_diff:+.2f}%, p={p_value:.4f} {significance}"
        )

    # Pairwise comparisons
    print()
    print("Pairwise Comparisons:")
    print()

    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            method1, method2 = methods[i], methods[j]
            scores1 = stats_dict[method1]["scores"]
            scores2 = stats_dict[method2]["scores"]

            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            mean_diff = np.mean(scores2) - np.mean(scores1)
            significance = (
                "***"
                if p_value < 0.001
                else "**"
                if p_value < 0.01
                else "*"
                if p_value < 0.05
                else "ns"
            )

            print(
                f"{method1:15s} vs {method2:15s}: Δ={mean_diff:+.2f}%, p={p_value:.4f} {significance}"
            )

    print()
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")


def print_detailed_statistics(stats_dict):
    print()
    print("Detailed Statistics Summary")
    print()
    print(
        f"{'Method':<15} {'Mean':<10} {'Std':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Q25-Q75':<15}"
    )
    print()

    for method, stats_data in stats_dict.items():
        print(
            f"{method:<15} "
            f"{stats_data['mean']:<10.2f} "
            f"{stats_data['std']:<10.2f} "
            f"{stats_data['median']:<10.2f} "
            f"{stats_data['min']:<10.2f} "
            f"{stats_data['max']:<10.2f} "
            f"{stats_data['q25']:.2f}-{stats_data['q75']:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyze TTA Results")

    parser.add_argument(
        "--results_path",
        type=str,
        default="./tta_visualizations_complete/all_results.pkl",
        help="Path to pickled results file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tta_analysis",
        help="Output directory for analysis plots",
    )

    args = parser.parse_args()

    # Check if results exist
    if not os.path.exists(args.results_path):
        print(f"Error: Results file not found at {args.results_path}")
        print()
        print("Please run the visualization script with --save_results flag first:")
        print("  python visualize_all_tta_complete.py --save_results")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print(f"Loading results from: {args.results_path}")
    all_results = load_results(args.results_path)

    print()
    print(f"Loaded results for {len(all_results)} methods:")
    for method in all_results.keys():
        print(f"  - {method}: {len(all_results[method])} samples")

    # Compute statistics
    print()
    print("Computing statistics...")
    stats_dict = compute_statistics(all_results)

    # Print detailed statistics
    print_detailed_statistics(stats_dict)

    # Statistical significance tests
    statistical_significance_test(stats_dict)

    # Generate plots
    print()
    print("Generating Plots")
    print()

    plot_dice_boxplot(stats_dict, os.path.join(args.output_dir, "dice_boxplot.png"))
    plot_dice_distribution(
        stats_dict, os.path.join(args.output_dir, "dice_distribution.png")
    )
    plot_method_comparison_bar(
        stats_dict, os.path.join(args.output_dir, "method_comparison.png")
    )
    plot_sample_wise_comparison(
        all_results, os.path.join(args.output_dir, "sample_wise_comparison.png")
    )

    print()
    print(f"Analysis complete! Plots saved to: {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
