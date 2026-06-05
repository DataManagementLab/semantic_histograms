from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from tqdm import tqdm
from mmce.dataset import Dataset
from mmce.embedder import TextEmbedder
from mmce.estimators.base_cardinality_estimator import (
    CardinalityEstimator,
)
from mmce.estimators.base_threshold_estimator import (
    ComboThresholdEstimator,
    FullEstimator,
    ThresholdBasedCardinalityEstimator,
)
from mmce.estimators.threshold_estimators.kv_based import PreLoadedKV
from mmce.models.threshold.model import ThresholdModel
from mmce.estimators.threshold_estimators.specificity_model import (
    SpecificityModelThresholdEstimator,
)
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FuncFormatter


SCALE_FACTOR = 1

logger = logging.getLogger(__name__)

SPECIFICITY_MODEL = "Ours 1 (Specificity Model)"
KV_BATCHING = "Ours 2 (Compr. KV-Cache Batching)"
ENSEMBLE = "Ours 1+2 (Ensemble)"
SAMPLING = "Sampling (Baseline)"


def offset(row):
    if row.iloc[1] in (KV_BATCHING, ENSEMBLE) and row.iloc[2] == 32:
        return (-15, -2)

    if row.iloc[1] in (KV_BATCHING, ENSEMBLE) and row.iloc[2] == 128:
        return (3, -2)

    return (-5, 5)


class Benchmark:
    def __init__(
        self,
        threshold_model_path=Path(
            "artifacts/models/wordnet_threshold_model.safetensors"
        ),
        output_dir=Path("artifacts/"),
    ):
        self.threshold_model_path = threshold_model_path
        self.device = torch.device("cuda:0")
        self.threshold_model = ThresholdModel.from_pretrained(
            self.threshold_model_path
        ).to(self.device)
        self.threshold_model.eval()

        self.output_dir = output_dir
        (output_dir / "csv").mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)

        pre_loaded_kv = PreLoadedKV()
        specificity_model = SpecificityModelThresholdEstimator(self.threshold_model)
        self.approaches: Sequence[CardinalityEstimator] = [
            # RandomOrder(),
            ThresholdBasedCardinalityEstimator(
                threshold_estimator=specificity_model,
                threshold_to_cardinality=FullEstimator(),
                determines_bucket_sizes="threshold_estimator",
                other_bucket_size=1000,
            ),
            # SigLipDecoderEstimator(),
            # RegressorEstimator(),
            # ClusterGaussianEstimator(),
            # BatchClusterGaussianEstimator(),
            # BatchCoresetEstimator(),
            # CoresetEstimator(),
            # GaussianMixtureEstimator(),
            # StratifiedSamplingEstimator(),
            # RandomProjectionEstimator(),
            # KVEstimator(0.6),
            ThresholdBasedCardinalityEstimator(
                threshold_estimator=pre_loaded_kv,
                threshold_to_cardinality=FullEstimator(),
                determines_bucket_sizes="threshold_estimator",
                other_bucket_size=1000,
            ),
            ThresholdBasedCardinalityEstimator(
                threshold_estimator=ComboThresholdEstimator(
                    pre_loaded_kv, specificity_model, 1000
                ),
                threshold_to_cardinality=FullEstimator(),
                determines_bucket_sizes="threshold_estimator",
                other_bucket_size=1000,
            ),
        ]
        self.setup_rng(seed=42)

    def setup_rng(self, seed: int):
        self.np_rng = np.random.default_rng(seed)
        self.torch_rng = torch.Generator(device="cpu")
        self.torch_rng.manual_seed(seed + 1)

    def run(
        self,
        dataset: Dataset,
        num_embeddings: List[int],
        num_kv_caches: List[int],
        sample_sizes: List[int],
        num_queries: int,
        num_filters: List[int],
        num_seeds: int = 20,
    ):
        bucket_sizes = {
            "sampling": sample_sizes,
            "num_kv_caches": num_kv_caches,
            "num_embeddings": num_embeddings,
        }
        qerror_data = self.run_qerror(
            dataset=dataset,
            bucket_sizes=bucket_sizes,
            num_seeds=num_seeds,
        )
        reorder_data = self.run_reorder(
            dataset=dataset,
            bucket_sizes=bucket_sizes,
            num_queries=num_queries,
            all_num_filters=num_filters,
            qerror_data=qerror_data,
            num_seeds=num_seeds,
        )
        qerror_data.to_csv(
            self.output_dir / "csv" / f"qerror-{dataset.name()}.csv",
            index=False,
        )
        reorder_data.to_csv(
            self.output_dir / "csv" / f"reorder-{dataset.name()}-q{num_queries}.csv",
            index=False,
        )

    def plot(self, datasets: Sequence[Dataset], num_queries: int):
        aliases = {
            "SpecificityModelThresholdEstimator-FullEstimator": SPECIFICITY_MODEL,
            "PreLoadedKV-FullEstimator": KV_BATCHING,
            "combo-PreLoadedKV-SpecificityModelThresholdEstimator-FullEstimator": ENSEMBLE,
            "sampling": SAMPLING,
        }
        q_error_plot_data = self.get_plot_data_q_error(datasets=datasets)
        self.plot_qerror(q_error_plot_data, aliases, [ENSEMBLE, SAMPLING])
        plot_data_reorder = self.get_plot_data_reorder(
            datasets=datasets, num_queries=num_queries
        )
        self.plot_reorder(
            plot_data_reorder,
            num_queries=num_queries,
            aliases=aliases,
            k_label_approaches=[
                ENSEMBLE,
                SAMPLING,
                KV_BATCHING,
            ],
        )
        self.plot_reorder_detailed(datasets=datasets, num_queries=num_queries)

    def get_plot_data_reorder(
        self, datasets: Sequence[Dataset], num_queries: int
    ) -> pd.DataFrame:
        if len(datasets) == 0:
            return pd.DataFrame()
        collected_data = []
        for dataset in datasets:
            # Read the benchmark CSV data specific to reorder
            data_path = (
                self.output_dir / "csv" / f"reorder-{dataset.name()}-q{num_queries}.csv"
            )
            benchmark_data = pd.read_csv(data_path)
            benchmark_data["dataset"] = dataset.name()
            collected_data.append(benchmark_data)
        all_data: pd.DataFrame = pd.concat(collected_data, ignore_index=True)  # pyright: ignore
        return all_data

    def plot_reorder(
        self,
        all_data: pd.DataFrame,
        num_queries: int,
        aliases: Optional[Dict[str, str]] = None,
        k_label_approaches: Optional[list[str]] = None,
        dataset_order=["artwork", "wildlife", "ecommerce"],
    ):
        all_data = all_data.copy()

        perfect_data: pd.DataFrame = all_data[all_data["approach"] == "perfect"]  # type: ignore
        non_perfect_data = all_data[all_data["approach"] != "perfect"]

        # 1. Merge and calculate overhead for BOTH metrics
        metrics = ["total_time", "profiling_time"]
        merged_data = non_perfect_data.merge(
            perfect_data[["dataset", "num_filters"] + metrics].rename(  # type: ignore
                columns={m: f"{m}_perfect" for m in metrics}
            ),
            on=["dataset", "num_filters"],
            how="left",
        )

        for m in metrics:
            merged_data[f"{m}_overhead"] = merged_data[m] - merged_data[f"{m}_perfect"]

        # Map raw approach names to human-readable aliases
        if aliases:
            merged_data["approach"] = merged_data["approach"].replace(aliases)

        # 2. Group by dataset, approach, filters, components, seed and sum across queries
        grouped_sum = merged_data.groupby(
            ["dataset", "approach", "num_filters", "num_components", "seed"],
            as_index=False,
        )[["total_time_overhead", "profiling_time_overhead"]].sum()

        # 3. Median across seeds
        means = grouped_sum.groupby(
            ["dataset", "approach", "num_filters", "num_components"], as_index=False
        )[["total_time_overhead", "profiling_time_overhead"]].median()

        # 4. Find the best num_components based on minimum total_time_overhead
        idx = means.groupby(["dataset", "approach", "num_filters"])[
            "total_time_overhead"
        ].idxmin()
        best_components = means.loc[idx].copy()[
            ["dataset", "approach", "num_filters", "num_components"]
        ]

        # Cast to string so categorical barplot mapping works predictably
        only_best_components = grouped_sum.merge(
            best_components, on=best_components.columns.tolist()
        )
        self.print_paper_stats_reorder(only_best_components)
        only_best_components["num_filters"] = only_best_components[
            "num_filters"
        ].astype(str)

        # Explicitly define the ordering of the X-axis to guarantee matching later
        filter_order = sorted(
            only_best_components["num_filters"].unique(), key=lambda x: float(x)
        )

        # --- PAPER READY STYLING ---
        sns.set_context("paper", font_scale=1.0)
        sns.set_style("whitegrid")

        # Build the exact hue order and apply aliases to it
        hue_order = [
            SAMPLING,
            SPECIFICITY_MODEL,
            KV_BATCHING,
            ENSEMBLE,
        ]

        palette_colors = sns.color_palette(n_colors=len(hue_order))

        g = sns.FacetGrid(
            only_best_components,
            col="dataset",
            col_wrap=3,
            sharey=False,
            height=2.3,
            aspect=1.0,
            col_order=dataset_order,
        )

        # 5. Define a custom mapping function to overlay the bars and add labels
        def plot_overlaid_bars(data, **kwargs):
            ax = plt.gca()

            # Plot the full bar (Total Time) first
            sns.barplot(
                data=data,
                x="num_filters",
                y="total_time_overhead",
                hue="approach",
                hue_order=hue_order,
                order=filter_order,
                palette=palette_colors,
                edgecolor="none",
                ax=ax,
                **{k: v for k, v in kwargs.items() if k != "color"},
            )

            # Plot the inner bar (Profiling Time) over it with a hatching pattern
            sns.barplot(
                data=data,
                x="num_filters",
                y="profiling_time_overhead",
                hue="approach",
                hue_order=hue_order,
                order=filter_order,
                palette=palette_colors,
                hatch="///",
                edgecolor="black",
                ax=ax,
                **{k: v for k, v in kwargs.items() if k != "color"},
            )

            if ax.get_legend() is not None:
                ax.get_legend().remove()  # type: ignore

            # --- NEW: Extract the max y-coordinate for every error bar ---
            error_bar_tops = {}
            for line in ax.lines:
                xs, ys = line.get_data()
                # Check if it's a vertical line
                if len(xs) > 1 and xs[0] == xs[1]:  # type: ignore
                    x_coord = xs[0]  # type: ignore
                    highest_y = max(ys)  # type: ignore
                    # Because there are two bar plots overlapping, we only save the highest error bar point per x-coord
                    if (
                        x_coord not in error_bar_tops
                        or highest_y > error_bar_tops[x_coord]
                    ):
                        error_bar_tops[x_coord] = highest_y

            total_containers = ax.containers[: len(hue_order)]
            max_height_in_plot = 0

            for i, app in enumerate(hue_order):
                bars = total_containers[i]
                for j, f_val in enumerate(filter_order):
                    bar = bars[j]
                    height = bar.get_height()

                    if pd.isna(height) or height <= 0:
                        continue

                    x_center = bar.get_x() + bar.get_width() / 2

                    # --- NEW: Determine the safe Y position for the annotation ---
                    y_pos = height
                    if error_bar_tops:
                        # Match the bar to its closest error bar coordinate
                        closest_x = min(
                            error_bar_tops.keys(), key=lambda k: abs(k - x_center)
                        )
                        # Use a small tolerance check so we don't accidentally match missing error bars to neighboring bars
                        if abs(closest_x - x_center) < 0.05:
                            y_pos = max(height, error_bar_tops[closest_x])

                    max_height_in_plot = max(max_height_in_plot, y_pos)

                    # --- CHECK: Only label if the approach is in the specified list ---
                    if k_label_approaches is not None and app not in k_label_approaches:
                        continue

                    # Find corresponding num_components for this specific bar
                    row = data[
                        (data["approach"] == app) & (data["num_filters"] == f_val)
                    ]
                    if not row.empty:
                        k_val = row["num_components"].values[0]
                        k_val = int(k_val) if k_val % 1 == 0 else k_val

                        # Annotate using y_pos (the top of the error bar) instead of height
                        ax.annotate(
                            f"{k_val}",
                            xy=(x_center, y_pos),
                            xytext=(0, 4),  # 4 points vertical offset
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            rotation=90,
                        )

            # Extend the y-axis limit slightly so the rotated labels don't get cut off
            if max_height_in_plot > 0:
                current_top = ax.get_ylim()[1]
                ax.set_ylim(top=max(current_top, max_height_in_plot * 1.30))

        g.map_dataframe(plot_overlaid_bars)

        g.set_titles("{col_name}")
        g.set_axis_labels("Number of Filters", "Runtime Overhead (s)")

        def compact_y_fmt(x, pos):
            if abs(x) >= 1e6:
                return f"{x * 1e-6:g}M"
            elif abs(x) >= 1e3:
                return f"{x * 1e-3:g}k"
            else:
                return f"{x:g}"

        compact_formatter = FuncFormatter(compact_y_fmt)

        for ax in g.axes.flat:
            ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
            ax.yaxis.set_major_formatter(compact_formatter)
            ax.tick_params(axis="y", pad=4)

        # 6. LEGENDFIX: Construct a custom legend
        legend_handles = []

        for app, color in zip(hue_order, palette_colors):
            legend_handles.append(mpatches.Patch(color=color, label=app))

        legend_handles.append(mpatches.Patch(color="white", label=""))
        legend_handles.append(mpatches.Patch(color="white", label="--- Breakdown ---"))

        legend_handles.append(
            mpatches.Patch(facecolor="gray", edgecolor="none", label="Total Overhead")
        )
        legend_handles.append(
            mpatches.Patch(
                facecolor="gray",
                hatch="///",
                edgecolor="black",
                label="Cardinality Estimation",
            )
        )

        g.fig.legend(
            handles=legend_handles,
            title="Approach",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            frameon=False,
        )

        g.fig.subplots_adjust(wspace=0.35, hspace=0.25)

        # 7. Save the plot
        plot_path = self.output_dir / "plots" / "reorder.pdf"

        try:
            plt.savefig(plot_path, bbox_inches="tight")
        except ValueError:
            plt.savefig(plot_path)

        plt.close()

        sns.reset_orig()

    def print_paper_stats_qerror(self, df):
        print(
            "Mean Latency Specificity model",
            df[df["approach"] == SPECIFICITY_MODEL]["duration_mean"].unique().mean(),
        )
        print(
            "Median Q-Error Ensemble",
            df[df["approach"] == ENSEMBLE]
            .groupby(["dataset", "n_components"])["q_error"]
            .median(),
        )
        print(
            "Ecommerce specificity q-error",
            df[(df["approach"] == SPECIFICITY_MODEL) & (df["dataset"] == "ecommerce")][
                "q_error"
            ].median(),
        )

    def print_paper_stats_reorder(self, df):
        df = (
            df.groupby(["dataset", "approach", "num_filters", "num_components"])
            .mean()
            .reset_index()[
                [
                    "dataset",
                    "approach",
                    "num_filters",
                    "total_time_overhead",
                ]
            ]
        )
        sampling = df[df["approach"] == SAMPLING][
            ["dataset", "num_filters", "total_time_overhead"]
        ]
        not_sampling = df[df["approach"] != SAMPLING]
        merged = not_sampling.merge(
            sampling, on=["dataset", "num_filters"], suffixes=("", "_sampling")
        )
        merged["percent_reduction"] = (
            100
            - (merged["total_time_overhead"] / merged["total_time_overhead_sampling"])
            * 100
        )
        print(
            "Max end-to-end runtime overhead reduction",
            merged["percent_reduction"].max(),
        )

    def plot_reorder_detailed(
        self,
        datasets: Sequence[Dataset],
        num_queries: int,
        metrics: List[str] = [
            "total_time",
            "total_processed_images",
            "execution_time",
            "execution_processed_images",
            "profiling_time",
            "profiling_processed_images",
        ],
    ):
        """Plots the results of the benchmark for the reorder task, per num_filters and metric."""

        for dataset in datasets:
            # Read the benchmark CSV data specific to reorder
            data_path = (
                self.output_dir / "csv" / f"reorder-{dataset.name()}-q{num_queries}.csv"
            )
            benchmark_data = pd.read_csv(data_path)

            # 1. Melt the dataframe to turn metric columns into a single 'metric' column
            id_vars = ["approach", "num_filters", "num_components", "seed"]
            melted_data = benchmark_data.melt(
                id_vars=id_vars,
                value_vars=metrics,
                var_name="metric",
                value_name="value",
            )

            # 2. Group by approach, num_components, seed, and metric, summing the values across queries
            grouped_sum = melted_data.groupby(
                ["approach", "num_filters", "num_components", "seed", "metric"],
                as_index=False,
            ).agg({"value": "sum"})

            # 3. Create the FacetGrid with rows for metrics and columns for num_filters
            g = sns.FacetGrid(
                grouped_sum,  # type: ignore
                row="metric",
                col="num_filters",
                sharey="row",  # Crucial: Allows metrics to have independent Y-axis scales
                height=3,
                aspect=1.5,
            )

            # 4. Plot using native Seaborn percentile intervals ("pi", 90 calculates 5th & 95th percentile)
            g.map_dataframe(
                sns.lineplot,
                x="num_components",
                y="value",
                hue="approach",
                marker="o",
                estimator="median",
                errorbar=("pi", 90),
            )

            # 5. Format axes, grid, and titles
            g.set_titles(row_template="{row_name}", col_template="{col_name} Filters")
            g.set_axis_labels("Number of Components", "")

            # Set specific Y-axis labels for the first column of each row
            for ax, metric in zip(g.axes[:, 0], g.row_names):
                ax.set_ylabel(f"Sum of {metric}")

            for ax in g.axes.flat:
                ax.grid(True, which="both", ls="--", lw=0.5)

            # Place a single legend outside the subplots
            g.add_legend(
                title="Approach", bbox_to_anchor=(1.02, 0.5), loc="center left"
            )
            g.fig.tight_layout()

            # 6. Save the plot
            plot_path = (
                self.output_dir
                / "plots"
                / f"details-{dataset.name()}-q{num_queries}.pdf"
            )
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()

    def run_reorder(
        self,
        dataset: Dataset,
        bucket_sizes: Dict[str, List[int]],
        num_queries: int,
        all_num_filters: List[int],
        qerror_data: pd.DataFrame,
        num_seeds: int,
    ) -> pd.DataFrame:
        embedder = TextEmbedder(10)
        list(embedder.embed_texts(["Starting"]))
        llm_responses = dataset.llm_responses
        all_images = sorted(llm_responses[list(llm_responses.keys())[0]])

        # Precompute GT cardinalities to avoid recomputing in loops
        gt_cardinalities = {
            f: sum(llm_responses[f][x]["keep"] for x in all_images)
            for f in dataset.filters()
        }

        # 1. Generate all queries
        queries = []
        dataset_filters = dataset.filters()
        items = torch.arange(len(dataset_filters))

        for num_filters in tqdm(all_num_filters, desc="Generating Queries", position=0):
            all_combos = torch.combinations(items, r=num_filters)
            random_indices = torch.randperm(
                all_combos.size(0), generator=self.torch_rng
            )[:num_queries]
            sampled_combos = all_combos[random_indices]

            for filter_mask in sampled_combos:
                filters: List[str] = self.np_rng.permutation(
                    [dataset_filters[i] for i in filter_mask]
                ).tolist()  # type: ignore
                query_name = "-".join(filters)
                queries.append(
                    {
                        "query": query_name,
                        "num_filters": num_filters,
                        "filters": filters,
                    }
                )

        queries_df = pd.DataFrame(queries)

        # 2. Explode queries to map each filter individually and track their permutation order
        queries_exploded = queries_df.explode("filters").rename(
            columns={"filters": "filter"}
        )
        queries_exploded["filter_idx"] = queries_exploded.groupby("query").cumcount()

        # 3. Merge with qerror_data to get all estimates efficiently
        merged = pd.merge(queries_exploded, qerror_data, on="filter", how="inner")

        # 4. Filter for valid n_components according to the approach type and valid seeds
        mask_sampling = merged["approach"].isin(["sampling"]) & merged[
            "n_components"
        ].isin(bucket_sizes["sampling"])
        mask_perfect_worst = merged["approach"].isin(["perfect", "worst"])
        mask_other = (~merged["approach"].isin(["sampling", "perfect", "worst"])) & (
            merged["n_components"].isin(
                bucket_sizes["num_kv_caches"] + bucket_sizes["num_embeddings"]
            )
        )
        mask_seeds = merged["seed"] < num_seeds

        merged = merged[(mask_sampling | mask_other | mask_perfect_worst) & mask_seeds]

        # 5. Sort by filter_idx before grouping to reconstruct the lists in the original permuted order
        merged = merged.sort_values(  # type: ignore
            ["query", "approach", "n_components", "seed", "filter_idx"]
        )

        grouped = merged.groupby(
            ["query", "num_filters", "approach", "n_components", "seed"], as_index=False
        ).agg(
            filters=("filter", list),
            estimated_cardinalities=("cardinality_estimate", list),
            profiling_time=("duration", "sum"),
        )

        # 6. Apply ground truth lookups map
        grouped["cardinalities_gt"] = grouped["filters"].apply(  # type: ignore
            lambda fs: [gt_cardinalities[f] for f in fs]
        )

        # 7. Calculate execution time via a fast row-wise apply
        tqdm.pandas(desc="Simulating Execution", position=1, leave=False)

        def simulate(row):
            exec_time, processed_imgs = self.execution_time(
                filters=row["filters"],
                images=all_images,
                llm_responses=llm_responses,
                cardinalities_sample=row["estimated_cardinalities"],
            )
            return pd.Series([exec_time, processed_imgs])

        grouped[["execution_time", "execution_processed_images"]] = (  # type: ignore
            grouped.progress_apply(simulate, axis=1)
        )

        # 8. Final scaling and metric allocations
        grouped["profiling_processed_images"] = (  # type: ignore
            grouped["n_components"]
            * grouped["num_filters"]
            * (grouped["approach"] == "sampling").astype(int)
        )
        grouped["total_time"] = (  # type: ignore
            grouped["profiling_time"] + grouped["execution_time"] * SCALE_FACTOR
        )
        grouped["total_processed_images"] = (  # type: ignore
            grouped["profiling_processed_images"]
            + grouped["execution_processed_images"] * SCALE_FACTOR
        )

        # Rename n_components to num_components to strictly match the requested schema
        grouped = grouped.rename(columns={"n_components": "num_components"})  # type: ignore

        final_cols = [
            "approach",
            "num_filters",
            "num_components",
            "execution_time",
            "execution_processed_images",
            "profiling_time",
            "profiling_processed_images",
            "total_time",
            "total_processed_images",
            "seed",
            "query",
            "estimated_cardinalities",
            "cardinalities_gt",
        ]

        return grouped[final_cols]  # type: ignore

    def execution_time(
        self,
        filters: Sequence[str],
        images: List[str],
        llm_responses: Dict[str, Dict[str, Dict[str, Any]]],
        cardinalities_sample: List[float],
    ):
        filter_order = np.argsort(cardinalities_sample)
        current_images = list(images)
        total_duration = 0.0
        processed_images = 0
        for i in filter_order:
            filter: str = filters[int(i)]
            keeps = [llm_responses[filter][img]["keep"] for img in current_images]
            durations = [
                llm_responses[filter][img]["duration"] for img in current_images
            ]
            current_images = [img for (keep, img) in zip(keeps, current_images) if keep]
            total_duration += sum(durations) / 1e9
            processed_images += len(durations)
        return total_duration, processed_images

    def run_qerror(
        self,
        dataset: Dataset,
        bucket_sizes: Dict[str, List[int]],
        num_seeds: int,
    ):
        embedder = TextEmbedder(10)
        list(embedder.embed_texts(["Starting"]))
        llm_responses = dataset.llm_responses
        image_embeddings = dataset.embeddings.to(self.device)
        all_images = sorted(llm_responses[list(llm_responses.keys())[0]])

        collected_data = []

        for this_filter in dataset.filters():
            cardinality_gt = sum(
                llm_responses[this_filter][x]["keep"] for x in all_images
            )
            if cardinality_gt == 0:
                logger.warning(f"Filter {this_filter} is empty")
                continue
            for s in bucket_sizes["sampling"]:
                for seed in range(num_seeds):
                    self.setup_rng(seed)
                    (cardinality_sample,), duration = self.run_sampling(
                        filters=[this_filter],
                        images=all_images,
                        sample_size=s,
                        llm_responses=llm_responses,
                    )
                    q_error = self.q_error(cardinality_gt, cardinality_sample)
                    collected_data.append(
                        {
                            "approach": "sampling",
                            "duration": duration,
                            "q_error": q_error,
                            "n_components": s,
                            "cardinality_estimate": cardinality_sample,
                            "filter": this_filter,
                            "seed": seed,
                        }
                    )
            collected_data.append(
                {
                    "approach": "perfect",
                    "duration": 0.0,
                    "q_error": 1.0,
                    "n_components": 0,
                    "cardinality_estimate": cardinality_gt,
                    "filter": this_filter,
                    "seed": 0,
                }
            )
            cardinality_worst = len(all_images) - cardinality_gt
            q_error_worst = self.q_error(cardinality_gt, cardinality_worst)
            collected_data.append(
                {
                    "approach": "worst",
                    "duration": 0.0,
                    "q_error": q_error_worst,
                    "n_components": 0,
                    "cardinality_estimate": cardinality_worst,
                    "filter": this_filter,
                    "seed": 0,
                }
            )

        for approach in tqdm(self.approaches, position=0, desc="Approaches"):
            num_buckets = approach.get_bucket_sizes(bucket_sizes)
            for b in tqdm(num_buckets, position=1, desc="Num Buckets", leave=False):
                if b > len(image_embeddings):
                    logger.warning(
                        f"Skipping approach {approach.name()} with num_buckets {b} because it exceeds the number of images."
                    )
                    continue
                this_num_seeds = num_seeds if not approach.is_deterministic() else 1

                for seed in tqdm(
                    range(this_num_seeds), position=2, desc="Seeds", leave=False
                ):
                    self.prepare_estimator(
                        approach=approach,
                        images=dataset.selected_subset,
                        image_embeddings=image_embeddings,
                        num_buckets=b,
                        seed=seed,
                    )
                    self.setup_rng(seed)
                    for this_filter in dataset.filters():
                        cardinality_gt = sum(
                            llm_responses[this_filter][x]["keep"] for x in all_images
                        )
                        (cardinality_estimate,), duration = self.run_estimator(
                            approach=approach,
                            filters=[this_filter],
                            num_buckets=b,
                            embedder=embedder,
                        )
                        q_error = self.q_error(cardinality_gt, cardinality_estimate)
                        if seed == 0:  # Only print for the first seed to avoid clutter
                            print(
                                this_filter,
                                approach.name(),
                                b,
                                "| Cardinality GT / Estimate:",
                                cardinality_gt,
                                cardinality_estimate,
                                "Q-Error:",
                                q_error,
                            )
                        collected_data.append(
                            {
                                "approach": approach.name(),
                                "duration": duration,
                                "q_error": q_error,
                                "n_components": b,
                                "cardinality_estimate": cardinality_estimate,
                                "filter": this_filter,
                                "seed": seed,
                            }
                        )
        df = pd.DataFrame(collected_data)
        return df

    def get_plot_data_q_error(self, datasets: Sequence[Dataset]):
        if len(datasets) == 0:
            return pd.DataFrame()
        all_data = []

        # 1. Process all datasets and collect them into a list
        for dataset in datasets:
            df = pd.read_csv(f"artifacts/csv/qerror-{dataset.name()}.csv")
            mean_duration = (
                df.groupby(["approach", "n_components"])
                .agg(
                    duration_mean=("duration", "mean"),
                )
                .reset_index()
            )
            merged = df.merge(
                mean_duration,
                on=["approach", "n_components"],
            )

            # Add a column to identify which dataset this data belongs to
            merged["dataset"] = dataset.name()
            all_data.append(merged)

        # 2. Combine all processed data into a single DataFrame
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df

    def plot_qerror(
        self,
        combined_df: pd.DataFrame,
        aliases: dict[str, str],
        k_label_approaches: Optional[list[str]] = None,
        dataset_order=["artwork", "wildlife", "ecommerce"],
    ):
        # Map raw approach names to human-readable aliases.
        combined_df = combined_df.copy()
        if aliases:
            combined_df["approach"] = combined_df["approach"].replace(aliases)

        self.print_paper_stats_qerror(combined_df)

        hue_order = [
            SAMPLING,
            SPECIFICITY_MODEL,
            KV_BATCHING,
            ENSEMBLE,
        ]

        sns.set_context("paper", font_scale=1.0)
        sns.set_style("whitegrid")

        col_wrap = 3
        g = sns.relplot(
            data=combined_df,
            x="duration_mean",
            y="q_error",
            hue="approach",
            col="dataset",
            col_wrap=col_wrap,
            kind="line",
            marker="o",
            estimator="median",
            errorbar=("pi", 90),
            err_style="bars",
            err_kws={"elinewidth": 0.7, "capsize": 4, "zorder": 1},
            hue_order=hue_order,  # type: ignore
            height=1.9,
            aspect=1.2,
            linewidth=1.5,
            markersize=6,
            facet_kws={"sharey": True, "sharex": True},
            col_order=dataset_order,
        )

        g.set(xscale="log", yscale="log")
        g.set_axis_labels("Mean Estimation Latency (s)", "Q-Error")

        scalar_formatter = ScalarFormatter()
        scalar_formatter.set_scientific(False)

        for i, ax in enumerate(g.axes.flatten()):
            ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.7)

            ax.xaxis.set_major_formatter(scalar_formatter)
            ax.yaxis.set_major_formatter(scalar_formatter)

            # Force hide y-tick labels for subplots NOT in the first column
            if i % col_wrap != 0:
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")

        g.legend.set_title("Approach")  # type: ignore

        # 4. Add floating text labels for n_components
        label_coords = (
            combined_df.groupby(["dataset", "approach", "n_components"])[
                ["duration_mean", "q_error"]
            ]
            .median()
            .reset_index()
        )

        label_coords = label_coords.dropna(
            subset=["duration_mean", "q_error", "n_components"]
        )
        label_coords = label_coords[
            (label_coords["duration_mean"] > 0) & (label_coords["q_error"] > 0)
        ]

        # Filter by specified approaches if provided
        if k_label_approaches is not None:
            label_coords = label_coords[
                label_coords["approach"].isin(k_label_approaches)
            ]

        # Filter to keep only the min and max k for each dataset/approach
        label_coords = label_coords.sort_values(
            by=["dataset", "approach", "n_components"]
        )

        def filter_first_last(group):
            if len(group) <= 2:
                return group
            return group.iloc[[0, -1]]

        if not label_coords.empty:
            label_coords = label_coords.groupby(
                ["dataset", "approach"], group_keys=False
            ).apply(filter_first_last)

        for dataset, ax in g.axes_dict.items():
            ds_coords = label_coords[label_coords["dataset"] == dataset]

            for _, row in ds_coords.iterrows():
                k_val = (
                    int(row["n_components"])
                    if row["n_components"] % 1 == 0
                    else row["n_components"]
                )

                ax.annotate(
                    f"{k_val}",
                    xy=(row["duration_mean"], row["q_error"]),
                    xytext=offset(row),
                    textcoords="offset points",
                    fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.5),
                )

        g.set_titles("{col_name}")

        # Squeeze the subplots closer together
        g.fig.subplots_adjust(wspace=0.08, hspace=0.25)

        try:
            g.fig.savefig("artifacts/plots/qerror.pdf", bbox_inches="tight")
        except ValueError:
            g.fig.savefig("artifacts/plots/qerror.pdf")

        plt.close()

        sns.reset_orig()

    def run_sampling(
        self,
        filters: Sequence[str],
        images: List[str],
        sample_size: int,
        llm_responses: Dict[str, Dict[str, Any]],
    ):
        sample = self.np_rng.choice(
            images,
            size=sample_size,
            replace=False,
        )
        sample_multiplier = len(images) / sample_size

        duration_all = 0.0
        cardinality_sample_all = []
        for filter in filters:
            cardinality_sample = (
                sum(llm_responses[filter][x]["keep"] for x in sample)
                * sample_multiplier
            )
            duration = sum(llm_responses[filter][x]["duration"] for x in sample) / 1e9
            cardinality_sample_all.append(cardinality_sample)
            duration_all += duration
        return cardinality_sample_all, duration_all

    def prepare_estimator(
        self,
        approach: CardinalityEstimator,
        num_buckets: int,
        images: List[Path],
        image_embeddings: torch.Tensor,
        seed: int,
    ):
        approach.fit(
            images=images,
            image_embeddings=image_embeddings,
            n_components=num_buckets,
            seed=seed,
        )

    def run_estimator(
        self,
        approach: CardinalityEstimator,
        filters: Sequence[str],
        num_buckets: int,
        embedder: TextEmbedder,
    ):
        start = datetime.now()
        query_embeddings = [None] * len(filters)
        if approach.embedding_based():
            query_embeddings = list(embedder.embed_texts(filters, use_tqdm=False))

        estimates_all = []
        for filter_str, query_embedding in zip(filters, query_embeddings):
            cardinality_estimate = approach.estimate(filter_str, query_embedding)
            estimates_all.append(float(cardinality_estimate))

        end = datetime.now()
        duration = (end - start).total_seconds()
        logger.debug(
            f"Estimated Cardinality via {approach.name()} (Num buckets: {num_buckets}):",
            estimates_all,
            "Time",
            duration,
        )
        return estimates_all, duration

    def read_estimator(
        self,
        approach_name: str,
        filters: Sequence[str],
        num_buckets: int,
        qerror_data: pd.DataFrame,
        seed: int,
    ) -> Tuple[List[float], float]:
        this_data = qerror_data[
            (qerror_data["approach"] == approach_name)
            & (qerror_data["filter"].apply(lambda x: x in filters))
            & (qerror_data["n_components"] == num_buckets)
            & (qerror_data["seed"] == seed)
        ]
        duration = this_data["duration"].sum()
        estimates_all: List[float] = [
            float(this_data[this_data["filter"] == f]["cardinality_estimate"].item())
            for f in filters
        ]
        return estimates_all, float(duration.item())

    def q_error(self, label: int, pred_cardinality: float):
        pred_cardinality = float(pred_cardinality)
        if (
            pred_cardinality <= 1
        ):  # Assumption: All filters output at least one tuple -> Estimators should predict at least 1.
            pred_cardinality = 1
        q_error = max((pred_cardinality / label, label / pred_cardinality))
        return q_error
