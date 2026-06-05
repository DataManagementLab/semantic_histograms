import argparse

from mmce.benchmark import Benchmark
from mmce.dataset import DATASETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "datasets",
        type=str,
        nargs="+",
        help="Name of dataset",
        choices=DATASETS.keys(),
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        nargs="+",
        help="How many embeddings to to use for the histogram",
        required=True,
    )
    parser.add_argument(
        "--num-kv-caches",
        type=int,
        nargs="+",
        help="How many kv-caches to use for the histogram",
        required=True,
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        help="How many samples to use for the sampling baseline",
        nargs="+",
        required=True,
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="How many queries to execute",
        required=True,
    )
    parser.add_argument(
        "--num-filters",
        type=int,
        nargs="+",
        help="Minimum and maximum number of features",
        required=True,
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        help="Size of the dataset to test",
        default=1000,
    )
    args = parser.parse_args()

    for dataset_name in args.datasets:
        dataset = DATASETS[dataset_name]()
        dataset.setup()
        dataset.select_subset(args.dataset_size)
        dataset.compute_embeddings()
        dataset.compute_llm_responses()

        benchmark = Benchmark()
        benchmark.run(
            dataset=dataset,
            num_embeddings=args.num_embeddings,
            num_kv_caches=args.num_kv_caches,
            sample_sizes=args.sample_sizes,
            num_queries=args.num_queries,
            num_filters=args.num_filters,
        )


if __name__ == "__main__":
    main()
