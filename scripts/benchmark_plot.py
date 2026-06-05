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
        "--num-queries",
        type=int,
        help="How many queries to execute",
        required=True,
    )
    args = parser.parse_args()

    datasets = []
    for d in args.datasets:
        dataset = DATASETS[d]()
        dataset.setup()
        dataset.compute_llm_responses()
        dataset.compute_embeddings()
        datasets.append(dataset)

    benchmark = Benchmark()
    benchmark.plot(datasets, args.num_queries)


if __name__ == "__main__":
    main()
