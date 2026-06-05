import argparse
from pathlib import Path

from mmce.embedder import ImageEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_path",
        type=Path,
        help="Path of images to embed. Either directory containing images or file that lists image files.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        help="Path to store the embeddings",
        default="artifacts/embeddings",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Whether to recursively look for images in the given directory/directories",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size of Embedder",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Only compute embedding of a sample of given size",
    )
    args = parser.parse_args()

    embedder = ImageEmbedder(args.batch_size)
    embedder.run_dir(
        in_path=args.in_path,
        out_path=args.out_path,
        recursive=args.recursive,
        sample=args.sample,
        image_paths_filename="images.txt",
    )


if __name__ == "__main__":
    main()
