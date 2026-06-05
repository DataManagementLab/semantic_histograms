import argparse
from tqdm import tqdm
import json
import math
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Prevent PIL from throwing DecompressionBomb errors on large images
Image.MAX_IMAGE_PIXELS = None


def estimate_processing_tokens(
    image_path: Path, max_dim: int = 1024, patch_size: int = 28
):
    """
    Simulates the resizing logic and calculates the exact number of vision tokens.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Warning: Could not load {image_path}: {e}")
        return None

    # Simulate the exact resizing logic from your LLM run method
    img.thumbnail((max_dim, max_dim))
    resized_size = img.size

    # Calculate grid patches
    width_patches = math.ceil(resized_size[0] / patch_size)
    height_patches = math.ceil(resized_size[1] / patch_size)

    return width_patches * height_patches


def find_image_files(directory: Path):
    result = {}
    # If not, look for any file with a matching stem (name without extension)
    for file_path in directory.iterdir():
        if file_path.is_file():
            result[file_path.stem] = file_path
            result[file_path.name] = file_path

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Plot estimated vision tokens against real LLM duration."
    )
    parser.add_argument(
        "-j",
        "--json_path",
        type=str,
        required=True,
        help="Path to the pre-computed JSON durations file.",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        type=str,
        required=True,
        help="Base directory containing the images.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="duration_correlation.png",
        help="Path to save the output plot (default: duration_correlation.png).",
    )
    parser.add_argument(
        "--skip-token-estimation",
        action="store_true",
        help="If true, only plot runtimes",
    )
    args = parser.parse_args()

    json_path = Path(args.json_path)
    image_dir = Path(args.image_dir)

    if not json_path.exists():
        print(f"Error: JSON file not found at {json_path}")
        return
    if not image_dir.exists():
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Load pre-computed data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    find_image_file_dict = find_image_files(image_dir)

    estimated_tokens_list = []
    real_durations_list = []
    prompt_ids = []

    print("Processing images and calculating token estimates...")

    cache = {}
    # Parse the nested dictionary: { prompt: { image_path: { "duration": value } } }
    for prompt_id, (prompt, images) in tqdm(
        enumerate(data.items()), desc="Processing prompts", unit="prompt", position=0
    ):
        for raw_img_name, metrics in tqdm(
            images.items(),
            desc="Processing images",
            unit="image",
            position=1,
            leave=False,
        ):
            duration = metrics.get("duration")
            if duration is None:
                continue

            # Treat %20 literally and auto-detect the file extension
            full_img_path = find_image_file_dict.get(raw_img_name)

            if not full_img_path:
                print(
                    f"Warning: Could not find an image matching '{raw_img_name}' in {image_dir}"
                )
                continue

            real_duration_sec = duration / 1e9
            if real_duration_sec > 1:
                print(
                    f"Warning: Unusually long duration ({real_duration_sec:.2f} seconds for {full_img_path} and prompt {prompt}). Check if this is an outlier or an error in the data."
                )
            if args.skip_token_estimation:
                real_durations_list.append(real_duration_sec)
                prompt_ids.append(prompt_id)
                continue

            # Estimate tokens
            if str(full_img_path) not in cache:
                cache[str(full_img_path)] = estimate_processing_tokens(full_img_path)
            tokens = cache[str(full_img_path)]

            if tokens is not None:
                estimated_tokens_list.append(tokens)
                real_durations_list.append(real_duration_sec)
                prompt_ids.append(prompt_id)

    if not real_durations_list:
        print(
            "Error: No valid data points found to plot. Check your image paths and JSON structure."
        )
        return

    print(
        f"Successfully processed {len(estimated_tokens_list)} image points. Generating plot..."
    )

    plt.figure(figsize=(10, 6))
    if args.skip_token_estimation:
        real_durations_list, prompt_ids = zip(
            *sorted(zip(real_durations_list, prompt_ids))
        )
        scatter = plt.scatter(
            list(range(len(real_durations_list))),
            real_durations_list,
            alpha=0.6,
            c=prompt_ids,
            cmap="tab10",
            edgecolors="k",
        )
        plt.colorbar(scatter, label="Prompt ID")

        plt.title("Processing Duration per Image")
        plt.ylabel("Real Processing Duration (Seconds)")
        plt.grid(True, linestyle="--", alpha=0.7)

    else:
        # different color per prompt id
        scatter = plt.scatter(
            estimated_tokens_list,
            real_durations_list,
            alpha=0.6,
            c=prompt_ids,
            cmap="tab10",
            edgecolors="k",
        )
        plt.colorbar(scatter, label="Prompt ID")

        # Calculate and plot a simple trendline (linear regression)
        if len(estimated_tokens_list) > 1:
            z = np.polyfit(estimated_tokens_list, real_durations_list, 1)
            p = np.poly1d(z)
            plt.plot(
                estimated_tokens_list,
                p(estimated_tokens_list),
                "r--",
                linewidth=2,
                label=f"Trendline (y={z[0]:.4f}x + {z[1]:.2f})",
            )
            plt.legend()

        plt.title("Correlation: Estimated Vision Tokens vs Real Processing Duration")
        plt.xlabel("Estimated Vision Tokens (Count)")
        plt.ylabel("Real Processing Duration (Seconds)")
        plt.grid(True, linestyle="--", alpha=0.7)

    # Save and show
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Plot saved successfully to {args.output}")


if __name__ == "__main__":
    main()
