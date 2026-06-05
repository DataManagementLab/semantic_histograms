from contextlib import ExitStack, contextmanager
from PIL import Image
from typing import Iterator, Sequence
from pathlib import Path
from huggingface_hub.utils.tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
import torch
import re
import numpy as np
from torch import Tensor
from safetensors.torch import save_file
from typing import List
import logging

logger = logging.getLogger(__name__)

# MODEL_ID = "google/siglip-so400m-patch14-384"
# PREPROCESSOR_ID = "google/siglip-so400m-patch14-384"
# EMBED_DIM = 1152
MODEL_ID = "google/siglip2-so400m-patch16-384"
PREPROCESSOR_ID = "google/siglip2-so400m-patch16-384"
EMBED_DIM = 1152


class Embedder:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.device = torch.device("cuda:0")
        self._processor = None
        self._model = None
        self.rng = np.random.default_rng(42)

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(MODEL_ID).to(self.device)
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                PREPROCESSOR_ID, use_fast=True
            )
        return self._processor

    def sanitize_name(self, name: str):
        name = re.sub(r"[^a-zA-Z0-9._-]", "", name.replace(" ", "_"))
        return name.strip("._")

    @property
    def embed_dim(self):
        return EMBED_DIM


class ImageEmbedder(Embedder):
    def run(
        self,
        image_paths: List[Path],
        out_path: Path,
    ):
        if out_path.name.endswith(".safetensors"):
            out_dir = out_path.parent
        else:
            out_dir = out_path
            out_path = out_dir / "embeddings.safetensors"

        out_dir.mkdir(exist_ok=True, parents=True)
        embeds = self.embed_images(image_paths)

        save_file(
            {"embeddings": torch.stack(tuple(embeds))},
            out_path,
        )

    def run_dir(
        self,
        in_path: Path,
        out_path: Path,
        recursive: bool,
        image_paths_filename: str,
        sample=0,
    ):
        if out_path.name.endswith(".safetensors"):
            out_dir = out_path.parent
        else:
            out_dir = out_path
            out_path = out_dir / "embeddings.safetensors"

        out_dir.mkdir(exist_ok=True, parents=True)
        paths = self.get_paths(in_path, recursive, sample)
        embeds = self.embed_images(paths)

        save_file(
            {"embeddings": torch.stack(tuple(embeds))},
            out_path,
        )
        with open(out_dir / image_paths_filename, "w") as f:
            for p in paths:
                print(str(p.absolute()), file=f)

    def get_paths(self, in_path: Path, recursive: bool, sample: int) -> Sequence[Path]:
        in_paths: Sequence[Path] = [in_path]
        if in_path.is_file():
            with open(in_path) as f:
                in_paths = [Path(line.strip()) for line in f.readlines()]
        paths = []
        for path in in_paths:
            if path.is_file():
                paths.append(path)
            elif recursive:
                for p in tqdm(
                    path.glob("**"),
                    total=1282168,
                    desc="recursive search for images",
                ):
                    if p.is_file():
                        paths.append(p)
            else:
                for p in tqdm(
                    path.glob("*"),
                    total=1282168,
                    desc="search for images",
                ):
                    if p.is_file():
                        paths.append(p)
        if sample and len(paths) > sample:
            indexes = self.rng.choice(len(paths), size=sample, replace=False)
            paths = [paths[i] for i in indexes]
        return paths

    def embed_images(self, paths: Sequence[Path]) -> Iterator[Tensor]:
        for i in tqdm(
            range(0, len(paths), self.batch_size), desc="Computing Embeddings"
        ):
            batch = paths[i : i + self.batch_size]
            embeds = self.embed_images_batch(batch)
            yield from embeds

    def embed_images_batch(self, paths: Sequence[Path]):
        with self.open_images(paths) as images:
            inputs = self.processor(
                text="",
                images=[img for img in images if img is not None],
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                vision_outputs = self.model.get_image_features(
                    pixel_values=inputs.pixel_values
                ).pooler_output

            image_embeds = F.normalize(vision_outputs, p=2, dim=-1)
            assert image_embeds.shape[1] == self.embed_dim, (
                f"Dimension does not match {image_embeds.shape[1]} != {self.embed_dim}"
            )
            result = (
                torch.rand(
                    (len(images), *image_embeds.shape[1:]),
                    dtype=image_embeds.dtype,
                    device=image_embeds.device,
                )
                * 0.00001  # avoid 0 vector
            )
            mask = torch.tensor(
                [img is not None for img in images], device=result.device
            )
            result[mask] = image_embeds
            return result

    @contextmanager
    def open_images(self, paths: Sequence[Path]):
        with ExitStack() as estack:
            images = []
            for path in paths:
                try:
                    images.append(estack.enter_context(Image.open(path).convert("RGB")))
                except Exception:
                    images.append(None)
                    logger.warning(f"Could not open {path}", exc_info=True)
            yield images


class TextEmbedder(Embedder):
    def embed_texts(self, texts: Sequence[str], use_tqdm=True) -> Iterator[Tensor]:
        iterator = range(0, len(texts), self.batch_size)
        if use_tqdm:
            iterator = tqdm(iterator, desc="Computing Embeddings")
        for i in iterator:
            batch = texts[i : i + self.batch_size]
            embeds = self.embed_texts_batch(batch)
            yield from embeds

    def remove_brackets(self, text_list: List[str]):
        # The regex pattern matches:
        # \[.*?\] for square brackets
        # \(.*?\) for parentheses
        # \{.*?\} for curly braces
        # The '?' makes the matching non-greedy so it stops at the first closing bracket
        pattern = r"\[.*?\]|\(.*?\)|\{.*?\}"

        cleaned_list = []
        for text in text_list:
            # Substitute the matched patterns with an empty string
            cleaned_text = re.sub(pattern, "", text)

            # Optional: Clean up any double spaces left behind and strip trailing whitespaces
            cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            cleaned_list.append(cleaned_text)

        return cleaned_list

    def embed_texts_batch(self, texts: Sequence[str]):
        # prompt = "A photo of a "
        prompt = ""
        texts = self.remove_brackets(list(texts))
        texts = [f"{prompt}{t}" for t in texts]
        inputs = self.processor(
            text=texts, padding="max_length", return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_outputs = self.model.get_text_features(**inputs).pooler_output

        text_embeds = F.normalize(text_outputs, p=2, dim=-1)
        assert text_embeds.shape[1] == self.embed_dim, (
            f"Dimension does not match {text_embeds.shape[1]} != {self.embed_dim}"
        )
        return text_embeds
