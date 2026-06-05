from typing import Dict, List, Optional, Sequence
import numpy as np
import gc
import copy
import asyncio
import torch
import requests
import logging
from pathlib import Path
from sklearn.cluster import KMeans
from transformers import AutoProcessor, DynamicCache, pipeline
from PIL import Image
from kvpress import KeyRerotationPress, ExpectedAttentionPress

from torch._prims_common import Tensor
from mmce.estimators.base_cardinality_estimator import CardinalityEstimator
from mmce.estimators.base_threshold_estimator import ThresholdEstimator


PORT_KV_VISION = {
    "llava-hf/llava-next-72b-hf": 5008,
    "llava-hf/llama3-llava-next-8b-hf": 5009,
}
PRESS = {
    "expected_attention": lambda compression_ratio: ExpectedAttentionPress(
        compression_ratio=compression_ratio
    ),
}
IMAGE_MAX_PIXELS = 14000 * 14000
logger = logging.getLogger(__name__)


def _iter_cache_layers(cache):
    """Yield (key_tensor, value_tensor) per layer for old and new DynamicCache."""
    for layer in cache.layers:
        yield layer.keys, layer.values


class KVEstimator(CardinalityEstimator):
    def __init__(self, compression_ratio=0.9):
        self.device = torch.device("cuda:0")
        self.model_name = "llava-hf/llama3-llava-next-8b-hf"
        self.gen = torch.Generator()
        self.gen.manual_seed(42)
        self.kv_cache_dir = Path("artifacts/kv_vision_cache/kv-image-qa-cache")
        self.compression_ratio = compression_ratio

    def setup_vision_model(self, n_components: int):
        self.kv_vision_model = KvVisionModel(
            model_id=self.model_name,
            compression_ratio=self.compression_ratio,
        )

    def fit(
        self,
        images: List[Path],
        image_embeddings: Tensor,
        n_components: int,
        seed: int,
    ):
        self.image_paths = images
        self.dataset_size = len(images)
        self.setup_vision_model(n_components)
        self.kv_vision_model.setup()
        asyncio.run(self._prepare_kv_vision_model(images))
        self.n_components = n_components

    async def _prepare_kv_vision_model(self, images: List[Path]):
        return await self.kv_vision_model.prepare(
            image_paths=images,
            cache_dir=self.kv_cache_dir,
        )

    def estimate_selectivity(
        self, predicate: str, idx: Optional[torch.Tensor] = None
    ) -> float:
        if idx is None:
            # first, randomly sample n_components images from the dataset
            idx = torch.multinomial(
                torch.ones(self.dataset_size),
                num_samples=self.n_components,
                generator=self.gen,
                replacement=False,
            )
        image_paths = [self.image_paths[i] for i in idx]

        # then, for each image, ask the kv_vision_model if the predicate is depicted in the image
        question = f"Is {predicate} depicted?"
        result = asyncio.run(
            self.kv_vision_model._invoke(
                question=question,
                image_paths=image_paths,
                cache_dir=self.kv_cache_dir,
                boolean_question=True,
            )
        )
        selectivity = sum(1 for r in result if r.strip().lower() == "1") / len(result)
        return selectivity

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[Tensor],
    ) -> float:
        selectivity = self.estimate_selectivity(predicate)
        cardinality = selectivity * self.dataset_size
        return cardinality

    def embedding_based(self) -> bool:
        return False

    def is_deterministic(self) -> bool:
        return True

    def get_bucket_sizes(self, bucket_sizes: Dict[str, List[int]]) -> List[int]:
        return bucket_sizes["num_embeddings"]


class PreLoadedKV(ThresholdEstimator):
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model_name = "llava-hf/llama3-llava-next-8b-hf"
        self.gen = torch.Generator()
        self.gen.manual_seed(42)
        self.kv_cache_dir = Path("artifacts/kv_vision_cache/kv-image-qa-cache")

        self.init_model()
        self.init_press([0.9, 0.8, 0.6])

    def init_model(self):
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        Image.MAX_IMAGE_PIXELS = 250000000

        # Initialize the pipeline
        args = [{"attn_implementation": "flash_attention_2"}, {}]
        self.pipe = None
        for x in args:
            try:
                self.pipe = pipeline(
                    "kv-press-text-generation",  # type: ignore
                    model=self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    model_kwargs=x,  # type: ignore
                )
                break
            except Exception as e:
                logger.warning(
                    f"Error initializing model with args {x}: {str(e)}", exc_info=True
                )
        assert self.pipe is not None, "Failed to initialize the model pipeline."
        self.pipe.model.eval()
        print(
            f"Model {self.model_name} loaded on {next(self.pipe.model.parameters()).device}",
        )

    def init_press(self, compression_ratios, press_name="expected_attention"):
        # Set up compression press
        if press_name not in PRESS:
            raise ValueError(
                f"Unknown press_name '{press_name}'. Available options: {list(PRESS.keys())}"
            )
        self.presses = {
            cr: KeyRerotationPress(PRESS[press_name](compression_ratio=cr))
            for cr in compression_ratios
        }

        print(
            f"Using press {press_name} with compression_ratios={compression_ratios}",
        )

    def select_buckets(
        self,
        images: List[Path],
        image_embeddings: Tensor,
        n_components: int,
        seed: int,
    ) -> List[Path]:
        path_exists_mask = torch.tensor(
            [
                Path(
                    f"{self.save_dir}/cache_entry_{self.hash_path(str(p))}.pt"
                ).exists()
                for p in images
            ]
        )
        images_exist = [p for p, exists in zip(images, path_exists_mask) if exists]
        image_embeddings_exist = image_embeddings[path_exists_mask]
        # cluster the image embeddings into n_components clusters, to get a diverse sample to pass to the kv_vision_model
        # to get a diverse sample we use the k-means++ algorithm to initialize the cluster centers, and then run k-means for a few iterations
        kmeans = KMeans(
            n_clusters=n_components, init="k-means++", max_iter=10, random_state=seed
        )
        kmeans.fit(image_embeddings_exist.cpu().numpy())
        cluster_centers = torch.from_numpy(kmeans.cluster_centers_).to(self.device)
        closest_indices = torch.cdist(cluster_centers, image_embeddings_exist).argmin(
            dim=1
        )
        paths = [images_exist[i] for i in closest_indices]
        return paths

    def hash_path(self, path: str) -> str:
        """Generate a sha256hash for a given path."""
        import hashlib

        return hashlib.sha256(path.encode()).hexdigest()

    def to_compression_tag(self, compression_ratio: float) -> str:
        """Convert compression ratio to a string tag for directory naming."""
        return (
            str(compression_ratio).replace(".", "_")
            if compression_ratio != 0.0
            else "0"
        )

    @property
    def save_dir(self) -> str:
        save_dir = f"{self.kv_cache_dir}/{self.model_name}/comp{self.to_compression_tag(self.compression_ratio)}"
        return save_dir

    def clear(self):
        self.padded_cache = None
        self.prefix_input_ids = None
        self.prefix_attention_masks = None

        gc.collect()
        torch.cuda.empty_cache()

    def fit(
        self,
        images: List[Path],
        image_embeddings: Tensor,
        n_components: int,
        seed: int,
    ):
        self.clear()
        self.compression_ratio = {
            128: 0.9,
            64: 0.8,
            32: 0.6,
        }[n_components]
        self.image_embeddings = image_embeddings

        self.selected_image_paths = self.select_buckets(
            images, image_embeddings, n_components, seed=seed
        )

        cache_files = []
        for i, image_path in enumerate(self.selected_image_paths):
            cache_name = self.hash_path(str(image_path))
            cache_filename = f"{self.save_dir}/cache_entry_{cache_name}.pt"
            cache_files.append(cache_filename)

        caches = []
        self.context_lengths = []

        for i in range(len(cache_files)):
            # Load the pre-generated cache to CPU first to avoid GPU memory accumulation
            print(cache_files[i])
            cache = torch.load(cache_files[i], map_location="cpu", weights_only=False)
            caches.append(cache)
            self.context_lengths.append(cache.layers[0].keys.shape[2])

        max_context_len = max(self.context_lengths)
        # Prepare inputs for each image in the batch
        collected_input_ids = []
        collected_masks = []
        for i, ctx_len in enumerate(self.context_lengths):
            padded_context_ids = torch.full(
                (1, ctx_len),
                self.pipe.tokenizer.pad_token_id + 1,  # type: ignore
                device=self.device,
            )
            pad_len = max_context_len - ctx_len
            padding_ids = torch.full(
                (1, pad_len),
                self.pipe.tokenizer.pad_token_id,  # type: ignore
                device=self.device,
            )
            padded_context = torch.cat([padding_ids, padded_context_ids], dim=1)

            # Create attention masks
            context_mask = torch.ones_like(padded_context_ids)
            padding_mask = torch.zeros_like(padding_ids)
            attention_mask = torch.cat([padding_mask, context_mask], dim=1)

            collected_input_ids.append(padded_context)
            collected_masks.append(attention_mask)

        # Batch the inputs
        self.prefix_input_ids = torch.cat(collected_input_ids, dim=0)
        self.prefix_attention_masks = torch.cat(collected_masks, dim=0)

        # Batch the caches
        batched_cache = []
        for layers in zip(*[_iter_cache_layers(c) for c in caches]):
            max_seq_len = max(k.shape[2] for k, _ in layers)
            keys_padded = []
            values_padded = []

            for k, v in layers:
                seq_len = k.shape[2]
                pad_len = max_seq_len - seq_len
                k_padded = (
                    torch.nn.functional.pad(k, (0, 0, pad_len, 0)) if pad_len > 0 else k
                )
                v_padded = (
                    torch.nn.functional.pad(v, (0, 0, pad_len, 0)) if pad_len > 0 else v
                )
                k_padded = k_padded.contiguous()
                v_padded = v_padded.contiguous()
                keys_padded.append(k_padded)
                values_padded.append(v_padded)

            keys_cat = torch.cat(keys_padded, dim=0)
            values_cat = torch.cat(values_padded, dim=0)
            batched_cache.append((keys_cat, values_cat))

        self.padded_cache = DynamicCache()
        for layer_idx, (keys, values) in enumerate(batched_cache):
            self.padded_cache.update(keys, values, layer_idx)

        # Move inputs to the device of the embedding layer (first layer)
        first_device = next(self.pipe.model.parameters()).device  # type: ignore
        self.prefix_input_ids = self.prefix_input_ids.to(first_device)  # type: ignore
        self.prefix_attention_masks = self.prefix_attention_masks.to(first_device)  # type: ignore

        # Move each cache layer to the same device as the corresponding model layer
        llm_layers = self.pipe.model.model.language_model.layers  # type: ignore
        for layer_idx, layer in enumerate(self.padded_cache.layers):
            layer_device = llm_layers[layer_idx].self_attn.q_proj.weight.device
            layer.keys = layer.keys.to(layer_device)  # type: ignore
            layer.values = layer.values.to(layer_device)  # type: ignore

        # pre-compute question suffix
        dummy = "#" * 100
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": dummy},
                ],
            }
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        _, self.question_suffix = prompt.split(dummy)

        # precompute KV-cache for prompt prefix
        prompt_prefix = "Answer the following question based on the image with '1' or '0'. Do not add any other comments. "

        prefix_ids = self.pipe.tokenizer.encode(  # type: ignore
            prompt_prefix, return_tensors="pt", add_special_tokens=False
        ).to(self.device)  # type: ignore

        prefix_mask = torch.ones_like(prefix_ids)

        # Expand to match the batch size
        prefix_ids = prefix_ids.expand(self.prefix_input_ids.shape[0], -1)
        prefix_mask = prefix_mask.expand(self.prefix_attention_masks.shape[0], -1)

        # The attention mask needs to cover the image cache + the new prefix
        full_prefix_mask = torch.cat([self.prefix_attention_masks, prefix_mask], dim=1)

        # Forward pass to inject the text prefix into the KV cache
        with torch.no_grad():
            outputs = self.pipe.model(  # type: ignore
                input_ids=prefix_ids,  # Pass ONLY the new text tokens
                attention_mask=full_prefix_mask,
                past_key_values=self.padded_cache,
                use_cache=True,
                return_dict=True,
            )

        # Update the stored states so get_responses() starts from here
        self.padded_cache = outputs.past_key_values
        self.prefix_input_ids = torch.cat([self.prefix_input_ids, prefix_ids], dim=1)
        self.prefix_attention_masks = full_prefix_mask

    def estimate(
        self,
        predicate: str,
        predicate_embedding: Optional[Tensor],
    ) -> float:
        assert predicate_embedding is not None, (
            "PreLoadedKV requires predicate_embedding to be not None."
        )
        responses = self.get_responses(predicate)
        selectivity_sample = sum(
            1 for r in responses if r.strip().lower() == "1"
        ) / len(responses)
        similarities = self.image_embeddings @ predicate_embedding.unsqueeze(1)
        threshold = torch.quantile(similarities, 1 - selectivity_sample)
        return threshold.item()

    def get_responses(self, predicate: str) -> List[str]:
        # Prepare inputs for each image in the batch
        answer_prefix = "Answer: "
        question = f"Is {predicate} depicted?"

        # prompt_prefix is removed here, as its KV representations are already in self.padded_cache
        question_text = question + self.question_suffix + answer_prefix
        question_ids = self.pipe.tokenizer.encode(  # type: ignore
            question_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)  # type: ignore
        question_mask = torch.ones_like(question_ids)

        question_ids = question_ids.expand(self.prefix_input_ids.shape[0], -1)  # type: ignore
        question_mask = question_mask.expand(self.prefix_attention_masks.shape[0], -1)  # type: ignore

        full_inputs = torch.cat([self.prefix_input_ids, question_ids], dim=1)  # type: ignore
        full_attention_mask = torch.cat(
            [self.prefix_attention_masks, question_mask],  # type: ignore
            dim=1,
        )  # type: ignore

        with torch.no_grad():
            generated = self.pipe.model.generate(  # type: ignore
                input_ids=full_inputs,
                attention_mask=full_attention_mask,
                past_key_values=copy.deepcopy(self.padded_cache),
                pad_token_id=self.pipe.tokenizer.eos_token_id,  # type: ignore
                do_sample=False,
                max_new_tokens=1,
            )

        decoded = self.pipe.tokenizer.batch_decode(  # type: ignore
            generated[:, full_inputs.shape[1] :],  # type: ignore
            skip_special_tokens=True,
        )
        torch.cuda.empty_cache()
        # print("Decoded responses:", decoded)

        return decoded

    def get_responses_naive(self, predicate: str) -> List[str]:

        # Ensure that fit() has been called and the paths were saved
        if not hasattr(self, "selected_image_paths") or not self.selected_image_paths:
            raise RuntimeError(
                "No images found. Please run fit() first and ensure `self.selected_image_paths` is stored."
            )

        # Reconstruct the exact prompt text used in your pre-computed pipeline
        prompt_prefix = "Answer the following question based on the image with '1' or '0'. Do not add any other comments. "
        question = f"Is {predicate} depicted?"
        prompt_text = prompt_prefix + question

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Append "Answer: " just like in the cached version's string concatenation
        prompt += "Answer: "

        responses = []

        # Iterate sequentially to bypass dynamic patching batching constraints in LLaVA-NeXT
        for image_path in self.selected_image_paths:
            try:
                # Load the raw image
                image = self.deserialize_image(image_path)
            except Exception as e:
                print(f"Error loading {image_path}: {e}")
                responses.append("0")  # Safe fallback if file is corrupted
                continue

            # Process text and image natively through the processor
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # Move inputs to device and cast vision tensors to the pipeline's dtype
            inputs = {
                k: v.to(self.device, dtype=torch.bfloat16)
                if torch.is_floating_point(v)
                else v.to(self.device)
                for k, v in inputs.items()
            }

            # Standard generation without pre-loaded past_key_values
            with torch.no_grad():
                generated_ids = self.pipe.model.generate(  # type: ignore
                    **inputs,
                    pad_token_id=self.pipe.tokenizer.eos_token_id,  # type: ignore
                    do_sample=False,
                    max_new_tokens=1,
                )

            # Slice the generated output to isolate just the new tokens
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0, input_len:]

            decoded = self.pipe.tokenizer.decode(  # type: ignore
                new_tokens, skip_special_tokens=True
            ).strip()  # type: ignore
            responses.append(decoded)

            # Clean up memory per iteration
            del inputs
            del generated_ids
            torch.cuda.empty_cache()

        return responses

    def deserialize_image(self, path):
        """Loads an image and downsizes it if it's too large."""
        try:
            # Handle file paths
            image = Image.open(path)
            image.load()  # Force load to catch truncation errors early
            image = image.convert("RGB")

            # Downsize image if too large
            if int(np.prod(image.size)) > IMAGE_MAX_PIXELS:
                ratio = np.sqrt(IMAGE_MAX_PIXELS / np.prod(image.size))
                image.thumbnail(
                    (int(image.size[0] * ratio), int(image.size[1] * ratio))
                )

            return image
        except Exception as e:
            logger.error(f"Error deserializing image {path}: {str(e)}")
            raise

    def embedding_based(self) -> bool:
        return True

    def is_deterministic(self) -> bool:
        return False

    def get_bucket_sizes(self, bucket_sizes: Dict[str, List[int]]) -> List[int]:
        return bucket_sizes["num_kv_caches"]


class KvVisionModel:
    def __init__(self, model_id: str, compression_ratio: float):
        self.compression_ratio = compression_ratio
        self.model_id = model_id

    def setup(
        self,
    ):
        result = requests.get(
            f"http://localhost:{PORT_KV_VISION.get(self.model_id)}/status"
        )
        assert result.status_code == 200
        json_response = result.json()
        assert json_response["status"] == "alive"
        assert json_response["model_name"] == self.model_id
        assert self.compression_ratio in json_response["compression_ratios"]
        logger.info(
            __name__,
            f"KV Vision model {self.model_id} with compression ratio {self.compression_ratio} is ready",
        )

    async def prepare(
        self,
        cache_dir: Path,
        image_paths: Sequence[Path],
    ):
        response = requests.post(
            f"http://localhost:{PORT_KV_VISION.get(self.model_id)}/prepare_caches",
            json={
                "column_name": "images",
                "image_paths": [str(p) for p in image_paths],
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir),
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        assert json_response["status"] == "cache_ready"

    async def wind_down(self):
        pass

    async def _invoke(
        self,
        question: str,
        image_paths: List[Path],
        cache_dir: Path,
        boolean_question: bool,
    ):
        print("Invoke KV Vision model with question:", question)
        response = requests.post(
            f"http://localhost:{PORT_KV_VISION.get(self.model_id)}/image_qa",
            json={
                "column_name": "images",
                "image_paths": [str(p) for p in image_paths],
                "question": question,
                "compression_ratio": self.compression_ratio,
                "cache_dir": str(cache_dir),
                "boolean": boolean_question,
            },
        )
        assert response.status_code == 200
        json_response = response.json()
        answers = json_response.get("answers", {})
        # log_odds = json_response.get("log_odds", {})
        result = []
        for image_path in image_paths:
            result_text = answers.get(str(image_path), "Not sure")
            # lo = log_odds.get(str(image_path), 0.0)
            result.append(result_text)
        return result
