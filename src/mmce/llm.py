from pathlib import Path
from typing import Optional
import ollama
from pydantic import BaseModel, ValidationError
from PIL import Image
import io
import logging
import random

logger = logging.getLogger(__name__)


Image.MAX_IMAGE_PIXELS = None


class FilterAnswer(BaseModel):
    answer: bool  # This forces a True/False (Yes/No) response
    confidence: float


class LLM:
    # def run_with_without_cache(
    #     self, prompts: List[str], image: Path, do_print=False
    # ) -> tuple[List[bool], List[float], List[float]]:
    #     # First make sure the image is cached by running once without modifying it
    #     self.run(
    #         "Do you like the image? (only answer yes/no)",
    #         image,
    #         do_print=False,
    #         force_cache_miss=False,
    #     )
    #     # Now we can measure the time with cache hit
    #     cache_hit_durations = []
    #     answers = []
    #     for prompt in prompts:
    #         answer, duration = self.run(
    #             prompt, image, do_print=do_print, force_cache_miss=False
    #         )
    #         cache_hit_durations.append(duration)
    #         answers.append(answer)

    #     # Now we can measure the time with cache miss by modifying the image slightly
    #     cache_miss_durations = []
    #     for prompt in prompts:
    #         _, duration = self.run(
    #             prompt, image, do_print=do_print, force_cache_miss=True
    #         )
    #         cache_miss_durations.append(duration)
    #     return answers, cache_hit_durations, cache_miss_durations

    def run(
        self, prompt: str, image: Path, do_print=False, force_cache_miss=False
    ) -> tuple[bool, float]:
        try:
            resized_image = self.resize_image(image, modify_image=force_cache_miss)
        except ValueError:
            return False, 0.0
        stream = ollama.chat(
            # model="qwen3.5:9b",
            model="qwen2.5vl:7b",
            format=FilterAnswer.model_json_schema(),
            messages=[
                {
                    "role": "user",
                    "content": "\n".join(
                        (
                            f"Is {prompt} depicted? Response format:",
                            "{",
                            '    "answer": <true or false>',
                            '    "confidence": <float between 0 and 1>',
                            "}",
                        )
                    ),
                    "images": [resized_image],
                }
            ],
            # options={"temperature": 0},  --> 0 leads to infite loops
            stream=True,
        )

        in_thinking = False
        content = ""
        thinking = ""
        duration = 0

        for chunk in stream:
            if chunk.message.thinking:
                if not in_thinking:
                    if do_print:
                        print("Thinking:\n", end="", flush=True)
                    in_thinking = True
                if do_print:
                    print(chunk.message.thinking, end="", flush=True)
                thinking += chunk.message.thinking

            elif chunk.message.content:
                if in_thinking:
                    if do_print:
                        print("\n\nAnswer:\n", end="", flush=True)
                    in_thinking = False
                if do_print:
                    print(chunk.message.content, end="", flush=True)
                content += chunk.message.content

            duration += (chunk.eval_duration or 0) + (chunk.prompt_eval_duration or 0)

        try:
            output = FilterAnswer.model_validate_json(content)
        except ValidationError:
            output = FilterAnswer(answer=False, confidence=0.1)
        if do_print:
            print()
            print("Duration:", duration)
        return output.answer, duration

    def run_arbitrary_prompt(
        self,
        prompt: str,
        image: Optional[Path],
        do_print=False,
        output_format: Optional[type[BaseModel]] = None,
        max_characters: Optional[int] = None,
    ) -> str:
        resized_image = None
        if image is not None:
            try:
                resized_image = self.resize_image(image)
            except ValueError:
                return ""

        stream = ollama.chat(
            # model="qwen3.5:9b",
            model="qwen2.5vl:7b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    **({"images": [resized_image]} if resized_image else {}),
                }
            ],
            format=output_format.model_json_schema() if output_format else None,
            stream=True,
            options={"temperature": 0.7},
        )

        in_thinking = False
        content = ""
        thinking = ""

        for chunk in stream:
            if chunk.message.thinking:
                if not in_thinking:
                    if do_print:
                        print("Thinking:\n", end="", flush=True)
                    in_thinking = True
                if do_print:
                    print(chunk.message.thinking, end="", flush=True)
                thinking += chunk.message.thinking

            elif chunk.message.content:
                if in_thinking:
                    if do_print:
                        print("\n\nAnswer:\n", end="", flush=True)
                    in_thinking = False
                if do_print:
                    print(chunk.message.content, end="", flush=True)
                content += chunk.message.content
            if max_characters and len(content) >= max_characters:
                break
        return content

    def resize_image(self, path: Path, modify_image: bool = False):
        buf = io.BytesIO()
        try:
            img = Image.open(path)
        except Exception:
            logger.error(f"Could not load image {path}", exc_info=True)
            raise ValueError("Could not load image")

        # Resize to a max dimension of 1024 while keeping aspect ratio
        img.thumbnail((1024, 1024))
        img = img.convert("RGB")

        # --- Inject invisible noise to break VLM cache ---
        if modify_image:
            pixels: Image.core.PixelAccess = img.load()  # type: ignore
            x = random.randint(0, img.width - 1)
            y = random.randint(0, img.height - 1)
            r, g, b = pixels[x, y]  # type: ignore
            # Tweak the red channel by 1 (wrapping around if it's 255)
            pixels[x, y] = ((r + 1) % 256, g, b)
        # -------------------------------------------------

        img.save(buf, format="JPEG")
        return buf.getvalue()
