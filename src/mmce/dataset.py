from collections import defaultdict
from typing import List
import numpy as np
from pathlib import Path
import sys
import json
import inspect
from abc import ABC, abstractmethod
from time import sleep
import pandas as pd
import requests
import logging
from tqdm import tqdm
from safetensors.torch import load_file
from bs4 import BeautifulSoup
import tarfile
import zipfile

from mmce.embedder import ImageEmbedder
from mmce.llm import LLM

logger = logging.getLogger(__name__)

DATASET_PATHS = {
    "artwork": Path("artifacts/datasets/artwork"),
    "ecommerce": Path("artifacts/datasets/ecommerce"),
    "wildlife": Path("artifacts/datasets/wildlife"),
    "cars": Path("artifacts/datasets/cars"),
}

ECOMMERCE_FILE_ID = "1fVV9PLgIMT-e-zxFM5ksdnN8xQlfgnmb"
WILDLIFE_FILE_ID = "1HG6tvXIA0BtpqbZqCR46oNSeY2yZ4Lko"

FILTERS = {
    "artwork": [  # should vary in selectivity
        "Madonna and Child",
        "saints identifiable by their halos",
        "a scene in which death is a dominant theme",
        "a religous scene",
        "a still life",
        "a scene of war",
        "an angel with wings",
        "a crucifixion scene",
        "a single seated figure",
        "a figure holding a book or scroll",
        "an animal as a central element",
        "a landscape with visible mountains",
        "an interior scene with architectural elements",
        "a figure wearing armor",
        "a scene involving water or the sea",
        "a figure playing a musical instrument",
        "symbolic objects such as skulls, hourglasses, or candles",
        "a mythological figure identifiable by attributes",
        "a royal or noble figure wearing a crown",
        "a battle or combat scene",
        "a domestic scene with everyday activities",
        "a figure in prayer",
        "architectural ruins",
        "a nighttime scene",
        "a figure with a visible halo or radiance",
        "a narrative scene from classical mythology",
    ],
    "ecommerce": [  # it is primarly a fashion dataset
        "a product",
        "a piece of clothing",
        "a t-shirt",
        "a t-shirt with a graphic design",
        "an item intended for a male audience",
        "a football or soccer team jersey",
        "a brand logo",
        "a long-sleeved item of clothing",
        "a short-sleeved item of clothing",
        "a sleeveless item of clothing",
        "a piece of outerwear",
        "a piece of formal wear",
        "a piece of casual wear",
        "a piece of athletic wear",
        "a piece of footwear",
        "pants",
        "a skirt",
        "a dress",
    ],
    "wildlife": [  # should vary in selectivity
        "trees in the background",
        "meadows or grasslands",
        "animals",
        "a herd of animals",
        "a mammal",
        "zebra",
        "impala",
        "warthog",
        "monkey",
        "bushbuck",
        "waterbuck",
        "antelope",
        "a predator",
        "a prey animal",
    ],
}


class Dataset(ABC):
    def __init__(self):
        self._selected_subset = []
        self._llm_responses = None
        self._embeddings = None

    def select_subset(self, num_images: int):
        all_images = np.array(sorted(list(DATASET_PATHS[self.name()].iterdir())))
        seed = 42
        rng = np.random.default_rng(seed)
        selected_images = rng.choice(all_images, size=num_images, replace=False)
        self._selected_subset = selected_images.tolist()

    @property
    def selected_subset(self) -> List[Path]:
        if not self._selected_subset:
            raise ValueError(
                "Must call select_subset before accessing selected_subset."
            )
        return self._selected_subset

    @property
    def llm_responses(self):
        if self._llm_responses is None:
            raise ValueError(
                "Must call compute_llm_responses before accessing llm_responses."
            )
        return self._llm_responses

    @property
    def embeddings(self):
        if self._embeddings is None:
            raise ValueError(
                "Must call compute_embeddings before accessing embeddings."
            )
        return self._embeddings

    def compute_llm_responses(self):
        if not self.response_path().exists():
            self._compute_llm_responses()
        with open(self.response_path()) as f:
            llm_responses = json.load(f)
        self._llm_responses = llm_responses

    def compute_embeddings(self):
        if not self.embedding_path().exists():
            self._compute_embeddings()
        embeddings = load_file(self.embedding_path())["embeddings"]
        self._embeddings = embeddings

    def _compute_embeddings(self):
        embedder = ImageEmbedder(100)
        embedder.run(
            image_paths=self._selected_subset,
            out_path=self.embedding_path(),
        )

    @classmethod
    def filters(cls):
        return FILTERS[cls.name()]

    def _compute_llm_responses(self):
        llm = LLM()

        tmp_path = self.response_path().parent / f"{self.response_path().name}.tmp"
        tmp_data = {}
        if tmp_path.exists():
            with open(tmp_path) as f:
                tmp_data = json.load(f)
        tmp_path.parent.mkdir(exist_ok=True, parents=False)

        collected = defaultdict(dict)

        prompts_tqdm = tqdm(
            self.filters(),
            leave=False,
            position=1,
        )
        for predicate in prompts_tqdm:
            prompts_tqdm.set_description(f"Prompts: {predicate}")
            images_tqdm = tqdm(self._selected_subset)
            for image in images_tqdm:
                images_tqdm.set_description(f"Image: {image.stem}")

                if image.stem in tmp_data and predicate in tmp_data[image.stem]:
                    collected[image.stem][predicate] = tmp_data[image.stem][predicate]
                else:
                    response, duration = llm.run(predicate, image)
                    collected[predicate][image.stem] = {
                        "keep": response,
                        "duration": duration,
                    }

                with open(tmp_path, "w") as f:
                    json.dump(collected, f)
        tmp_path.rename(self.response_path())

    @classmethod
    def name(cls):
        return cls.__name__.split("Dataset")[0].lower()

    @classmethod
    def path(cls):
        return DATASET_PATHS[cls.name()]

    @classmethod
    def raw_path(cls):
        path = cls.path()
        raw_path = path.parent.parent / "raw" / path.name
        return raw_path

    @classmethod
    def response_path(cls):
        dataset_path = cls.path()
        return dataset_path.parent.parent / "responses" / f"{cls.name()}.json"

    @classmethod
    def embedding_path(cls):
        dataset_path = cls.path()
        return dataset_path.parent.parent / "embeddings" / f"{cls.name()}.safetensors"

    @abstractmethod
    def setup(self):
        pass

    def download_file_from_google_drive(self, file_id, destination):
        # This is the base URL for downloading Google Drive files
        URL = "https://drive.google.com/uc?export=download"

        # Start a session so we can keep the cookies
        session = requests.Session()

        # Step 1: Make the initial request
        response = session.get(URL, params={"id": file_id}, stream=True)

        # Step 2: Check for the virus scan warning token
        response = self.handle_big_file_warning(session, response)

        # Step 3: Save the file in chunks
        self.save_response_content(response, destination)
        print(f"Download complete: {destination}")

    def handle_big_file_warning(self, session, response):
        # Google Drive's warning cookie starts with 'download_warning'
        soup = BeautifulSoup(response.text, "html.parser")
        form = soup.find("form", id="download-form")

        if form:
            print("Big file warning intercepted. Extracting bypass tokens...")

            # Extract all the hidden inputs (id, export, confirm, uuid)
            params = {
                input_tag["name"]: input_tag["value"]
                for input_tag in form.find_all("input", type="hidden")
            }

            # Get the URL to submit them to
            action_url = form["action"]

            sleep(1)  # Be polite and wait a bit before making the next request
            response = session.get(action_url, params=params, stream=True)
        return response

    def save_response_content(self, response, destination):
        # 32KB chunks are standard, but you can increase this for larger files
        CHUNK_SIZE = 32768
        total_num_chunks = int(response.headers.get("content-length", 0)) // CHUNK_SIZE

        with open(destination, "wb") as f:
            for chunk in tqdm(
                response.iter_content(CHUNK_SIZE),
                desc=f"Downloading to {destination.name}",
                total=total_num_chunks,
                unit="chunk",
            ):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)


class ArtworkDataset(Dataset):
    def setup(self):
        csv_path = Path("data/paintings_medium.csv")
        column_name = "image_url"
        df = pd.read_csv(csv_path)
        image_urls = df[column_name]
        self.download_wikidata(image_urls)

    def download_wikidata(self, image_urls):
        self.path().mkdir(exist_ok=True, parents=True)
        for url in tqdm(image_urls, desc="Download Artwork Dataset"):
            file_name = url.split("/")[-1]
            file_path = self.path() / file_name
            if file_path.exists():
                continue
            r = requests.get(
                url,
                stream=True,
                headers={
                    "User-Agent": "MMCE/0.1 (multi-modal cardinality estimation; mailto:matthias.urban@tu-darmstadt.de)"
                },
            )
            if r.status_code != 200:
                logger.error(
                    __name__,
                    f"Failed to download {url}. Status code: {r.status_code}",
                )
            with open(file_path, "wb") as f:
                try:
                    total_length = int(r.headers.get("content-length"))  # type: ignore
                except TypeError:
                    total_length = 0
                for chunk in tqdm(
                    r.iter_content(chunk_size=1024),
                    total=total_length / 1024,
                    unit="KB",
                    desc=f"Downloading {url}",
                    leave=False,
                    position=1,
                ):
                    if chunk:
                        f.write(chunk)
            logger.debug(__name__, f"Downloaded {url} to {file_name}")
            sleep(1)


class EcommerceDataset(Dataset):
    def setup(self):
        self.raw_path().mkdir(exist_ok=True, parents=True)
        self.path().mkdir(exist_ok=True, parents=True)
        destination = self.raw_path() / "ecomm.tar.gz"
        if not destination.exists():
            self.download_file_from_google_drive(ECOMMERCE_FILE_ID, destination)

        print(f"Extracting {destination} to {self.raw_path()}...")
        if not any(self.raw_path().iterdir()):
            with tarfile.open(destination, "r:gz") as tar:
                tar.extractall(path=self.raw_path())

        # now copy the images from <raw_path>/1/fashion-dataset/images to <path>
        if not any(self.path().iterdir()):
            source = self.raw_path() / "1" / "fashion-dataset" / "images"
            for image in tqdm(list(source.iterdir()), desc="Copying Ecommerce Dataset"):
                destination = self.path() / image.name
                if not destination.exists():
                    destination.symlink_to(image.resolve())


class WildlifeDataset(Dataset):
    def setup(self):
        self.raw_path().mkdir(exist_ok=True, parents=True)
        self.path().mkdir(exist_ok=True, parents=True)
        destination = self.raw_path() / "wildlife.zip"
        if not destination.exists():
            self.download_file_from_google_drive(WILDLIFE_FILE_ID, destination)
        if not any(self.raw_path().iterdir()):
            with zipfile.ZipFile(destination, "r") as zip_ref:
                zip_ref.extractall(self.raw_path())

        # now copy the images from <raw_path>/hypnotu/dsail-porini/versions/2/DSAIL-Porini\ Annotated\ camera\ trap\ images\ of\ wildlife\ species\ from\ a\ conservancy\ in\ Kenya/OpenMV_images to <path>
        if not any(self.path().iterdir()):
            sources = list(
                (
                    self.raw_path()
                    / "hypnotu"
                    / "dsail-porini"
                    / "versions"
                    / "2"
                    / "DSAIL-Porini Annotated camera trap images of wildlife species from a conservancy in Kenya"
                    / "RaspberryPi_images"
                ).iterdir()
            )
            for source in tqdm(sources, desc="Copying Wildlife Dataset", position=0):
                for image in tqdm(
                    list(source.iterdir()),
                    desc=f"Copying from {source.name}",
                    position=1,
                    leave=False,
                ):
                    destination = self.path() / image.name
                    if not destination.exists():
                        destination.symlink_to(image.resolve())


current_module = sys.modules[__name__]
classes = inspect.getmembers(current_module, inspect.isclass)
DATASETS = {
    cls.name(): cls
    for _, cls in classes
    if issubclass(cls, Dataset) and cls is not Dataset and cls.__module__ == __name__
}
