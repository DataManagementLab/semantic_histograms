import faiss
import json
from pathlib import Path
import nltk
import torch
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from typing import List, Set, Union
from mmce.embedder import TextEmbedder, EMBED_DIM
from safetensors.torch import load_file, save_file
from nltk.corpus import wordnet as wn
from collections import defaultdict
from tqdm import tqdm
from mmce.models.threshold.model import ThresholdModel
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn
import matplotlib.pyplot as plt
import torch.optim as optim
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


nltk.download("wordnet")


class ThresholdLearner:
    def __init__(
        self,
        dataset_name: str,
        embedding_dir=Path("artifacts/embeddings"),
        model_dir=Path("artifacts/models"),
        responses_dir=Path("artifacts/responses"),
    ):
        self.dataset_name = dataset_name
        self.embedding_dir = embedding_dir
        self.embedder = TextEmbedder(100)
        self.responses_dir = responses_dir
        self.thresholds_path = (
            self.embedding_dir / f"{dataset_name}_thresholds.safetensors"
        )
        threshold_model_path = model_dir / f"{dataset_name}_threshold_model.safetensors"
        self.threshold_model_path = threshold_model_path
        self.threshold_model_path_joblib = threshold_model_path.parent / (
            threshold_model_path.stem + ".joblib"
        )

        self.rng = np.random.default_rng(42)
        self.device = torch.device("cuda:0")

    def get_wordnet_id_from_path(self, image_path: Path):
        wordnet_id = int(image_path.name.split("_")[0][1:])
        return wordnet_id

    def get_training_set(self):
        if self.dataset_name == "wordnet":
            self._get_training_set_wordnet()
        else:
            self._get_training_set_generic()

    def _get_training_set_generic(self):
        index, _ = self.load_index()
        with open(self.responses_dir / f"{self.dataset_name}.json") as f:
            responses = json.load(f)
        predicates = sorted(responses.keys())
        predicate_embeddings = self.embedder.embed_texts(predicates)

        collect_thresholds = []
        collect_embeddings = []

        for predicate, predicate_embedding in zip(predicates, predicate_embeddings):
            selected_images = [
                img for img, info in responses[predicate].items() if info["keep"]
            ]
            num = len(selected_images)
            distances, idx = index.search(
                predicate_embedding.unsqueeze(0).cpu().numpy(), num * 2
            )  # type: ignore
            threshold = float(distances[0][num - 1])
            collect_thresholds.append(threshold)
            collect_embeddings.append(predicate_embedding)
        save_file(
            {
                "embeddings": torch.stack(collect_embeddings),
                "thresholds": torch.tensor(collect_thresholds),
            },
            self.thresholds_path,
        )

    def _get_training_set_wordnet(self):
        index, _ = self.load_index()
        wordnet_ids, paths = self.load_wordnet_ids()
        predicates_map = self.get_predicates(wordnet_ids)
        sorted_predicates = sorted(predicates_map)
        predicate_embeddings = self.embedder.embed_texts(sorted_predicates)
        num_results = self.get_num_results(
            wordnet_ids, sorted_predicates, predicates_map
        )

        collect_thresholds = []
        collect_embeddings = []
        for predicate, embedding, num in zip(
            sorted_predicates, predicate_embeddings, num_results
        ):
            distances, idx = index.search(embedding.unsqueeze(0).cpu().numpy(), num * 2)  # type: ignore
            threshold = float(distances[0][num - 1])
            retrieved_wordnet_ids = [wordnet_ids[int(i)] for i in idx[0][:num]]
            allowed_ids = predicates_map[predicate]
            precision = sum([w in allowed_ids for w in retrieved_wordnet_ids]) / len(
                retrieved_wordnet_ids
            )
            collect_thresholds.append(threshold)
            collect_embeddings.append(embedding)
            print(
                " ",
                predicate,
                "(",
                predicates_map[predicate],
                ")",
                "Thresh:",
                threshold,
                "Prec:",
                precision,
                "Num",
                num,
                " " * 10,
                embedding.shape,
            )
            if precision != 1:
                retrieved_paths = [paths[int(i)] for i in idx[0][:num]]
                first_incorrect = [
                    str(p)
                    for w, p in zip(retrieved_wordnet_ids, retrieved_paths)
                    if w not in allowed_ids
                ][:3]
                print("First few incorrect:\n", ",\n".join(first_incorrect))
        save_file(
            {
                "embeddings": torch.stack(collect_embeddings),
                "thresholds": torch.tensor(collect_thresholds),
            },
            self.thresholds_path,
        )

    def load_index(self):
        embeddings = load_file(self.embedding_dir / f"{self.dataset_name}.safetensors")[
            "embeddings"
        ]
        index = faiss.IndexFlatIP(self.embedder.embed_dim)
        index.add(embeddings)  # type: ignore
        return index, embeddings

    def load_wordnet_ids(self):
        wordnet_ids = []
        paths = []
        with open(self.embedding_dir / "images.txt") as f:
            for line in f:
                path = Path(line.strip())
                wordnet_ids.append(self.get_wordnet_id_from_path(path))
                paths.append(path)
        return wordnet_ids, paths

    def get_predicates(self, wordnet_ids: Union[List[int], Set[int]]):
        wordnet_ids = set(wordnet_ids)
        result = defaultdict(set)
        for wid in wordnet_ids:
            synset = wn.synset_from_pos_and_offset("n", wid)
            assert synset is not None
            collected_parents = set()
            for path in synset.hypernym_paths():
                for parent in path:
                    collected_parents.add(parent)
            for parent in collected_parents:
                result[parent.definition()].add(wid)
                for name in parent.lemma_names():
                    result[name.replace("_", " ")].add(wid)
        return result

    def get_num_results(self, wordnet_ids, sorted_predicates, predicates_map):
        for p in tqdm(sorted_predicates):
            yield sum([w in predicates_map[p] for w in wordnet_ids])

    def get_allowed_ids(self, wordnet_ids, predicates):
        for p in tqdm(predicates):
            yield sum([w in predicates[p] for w in wordnet_ids])

    def visualize(self, model_path: Path, name: str):
        state_dict = load_file(model_path)
        has_corrector_layer = self.dataset_name != "wordnet"
        model = ThresholdModel(EMBED_DIM, has_corrector_layer=has_corrector_layer).to(
            self.device
        )
        model.load_state_dict(state_dict)
        model.eval()

        data = load_file(self.thresholds_path)
        X = data["embeddings"]
        y = data["thresholds"].float().unsqueeze(1)

        dataset_size, embed_dim = X.shape

        # Create Train/Val Split (90/10)
        full_dataset = TensorDataset(X, y)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        gen = torch.Generator()
        gen.manual_seed(42)
        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size], generator=gen
        )
        train_loader = DataLoader(train_ds, batch_size=128)
        val_loader = DataLoader(val_ds, batch_size=128)

        val_preds = []
        val_thresholds = []
        for b_X, b_y in val_loader:
            pred = model(b_X.to(self.device)).view(-1).detach().cpu().numpy()
            val_preds.append(pred)
            val_thresholds.append(b_y.numpy())
        train_preds = []
        train_thresholds = []
        for b_X, b_y in train_loader:
            pred = model(b_X.to(self.device)).view(-1).detach().cpu().numpy()
            train_preds.append(pred)
            train_thresholds.append(b_y.numpy())

        plt.figure(figsize=(12, 7))
        plt.scatter(np.concat(val_preds), np.concat(val_thresholds), color="red")
        plt.scatter(np.concat(train_preds), np.concat(train_thresholds), color="blue")
        plt.title(f"Predicted {name} vs. Threshold", fontsize=14)
        plt.xlabel(f"Predicted {name}", fontsize=12)
        plt.ylabel("Threshold", fontsize=12)
        plt.savefig(
            f"artifacts/plots/{self.dataset_name}_predicted_{name}_vs_threshold.pdf"
        )

    def visualize_joblib(self, model_path: Path, name: str):
        data = load_file(self.thresholds_path)
        model_lgb = joblib.load(model_path)
        X = data["embeddings"].numpy()
        y = data["thresholds"].float().unsqueeze(1).numpy()

        # Create Train/Val Split (90/10)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        y_test_pred = model_lgb.predict(X_test)
        y_train_pred = model_lgb.predict(X_train)

        plt.figure(figsize=(12, 7))
        plt.scatter(y_test_pred, y_test, color="red")
        plt.scatter(y_train_pred, y_train, color="blue")
        plt.title(f"Predicted {name} vs. Threshold", fontsize=14)
        plt.xlabel(f"Predicted {name}", fontsize=12)
        plt.ylabel("Threshold", fontsize=12)
        plt.savefig(f"artifacts/plots/predicted_{name}_vs_threshold.pdf")

    def train_nn(self):
        # 1. Load and freeze model
        if self.dataset_name == "wordnet":
            model = ThresholdModel(EMBED_DIM).to(self.device)
        else:  # load pre-trained wordnet model
            state_dict = load_file(
                Path("artifacts/models/wordnet_threshold_model.safetensors")
            )
            model = ThresholdModel(EMBED_DIM, has_corrector_layer=True).to(self.device)
            model.load_state_dict(state_dict, strict=False)
            for param in model.parameters():
                param.requires_grad = False
            # unfreeze network.10.corrector
            for param in model.network[-1].corrector.parameters():  # type: ignore
                param.requires_grad = True

        # 2. Load and Prepare Data
        data = load_file(self.thresholds_path)
        X = data["embeddings"]
        y = data["thresholds"].float().unsqueeze(1)

        # Create Train/Val Split (90/10)
        full_dataset = TensorDataset(X, y)
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        gen = torch.Generator()
        gen.manual_seed(42)
        train_ds, val_ds = random_split(
            full_dataset, [train_size, val_size], generator=gen
        )

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)

        # 3. Setup
        criterion = nn.MSELoss()  # Best for regression
        lr = 1e-3 if self.dataset_name == "wordnet" else 1e-3
        weight_decay = 1e-2 if self.dataset_name == "wordnet" else 1e-5
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        # 4. Training Loop
        for epoch in range(500):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            abs_errors = []
            with torch.no_grad():
                for b_X, b_y in val_loader:
                    b_X, b_y = b_X.to(self.device), b_y.to(self.device)
                    p = model(b_X)
                    batch_abs_error = torch.abs(p - b_y)
                    abs_errors.extend(batch_abs_error.cpu().numpy().flatten())

            abs_errors = np.array(abs_errors)
            mae = np.mean(abs_errors)
            p50 = np.percentile(abs_errors, 50)  # Median
            p75 = np.percentile(abs_errors, 75)
            p90 = np.percentile(abs_errors, 90)
            p95 = np.percentile(abs_errors, 95)
            print(
                f"Epoch {epoch + 1} | MSE: {train_loss / len(train_loader):.4f} | MAE: {mae:.4f} | P50: {p50:.4f} | P75: {p75:.4f} | P90: {p90:.4f} | P95: {p95:.4f} "
            )

        # 5. Save Weights
        save_file(model.state_dict(), self.threshold_model_path)

    def train_gb(self):
        data = load_file(self.thresholds_path)
        X = data["embeddings"].numpy()
        y = data["thresholds"].float().unsqueeze(1).numpy()

        # Create Train/Val Split (90/10)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=42
        )

        # Train model
        # model = xgb.XGBRegressor(
        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,  # 6
            num_leaves=31,
            # objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            eval_metric="rmse",
        )

        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = np.sqrt(mean_absolute_error(y_test, y_pred))
        r2 = np.sqrt(r2_score(y_test, y_pred))

        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

        # Save Weights
        joblib.dump(model, self.threshold_model_path_joblib)

    def interactive_test(self):
        model = ThresholdModel.from_pretrained(self.threshold_model_path).to(
            self.device
        )
        model.eval()

        while True:
            word = input("Input word to test > ")
            embedding = list(self.embedder.embed_texts([word]))[0]
            print(model(embedding.unsqueeze(0)))
