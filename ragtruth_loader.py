"""RAGTruth dataset loader for TSV profiling.

Loads RAGTruth data (response.jsonl + source_info.jsonl) into the exact format
TSV expects: list of tokenized prompt tensors + numpy label arrays.

Uses teacher-forcing mode: prompt + response are concatenated and tokenized together.

Label convention:
  - TSV: 1 = truthful, 0 = hallucinated
  - RAGTruth: labels=[] means clean, labels=[spans] means hallucinated
  - This loader flips: no spans -> 1 (truthful), has spans -> 0 (hallucinated)
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default quality values to exclude (problematic samples)
DEFAULT_EXCLUDE_QUALITY = {"incorrect_refusal", "truncated"}


class RAGTruthLoader:
    """Load RAGTruth dataset into TSV-compatible format.

    Args:
        data_dir: Path to directory containing response.jsonl and source_info.jsonl
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum token sequence length (truncates longer sequences)
        task_types: List of task types to include (None = all). Options: "QA", "Summary", "Data2txt"
        exclude_quality: Set of quality tags to exclude
        split_filter: Only include samples with this split value (e.g. "train", "test"). None = all.
        model_filter: Only include samples from these source models (partial match). None = all.
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        task_types: Optional[List[str]] = None,
        exclude_quality: Optional[set] = None,
        split_filter: Optional[str] = None,
        model_filter: Optional[List[str]] = None,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task_types = set(task_types) if task_types else None
        self.exclude_quality = exclude_quality if exclude_quality is not None else DEFAULT_EXCLUDE_QUALITY
        self.split_filter = split_filter
        self.model_filter = model_filter

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_source_info(self) -> Dict[str, Dict[str, Any]]:
        """Load source_info.jsonl into a dict keyed by source_id."""
        source_map = {}
        source_file = os.path.join(self.data_dir, "source_info.jsonl")

        with open(source_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                source_id = item.get("source_id")
                if source_id:
                    source_map[source_id] = item

        logger.info(f"Loaded {len(source_map)} source entries from {source_file}")
        return source_map

    def _build_prompt(self, source_info: Dict[str, Any], task_type: str) -> str:
        """Build prompt text from source info, following hallucination-detection conventions."""
        source_data = source_info.get("source_info", {})

        if task_type == "QA":
            if isinstance(source_data, dict):
                question = source_data.get("question", "")
                # Handle both "passages" and "context" field names
                passages = source_data.get("passages", source_data.get("context", ""))
                if isinstance(passages, list):
                    passages = "\n\n".join(str(p) for p in passages)
                if passages:
                    return f"Context:\n{passages}\n\nQuestion: {question}"
                return question
            elif isinstance(source_data, str):
                return source_data

        elif task_type == "Summary":
            if isinstance(source_data, str) and source_data:
                return f"Summarize the following:\n\n{source_data}"
            elif isinstance(source_data, dict):
                # Some summary tasks store text in a "text" or "document" field
                text = source_data.get("text", source_data.get("document", ""))
                if text:
                    return f"Summarize the following:\n\n{text}"
            # Fallback to prompt field
            return source_info.get("prompt", "")

        elif task_type == "Data2txt":
            if isinstance(source_data, dict):
                return f"Describe the following data:\n\n{json.dumps(source_data, ensure_ascii=False, indent=2)}"
            elif isinstance(source_data, str) and source_data:
                return f"Describe the following data:\n\n{source_data}"

        # Generic fallback
        if isinstance(source_data, str) and source_data:
            return source_data
        return source_info.get("prompt", "")

    def _should_include(self, item: Dict[str, Any], task_type: str) -> bool:
        """Check if a response item passes all filters."""
        # Quality filter
        if item.get("quality", "good") in self.exclude_quality:
            return False
        # Task type filter
        if self.task_types and task_type not in self.task_types:
            return False
        # Split filter (RAGTruth has train/test/validation splits)
        if self.split_filter and item.get("split", "") != self.split_filter:
            return False
        # Model filter (partial match against source model name)
        if self.model_filter:
            item_model = item.get("model", "")
            if not any(mf.lower() in item_model.lower() for mf in self.model_filter):
                return False
        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[List[torch.Tensor], np.ndarray, Dict[str, Any]]:
        """Load and tokenize RAGTruth data.

        Returns:
            prompts: List of tokenized tensors, each [1, seq_len] (on CPU)
            labels: numpy array of int, 1=truthful 0=hallucinated (TSV convention)
            stats: Dataset statistics dict
        """
        source_map = self._load_source_info()
        response_file = os.path.join(self.data_dir, "response.jsonl")

        prompts: List[torch.Tensor] = []
        labels: List[int] = []
        seq_lengths: List[int] = []
        task_counts: Dict[str, int] = {}
        label_counts = {0: 0, 1: 0}

        with open(response_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                source_id = item.get("source_id", "")
                source_info = source_map.get(source_id)
                if source_info is None:
                    continue

                task_type = source_info.get("task_type", "QA")

                if not self._should_include(item, task_type):
                    continue

                # Build full text (teacher-forcing: prompt + response)
                prompt_text = self._build_prompt(source_info, task_type)
                response_text = item.get("response", "")
                full_text = prompt_text + "\n" + response_text

                # Tokenize
                tokens = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                ).input_ids  # [1, seq_len]

                prompts.append(tokens)
                seq_lengths.append(tokens.shape[1])

                # Label: TSV convention — 1=truthful, 0=hallucinated
                hallucination_spans = item.get("labels", [])
                label = 0 if hallucination_spans else 1
                labels.append(label)
                label_counts[label] += 1

                # Task stats
                task_counts[task_type] = task_counts.get(task_type, 0) + 1

        labels_arr = np.array(labels, dtype=np.int32)

        stats = {
            "total_samples": len(prompts),
            "task_type_distribution": task_counts,
            "label_distribution": {
                "truthful": label_counts[1],
                "hallucinated": label_counts[0],
            },
            "avg_sequence_length": round(np.mean(seq_lengths), 1) if seq_lengths else 0,
            "max_sequence_length": int(np.max(seq_lengths)) if seq_lengths else 0,
            "min_sequence_length": int(np.min(seq_lengths)) if seq_lengths else 0,
            "max_token_length_setting": self.max_length,
            "filters_applied": {
                "split_filter": self.split_filter,
                "model_filter": self.model_filter,
                "task_types": list(self.task_types) if self.task_types else None,
            },
        }

        logger.info(
            f"Loaded {len(prompts)} samples from RAGTruth "
            f"(truthful={label_counts[1]}, hallucinated={label_counts[0]}, "
            f"avg_len={stats['avg_sequence_length']})"
        )
        return prompts, labels_arr, stats

    def load_with_builtin_splits(
        self,
        exemplar_size: int = 32,
        seed: int = 42,
    ) -> Tuple[
        Tuple[List[torch.Tensor], np.ndarray],
        Tuple[List[torch.Tensor], np.ndarray],
        Tuple[List[torch.Tensor], np.ndarray],
        Dict[str, Any],
    ]:
        """Load RAGTruth using its built-in train/test splits.

        Train split is further divided into wild (train) pool and exemplar set.

        Returns:
            (test_prompts, test_labels),
            (train_prompts, train_labels),
            (exemplar_prompts, exemplar_labels),
            stats
        """
        # Save original filter and load each split separately
        orig_split = self.split_filter

        self.split_filter = "test"
        test_prompts, test_labels, test_stats = self.load()

        self.split_filter = "train"
        train_prompts, train_labels, train_stats = self.load()

        self.split_filter = orig_split

        # Sample exemplars from train set (balanced by class)
        rng = np.random.RandomState(seed)
        n_train = len(train_prompts)
        exemplar_size = min(exemplar_size, n_train)

        perm = rng.permutation(n_train)
        exemplar_idx = set(perm[:exemplar_size].tolist())
        wild_idx = set(range(n_train)) - exemplar_idx

        exemplar_prompts = [train_prompts[i] for i in sorted(exemplar_idx)]
        exemplar_labels = train_labels[np.array(sorted(exemplar_idx))]
        wild_prompts = [train_prompts[i] for i in sorted(wild_idx)]
        wild_labels = train_labels[np.array(sorted(wild_idx))]

        stats = {
            "total_samples": len(test_prompts) + len(train_prompts),
            "test_samples": len(test_prompts),
            "train_samples": len(wild_prompts),
            "exemplar_samples": len(exemplar_prompts),
            "task_type_distribution": {
                k: train_stats["task_type_distribution"].get(k, 0)
                + test_stats["task_type_distribution"].get(k, 0)
                for k in set(list(train_stats["task_type_distribution"].keys())
                             + list(test_stats["task_type_distribution"].keys()))
            },
            "label_distribution": {
                "truthful": train_stats["label_distribution"]["truthful"]
                + test_stats["label_distribution"]["truthful"],
                "hallucinated": train_stats["label_distribution"]["hallucinated"]
                + test_stats["label_distribution"]["hallucinated"],
            },
            "avg_sequence_length": round(
                (train_stats["avg_sequence_length"] * len(train_prompts)
                 + test_stats["avg_sequence_length"] * len(test_prompts))
                / max(len(train_prompts) + len(test_prompts), 1), 1
            ),
            "max_token_length_setting": self.max_length,
            "split_mode": "builtin",
        }

        logger.info(
            f"Loaded with built-in splits: test={len(test_prompts)}, "
            f"train(wild)={len(wild_prompts)}, exemplar={len(exemplar_prompts)}"
        )

        return (test_prompts, test_labels), (wild_prompts, wild_labels), \
            (exemplar_prompts, exemplar_labels), stats

    def split_data(
        self,
        prompts: List[torch.Tensor],
        labels: np.ndarray,
        exemplar_size: int = 32,
        wild_ratio: float = 0.75,
        seed: int = 42,
    ) -> Tuple[
        Tuple[List[torch.Tensor], np.ndarray],
        Tuple[List[torch.Tensor], np.ndarray],
        Tuple[List[torch.Tensor], np.ndarray],
    ]:
        """Split data into test / train(wild) / exemplar, mirroring TSV's splitting logic.

        Args:
            prompts: List of tokenized tensors
            labels: numpy label array
            exemplar_size: Number of labeled exemplar samples
            wild_ratio: Fraction of data used as wild (train) pool
            seed: Random seed for reproducibility

        Returns:
            (test_prompts, test_labels),
            (train_prompts, train_labels),
            (exemplar_prompts, exemplar_labels)
        """
        rng = np.random.RandomState(seed)
        n = len(prompts)

        index = rng.permutation(n)
        n_wild = int(wild_ratio * n)

        # Wild pool (first wild_ratio fraction)
        wild_indices = set(index[:n_wild].tolist())

        # Exemplar: sample from wild pool, ensuring balanced classes
        wild_list = index[:n_wild]
        exemplar_indices = set(rng.choice(wild_list, size=min(exemplar_size, len(wild_list)), replace=False).tolist())

        # Remaining wild (exclude exemplar and last 100 as buffer, matching original TSV)
        buffer = min(100, n_wild // 5)
        train_indices = set(index[: n_wild - buffer].tolist()) - exemplar_indices

        # Test: everything not in wild pool
        test_indices = set(range(n)) - wild_indices

        def gather(indices_set):
            idx_list = sorted(indices_set)
            p = [prompts[i] for i in idx_list]
            l = labels[np.array(idx_list)]
            return p, l

        test_prompts, test_labels = gather(test_indices)
        train_prompts, train_labels = gather(train_indices)
        exemplar_prompts, exemplar_labels = gather(exemplar_indices)

        logger.info(
            f"Data split: test={len(test_prompts)}, "
            f"train(wild)={len(train_prompts)}, "
            f"exemplar={len(exemplar_prompts)}"
        )

        return (test_prompts, test_labels), (train_prompts, train_labels), (exemplar_prompts, exemplar_labels)
