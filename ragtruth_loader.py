"""RAGTruth dataset loader for TSV hallucination detection.

Loads RAGTruth data (response.jsonl + source_info.jsonl) into the exact format
TSV expects: list of tokenized prompt tensors + numpy label arrays.

Uses teacher-forcing mode: source prompt + response are concatenated and tokenized.

Label convention (aligned with TSV):
  - 1 = truthful  (RAGTruth: labels == [])
  - 0 = hallucinated  (RAGTruth: labels contains span annotations)

Prompt format (aligned with TSV paper):
  For all task types, we use the original prompt from source_info.jsonl
  (which already contains context + instruction) concatenated with the response.
  This mirrors TSV's "Q: {question} A:{answer}" teacher-forcing approach.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Quality values to exclude (problematic samples)
DEFAULT_EXCLUDE_QUALITY = {"incorrect_refusal", "truncated"}


class RAGTruthLoader:
    """Load RAGTruth dataset into TSV-compatible format.

    Args:
        data_dir: Path to directory containing response.jsonl and source_info.jsonl
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum token sequence length. None or 0 = no truncation.
        task_types: List of task types to include (None = all). Options: "QA", "Summary", "Data2txt"
        exclude_quality: Set of quality tags to exclude
        model_filter: Only include samples from these source models (partial match). None = all.
        exclude_implicit_true: If True, treat samples where ALL hallucination spans
            are implicit_true as truthful (factually correct but unsupported by context).
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: Optional[int] = None,
        task_types: Optional[List[str]] = None,
        exclude_quality: Optional[set] = None,
        model_filter: Optional[List[str]] = None,
        exclude_implicit_true: bool = False,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length if max_length and max_length > 0 else None
        self.task_types = set(task_types) if task_types else None
        self.exclude_quality = exclude_quality if exclude_quality is not None else DEFAULT_EXCLUDE_QUALITY
        self.model_filter = model_filter
        self.exclude_implicit_true = exclude_implicit_true

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
                if source_id is not None:
                    source_map[str(source_id)] = item

        logger.info(f"Loaded {len(source_map)} source entries from {source_file}")
        return source_map

    def _build_full_text(self, source_info: Dict[str, Any], response_text: str) -> str:
        """Build full tokenization text: prompt + response.

        Uses the `prompt` field from source_info.jsonl, which already contains
        the task instruction + context/question for all three task types.
        This is analogous to TSV's "Q: {question} A:{answer}" format.
        """
        # The prompt field contains the exact instruction sent to the LLM,
        # including context passages, questions, data, etc.
        prompt = source_info.get("prompt", "")

        if not prompt:
            # Fallback: construct from source_info content
            task_type = source_info.get("task_type", "")
            source_data = source_info.get("source_info", "")

            if task_type == "QA" and isinstance(source_data, dict):
                question = source_data.get("question", "")
                passages = source_data.get("passages", "")
                if isinstance(passages, list):
                    passages = "\n\n".join(str(p) for p in passages)
                prompt = f"Q: {question}\nContext: {passages}"

            elif task_type == "Summary":
                text = source_data if isinstance(source_data, str) else str(source_data)
                prompt = f"Summarize: {text}"

            elif task_type == "Data2txt" and isinstance(source_data, dict):
                prompt = f"Describe: {json.dumps(source_data, ensure_ascii=False)}"

            else:
                prompt = str(source_data) if source_data else ""

        # Concatenate prompt + response (teacher-forcing, like TSV)
        full_text = prompt.strip() + "\n" + response_text.strip()
        return full_text

    def _get_label(self, item: Dict[str, Any]) -> int:
        """Determine binary label for a response.

        Returns:
            1 = truthful, 0 = hallucinated (TSV convention)
        """
        labels = item.get("labels", [])

        if not labels:
            return 1  # No hallucination spans -> truthful

        if self.exclude_implicit_true:
            # Filter out spans that are factually correct but unsupported
            real_hallucinations = [
                span for span in labels
                if not span.get("implicit_true", False)
            ]
            if not real_hallucinations:
                return 1  # All spans were implicit_true -> treat as truthful

        return 0  # Has hallucination spans -> hallucinated

    def _should_include(self, item: Dict[str, Any], task_type: str) -> bool:
        """Check if a response item passes all filters."""
        # Quality filter
        if item.get("quality", "good") in self.exclude_quality:
            return False
        # Task type filter
        if self.task_types and task_type not in self.task_types:
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

    def load(self, split_filter: Optional[str] = None) -> Tuple[List[torch.Tensor], np.ndarray, Dict[str, Any]]:
        """Load and tokenize RAGTruth data.

        Args:
            split_filter: Only include samples with this split value ("train" or "test").
                          None = load all splits.

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

                # Split filter
                if split_filter and item.get("split", "") != split_filter:
                    continue

                source_id = str(item.get("source_id", ""))
                source_info = source_map.get(source_id)
                if source_info is None:
                    continue

                task_type = source_info.get("task_type", "QA")

                if not self._should_include(item, task_type):
                    continue

                # Build full text: prompt + response
                response_text = item.get("response", "")
                full_text = self._build_full_text(source_info, response_text)

                # Tokenize (no truncation by default, or use max_length if set)
                tokenize_kwargs = {
                    "return_tensors": "pt",
                }
                if self.max_length is not None:
                    tokenize_kwargs["truncation"] = True
                    tokenize_kwargs["max_length"] = self.max_length

                tokens = self.tokenizer(full_text, **tokenize_kwargs).input_ids  # [1, seq_len]

                prompts.append(tokens)
                seq_lengths.append(tokens.shape[1])

                # Label
                label = self._get_label(item)
                labels.append(label)
                label_counts[label] += 1

                # Task stats
                task_counts[task_type] = task_counts.get(task_type, 0) + 1

        labels_arr = np.array(labels, dtype=np.int32)

        stats = {
            "dataset": "RAGTruth",
            "total_samples": len(prompts),
            "task_type_distribution": task_counts,
            "label_distribution": {
                "truthful": label_counts[1],
                "hallucinated": label_counts[0],
            },
            "avg_sequence_length": round(float(np.mean(seq_lengths)), 1) if seq_lengths else 0,
            "max_sequence_length": int(np.max(seq_lengths)) if seq_lengths else 0,
            "min_sequence_length": int(np.min(seq_lengths)) if seq_lengths else 0,
            "max_token_length_setting": self.max_length or "unlimited",
            "split_filter": split_filter,
        }

        logger.info(
            f"Loaded {len(prompts)} samples from RAGTruth "
            f"(truthful={label_counts[1]}, hallucinated={label_counts[0]}, "
            f"avg_len={stats['avg_sequence_length']}, max_len={stats['max_sequence_length']})"
        )
        return prompts, labels_arr, stats

    def load_splits(
        self,
        exemplar_size: int = 32,
        wild_ratio: float = 0.75,
        seed: int = 42,
    ) -> Tuple[
        Tuple[List[torch.Tensor], np.ndarray],  # test
        Tuple[List[torch.Tensor], np.ndarray],  # train (wild)
        Tuple[List[torch.Tensor], np.ndarray],  # exemplar
        Dict[str, Any],  # stats
    ]:
        """Load RAGTruth using its built-in train/test splits.

        Train split is further divided into wild (unlabeled) pool and exemplar (labeled) set,
        following TSV's data partitioning logic.

        Args:
            exemplar_size: Number of labeled exemplar samples to select from train.
            wild_ratio: Not used here (RAGTruth has its own train/test split).
                        Train split is used entirely as the wild pool.
            seed: Random seed for exemplar selection.

        Returns:
            (test_prompts, test_labels),
            (wild_prompts, wild_labels),
            (exemplar_prompts, exemplar_labels),
            stats
        """
        # Load each split using RAGTruth's built-in "split" field
        test_prompts, test_labels, test_stats = self.load(split_filter="test")
        train_prompts, train_labels, train_stats = self.load(split_filter="train")

        n_train = len(train_prompts)
        if n_train == 0:
            raise ValueError("No training samples loaded. Check data_dir and filters.")

        # Sample exemplars from train set
        rng = np.random.RandomState(seed)
        exemplar_size = min(exemplar_size, n_train)

        perm = rng.permutation(n_train)
        exemplar_idx = sorted(perm[:exemplar_size].tolist())
        wild_idx = sorted(set(range(n_train)) - set(exemplar_idx))

        exemplar_prompts = [train_prompts[i] for i in exemplar_idx]
        exemplar_labels = train_labels[np.array(exemplar_idx)]
        wild_prompts = [train_prompts[i] for i in wild_idx]
        wild_labels = train_labels[np.array(wild_idx)]

        # Merge stats
        stats = {
            "dataset": "RAGTruth",
            "total_samples": len(test_prompts) + len(train_prompts),
            "test_samples": len(test_prompts),
            "train_wild_samples": len(wild_prompts),
            "exemplar_samples": len(exemplar_prompts),
            "task_type_distribution": {
                k: train_stats["task_type_distribution"].get(k, 0)
                + test_stats["task_type_distribution"].get(k, 0)
                for k in set(
                    list(train_stats["task_type_distribution"].keys())
                    + list(test_stats["task_type_distribution"].keys())
                )
            },
            "label_distribution": {
                "truthful": train_stats["label_distribution"]["truthful"]
                + test_stats["label_distribution"]["truthful"],
                "hallucinated": train_stats["label_distribution"]["hallucinated"]
                + test_stats["label_distribution"]["hallucinated"],
            },
            "avg_sequence_length": round(
                (train_stats["avg_sequence_length"] * n_train
                 + test_stats["avg_sequence_length"] * len(test_prompts))
                / max(n_train + len(test_prompts), 1), 1
            ),
            "max_sequence_length": max(
                train_stats.get("max_sequence_length", 0),
                test_stats.get("max_sequence_length", 0),
            ),
            "max_token_length_setting": self.max_length or "unlimited",
            "split_mode": "builtin_train_test",
        }

        logger.info(
            f"RAGTruth splits: test={len(test_prompts)}, "
            f"wild={len(wild_prompts)}, exemplar={len(exemplar_prompts)}"
        )

        return (test_prompts, test_labels), (wild_prompts, wild_labels), \
            (exemplar_prompts, exemplar_labels), stats