"""Profiled TSV training entry point.

Replicates the training logic from tsv_main.py with profiling instrumentation
at phase / epoch / operation granularity. Supports RAGTruth dataset.

Usage:
    # RAGTruth
    CUDA_VISIBLE_DEVICES=0 python run_profiled.py \
        --model_name llama3.1-8B \
        --dataset_name ragtruth \
        --ragtruth_data_dir /path/to/RAGTruth/dataset \
        --batch_size 8 \
        --profile_output_dir ./profiling_results/

    # Original TQA (for validation)
    CUDA_VISIBLE_DEVICES=0 python run_profiled.py \
        --model_name llama3.1-8B \
        --dataset_name tqa \
        --batch_size 128 \
        --profile_output_dir ./profiling_results/
"""

import os
import argparse
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score

# TSV imports (from existing project files)
from train_utils import (
    collate_fn,
    get_last_non_padded_token_rep,
    compute_ot_loss_cos,
    compute_entropy,
    update_centroids_ema,
    update_centroids_ema_hard,
)
from llm_layers import add_tsv_layers
from sinkhorn_knopp import SinkhornKnopp_imb
from tsv_main import test_model, seed_everything, HF_NAMES

# Profiling imports
from profiler import TSVProfiler, clear_gpu_memory
from ragtruth_loader import RAGTruthLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Profiled TSV training")

    # Original TSV args
    parser.add_argument("--model_name", type=str, default="llama3.1-8B")
    parser.add_argument("--model_prefix", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="ragtruth",
                        choices=["ragtruth", "tqa", "triviaqa", "sciq", "nq_open"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cos_temp", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--str_layer", type=int, default=9)
    parser.add_argument("--component", type=str, default="res")
    parser.add_argument("--lam", type=float, default=5)
    parser.add_argument("--init_num_epochs", type=int, default=20)
    parser.add_argument("--aug_num_epochs", type=int, default=20)
    parser.add_argument("--num_exemplars", type=int, default=32)
    parser.add_argument("--num_selected_data", type=int, default=128)
    parser.add_argument("--wild_ratio", type=float, default=0.75)
    parser.add_argument("--most_likely", type=int, default=0)
    parser.add_argument("--thres_gt", type=float, default=0.5)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="AdamW")

    # RAGTruth-specific args
    parser.add_argument("--ragtruth_data_dir", type=str, default=None,
                        help="Path to RAGTruth dataset dir (containing response.jsonl + source_info.jsonl)")
    parser.add_argument("--ragtruth_task_types", type=str, nargs="+", default=None,
                        help="Task types to include: QA Summary Data2txt")
    parser.add_argument("--ragtruth_max_length", type=int, default=2048,
                        help="Max token length for RAGTruth samples")
    parser.add_argument("--ragtruth_split_filter", type=str, default=None,
                        help="Only include samples from this split (train/test/validation)")
    parser.add_argument("--ragtruth_model_filter", type=str, nargs="+", default=None,
                        help="Only include samples from these source models (partial match)")
    parser.add_argument("--ragtruth_use_builtin_splits", action="store_true",
                        help="Use RAGTruth built-in train/test splits instead of random split")

    # Profiling args
    parser.add_argument("--profile_output_dir", type=str, default="./profiling_results/")
    parser.add_argument("--gpu_util_interval", type=float, default=2.0,
                        help="GPU utilization sampling interval in seconds")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading (original datasets, for validation)
# ---------------------------------------------------------------------------

def load_original_dataset_data(args, tokenizer):
    """Load data using the original TSV data pipeline (for tqa/triviaqa/sciq/nq_open).

    Returns (prompts, labels, stats) in the same format as RAGTruthLoader.load().
    """
    from datasets import load_dataset

    if args.dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", "generation")["validation"]
    elif args.dataset_name == "triviaqa":
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()
        def remove_dups(batch):
            if batch["question_id"][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch["question_id"][0])
            return batch
        dataset = dataset.map(remove_dups, batch_size=1, batched=True, load_from_cache_file=False)
    elif args.dataset_name == "sciq":
        dataset = load_dataset("allenai/sciq", split="validation")
    elif args.dataset_name == "nq_open":
        dataset = load_dataset("google-research-datasets/nq_open", split="validation")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    length = len(dataset)

    # Load pre-generated answers and BLEURT scores
    prompts = []
    for i in range(length):
        question = dataset[i]["question"]
        answers = np.load(
            f"./save_for_eval/{args.dataset_name}_hal_det/answers/"
            f"most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
        )
        for anw in answers:
            prompt = tokenizer(
                f"Answer the question concisely. Q: {question} A:" + anw,
                return_tensors="pt",
            ).input_ids.cuda()
            prompts.append(prompt)

    gts = np.load(f"./ml_{args.dataset_name}_bleurt_score.npy")

    if args.dataset_name in ("tqa", "triviaqa"):
        thres = 0.5
    else:
        thres = 0.2
    labels = np.asarray(gts > thres, dtype=np.int32)

    stats = {"total_samples": len(prompts), "dataset": args.dataset_name}
    return prompts, labels, stats, dataset, length


def split_original_data(args, prompts, labels, length):
    """Split original dataset data using pre-saved indices (matching tsv_main.py)."""
    index = np.load(f"data_indices/data_index_{args.dataset_name}.npy")
    exemplar_index = np.load(f"data_indices/exemplar_idx_{args.dataset_name}.npy")
    wild_q_indices = index[: int(args.wild_ratio * length)]
    wild_q_indices1 = wild_q_indices[: len(wild_q_indices) - 100]

    args.num_exemplars = len(exemplar_index)

    test_prompts, train_prompts, exemplar_prompts = [], [], []
    test_labels, train_labels, exemplar_labels = [], [], []

    for i in range(length):
        if i not in wild_q_indices:
            test_labels.extend(labels[i: i + 1])
            test_prompts.extend(prompts[i: i + 1])
        elif i in exemplar_index:
            exemplar_labels.extend(labels[i: i + 1])
            exemplar_prompts.extend(prompts[i: i + 1])
        elif i in wild_q_indices1:
            train_labels.extend(labels[i: i + 1])
            train_prompts.extend(prompts[i: i + 1])

    return (
        (test_prompts, np.asarray(test_labels)),
        (train_prompts, np.asarray(train_labels)),
        (exemplar_prompts, np.asarray(exemplar_labels)),
    )


# ---------------------------------------------------------------------------
# Profiled get_ex_data (split into sub-operations for timing)
# ---------------------------------------------------------------------------

def get_ex_data_profiled(model, prompts, labels, batch_size, centroids, sinkhorn,
                         num_selected_data, cls_dist, args, phase_profiler):
    """Profiled version of get_ex_data — splits into embedding extraction, sinkhorn, selection."""
    all_embeddings = []
    all_labels = []
    num_samples = len(prompts)

    with torch.no_grad():
        with autocast(dtype=torch.float16):
            # Sub-operation: embedding extraction
            with phase_profiler.operation("embedding_extraction"):
                for batch_start in tqdm(range(0, num_samples, batch_size), desc="Embedding extraction"):
                    batch_prompts = prompts[batch_start: batch_start + batch_size]
                    batch_labels = labels[batch_start: batch_start + batch_size]
                    batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)
                    attention_mask = (batch_prompts != 0).half()
                    batch_prompts = batch_prompts.cuda()
                    batch_labels = batch_labels.cuda()
                    attention_mask = attention_mask.to(batch_prompts.device)
                    all_labels.append(batch_labels.cpu().numpy())

                    output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(),
                                   output_hidden_states=True)
                    hidden_states = output.hidden_states
                    hidden_states = torch.stack(hidden_states, dim=0).squeeze()
                    last_layer_hidden_state = hidden_states[-1]
                    last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state, attention_mask.squeeze())
                    all_embeddings.append(last_token_rep)

                all_embeddings = F.normalize(torch.concat(all_embeddings), p=2, dim=-1)

            # Sub-operation: sinkhorn
            with phase_profiler.operation("sinkhorn"):
                pseudo_label = sinkhorn(all_embeddings, centroids)

            # Sub-operation: data selection
            with phase_profiler.operation("data_selection"):
                selected_indices = compute_entropy(
                    all_embeddings, centroids, pseudo_label,
                    num_selected_data, cls_dist, args
                )
                selected_labels_soft = pseudo_label[selected_indices]

    return selected_indices, selected_labels_soft


# ---------------------------------------------------------------------------
# Main profiled training
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    seed_everything(42)

    model_name_or_path = HF_NAMES[args.model_prefix + args.model_name]
    device = torch.device("cuda")

    profiler = TSVProfiler(output_dir=args.profile_output_dir)

    # ===== Phase: Model Loading =====
    with profiler.phase("model_loading", args.gpu_util_interval) as p:
        logger.info(f"Loading model: {model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, low_cpu_mem_usage=True,
            torch_dtype=torch.float16, device_map="auto", token=""
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, token="")
        profiler.set_model_info(model, model_name_or_path)

    # ===== Phase: Data Loading =====
    with profiler.phase("data_loading", args.gpu_util_interval) as p:
        if args.dataset_name == "ragtruth":
            if not args.ragtruth_data_dir:
                raise ValueError("--ragtruth_data_dir is required for ragtruth dataset")
            loader = RAGTruthLoader(
                data_dir=args.ragtruth_data_dir,
                tokenizer=tokenizer,
                max_length=args.ragtruth_max_length,
                task_types=args.ragtruth_task_types,
                split_filter=args.ragtruth_split_filter,
                model_filter=args.ragtruth_model_filter,
            )
            if args.ragtruth_use_builtin_splits:
                (test_prompts, test_labels), (train_prompts, train_labels), \
                    (exemplar_prompts, exemplar_labels), stats = loader.load_with_builtin_splits(
                        exemplar_size=args.num_exemplars,
                    )
            else:
                all_prompts, all_labels, stats = loader.load()
                (test_prompts, test_labels), (train_prompts, train_labels), \
                    (exemplar_prompts, exemplar_labels) = loader.split_data(
                        all_prompts, all_labels,
                        exemplar_size=args.num_exemplars,
                        wild_ratio=args.wild_ratio,
                    )
            args.num_exemplars = len(exemplar_prompts)
            profiler.set_dataset_info(stats)
        else:
            all_prompts, all_labels, stats, dataset, length = load_original_dataset_data(args, tokenizer)
            (test_prompts, test_labels), (train_prompts, train_labels), \
                (exemplar_prompts, exemplar_labels) = split_original_data(args, all_prompts, all_labels, length)
            profiler.set_dataset_info(stats)

    logger.info(f"Data: test={len(test_prompts)}, train={len(train_prompts)}, exemplar={len(exemplar_prompts)}")

    # ===== Model Setup (freeze + TSV injection) =====
    for param in model.parameters():
        param.requires_grad = False

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    tsv = nn.ParameterList(
        [nn.Parameter(torch.zeros(hidden_size), requires_grad=True) for _ in range(num_layers)]
    )
    tsv.to(device)
    add_tsv_layers(model, tsv, [args.lam], args)

    optimizer = torch.optim.AdamW(list(tsv.parameters()), lr=args.lr)
    scaler = GradScaler()
    layer_number = -1
    batch_size = args.batch_size
    best_test_auroc = -1
    best_test_epoch = -1

    # Sinkhorn setup
    args.num_iters_sk = 3
    args.epsilon_sk = 0.05
    num_exemplars = args.num_exemplars
    ex_hallu = (num_exemplars - exemplar_labels[:num_exemplars].sum()) / num_exemplars
    ex_true = exemplar_labels[:num_exemplars].sum() / num_exemplars
    cls_dist = torch.tensor([ex_hallu, ex_true]).float().cuda().view(-1, 1)
    sinkhorn = SinkhornKnopp_imb(args, cls_dist)

    # Centroids
    centroids = torch.randn((2, hidden_size)).half().cuda()
    centroids = F.normalize(centroids, p=2, dim=1)

    # Save unpadded exemplar data for later augmentation
    exemplar_prompts_raw, exemplar_labels_raw = exemplar_prompts, exemplar_labels
    exemplar_prompts_padded, exemplar_labels_padded = collate_fn(exemplar_prompts, exemplar_labels)

    # ===== Phase: Init Training (exemplar) =====
    with profiler.phase("init_training", args.gpu_util_interval) as p:
        for epoch in range(args.init_num_epochs):
            with p.epoch_scope():
                running_loss = 0.0
                total = 0

                for batch_start in range(0, num_exemplars, batch_size):
                    batch_prompts = exemplar_prompts_padded[batch_start: batch_start + batch_size]
                    batch_labels = exemplar_labels_padded[batch_start: batch_start + batch_size]
                    attention_mask = (batch_prompts != 0).half()
                    batch_prompts = batch_prompts.to(device)
                    batch_labels = batch_labels.to(device)
                    attention_mask = attention_mask.to(device)

                    # Forward
                    with p.operation("forward"):
                        with autocast(dtype=torch.float16):
                            output = model(batch_prompts.squeeze(), attention_mask=attention_mask.squeeze(),
                                           output_hidden_states=True)
                            hidden_states = torch.stack(output.hidden_states, dim=0).squeeze()
                            last_layer_hidden_state = hidden_states[layer_number]
                            last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state,
                                                                          attention_mask.squeeze())

                    # OT loss
                    with p.operation("ot_loss"):
                        with autocast(dtype=torch.float16):
                            batch_labels_oh = F.one_hot(batch_labels, num_classes=2)
                            ot_loss, similarities = compute_ot_loss_cos(
                                last_token_rep, centroids, batch_labels_oh, batch_size, args
                            )
                            loss = ot_loss

                    # Centroid update
                    with p.operation("centroid_update"):
                        with torch.no_grad():
                            centroids = update_centroids_ema_hard(centroids, last_token_rep, batch_labels_oh, args)

                    # Backward
                    with p.operation("backward"):
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    total += batch_labels.size(0)
                    running_loss += loss.item() * batch_labels.size(0)

                epoch_loss = running_loss / total if total > 0 else 0

                # Test evaluation
                with p.operation("test_eval"):
                    test_preds, test_labels_combined = test_model(
                        model, centroids, test_prompts, test_labels, device, batch_size, layer_number
                    )
                    test_auroc = roc_auc_score(
                        test_labels_combined.cpu().numpy() if torch.is_tensor(test_labels_combined) else test_labels_combined,
                        test_preds.cpu().numpy() if torch.is_tensor(test_preds) else test_preds,
                    )

                if test_auroc > best_test_auroc:
                    best_test_auroc = test_auroc
                    best_test_epoch = epoch
                logger.info(f"[Init] Epoch {epoch + 1}/{args.init_num_epochs}, "
                            f"Loss: {epoch_loss:.4f}, Test AUROC: {test_auroc:.4f}, "
                            f"Best: {best_test_auroc:.4f} @ epoch {best_test_epoch}")
                model.train()

    # ===== Phase: Semi-supervised Data Selection =====
    with profiler.phase("ss_data_selection", args.gpu_util_interval) as p:
        with torch.no_grad():
            selected_indices, selected_labels_soft = get_ex_data_profiled(
                model, train_prompts, train_labels, batch_size,
                centroids, sinkhorn, args.num_selected_data, cls_dist, args, p
            )

    # Build augmented dataset
    selected_prompts = [train_prompts[i] for i in selected_indices]
    augmented_prompts = selected_prompts + list(exemplar_prompts_raw)
    exemplar_label_tensor = torch.tensor(exemplar_labels_raw).cuda()
    exemplar_labels_oh = F.one_hot(exemplar_label_tensor.to(torch.int64), num_classes=2)
    augmented_labels = torch.concat((selected_labels_soft, exemplar_labels_oh.clone().float().cuda()))
    num_augmented = len(augmented_prompts)

    # ===== Phase: Augmented Training =====
    with profiler.phase("augmented_training", args.gpu_util_interval) as p:
        with autocast(dtype=torch.float16):
            for epoch in range(args.aug_num_epochs):
                with p.epoch_scope():
                    running_loss = 0.0
                    total = 0

                    for batch_start in range(0, num_augmented, batch_size):
                        batch_prompts_list = augmented_prompts[batch_start: batch_start + batch_size]
                        batch_labels_aug = augmented_labels[batch_start: batch_start + batch_size]
                        # Pad prompts only; keep soft labels as float (collate_fn would
                        # convert to torch.long, destroying pseudo-label probabilities)
                        batch_prompts_t, _ = collate_fn(batch_prompts_list, batch_labels_aug)
                        batch_labels_t = batch_labels_aug  # preserve float soft labels
                        attention_mask = (batch_prompts_t != 0).half()
                        batch_prompts_t = batch_prompts_t.to(device)
                        batch_labels_t = batch_labels_t.to(device)
                        attention_mask = attention_mask.to(device)

                        # Forward
                        with p.operation("forward"):
                            output = model(batch_prompts_t.squeeze(), attention_mask=attention_mask.squeeze(),
                                           output_hidden_states=True)
                            hidden_states = torch.stack(output.hidden_states, dim=0).squeeze()
                            last_layer_hidden_state = hidden_states[layer_number]
                            last_token_rep = get_last_non_padded_token_rep(last_layer_hidden_state,
                                                                          attention_mask.squeeze())

                        # OT loss
                        with p.operation("ot_loss"):
                            ot_loss, similarities = compute_ot_loss_cos(
                                last_token_rep, centroids, batch_labels_t, batch_size, args
                            )
                            loss = ot_loss

                        # Centroid update (soft)
                        with p.operation("centroid_update"):
                            with torch.no_grad():
                                centroids = update_centroids_ema(centroids, last_token_rep,
                                                                 batch_labels_t.half(), args)

                        # Backward
                        with p.operation("backward"):
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                        total += batch_labels_t.size(0)
                        running_loss += loss.item() * batch_labels_t.size(0)

                    epoch_loss = running_loss / total if total > 0 else 0

                    # Test evaluation
                    with torch.no_grad():
                        with p.operation("test_eval"):
                            test_preds, test_labels_combined = test_model(
                                model, centroids, test_prompts, test_labels, device, batch_size, layer_number
                            )
                            test_auroc = roc_auc_score(test_labels_combined, test_preds)

                    if test_auroc > best_test_auroc:
                        best_test_auroc = test_auroc
                        best_test_epoch = epoch + args.init_num_epochs
                    logger.info(f"[Aug] Epoch {epoch + 1}/{args.aug_num_epochs}, "
                                f"Loss: {epoch_loss:.4f}, Test AUROC: {test_auroc:.4f}, "
                                f"Best: {best_test_auroc:.4f} @ epoch {best_test_epoch}")
                    model.train()

    # ===== Save profiling results =====
    logger.info(f"Training complete. Best AUROC: {best_test_auroc:.4f} @ epoch {best_test_epoch}")
    profiler.save()
    logger.info(f"Profiling results saved to {args.profile_output_dir}")


if __name__ == "__main__":
    main()
