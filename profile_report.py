"""Report formatting and output for TSV profiling results.

Generates human-readable reports and structured JSON output from ProfilingResults.
"""

import os
import json
import logging
from typing import List

from profiler import ProfilingResults, PhaseMetrics

logger = logging.getLogger(__name__)


def format_phase_table(results: ProfilingResults) -> str:
    """Format phase-level summary as a table.

    Columns: Phase | Time(s) | Time% | GPU Peak(MB) | CPU Peak(MB) | GPU Util%
    """
    total = results.total_time_seconds or 1e-9

    header = f"{'Phase':<25} {'Time(s)':>9} {'Time%':>7} {'GPU Peak(MB)':>13} {'CPU Peak(MB)':>12} {'GPU Util%':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for name, pm in results.phases.items():
        pct = pm.total_time_seconds / total * 100
        lines.append(
            f"{name:<25} {pm.total_time_seconds:>9.2f} {pct:>6.1f}% "
            f"{pm.gpu_peak_allocated_mb:>13.1f} {pm.cpu_peak_mb:>12.1f} "
            f"{pm.gpu_utilization_avg:>9.1f}%"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_operation_breakdown(phase: PhaseMetrics) -> str:
    """Format per-operation breakdown for a single phase."""
    if not phase.operations:
        return "  (no operations recorded)"

    total_phase = phase.total_time_seconds or 1e-9

    header = f"  {'Operation':<22} {'Avg(ms)':>9} {'Total(s)':>9} {'Time%':>7} {'Count':>7} {'Min(ms)':>9} {'Max(ms)':>9}"
    sep = "  " + "-" * (len(header) - 2)
    lines = [sep, header, sep]

    # Sort by total time descending
    sorted_ops = sorted(phase.operations.values(), key=lambda x: x.total_seconds, reverse=True)

    for op in sorted_ops:
        pct = op.total_seconds / total_phase * 100
        lines.append(
            f"  {op.name:<22} {op.avg_ms:>9.2f} {op.total_seconds:>9.3f} {pct:>6.1f}% "
            f"{op.count:>7} {op.min_ms:>9.2f} {op.max_ms:>9.2f}"
        )

    lines.append(sep)
    return "\n".join(lines)


def format_epoch_breakdown(phase: PhaseMetrics) -> str:
    """Format per-epoch timing for a phase."""
    if not phase.epoch_times:
        return "  (no epoch data)"

    lines = []
    for i, t in enumerate(phase.epoch_times):
        lines.append(f"  Epoch {i + 1:>3}: {t:>8.2f}s")

    avg = sum(phase.epoch_times) / len(phase.epoch_times)
    lines.append(f"  {'Average':>9}: {avg:>8.2f}s")
    return "\n".join(lines)


def format_throughput_summary(results: ProfilingResults) -> str:
    """Format throughput metrics: samples/sec for training phases."""
    ds = results.dataset_info
    total_samples = ds.get("total_samples", 0)
    if total_samples == 0:
        return "  (no dataset info for throughput calculation)"

    lines = []
    # Estimate sample counts per phase based on dataset info
    exemplar_count = ds.get("exemplar_samples", ds.get("label_distribution", {}).get("truthful", 0) +
                            ds.get("label_distribution", {}).get("hallucinated", 0))

    for name, pm in results.phases.items():
        if pm.total_time_seconds <= 0:
            continue

        n_epochs = len(pm.epoch_times) if pm.epoch_times else 0

        # Estimate forward passes from operation counts
        fwd_op = pm.operations.get("forward")
        if fwd_op and fwd_op.count > 0:
            total_fwd_samples = fwd_op.count  # batches, not samples
            throughput = total_fwd_samples / pm.total_time_seconds
            lines.append(f"  {name:<25} {throughput:>8.2f} batches/s  "
                         f"({fwd_op.count} forward passes in {pm.total_time_seconds:.1f}s)")
            if n_epochs > 0:
                lines.append(f"  {'':25} {fwd_op.avg_ms:>8.1f} ms/batch avg forward")
        elif pm.total_time_seconds > 0:
            lines.append(f"  {name:<25} {pm.total_time_seconds:>8.2f}s total")

    return "\n".join(lines) if lines else "  (no throughput data)"


def format_full_report(results: ProfilingResults) -> str:
    """Generate complete human-readable profiling report."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("TSV Profiling Report")
    lines.append("=" * 70)
    lines.append(f"Timestamp:  {results.timestamp}")
    lines.append(f"Model:      {results.model_name}")
    lines.append(f"Parameters: {results.total_params:,} total | {results.trainable_params:,} trainable (TSV)")

    # Hardware
    hw = results.hardware_info
    if hw.get("cuda_available"):
        lines.append(f"GPU:        {hw.get('device_name', 'N/A')} ({hw.get('gpu_memory_total_gb', 0):.1f} GB)")

    # Dataset
    ds = results.dataset_info
    if ds:
        ds_name = ds.get("dataset", "RAGTruth")
        lines.append(f"Dataset:    {ds_name} ({ds.get('total_samples', 0)} samples, "
                      f"avg seq len: {ds.get('avg_sequence_length', 0)})")
        td = ds.get("task_type_distribution", {})
        if td:
            parts = [f"{k}={v}" for k, v in td.items()]
            lines.append(f"            Task types: {', '.join(parts)}")
        ld = ds.get("label_distribution", {})
        if ld:
            lines.append(f"            Labels: truthful={ld.get('truthful', 0)}, "
                          f"hallucinated={ld.get('hallucinated', 0)}")

    lines.append(f"\nTotal Time: {results.total_time_seconds:.2f}s")
    lines.append("")

    # Phase summary table
    lines.append("Phase Breakdown:")
    lines.append(format_phase_table(results))
    lines.append("")

    # Throughput summary
    lines.append("Throughput:")
    lines.append(format_throughput_summary(results))
    lines.append("")

    # Per-phase operation breakdowns
    for name, pm in results.phases.items():
        if pm.operations:
            n_epochs = len(pm.epoch_times) if pm.epoch_times else 0
            epoch_info = f" ({n_epochs} epochs)" if n_epochs > 0 else ""
            lines.append(f"{name}{epoch_info} -- Operation Breakdown:")
            lines.append(format_operation_breakdown(pm))
            lines.append("")

        if pm.epoch_times and len(pm.epoch_times) > 1:
            lines.append(f"{name} -- Per-Epoch Times:")
            lines.append(format_epoch_breakdown(pm))
            lines.append("")

    lines.append("=" * 70)
    return "\n".join(lines)


def save_results(results: ProfilingResults, output_dir: str) -> None:
    """Save profiling results in multiple formats.

    Creates:
        profiling_results.json  - Full structured data
        profiling_report.txt    - Human-readable report
        profiling_summary.json  - Phase-level totals only
    """
    os.makedirs(output_dir, exist_ok=True)

    # Full JSON
    full_path = os.path.join(output_dir, "profiling_results.json")
    results.to_json(full_path)
    logger.info(f"Full results saved to {full_path}")

    # Human-readable report
    report = format_full_report(results)
    report_path = os.path.join(output_dir, "profiling_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Print report to console
    print(report)

    # Compact summary JSON
    summary = {
        "model": results.model_name,
        "total_time_seconds": round(results.total_time_seconds, 2),
        "total_params": results.total_params,
        "trainable_params": results.trainable_params,
        "phases": {},
    }
    for name, pm in results.phases.items():
        phase_summary = {
            "time_seconds": round(pm.total_time_seconds, 2),
            "gpu_peak_mb": round(pm.gpu_peak_allocated_mb, 1),
            "cpu_peak_mb": round(pm.cpu_peak_mb, 1),
            "gpu_util_avg": round(pm.gpu_utilization_avg, 1),
            "num_epochs": len(pm.epoch_times),
        }
        fwd_op = pm.operations.get("forward")
        if fwd_op and fwd_op.count > 0 and pm.total_time_seconds > 0:
            phase_summary["forward_batches"] = fwd_op.count
            phase_summary["forward_avg_ms"] = round(fwd_op.avg_ms, 2)
            phase_summary["batches_per_sec"] = round(fwd_op.count / pm.total_time_seconds, 2)
        summary["phases"][name] = phase_summary

    summary_path = os.path.join(output_dir, "profiling_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to {summary_path}")
