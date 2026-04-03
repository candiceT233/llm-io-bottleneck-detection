import os
import json
import glob
import copy
import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
print(client)
BOTTLENECK_CLASSES = [
    "storage_bandwidth_saturation",
    "metadata_contention",
    "lock_contention",
    "network_io_bottleneck",
    "serialized_io",
    "checkpointing_overhead",
    "compute_bound",
]

SYSTEM_PROMPT = "You are an expert HPC systems engineer specializing in I/O performance analysis."

USER_PROMPT_TEMPLATE = """\
Analyze the following HPC workflow execution snapshot and diagnose the primary bottleneck.

Choose exactly one bottleneck class from this list:
- storage_bandwidth_saturation: storage device bandwidth is saturated (bw_utilization_ratio near 1.0)
- metadata_contention: too many small file operations overwhelming the metadata server (high iops, tiny avg_io_size_kb)
- lock_contention: multiple tasks competing for write locks on shared files (many writers per shared file, low bandwidth despite active I/O)
- network_io_bottleneck: data movement over the network is the limiting factor (network_util_ratio near 1.0, often with remote/object storage)
- serialized_io: I/O is forced through a single writer/reader, preventing parallelism (io_parallelism=1)
- checkpointing_overhead: periodic checkpointing dominates stage time (large checkpoint_size_mb, frequent intervals)
- compute_bound: CPU computation is the bottleneck, I/O is not a significant factor (high cpu_util_pct, low io_time_ratio)

Respond with a JSON object only — no markdown, no extra text:
{{
  "bottleneck": "<one class from the list above>",
  "confidence": "<high | medium | low>",
  "key_signals": ["<metric: value>", ...],
  "explanation": "<concise reasoning, 2-4 sentences>"
}}

Snapshot:
{snapshot_json}
"""

CLASSIFY_ONLY_PROMPT_TEMPLATE = """\
Analyze the following HPC workflow execution snapshot and diagnose the primary bottleneck.

Choose exactly one bottleneck class from this list:
- storage_bandwidth_saturation
- metadata_contention
- lock_contention
- network_io_bottleneck
- serialized_io
- checkpointing_overhead
- compute_bound

Respond with a JSON object only — no markdown, no extra text:
{{"bottleneck": "<one class from the list above>"}}

Snapshot:
{snapshot_json}
"""


def strip_annotation(snapshot: dict) -> dict:
    """Remove ground-truth annotation before sending to the LLM."""
    clean = copy.deepcopy(snapshot)
    clean.pop("annotation", None)
    return clean


def build_prompt(snapshot: dict, strategy: str = "zero_shot") -> str:
    clean = strip_annotation(snapshot)
    snapshot_json = json.dumps(clean, indent=2)
    if strategy == "classify_only":
        return CLASSIFY_ONLY_PROMPT_TEMPLATE.format(snapshot_json=snapshot_json)
    return USER_PROMPT_TEMPLATE.format(snapshot_json=snapshot_json)


def rule_based_diagnose(snapshot: dict) -> dict:
    exe = snapshot.get("execution", {})
    io = snapshot.get("io_metrics", {})
    res = snapshot.get("resource_utilization", {})
    storage = snapshot.get("storage", {})

    io_time_ratio = exe.get("io_time_ratio", 0)
    cpu_util = res.get("cpu_util_pct", 0)
    bw_util = io.get("bw_utilization_ratio", 0)
    network_util = res.get("network_util_ratio", 0)
    avg_io_size_kb = io.get("avg_io_size_kb", 1024)
    iops = io.get("iops", 0)
    parallelism = exe.get("parallelism", 999)
    checkpoint_size_mb = exe.get("checkpoint_size_mb", 0)
    shared = storage.get("shared", False)

    # Priority-ordered rules
    if cpu_util > 70 and io_time_ratio < 0.3:
        bottleneck = "compute_bound"
    elif network_util > 0.8 and io_time_ratio > 0.5:
        bottleneck = "network_io_bottleneck"
    elif bw_util > 0.85 and io_time_ratio > 0.5:
        bottleneck = "storage_bandwidth_saturation"
    elif avg_io_size_kb < 64 and iops > 500 and io_time_ratio > 0.5:
        bottleneck = "metadata_contention"
    elif parallelism == 1 and io_time_ratio > 0.5:
        bottleneck = "serialized_io"
    elif checkpoint_size_mb > 100 and io_time_ratio > 0.5:
        bottleneck = "checkpointing_overhead"
    elif shared and io_time_ratio > 0.5:
        bottleneck = "lock_contention"
    else:
        bottleneck = "storage_bandwidth_saturation"

    return {
        "bottleneck": bottleneck,
        "confidence": "medium",
        "key_signals": [
            f"io_time_ratio: {io_time_ratio}",
            f"cpu_util_pct: {cpu_util}",
            f"bw_utilization_ratio: {bw_util}",
            f"network_util_ratio: {network_util}",
            f"avg_io_size_kb: {avg_io_size_kb}",
        ],
        "explanation": "Rule-based classification using metric thresholds.",
    }


def diagnose(snapshot: dict, model: str = "gpt-4.1-mini", strategy: str = "zero_shot") -> dict:
    if strategy == "rule_based":
        return rule_based_diagnose(snapshot)

    prompt = build_prompt(snapshot, strategy=strategy)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


def load_snapshots(snapshots_dir: str = "data/snapshots") -> list[dict]:
    paths = sorted(glob.glob(os.path.join(snapshots_dir, "snap_*.json")))
    snapshots = []
    for p in paths:
        with open(p) as f:
            snapshots.append(json.load(f))
    return snapshots


def evaluate(snapshots: list[dict], model: str = "gpt-4.1-mini", strategy: str = "zero_shot") -> dict:
    results = []

    for i, snap in enumerate(snapshots, 1):
        snap_id = snap.get("id", f"snap_{i:03d}")
        gt = snap["annotation"]

        print(f"[{i:2d}/{len(snapshots)}] {snap_id} ...", end=" ", flush=True)

        prediction = diagnose(snap, model=model, strategy=strategy)

        correct = prediction.get("bottleneck") == gt["bottleneck"]
        print("correct" if correct else f"WRONG (predicted: {prediction.get('bottleneck')})")

        results.append({
            "id": snap_id,
            "ground_truth": gt["bottleneck"],
            "gt_confidence": gt["confidence"],
            "predicted": prediction.get("bottleneck"),
            "llm_confidence": prediction.get("confidence"),
            "correct": correct,
            "gt_key_signals": gt.get("key_signals", []),
            "llm_key_signals": prediction.get("key_signals", []),
            "explanation": prediction.get("explanation", ""),
        })

    summary = _compute_summary(results)
    return {"model": model, "strategy": strategy, "results": results, "summary": summary}


def _compute_summary(results: list[dict]) -> dict:
    total = len(results)
    correct = sum(r["correct"] for r in results)
    accuracy = correct / total if total > 0 else 0.0

    # Per-class counts for precision / recall / F1
    classes = BOTTLENECK_CLASSES
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    for r in results:
        gt = r["ground_truth"]
        pred = r["predicted"]
        if gt == pred:
            tp[gt] = tp.get(gt, 0) + 1
        else:
            fn[gt] = fn.get(gt, 0) + 1
            fp[pred] = fp.get(pred, 0) + 1

    per_class = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[c] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)}

    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(per_class)

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
    }


def save_results(run: dict, results_dir: str = "data/results") -> str:
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run['model']}_{run['strategy']}_{ts}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(run, f, indent=2)
    return path


def print_summary(summary: dict) -> None:
    print("\n=== Summary ===")
    print(f"Accuracy : {summary['accuracy']:.1%}  ({summary['correct']}/{summary['total']})")
    print(f"Macro F1 : {summary['macro_f1']:.3f}")
    print("\nPer-class F1:")
    for cls, metrics in summary["per_class"].items():
        print(f"  {cls:<35} P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")


if __name__ == "__main__":
    MODEL = "gpt-4.1-mini"
    STRATEGIES = ["rule_based", "classify_only", "zero_shot"]

    print(f"Loading snapshots...")
    snapshots = load_snapshots()
    print(f"Loaded {len(snapshots)} snapshots.\n")

    for strategy in STRATEGIES:
        label = f"{MODEL} / {strategy}" if strategy != "rule_based" else "rule_based (no LLM)"
        print(f"{'='*55}")
        print(f"Strategy: {label}")
        print(f"{'='*55}")
        run = evaluate(snapshots, model=MODEL, strategy=strategy)
        print_summary(run["summary"])
        path = save_results(run)
        print(f"\nResults saved to: {path}\n")
