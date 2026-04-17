import os
import gc
import json
import glob
import copy
import time
import resource
import datetime
import psutil
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
BOTTLENECK_CLASSES = [
    "storage_bandwidth_saturation",
    "metadata_contention",
    "lock_contention",
    "network_io_bottleneck",
    "serialized_io",
    "checkpointing_overhead",
    "compute_bound",
    "read_bandwidth_saturation",
    "io_interference",
    "data_skew",
    "staging_inefficiency",
]

SYSTEM_PROMPT = "You are an expert HPC systems engineer specializing in I/O performance analysis."

PROMPT_HEADER = """\
Analyze the following HPC workflow execution snapshot and diagnose the primary bottleneck.

Choose exactly one bottleneck class from this list:
- storage_bandwidth_saturation: write bandwidth saturates the storage device (bw_utilization_ratio near 1.0 on writes, low cpu_util)
- read_bandwidth_saturation: read bandwidth saturates the storage device (bw_utilization_ratio near 1.0 on reads, read_bw_mb_s close to peak, low cpu_util)
- metadata_contention: too many small file operations overwhelming the metadata server (high iops, tiny avg_io_size_kb, low bw_utilization_ratio)
- lock_contention: multiple tasks competing for write locks on shared files (shared storage, many writers, very low bw_utilization_ratio despite high parallelism)
- network_io_bottleneck: data movement over the network is the limiting factor (network_util_ratio near 1.0, remote/object storage such as s3, hdfs, nfs)
- serialized_io: I/O is forced through a single writer/reader, preventing parallelism (parallelism=1 in execution, low bw_utilization_ratio)
- checkpointing_overhead: periodic checkpointing dominates stage time (checkpoint_size_mb and num_checkpoints present, large aggregate_size_mb)
- compute_bound: CPU computation is the bottleneck, I/O is not a significant factor (high cpu_util_pct, low io_time_ratio)
- io_interference: external competing jobs degrade observed bandwidth below the workflow's fair share (competing_jobs field present, bw_utilization_ratio low despite shared filesystem being the storage)
- data_skew: I/O load is unevenly distributed across nodes, creating hot spots (io_imbalance_ratio and hot_node_count fields present, bw_utilization_ratio moderately low)
- staging_inefficiency: data was not pre-staged to a fast local tier before execution (data_staged=false, remote storage type such as s3/nfs/hdfs, high remote_access_latency_ms, low bw_utilization_ratio)
"""

RESPONSE_FORMAT = """\
Respond with a JSON object only — no markdown, no extra text:
{{
  "bottleneck": "<one class from the list above>",
  "confidence": "<high | medium | low>",
  "key_signals": ["<metric: value>", ...],
  "explanation": "<concise reasoning, 2-4 sentences>"
}}"""

USER_PROMPT_TEMPLATE = PROMPT_HEADER + "\n" + RESPONSE_FORMAT + """

Snapshot:
{snapshot_json}
"""

COT_PROMPT_TEMPLATE = PROMPT_HEADER + """\

Reason step by step through the snapshot before concluding:
Step 1 — Compute vs I/O: compare cpu_util_pct and io_time_ratio. Is the stage compute-bound or I/O-bound?
Step 2 — Network: check network_util_ratio and storage type. Is the network the bottleneck?
Step 3 — Staging: check data_staged and remote_access_latency_ms. Was data pre-staged?
Step 4 — Interference: check competing_jobs. Are external jobs degrading bandwidth?
Step 5 — Data skew: check io_imbalance_ratio and hot_node_count. Is load unevenly distributed?
Step 6 — Bandwidth saturation: compare read_bw_mb_s and write_bw_mb_s against peak_storage_bw_mb_s. Which path is saturated?
Step 7 — Access pattern: check avg_io_size_kb and iops. Are small files overwhelming the metadata server?
Step 8 — Parallelism: check parallelism field. Is I/O serialized through a single task?
Step 9 — Checkpointing: check checkpoint_size_mb and num_checkpoints. Does checkpointing dominate I/O?
Step 10 — Lock contention: check num_shared_files, parallelism, and bw_utilization_ratio. Are tasks contending for shared file locks?
Step 11 — Conclude: state which single bottleneck class best fits the evidence.

""" + RESPONSE_FORMAT + """

Snapshot:
{snapshot_json}
"""

# Few-shot examples: hand-picked to cover the most commonly confused classes.
# Each example is a (snapshot_without_annotation, annotation) pair.
FEW_SHOT_EXAMPLES = [
    # read_bandwidth_saturation — high read_bw near peak, low cpu
    (
        {
            "id": "snap_003", "workflow": {"name": "cryo_em_pipeline", "type": "imaging", "num_stages": 6},
            "stage": {"name": "reconstruction", "order": 4, "predecessors": ["particle_picking"], "operation": "read"},
            "execution": {"num_nodes": 16, "num_tasks": 128, "parallelism": 128, "total_time_s": 1820,
                          "io_time_s": 1500, "compute_time_s": 320, "io_time_ratio": 0.82,
                          "transfer_size_kb": 2048, "aggregate_size_mb": 570000, "op_count": 285000},
            "io_metrics": {"read_bw_mb_s": 380, "write_bw_mb_s": 0, "peak_storage_bw_mb_s": 400,
                           "bw_utilization_ratio": 0.95, "iops": 190, "avg_io_size_kb": 2048, "sequential_ratio": 0.96},
            "resource_utilization": {"cpu_util_pct": 22, "memory_util_pct": 48, "network_bw_mb_s": 390,
                                     "network_peak_bw_mb_s": 10000, "network_util_ratio": 0.039},
            "storage": {"type": "gpfs", "stripe_count": 16, "shared": True},
        },
        {"bottleneck": "read_bandwidth_saturation", "confidence": "high",
         "key_signals": ["bw_utilization_ratio: 0.95", "read_bw_mb_s: 380", "peak_storage_bw_mb_s: 400",
                         "cpu_util_pct: 22", "io_time_ratio: 0.82"],
         "explanation": "Read bandwidth of 380 MB/s reaches 95% of the GPFS peak (400 MB/s), saturating the read path. CPU at 22% confirms tasks are stalled on I/O, not compute. The io_time_ratio of 0.82 shows I/O dominates stage time."},
    ),
    # lock_contention — high parallelism, low bw_util, shared file writes
    (
        {
            "id": "snap_015", "workflow": {"name": "genome_assembly", "type": "genomics", "num_stages": 6},
            "stage": {"name": "merge_results", "order": 5, "predecessors": ["contig_assembly"], "operation": "write"},
            "execution": {"num_nodes": 20, "num_tasks": 200, "parallelism": 200, "total_time_s": 800,
                          "io_time_s": 680, "compute_time_s": 120, "io_time_ratio": 0.85,
                          "transfer_size_kb": 512, "aggregate_size_mb": 57800, "op_count": 115600,
                          "num_shared_files": 3},
            "io_metrics": {"read_bw_mb_s": 0, "write_bw_mb_s": 85, "peak_storage_bw_mb_s": 400,
                           "bw_utilization_ratio": 0.21, "iops": 170, "avg_io_size_kb": 512, "sequential_ratio": 0.72},
            "resource_utilization": {"cpu_util_pct": 12, "memory_util_pct": 35, "network_bw_mb_s": 88,
                                     "network_peak_bw_mb_s": 5000, "network_util_ratio": 0.018},
            "storage": {"type": "beegfs", "stripe_count": 8, "shared": True},
        },
        {"bottleneck": "lock_contention", "confidence": "high",
         "key_signals": ["num_shared_files: 3", "num_tasks: 200", "bw_utilization_ratio: 0.21",
                         "cpu_util_pct: 12", "io_time_ratio: 0.85"],
         "explanation": "200 tasks compete for write locks on 3 shared files, serializing access and capping bandwidth at 85 MB/s (21% of the 400 MB/s BeeGFS peak). CPU at 12% confirms tasks are waiting for lock release, not computing."},
    ),
    # io_interference — low observed bw, competing_jobs present, local filesystem
    (
        {
            "id": "snap_091", "workflow": {"name": "namd", "type": "molecular_dynamics", "num_stages": 6},
            "stage": {"name": "fetch_input", "order": 1, "predecessors": ["initial_data"], "operation": "read"},
            "execution": {"num_nodes": 21, "num_tasks": 84, "parallelism": 84, "total_time_s": 732,
                          "io_time_s": 622, "compute_time_s": 110, "io_time_ratio": 0.85,
                          "transfer_size_kb": 2048, "aggregate_size_mb": 3861, "op_count": 1930,
                          "competing_jobs": 7},
            "io_metrics": {"read_bw_mb_s": 107.5, "write_bw_mb_s": 0, "peak_storage_bw_mb_s": 300,
                           "bw_utilization_ratio": 0.36, "iops": 53, "avg_io_size_kb": 2048, "sequential_ratio": 0.71},
            "resource_utilization": {"cpu_util_pct": 18, "memory_util_pct": 42, "network_bw_mb_s": 110,
                                     "network_peak_bw_mb_s": 5000, "network_util_ratio": 0.022},
            "storage": {"type": "lustre", "stripe_count": 8, "shared": True},
        },
        {"bottleneck": "io_interference", "confidence": "high",
         "key_signals": ["competing_jobs: 7", "bw_utilization_ratio: 0.36", "peak_storage_bw_mb_s: 300",
                         "cpu_util_pct: 18", "io_time_ratio: 0.85"],
         "explanation": "7 competing jobs on the shared Lustre filesystem degrade observed bandwidth to 107.5 MB/s (36% of the 300 MB/s peak). CPU at 18% rules out compute bottleneck. The low bandwidth is caused by external interference, not the workflow's own I/O pattern."},
    ),
    # staging_inefficiency — data_staged=false, remote storage, high latency
    (
        {
            "id": "snap_171", "workflow": {"name": "montage", "type": "astronomy", "num_stages": 7},
            "stage": {"name": "scan_catalog", "order": 4, "predecessors": ["load_reference"], "operation": "read"},
            "execution": {"num_nodes": 37, "num_tasks": 481, "parallelism": 481, "total_time_s": 1076,
                          "io_time_s": 796, "compute_time_s": 280, "io_time_ratio": 0.74,
                          "transfer_size_kb": 128, "aggregate_size_mb": 6152, "op_count": 49216,
                          "data_staged": False, "intended_staging_tier": "node_local_ssd",
                          "remote_access_latency_ms": 183},
            "io_metrics": {"read_bw_mb_s": 93.2, "write_bw_mb_s": 0, "peak_storage_bw_mb_s": 400,
                           "bw_utilization_ratio": 0.23, "iops": 728, "avg_io_size_kb": 128, "sequential_ratio": 0.62},
            "resource_utilization": {"cpu_util_pct": 14, "memory_util_pct": 28, "network_bw_mb_s": 96,
                                     "network_peak_bw_mb_s": 5000, "network_util_ratio": 0.019},
            "storage": {"type": "nfs", "stripe_count": 2, "shared": True},
        },
        {"bottleneck": "staging_inefficiency", "confidence": "high",
         "key_signals": ["data_staged: false", "remote_access_latency_ms: 183", "bw_utilization_ratio: 0.23",
                         "storage_type: nfs", "cpu_util_pct: 14"],
         "explanation": "Data was not pre-staged to node_local_ssd before execution. Tasks access NFS directly with 183 ms remote latency, yielding only 93 MB/s (23% of peak). CPU at 14% confirms tasks are stalled on remote I/O."},
    ),
    # metadata_contention — tiny avg_io_size, high iops, low bw_util
    (
        {
            "id": "snap_008", "workflow": {"name": "1000_genomes", "type": "genomics", "num_stages": 5},
            "stage": {"name": "sifting", "order": 2, "predecessors": ["individuals"], "operation": "write"},
            "execution": {"num_nodes": 2, "num_tasks": 10, "parallelism": 10, "total_time_s": 280,
                          "io_time_s": 240, "compute_time_s": 40, "io_time_ratio": 0.86,
                          "transfer_size_kb": 4, "aggregate_size_mb": 504, "op_count": 129024},
            "io_metrics": {"read_bw_mb_s": 0, "write_bw_mb_s": 2.1, "peak_storage_bw_mb_s": 200,
                           "bw_utilization_ratio": 0.01, "iops": 538, "avg_io_size_kb": 4, "sequential_ratio": 0.12},
            "resource_utilization": {"cpu_util_pct": 25, "memory_util_pct": 38, "network_bw_mb_s": 3,
                                     "network_peak_bw_mb_s": 5000, "network_util_ratio": 0.001},
            "storage": {"type": "beegfs", "stripe_count": 2, "shared": True},
        },
        {"bottleneck": "metadata_contention", "confidence": "high",
         "key_signals": ["iops: 538", "avg_io_size_kb: 4", "bw_utilization_ratio: 0.01",
                         "io_time_ratio: 0.86", "op_count: 129024"],
         "explanation": "538 IOPS with an average I/O size of only 4 KB floods the BeeGFS metadata server with open/close/stat operations. Despite 86% of time spent in I/O, bandwidth utilization is only 1% because the bottleneck is metadata throughput, not data transfer."},
    ),
]


def build_few_shot_prompt(snapshot: dict) -> str:
    lines = [PROMPT_HEADER, "Here are examples of correctly diagnosed snapshots:"]

    for i, (ex_snap, ex_ann) in enumerate(FEW_SHOT_EXAMPLES, 1):
        lines.append(f"\n--- Example {i} ---")
        lines.append(f"Snapshot:\n{json.dumps(ex_snap, indent=2)}")
        lines.append(f"Diagnosis:\n{json.dumps(ex_ann, indent=2)}")

    clean = strip_annotation(snapshot)
    lines.append("\n--- Now diagnose this snapshot ---")
    lines.append(f"Snapshot:\n{json.dumps(clean, indent=2)}")
    lines.append("\n" + RESPONSE_FORMAT)
    return "\n".join(lines)


CLASSIFY_ONLY_PROMPT_TEMPLATE = """\
Analyze the following HPC workflow execution snapshot and diagnose the primary bottleneck.

Choose exactly one bottleneck class from this list:
- storage_bandwidth_saturation
- read_bandwidth_saturation
- metadata_contention
- lock_contention
- network_io_bottleneck
- serialized_io
- checkpointing_overhead
- compute_bound
- io_interference
- data_skew
- staging_inefficiency

""" + RESPONSE_FORMAT + """

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
    if strategy == "chain_of_thought":
        return COT_PROMPT_TEMPLATE.format(snapshot_json=snapshot_json)
    if strategy == "few_shot":
        return build_few_shot_prompt(snapshot)
    return USER_PROMPT_TEMPLATE.format(snapshot_json=snapshot_json)


def rule_based_diagnose(snapshot: dict) -> dict:
    exe     = snapshot.get("execution", {})
    io      = snapshot.get("io_metrics", {})
    res     = snapshot.get("resource_utilization", {})
    storage = snapshot.get("storage", {})

    io_time_ratio      = exe.get("io_time_ratio", 0)
    cpu_util           = res.get("cpu_util_pct", 0)
    bw_util            = io.get("bw_utilization_ratio", 0)
    network_util       = res.get("network_util_ratio", 0)
    avg_io_size_kb     = io.get("avg_io_size_kb", 1024)
    iops               = io.get("iops", 0)
    read_bw            = io.get("read_bw_mb_s", 0)
    write_bw           = io.get("write_bw_mb_s", 0)
    parallelism        = exe.get("parallelism", 999)
    checkpoint_size_mb = exe.get("checkpoint_size_mb", 0)
    shared             = storage.get("shared", False)
    stor_type          = storage.get("type", "")
    competing_jobs     = exe.get("competing_jobs", 0)
    imbalance_ratio    = exe.get("io_imbalance_ratio", 1.0)
    data_staged        = exe.get("data_staged", True)   # absent → assume staged
    latency_ms         = exe.get("remote_access_latency_ms", 0)

    REMOTE_TYPES = {"s3", "hdfs", "nfs", "ceph"}

    # Priority-ordered rules
    if cpu_util > 70 and io_time_ratio < 0.3:
        bottleneck = "compute_bound"
    elif network_util > 0.8 and io_time_ratio > 0.5:
        bottleneck = "network_io_bottleneck"
    elif data_staged is False and stor_type in REMOTE_TYPES and latency_ms > 30:
        bottleneck = "staging_inefficiency"
    elif competing_jobs > 2 and bw_util < 0.6 and io_time_ratio > 0.5:
        bottleneck = "io_interference"
    elif imbalance_ratio > 2.5 and bw_util < 0.65 and io_time_ratio > 0.5:
        bottleneck = "data_skew"
    elif bw_util > 0.85 and io_time_ratio > 0.5 and read_bw > write_bw:
        bottleneck = "read_bandwidth_saturation"
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
            f"competing_jobs: {competing_jobs}",
            f"io_imbalance_ratio: {imbalance_ratio}",
            f"data_staged: {data_staged}",
        ],
        "explanation": "Rule-based classification using metric thresholds.",
    }


def diagnose(snapshot: dict, model: str = "gpt-4.1-mini", strategy: str = "zero_shot") -> dict:
    if strategy == "rule_based":
        result = rule_based_diagnose(snapshot)
        result["duration_s"] = 0.0
        result["token_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return result

    prompt = build_prompt(snapshot, strategy=strategy)

    t0 = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    duration_s = time.time() - t0

    raw = response.choices[0].message.content
    result = json.loads(raw)
    result["duration_s"] = round(duration_s, 3)
    result["token_usage"] = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
    return result


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_snapshots(snapshots_dir: str = "data/snapshots") -> list[dict]:
    snapshots_dir = os.path.join(ROOT, snapshots_dir)
    paths = sorted(glob.glob(os.path.join(snapshots_dir, "snap_*.json")))
    snapshots = []
    for p in paths:
        with open(p) as f:
            snapshots.append(json.load(f))
    return snapshots


FEW_SHOT_IDS = {ex["id"] for ex, _ in FEW_SHOT_EXAMPLES}


def _flush_and_snapshot() -> dict:
    """Force GC, then capture baseline CPU/memory for the current process."""
    gc.collect()
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # first call initialises the counter; returns 0.0
    return {
        "proc": proc,
        "mem_start_mb": proc.memory_info().rss / 1024 / 1024,
    }


def evaluate(snapshots: list[dict], model: str = "gpt-4.1-mini", strategy: str = "zero_shot") -> dict:
    results = []

    ctx = _flush_and_snapshot()
    start_time = datetime.datetime.now()

    # Exclude few-shot examples from their own evaluation to avoid data leakage
    if strategy == "few_shot":
        eval_snaps = [s for s in snapshots if s.get("id") not in FEW_SHOT_IDS]
        print(f"  (few_shot: excluding {len(snapshots) - len(eval_snaps)} example snapshots from evaluation)")
    else:
        eval_snaps = snapshots

    for i, snap in enumerate(eval_snaps, 1):
        snap_id = snap.get("id", f"snap_{i:03d}")
        gt = snap["annotation"]

        print(f"[{i:2d}/{len(eval_snaps)}] {snap_id} ...", end=" ", flush=True)

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
            "duration_s": prediction.get("duration_s", 0.0),
            "token_usage": prediction.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        })

    end_time = datetime.datetime.now()
    total_duration_s = (end_time - start_time).total_seconds()

    mem_end_mb = ctx["proc"].memory_info().rss / 1024 / 1024
    mem_peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB → MB on Linux
    cpu_pct = ctx["proc"].cpu_percent(interval=None)

    resource_usage = {
        "cpu_pct": round(cpu_pct, 1),
        "mem_start_mb": round(ctx["mem_start_mb"], 1),
        "mem_end_mb": round(mem_end_mb, 1),
        "mem_peak_mb": round(mem_peak_mb, 1),
    }

    summary = _compute_summary(results)
    return {
        "model": model,
        "strategy": strategy,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_s": round(total_duration_s, 1),
        "resource_usage": resource_usage,
        "results": results,
        "summary": summary,
        "num_evaluated": len(eval_snaps),
    }


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

    n = len(results)
    total_prompt  = sum(r["token_usage"]["prompt_tokens"]     for r in results)
    total_compl   = sum(r["token_usage"]["completion_tokens"] for r in results)
    total_tok     = sum(r["token_usage"]["total_tokens"]      for r in results)
    total_dur     = sum(r["duration_s"]                       for r in results)

    # Per-class token and timing aggregation (keyed by ground-truth class)
    class_buckets: dict[str, list] = {c: [] for c in classes}
    for r in results:
        class_buckets[r["ground_truth"]].append(r)

    per_class_tokens = {}
    for c, bucket in class_buckets.items():
        m = len(bucket)
        if m == 0:
            continue
        c_prompt = sum(r["token_usage"]["prompt_tokens"]     for r in bucket)
        c_compl  = sum(r["token_usage"]["completion_tokens"] for r in bucket)
        c_tok    = sum(r["token_usage"]["total_tokens"]      for r in bucket)
        c_dur    = sum(r["duration_s"]                       for r in bucket)
        per_class_tokens[c] = {
            "count":                      m,
            "total_tokens":               c_tok,
            "avg_prompt_tokens":          round(c_prompt / m, 1),
            "avg_completion_tokens":      round(c_compl  / m, 1),
            "avg_total_tokens":           round(c_tok    / m, 1),
            "avg_duration_s":             round(c_dur    / m, 3),
        }

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 3),
        "macro_f1": round(macro_f1, 3),
        "per_class": per_class,
        "tokens": {
            "total_prompt":               total_prompt,
            "total_completion":           total_compl,
            "total":                      total_tok,
            "avg_prompt_per_snapshot":    round(total_prompt / n, 1) if n else 0,
            "avg_completion_per_snapshot":round(total_compl  / n, 1) if n else 0,
            "avg_total_per_snapshot":     round(total_tok    / n, 1) if n else 0,
        },
        "per_class_tokens": per_class_tokens,
        "total_api_time_s": round(total_dur, 1),
    }


def save_results(run: dict, results_dir: str = "data/results") -> str:
    results_dir = os.path.join(ROOT, results_dir)
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run['model']}_{run['strategy']}_{ts}.json"
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(run, f, indent=2)
    return path


def print_summary(run: dict) -> None:
    summary = run["summary"]
    res     = run.get("resource_usage", {})
    tok     = summary.get("tokens", {})

    print("\n=== Summary ===")
    print(f"Accuracy : {summary['accuracy']:.1%}  ({summary['correct']}/{summary['total']})")
    print(f"Macro F1 : {summary['macro_f1']:.3f}")

    print("\nPer-class F1:")
    for cls, metrics in summary["per_class"].items():
        print(f"  {cls:<35} P={metrics['precision']:.2f}  R={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")

    print("\n=== Timing ===")
    print(f"  Wall-clock duration  : {run.get('duration_s', 0):.1f} s")
    print(f"  Total API time       : {summary.get('total_api_time_s', 0):.1f} s")
    print(f"  Start                : {run.get('start_time', 'n/a')}")
    print(f"  End                  : {run.get('end_time',   'n/a')}")

    if tok:
        print("\n=== Token Usage (overall) ===")
        print(f"  Total prompt tokens      : {tok.get('total_prompt', 0):,}")
        print(f"  Total completion tokens  : {tok.get('total_completion', 0):,}")
        print(f"  Total tokens             : {tok.get('total', 0):,}")
        print(f"  Avg prompt / snapshot    : {tok.get('avg_prompt_per_snapshot', 0):.1f}")
        print(f"  Avg completion / snapshot: {tok.get('avg_completion_per_snapshot', 0):.1f}")
        print(f"  Avg total / snapshot     : {tok.get('avg_total_per_snapshot', 0):.1f}")

    pct = summary.get("per_class_tokens", {})
    if pct:
        print("\n=== Token Usage per Bottleneck Class ===")
        header = f"  {'Class':<35} {'N':>4}  {'Avg total':>9}  {'Avg prompt':>10}  {'Avg compl':>9}  {'Avg latency':>11}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        # Sort by avg_total_tokens descending so the most expensive class is at the top
        for c, m in sorted(pct.items(), key=lambda kv: kv[1]["avg_total_tokens"], reverse=True):
            print(
                f"  {c:<35} {m['count']:>4}  "
                f"{m['avg_total_tokens']:>9.1f}  "
                f"{m['avg_prompt_tokens']:>10.1f}  "
                f"{m['avg_completion_tokens']:>9.1f}  "
                f"{m['avg_duration_s']:>10.3f}s"
            )

    if res:
        print("\n=== Resource Usage ===")
        print(f"  CPU (process, whole run) : {res.get('cpu_pct', 0):.1f} %")
        print(f"  Memory start (RSS)       : {res.get('mem_start_mb', 0):.1f} MB")
        print(f"  Memory end   (RSS)       : {res.get('mem_end_mb', 0):.1f} MB")
        print(f"  Memory peak  (RSS)       : {res.get('mem_peak_mb', 0):.1f} MB")


class Tee:
    """Write to both stdout and a file simultaneously."""
    def __init__(self, file_path: str):
        import sys
        self._file = open(file_path, "w")
        self._stdout = sys.stdout
        import sys as _sys
        _sys.stdout = self

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        import sys
        sys.stdout = self._stdout
        self._file.close()


if __name__ == "__main__":
    import sys

    MODEL = "gpt-4.1-mini"
    STRATEGIES = ["rule_based", "classify_only", "zero_shot", "few_shot", "chain_of_thought"] 

    os.makedirs(os.path.join(ROOT, "data/results"), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(ROOT, f"data/results/run_{ts}.log")
    tee = Tee(log_path)

    print(f"Loading snapshots...")
    snapshots = load_snapshots()
    print(f"Loaded {len(snapshots)} snapshots.\n")
    print(f"Log file: {log_path}\n")

    for strategy in STRATEGIES:
        label = f"{MODEL} / {strategy}" if strategy != "rule_based" else "rule_based (no LLM)"
        print(f"{'='*55}")
        print(f"Strategy: {label}")
        print(f"{'='*55}")
        run = evaluate(snapshots, model=MODEL, strategy=strategy)
        print_summary(run)
        path = save_results(run)
        print(f"\nResults saved to: {path}\n")

    tee.close()
