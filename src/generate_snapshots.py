#!/usr/bin/env python3
"""Generate synthetic HPC workflow snapshots for LLM bottleneck detection dataset.
Snapshots snap_051 to snap_385 (335 total).
"""

import json
import os
import random

random.seed(42)

OUTPUT_DIR = "data/snapshots"

WORKFLOWS = [
    {"name": "1000_genomes",  "type": "genomics",            "num_stages": 5},
    {"name": "montage",       "type": "astronomy",            "num_stages": 7},
    {"name": "epigenomics",   "type": "genomics",             "num_stages": 6},
    {"name": "sipht",         "type": "bioinformatics",       "num_stages": 8},
    {"name": "cybershake",    "type": "seismology",           "num_stages": 4},
    {"name": "ligo",          "type": "gravitational_waves",  "num_stages": 5},
    {"name": "blast",         "type": "genomics",             "num_stages": 3},
    {"name": "namd",          "type": "molecular_dynamics",   "num_stages": 6},
    {"name": "wrf",           "type": "climate",              "num_stages": 5},
    {"name": "swift_t",       "type": "data_analysis",        "num_stages": 9},
    {"name": "pegasus_wms",   "type": "data_processing",      "num_stages": 7},
    {"name": "spark_etl",     "type": "etl",                  "num_stages": 4},
]

READ_STAGES  = ["load_data", "fetch_input", "read_checkpoint", "preprocess",
                "data_ingestion", "scan_catalog", "load_reference", "restore_state"]
WRITE_STAGES = ["output_results", "write_checkpoint", "save_output", "store_results",
                "dump_state", "flush_buffer", "archive_data", "commit_results"]
RW_STAGES    = ["transform", "merge", "filter", "sort", "join", "aggregate", "reduce"]

LOCAL_STORAGE  = ["beegfs", "lustre", "gpfs", "pvfs2", "wekafs"]
REMOTE_STORAGE = ["hdfs", "s3", "nfs", "ceph"]
ALL_STORAGE    = LOCAL_STORAGE + REMOTE_STORAGE


def ri(lo, hi):         return random.randint(lo, hi)
def rf(lo, hi, d=2):   return round(random.uniform(lo, hi), d)
def pick(lst):          return random.choice(lst)


def make_workflow():
    return pick(WORKFLOWS).copy()


def make_stage(op):
    names = {"read": READ_STAGES, "write": WRITE_STAGES}.get(op, RW_STAGES)
    order = ri(1, 6)
    preds = [pick(READ_STAGES)] if order > 1 else ["initial_data"]
    return {"name": pick(names), "order": order, "predecessors": preds, "operation": op}


# ───────────────────────────── NEW BOTTLENECK TYPES ─────────────────────────

def gen_read_bandwidth_saturation(snap_id):
    num_nodes  = ri(8, 64)
    num_tasks  = num_nodes * ri(4, 20)
    peak_bw    = pick([100, 150, 200, 300, 400, 500])
    bw_util    = rf(0.88, 0.99)
    read_bw    = round(peak_bw * bw_util, 1)
    avg_io_kb  = pick([256, 512, 1024, 2048, 4096])
    iops       = max(1, round(read_bw * 1024 / avg_io_kb))
    agg_mb     = round(num_tasks * avg_io_kb / 1024)
    op_count   = round(agg_mb * 1024 / avg_io_kb)
    io_ratio   = rf(0.75, 0.92)
    total_t    = ri(100, 800)
    io_t       = round(total_t * io_ratio)
    cpu        = ri(5, 20)
    mem        = ri(20, 55)
    net_peak   = pick([1000, 5000, 10000, 25000])
    net_bw     = round(read_bw * rf(0.8, 1.05), 1)
    net_util   = round(net_bw / net_peak, 3)
    stor       = pick(LOCAL_STORAGE)
    stripes    = pick([4, 8, 16, 32])

    expl = (
        f"Read bandwidth of {read_bw} MB/s reaches {round(bw_util*100)}% of the {stor.upper()} "
        f"storage peak ({peak_bw} MB/s), saturating the filesystem on the read path. "
        f"CPU utilization of only {cpu}% confirms compute cores are stalled waiting on data. "
        f"The io_time_ratio of {io_ratio} confirms that I/O dominates the stage duration."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage("read"),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": 0,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": round(bw_util, 2),
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.75, 0.98),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "read_bandwidth_saturation", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"bw_utilization_ratio: {round(bw_util, 2)}",
                f"read_bw_mb_s: {read_bw}",
                f"peak_storage_bw_mb_s: {peak_bw}",
                f"cpu_util_pct: {cpu}",
                f"io_time_ratio: {io_ratio}",
            ],
        },
    }


def gen_io_interference(snap_id):
    num_nodes    = ri(4, 32)
    num_tasks    = num_nodes * ri(2, 10)
    competing    = ri(3, 15)
    peak_bw      = pick([200, 300, 400, 500])
    interf_frac  = rf(0.30, 0.65)
    obs_bw       = round(peak_bw * interf_frac * rf(0.70, 0.95), 1)
    bw_util      = round(obs_bw / peak_bw, 2)
    op           = pick(["read", "write"])
    avg_io_kb    = pick([256, 512, 1024, 2048])
    iops         = max(1, round(obs_bw * 1024 / avg_io_kb))
    agg_mb       = ri(500, 20000)
    op_count     = round(agg_mb * 1024 / avg_io_kb)
    io_ratio     = rf(0.65, 0.88)
    total_t      = ri(150, 1200)
    io_t         = round(total_t * io_ratio)
    cpu          = ri(10, 30)
    mem          = ri(25, 60)
    net_peak     = pick([1000, 5000, 10000])
    net_bw       = round(obs_bw * rf(0.90, 1.05), 1)
    net_util     = round(net_bw / net_peak, 3)
    stor         = pick(LOCAL_STORAGE)
    stripes      = pick([4, 8, 16])
    read_bw      = obs_bw if op == "read" else 0
    write_bw     = obs_bw if op == "write" else 0

    expl = (
        f"Despite the storage system having a peak of {peak_bw} MB/s, the workflow only observes "
        f"{obs_bw} MB/s ({round(bw_util*100)}% apparent utilization) due to {competing} competing jobs "
        f"concurrently saturating the shared {stor.upper()} filesystem. "
        f"CPU utilization of {cpu}% confirms tasks are stalled on I/O rather than being compute-limited. "
        f"The degradation is caused by external interference, not the workflow's own I/O pattern."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
            "competing_jobs": competing,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.50, 0.85),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "io_interference", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"competing_jobs: {competing}",
                f"bw_utilization_ratio: {bw_util}",
                f"observed_bw_mb_s: {obs_bw}",
                f"peak_storage_bw_mb_s: {peak_bw}",
                f"cpu_util_pct: {cpu}",
            ],
        },
    }


def gen_data_skew(snap_id):
    num_nodes      = ri(16, 128)
    num_tasks      = num_nodes * ri(4, 20)
    imbalance_r    = rf(3.0, 10.0, 1)
    hot_nodes      = ri(1, max(1, num_nodes // 10))
    hot_pct        = round(hot_nodes / num_nodes * 100, 1)
    peak_bw        = pick([200, 400, 600, 800])
    avg_bw         = round(peak_bw * rf(0.30, 0.60), 1)
    bw_util        = round(avg_bw / peak_bw, 2)
    op             = pick(["read", "write"])
    avg_io_kb      = pick([128, 256, 512, 1024])
    iops           = max(1, round(avg_bw * 1024 / avg_io_kb))
    agg_mb         = ri(1000, 40000)
    op_count       = round(agg_mb * 1024 / avg_io_kb)
    io_ratio       = rf(0.65, 0.88)
    total_t        = ri(200, 1500)
    io_t           = round(total_t * io_ratio)
    cpu            = ri(15, 45)
    mem            = ri(30, 65)
    net_peak       = pick([5000, 10000, 25000])
    net_bw         = round(avg_bw * rf(0.80, 1.10), 1)
    net_util       = round(net_bw / net_peak, 3)
    stor           = pick(LOCAL_STORAGE)
    stripes        = pick([4, 8, 16])
    read_bw        = avg_bw if op == "read" else 0
    write_bw       = avg_bw if op == "write" else 0

    expl = (
        f"I/O load is highly skewed: {hot_nodes} of {num_nodes} nodes ({hot_pct}% of the cluster) "
        f"handle a disproportionate share of I/O, with an imbalance ratio of {imbalance_r}x. "
        f"These hot nodes saturate their local I/O paths while the rest sit largely idle, "
        f"pulling aggregate bandwidth ({avg_bw} MB/s) well below the system peak ({peak_bw} MB/s). "
        f"The bottleneck is structural: data partitioning assigns unequal volumes across nodes."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
            "io_imbalance_ratio": imbalance_r, "hot_node_count": hot_nodes,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.50, 0.85),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "data_skew", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"io_imbalance_ratio: {imbalance_r}",
                f"hot_node_count: {hot_nodes}",
                f"hot_node_pct: {hot_pct}%",
                f"bw_utilization_ratio: {bw_util}",
                f"io_time_ratio: {io_ratio}",
            ],
        },
    }


def gen_staging_inefficiency(snap_id):
    num_nodes   = ri(8, 64)
    num_tasks   = num_nodes * ri(4, 16)
    stor        = pick(REMOTE_STORAGE)
    latency_ms  = ri(50, 500)
    peak_bw     = pick([200, 400, 600])
    stage_eff   = rf(0.20, 0.55)
    obs_bw      = round(peak_bw * stage_eff, 1)
    bw_util     = round(obs_bw / peak_bw, 2)
    avg_io_kb   = pick([128, 256, 512, 1024])
    iops        = max(1, round(obs_bw * 1024 / avg_io_kb))
    agg_mb      = ri(2000, 80000)
    op_count    = round(agg_mb * 1024 / avg_io_kb)
    io_ratio    = rf(0.72, 0.92)
    total_t     = ri(300, 2000)
    io_t        = round(total_t * io_ratio)
    cpu         = ri(8, 25)
    mem         = ri(20, 50)
    net_peak    = pick([1000, 5000, 10000])
    net_bw      = round(obs_bw * rf(0.85, 1.05), 1)
    net_util    = round(net_bw / net_peak, 3)
    op          = pick(["read", "write"])
    read_bw     = obs_bw if op == "read" else 0
    write_bw    = obs_bw if op == "write" else 0
    tier        = pick(["burst_buffer", "local_nvme", "node_local_ssd"])
    stripes     = pick([1, 2, 4])

    expl = (
        f"Data was not pre-staged to a fast local storage tier ({tier}) before execution began. "
        f"Tasks access {stor.upper()} directly, which carries a remote latency of {latency_ms} ms, "
        f"yielding only {obs_bw} MB/s effective bandwidth ({round(bw_util*100)}% of the {peak_bw} MB/s peak). "
        f"CPU utilization of {cpu}% confirms tasks spend most time waiting on remote I/O "
        f"rather than performing useful computation."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
            "data_staged": False, "intended_staging_tier": tier,
            "remote_access_latency_ms": latency_ms,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.40, 0.80),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "staging_inefficiency", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                "data_staged: false",
                f"remote_access_latency_ms: {latency_ms}",
                f"bw_utilization_ratio: {bw_util}",
                f"storage_type: {stor}",
                f"cpu_util_pct: {cpu}",
                f"io_time_ratio: {io_ratio}",
            ],
        },
    }


# ───────────────────────────── EXISTING BOTTLENECK TYPES ────────────────────

def gen_compute_bound(snap_id):
    num_nodes  = ri(8, 128)
    num_tasks  = num_nodes * ri(4, 16)
    cpu        = ri(78, 98)
    io_ratio   = rf(0.05, 0.22)
    total_t    = ri(200, 2000)
    io_t       = round(total_t * io_ratio)
    peak_bw    = pick([200, 400, 600])
    bw_util    = rf(0.05, 0.25)
    eff_bw     = round(peak_bw * bw_util, 1)
    op         = pick(["read", "write"])
    avg_io_kb  = pick([64, 128, 256, 512])
    iops       = max(1, round(eff_bw * 1024 / avg_io_kb))
    agg_mb     = ri(100, 5000)
    op_count   = round(agg_mb * 1024 / avg_io_kb)
    mem        = ri(40, 85)
    net_peak   = pick([1000, 5000, 10000])
    net_bw     = round(eff_bw * rf(0.50, 1.20), 1)
    net_util   = round(net_bw / net_peak, 3)
    stor       = pick(ALL_STORAGE)
    stripes    = pick([4, 8, 16])
    read_bw    = eff_bw if op == "read" else 0
    write_bw   = eff_bw if op == "write" else 0

    expl = (
        f"CPU utilization of {cpu}% confirms that computation is the dominant bottleneck. "
        f"The io_time_ratio of {io_ratio} shows that I/O accounts for a small fraction of the total stage time. "
        f"Bandwidth utilization of {round(bw_util*100)}% leaves significant I/O headroom unused, "
        f"indicating the system is not I/O-limited."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": round(bw_util, 2),
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.50, 0.90),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": pick([True, False])},
        "annotation": {
            "bottleneck": "compute_bound", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"cpu_util_pct: {cpu}",
                f"io_time_ratio: {io_ratio}",
                f"bw_utilization_ratio: {round(bw_util, 2)}",
            ],
        },
    }


def gen_network_io_bottleneck(snap_id):
    num_nodes  = ri(8, 64)
    num_tasks  = num_nodes * ri(4, 16)
    net_peak   = pick([1000, 2500, 5000])
    net_util   = rf(0.85, 0.99)
    net_bw     = round(net_peak * net_util, 1)
    peak_bw    = pick([500, 800, 1000])
    eff_bw     = min(net_bw, round(peak_bw * rf(0.50, 0.80), 1))
    bw_util    = round(eff_bw / peak_bw, 2)
    op         = pick(["read", "write"])
    avg_io_kb  = pick([512, 1024, 2048, 4096])
    iops       = max(1, round(eff_bw * 1024 / avg_io_kb))
    agg_mb     = ri(5000, 100000)
    op_count   = round(agg_mb * 1024 / avg_io_kb)
    io_ratio   = rf(0.72, 0.92)
    total_t    = ri(200, 1500)
    io_t       = round(total_t * io_ratio)
    cpu        = ri(8, 25)
    mem        = ri(25, 55)
    stor       = pick(REMOTE_STORAGE)
    stripes    = pick([1, 2, 4])
    read_bw    = eff_bw if op == "read" else 0
    write_bw   = eff_bw if op == "write" else 0

    expl = (
        f"Network bandwidth of {net_bw} MB/s reaches {round(net_util*100)}% of the "
        f"network peak ({net_peak} MB/s), making the interconnect the primary bottleneck. "
        f"Data is accessed from {stor.upper()}, requiring all transfers to traverse the network. "
        f"CPU utilization of {cpu}% confirms tasks are stalled waiting on data transfer "
        f"rather than performing computation."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.60, 0.90),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": round(net_util, 2),
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "network_io_bottleneck", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"network_util_ratio: {round(net_util, 2)}",
                f"network_bw_mb_s: {net_bw}",
                f"network_peak_bw_mb_s: {net_peak}",
                f"cpu_util_pct: {cpu}",
                f"io_time_ratio: {io_ratio}",
                f"storage_type: {stor}",
            ],
        },
    }


def gen_storage_bandwidth_saturation(snap_id):
    num_nodes  = ri(8, 64)
    num_tasks  = num_nodes * ri(4, 20)
    peak_bw    = pick([100, 150, 200, 300, 400, 500])
    bw_util    = rf(0.88, 0.99)
    eff_bw     = round(peak_bw * bw_util, 1)
    op         = pick(["write", "read"])
    avg_io_kb  = pick([512, 1024, 2048, 4096])
    iops       = max(1, round(eff_bw * 1024 / avg_io_kb))
    agg_mb     = ri(5000, 100000)
    op_count   = round(agg_mb * 1024 / avg_io_kb)
    io_ratio   = rf(0.78, 0.95)
    total_t    = ri(100, 800)
    io_t       = round(total_t * io_ratio)
    cpu        = ri(5, 18)
    mem        = ri(30, 60)
    net_peak   = pick([1000, 5000, 10000])
    net_bw     = round(eff_bw * rf(0.85, 1.05), 1)
    net_util   = round(net_bw / net_peak, 3)
    stor       = pick(LOCAL_STORAGE)
    stripes    = pick([4, 8, 16, 32])
    read_bw    = eff_bw if op == "read" else 0
    write_bw   = eff_bw if op == "write" else 0

    expl = (
        f"{'Write' if op == 'write' else 'Read'} bandwidth of {eff_bw} MB/s reaches "
        f"{round(bw_util*100)}% of the {stor.upper()} storage peak ({peak_bw} MB/s), "
        f"saturating the filesystem. CPU utilization of only {cpu}% confirms compute cores are "
        f"idle waiting on I/O. The io_time_ratio of {io_ratio} further shows that I/O dominates "
        f"the stage duration."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": round(bw_util, 2),
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.75, 0.98),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "storage_bandwidth_saturation", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"bw_utilization_ratio: {round(bw_util, 2)}",
                f"{'write' if op == 'write' else 'read'}_bw_mb_s: {eff_bw}",
                f"peak_storage_bw_mb_s: {peak_bw}",
                f"cpu_util_pct: {cpu}",
                f"io_time_ratio: {io_ratio}",
            ],
        },
    }


def gen_metadata_contention(snap_id):
    num_nodes  = ri(16, 128)
    num_tasks  = num_nodes * ri(8, 32)
    avg_io_kb  = pick([1, 2, 4, 8, 16, 32])
    iops       = ri(5000, 80000)
    peak_bw    = pick([200, 400, 600])
    bw_util    = round(min((iops * avg_io_kb / 1024) / peak_bw, 0.38), 2)
    eff_bw     = round(peak_bw * bw_util, 1)
    agg_mb     = round(iops * avg_io_kb / 1024)
    op_count   = iops * ri(50, 300)
    io_ratio   = rf(0.72, 0.92)
    total_t    = ri(200, 1500)
    io_t       = round(total_t * io_ratio)
    cpu        = ri(15, 35)
    mem        = ri(25, 55)
    net_peak   = pick([1000, 5000, 10000])
    net_bw     = round(eff_bw * rf(0.70, 1.10), 1)
    net_util   = round(net_bw / net_peak, 3)
    op         = pick(["read", "write"])
    stor       = pick(LOCAL_STORAGE)
    stripes    = ri(1, 4)
    read_bw    = eff_bw if op == "read" else 0
    write_bw   = eff_bw if op == "write" else 0

    expl = (
        f"The stage issues {iops} IOPS with an average I/O size of only {avg_io_kb} KB, "
        f"overwhelming the metadata server with open/close/stat operations. "
        f"Despite the high operation count, bandwidth utilization is only {round(bw_util*100)}% "
        f"({eff_bw} MB/s of {peak_bw} MB/s peak) because data transfer is not the bottleneck — "
        f"the metadata server queue is. The io_time_ratio of {io_ratio} confirms I/O dominates stage time."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.10, 0.40),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "metadata_contention", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"iops: {iops}",
                f"avg_io_size_kb: {avg_io_kb}",
                f"bw_utilization_ratio: {bw_util}",
                f"io_time_ratio: {io_ratio}",
                f"op_count: {op_count}",
            ],
        },
    }


def gen_serialized_io(snap_id):
    num_nodes  = ri(16, 128)
    num_tasks  = num_nodes * ri(4, 20)
    peak_bw    = pick([200, 400, 600])
    single_bw  = round(peak_bw * rf(0.08, 0.28), 1)
    bw_util    = round(single_bw / peak_bw, 2)
    op         = pick(["write", "read"])
    avg_io_kb  = pick([512, 1024, 2048])
    iops       = max(1, round(single_bw * 1024 / avg_io_kb))
    agg_mb     = ri(1000, 30000)
    op_count   = round(agg_mb * 1024 / avg_io_kb)
    io_ratio   = rf(0.78, 0.95)
    total_t    = ri(300, 2000)
    io_t       = round(total_t * io_ratio)
    cpu        = ri(5, 20)
    mem        = ri(15, 40)
    net_peak   = pick([1000, 5000, 10000])
    net_bw     = round(single_bw * rf(0.90, 1.10), 1)
    net_util   = round(net_bw / net_peak, 3)
    stor       = pick(LOCAL_STORAGE)
    stripes    = pick([4, 8, 16])
    read_bw    = single_bw if op == "read" else 0
    write_bw   = single_bw if op == "write" else 0

    expl = (
        f"I/O parallelism is forced to 1 — only a single {'writer' if op == 'write' else 'reader'} "
        f"operates at a time despite {num_tasks} tasks available. "
        f"This limits effective bandwidth to {single_bw} MB/s ({round(bw_util*100)}% of the "
        f"{peak_bw} MB/s storage peak) since parallel filesystem capacity goes unused. "
        f"The remaining {num_tasks - 1} tasks wait idle while serialized I/O completes."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage(op),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": 1,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
        },
        "io_metrics": {
            "read_bw_mb_s": read_bw, "write_bw_mb_s": write_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.80, 1.00),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "serialized_io", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                "parallelism: 1",
                f"num_tasks: {num_tasks}",
                f"bw_utilization_ratio: {bw_util}",
                f"io_time_ratio: {io_ratio}",
                f"{'write' if op == 'write' else 'read'}_bw_mb_s: {single_bw}",
            ],
        },
    }


def gen_checkpointing_overhead(snap_id):
    num_nodes    = ri(16, 256)
    num_tasks    = num_nodes * ri(4, 16)
    ckpt_mb      = ri(2000, 100000)
    ckpt_intv    = ri(60, 600)
    num_ckpts    = ri(3, 20)
    peak_bw      = pick([200, 400, 600, 800])
    ckpt_bw      = round(peak_bw * rf(0.60, 0.92), 1)
    bw_util      = round(ckpt_bw / peak_bw, 2)
    avg_io_kb    = pick([1024, 2048, 4096])
    iops         = max(1, round(ckpt_bw * 1024 / avg_io_kb))
    agg_mb       = ckpt_mb * num_ckpts
    op_count     = round(agg_mb * 1024 / avg_io_kb)
    io_ratio     = rf(0.65, 0.88)
    total_t      = ri(500, 3000)
    io_t         = round(total_t * io_ratio)
    cpu          = ri(10, 30)
    mem          = ri(35, 70)
    net_peak     = pick([5000, 10000, 25000])
    net_bw       = round(ckpt_bw * rf(0.85, 1.05), 1)
    net_util     = round(net_bw / net_peak, 3)
    stor         = pick(LOCAL_STORAGE)
    stripes      = pick([8, 16, 32])

    expl = (
        f"The stage performs {num_ckpts} checkpoint writes of {ckpt_mb} MB each "
        f"at {ckpt_intv}s intervals, contributing {agg_mb} MB of total checkpoint I/O. "
        f"These periodic bursts saturate the storage at {ckpt_bw} MB/s ({round(bw_util*100)}% of peak), "
        f"dominating the stage I/O budget. The io_time_ratio of {io_ratio} reflects cumulative "
        f"time flushing checkpoints rather than performing productive computation."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage("write"),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
            "checkpoint_size_mb": ckpt_mb, "checkpoint_interval_s": ckpt_intv,
            "num_checkpoints": num_ckpts,
        },
        "io_metrics": {
            "read_bw_mb_s": 0, "write_bw_mb_s": ckpt_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": bw_util,
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.85, 1.00),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "checkpointing_overhead", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"checkpoint_size_mb: {ckpt_mb}",
                f"checkpoint_interval_s: {ckpt_intv}",
                f"num_checkpoints: {num_ckpts}",
                f"aggregate_size_mb: {agg_mb}",
                f"io_time_ratio: {io_ratio}",
            ],
        },
    }


def gen_lock_contention(snap_id):
    num_nodes    = ri(16, 128)
    num_tasks    = num_nodes * ri(8, 32)
    peak_bw      = pick([200, 400, 600])
    bw_util      = rf(0.05, 0.30)
    eff_bw       = round(peak_bw * bw_util, 1)
    avg_io_kb    = pick([256, 512, 1024])
    iops         = max(1, round(eff_bw * 1024 / avg_io_kb))
    agg_mb       = ri(500, 20000)
    op_count     = round(agg_mb * 1024 / avg_io_kb)
    io_ratio     = rf(0.72, 0.92)
    total_t      = ri(300, 2000)
    io_t         = round(total_t * io_ratio)
    cpu          = ri(5, 20)
    mem          = ri(20, 50)
    net_peak     = pick([1000, 5000, 10000])
    net_bw       = round(eff_bw * rf(0.80, 1.10), 1)
    net_util     = round(net_bw / net_peak, 3)
    stor         = pick(LOCAL_STORAGE)
    stripes      = pick([4, 8, 16])
    shared_files = ri(1, 10)

    expl = (
        f"{num_tasks} tasks compete for write locks on {shared_files} shared file(s) on the "
        f"shared {stor.upper()} filesystem. Lock acquisition serializes access, resulting in "
        f"only {eff_bw} MB/s effective bandwidth ({round(bw_util*100)}% of the {peak_bw} MB/s peak) "
        f"despite full task parallelism. CPU utilization of {cpu}% confirms tasks are stalled "
        f"waiting for lock release rather than performing computation."
    )
    return {
        "id": snap_id,
        "workflow": make_workflow(),
        "stage": make_stage("write"),
        "execution": {
            "num_nodes": num_nodes, "num_tasks": num_tasks, "parallelism": num_tasks,
            "total_time_s": total_t, "io_time_s": io_t, "compute_time_s": total_t - io_t,
            "io_time_ratio": io_ratio, "transfer_size_kb": avg_io_kb,
            "aggregate_size_mb": agg_mb, "op_count": op_count,
            "num_shared_files": shared_files,
        },
        "io_metrics": {
            "read_bw_mb_s": 0, "write_bw_mb_s": eff_bw,
            "peak_storage_bw_mb_s": peak_bw, "bw_utilization_ratio": round(bw_util, 2),
            "iops": iops, "avg_io_size_kb": avg_io_kb, "sequential_ratio": rf(0.50, 0.85),
        },
        "resource_utilization": {
            "cpu_util_pct": cpu, "memory_util_pct": mem,
            "network_bw_mb_s": net_bw, "network_peak_bw_mb_s": net_peak,
            "network_util_ratio": net_util,
        },
        "storage": {"type": stor, "stripe_count": stripes, "shared": True},
        "annotation": {
            "bottleneck": "lock_contention", "confidence": "high",
            "explanation": expl,
            "key_signals": [
                f"num_shared_files: {shared_files}",
                f"num_tasks: {num_tasks}",
                f"bw_utilization_ratio: {round(bw_util, 2)}",
                f"cpu_util_pct: {cpu}",
                f"io_time_ratio: {io_ratio}",
                "storage.shared: true",
            ],
        },
    }


# ───────────────────────────── GENERATION SCHEDULE ──────────────────────────

SCHEDULE = [
    (gen_read_bandwidth_saturation,   40),
    (gen_io_interference,             40),
    (gen_data_skew,                   40),
    (gen_staging_inefficiency,        40),
    (gen_compute_bound,               25),
    (gen_network_io_bottleneck,       25),
    (gen_storage_bandwidth_saturation,25),
    (gen_metadata_contention,         25),
    (gen_serialized_io,               25),
    (gen_checkpointing_overhead,      25),
    (gen_lock_contention,             25),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    snap_num = 51
    total = sum(c for _, c in SCHEDULE)

    for gen_fn, count in SCHEDULE:
        label = gen_fn.__name__.replace("gen_", "")
        print(f"  {label:<35} x{count}")
        for _ in range(count):
            snap_id = f"snap_{snap_num:03d}"
            snap = gen_fn(snap_id)
            path = os.path.join(OUTPUT_DIR, f"{snap_id}.json")
            with open(path, "w") as f:
                json.dump(snap, f, indent=2)
            snap_num += 1

    print(f"\nDone. {total} snapshots written: snap_051 to snap_{snap_num - 1:03d}.")


if __name__ == "__main__":
    main()
