import re
import math

# Parse fields independently: auxiliary losses and diagnostics can be inserted
# between them without making the entire step disappear from the graphs.
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
STEP_RE = re.compile(
    r"^(?P<unit>e|stage)(?P<epoch>\d+)\s+step\s+(?P<step>\d+)/(?P<total>\d+)"
    r"(?:\s+\((?P<pct>\d+)%\))?"
    rf"(?=[^\r\n]*?\bloss\s+(?P<loss>{NUMBER})(?=\s|$))"
    rf"(?=[^\r\n]*?\blr\s+(?P<lr>{NUMBER}(?:/{NUMBER})*)(?=\s|$))"
    rf"(?=[^\r\n]*?\s(?P<sit>{NUMBER})s/it(?=\s|$))"
    rf"(?=[^\r\n]*?\b(?:peak|peak_alloc)\s+(?P<peak>{NUMBER}(?:/{NUMBER})*)(?P<memory_unit>GiB|GB)(?=\s|$))"
    rf"(?:(?=[^\r\n]*?\(mse\s+(?P<mse>{NUMBER})\s+\+\s+hf\s+(?P<hf>{NUMBER})\)))?"
    rf"(?:(?=[^\r\n]*?\bot\s+(?P<ot>{NUMBER})(?=\s|$)))?"
    r"(?:(?=[^\r\n]*?\beta\s+(?P<eta>\S+)))?"
    r"[^\r\n]*$"
)

# Startup summary lines worth surfacing on the metrics tab rather than leaving in the console.
REPORT_RE = re.compile(
    r"^(dataset|steps|stage|mode|params|quant|flow|sched|compile|optim|hf loss)\s{2,}"
)


def parse_step(text):
    m = STEP_RE.match(text.strip())
    if not m:
        return None
    d = m.groupdict()
    step = int(d["step"])
    rec = {
        ("stage" if d["unit"] == "stage" else "epoch"): int(d["epoch"]),
        "step": step,
        "total": int(d["total"]),
        "loss": float(d["loss"]),
        # The graph plots the first group's LR; the schedule shape is identical across
        # groups (they differ by a constant ratio), so one curve describes all of them.
        "lr": float(d["lr"].split("/")[0]),
        "lrs": [float(v) for v in d["lr"].split("/") if v],
        "sit": float(d["sit"]),
        "peak": [float(p) for p in d["peak"].split("/") if p],
    }
    if d["memory_unit"] == "GB":
        rec["peak"] = [p * 1e9 / 2**30 for p in rec["peak"]]
    for key in ("allocated", "reserved", "device_used"):
        field = re.search(rf"\b{key}\s+({NUMBER}(?:/{NUMBER})*)GiB", text)
        if field:
            rec[key] = [float(v) for v in field[1].split("/")]
    if d["mse"] is not None:
        rec["mse"] = float(d["mse"])
        rec["hf"] = float(d["hf"])
    if d["ot"] is not None:
        rec["ot"] = float(d["ot"])
    if d["eta"] is not None:
        rec["eta"] = d["eta"]
    return rec


def compact_records(records, limit=60000):
    """Bound plot memory while retaining the whole run and local extrema."""
    if len(records) <= limit:
        return records
    stride = max(32, math.ceil(len(records) * 20 / limit))
    kept = []
    for start in range(0, len(records), stride):
        block = records[start : start + stride]
        indices = {0, len(block) - 1}
        for key in ("loss", "ema", "mse", "hf", "lr", "sit", "peak0", "peak1"):
            values = []
            for index, record in enumerate(block):
                value = (
                    (record.get("peak") or [])[int(key[-1])]
                    if key.startswith("peak")
                    and len(record.get("peak") or []) > int(key[-1])
                    else record.get(key)
                )
                if isinstance(value, (int, float)) and math.isfinite(value):
                    values.append((value, index))
            if values:
                indices.update((min(values)[1], max(values)[1]))
        kept.extend(block[index] for index in sorted(indices))
    return kept
