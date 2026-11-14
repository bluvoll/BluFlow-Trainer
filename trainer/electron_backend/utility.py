"""Isolated CPU utility worker. Input is a session-owned JSON file."""

import json
import math
import sys
from pathlib import Path


def operate(kind, args):
    if kind == "alpha-inspect":
        from ..ui_core.lora_alpha import LoraFile

        item = LoraFile(Path(args["source"]))
        return dict(
            groups=item.groups(), modules=len(item.alphas), dora=item.dora_modules
        )
    if kind == "alpha":
        from ..ui_core.lora_alpha import BaseModel, LoraFile

        source = Path(args["source"]).resolve()
        target = Path(args["destination"]).resolve()
        if target == source or (target.exists() and not args.get("overwrite")):
            raise ValueError(
                "Choose a new output filename, or explicitly confirm overwriting an output copy."
            )
        item = LoraFile(source)
        mode, value = args.get("mode", "bake"), float(args.get("value", 1.5))
        if mode not in ("bake", "set", "dim") or not math.isfinite(value) or value < 0:
            raise ValueError("Choose a valid alpha mode and finite nonnegative value")
        notes = []
        if item.dora_modules and mode == "bake" and args.get("base"):
            patches, notes = item.dora_bake(value, BaseModel(Path(args["base"])))
            if not patches:
                raise ValueError("No DoRA modules match this base model")
            item.save_as(
                target,
                patches=patches,
                extra_metadata={
                    "ss_dora_baked_weight": str(value),
                    "ss_dora_bake_base": Path(args["base"]).name,
                },
            )
        else:

            def alpha(name, current, rank):
                return (
                    current * value
                    if mode == "bake"
                    else value
                    if mode == "set"
                    else float(rank)
                    if rank is not None
                    else current
                )

            item.save_as(target, new_alpha_for=alpha)
            if item.dora_modules:
                notes.append(
                    "DoRA alpha edits do not reproduce inference weight. Supply the training base model for exact strength baking."
                )
        return dict(path=str(target), notes=notes)
    if kind == "qkv":
        from ..tools.lora_qkv_convert import convert_file

        result = convert_file(
            args["source"],
            args.get("destination") or None,
            layout=args.get("layout", "fused"),
        )
    elif kind == "resize":
        from ..tools.lora_resize import resize_file

        result = resize_file(
            args["source"],
            args.get("destination") or None,
            args.get("rank"),
            progress=lambda done, total, name: print(
                f"{done}/{total} {name}", flush=True
            ),
            method=args.get("method", "sv_fro"),
            threshold=float(args.get("threshold", 0.99)),
        )
    elif kind == "recover":
        from ..tools.recover_config import recover_file

        result = recover_file(args["source"], args.get("destination") or None)
    else:
        raise ValueError("Unknown utility")
    return (
        dict(path=str(result[0]), report=result[1])
        if isinstance(result, tuple)
        else dict(path=str(result))
    )


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        result = operate(request["kind"], request["args"])
        print(
            "BLUFLOW_UTILITY_RESULT "
            + json.dumps(dict(result=result), default=str, ensure_ascii=True),
            flush=True,
        )
        return 0
    except Exception as exc:
        print(
            "BLUFLOW_UTILITY_RESULT "
            + json.dumps(dict(error=f"{type(exc).__name__}: {exc}"), ensure_ascii=True),
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
