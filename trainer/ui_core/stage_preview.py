"""On-demand dataset scans shared by both GUIs, never run on each keystroke."""
from pathlib import Path
import tempfile


def preview_stages(flat, world):
    from ..gui.bridge import dump_toml
    from ..training.config import load_config
    from ..tools.stage_plan import inspect_plan
    with tempfile.TemporaryDirectory(prefix='bluflow-stage-plan-') as directory:
        path = Path(directory) / 'preview.toml'
        path.write_text(dump_toml(flat), encoding='utf-8')
        return inspect_plan(load_config(path), world)
