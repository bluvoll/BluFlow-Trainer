"""Qt adapter for the shared desktop field definitions."""

from dataclasses import dataclass
from typing import Callable
from . import fields as F
from .widgets import RAW_GROUP_TITLES, TRANSFORMED_GROUP_TITLES
from ..ui_core import schema as shared


@dataclass(frozen=True)
class Spec:
    label: str
    tooltip: str
    make: Callable
    inline_label: bool = False


def adapt(spec):
    editor = spec.make()
    return Spec(
        spec.label,
        spec.tooltip,
        lambda: getattr(F, editor.kind)(*editor.args, **editor.kwargs),
        spec.inline_label,
    )


SPEC = {key: adapt(spec) for key, spec in shared.SPEC.items()}
LAYOUT = shared.LAYOUT
layout_keys = shared.layout_keys
COMPONENT_TOOLTIPS = shared.COMPONENT_TOOLTIPS
RAW_GROUP_TITLES.update(shared.RAW_GROUP_TITLES)
TRANSFORMED_GROUP_TITLES.update(shared.TRANSFORMED_GROUP_TITLES)
