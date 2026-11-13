"""Serializable editor factories used by both desktop frontends."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Editor:
    kind: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)

    def describe(self):
        def serial(value):
            if isinstance(value, type):
                return value.__name__
            if isinstance(value, (tuple, list)):
                return [serial(v) for v in value]
            if isinstance(value, dict):
                return {k: serial(v) for k, v in value.items()}
            return value

        return dict(kind=self.kind, args=serial(self.args), kwargs=serial(self.kwargs))


KINDS = (
    "TextEditor",
    "OptTextEditor",
    "PathEditor",
    "IntEditor",
    "OptIntEditor",
    "ImageTargetEditor",
    "FloatEditor",
    "SciEditor",
    "BoolEditor",
    "TrainComponentEditor",
    "ChoiceEditor",
    "OptimizerEditor",
    "NumListEditor",
    "IntListEditor",
    "FloatListEditor",
    "StrListEditor",
    "CheckSetEditor",
    "BucketMapEditor",
    "WeightsEditor",
    "CanvasListEditor",
    "CurriculumEditor",
    "SubsetEditor",
    "BlockLRRangesEditor",
    "ResolutionStagesEditor",
)


def factory(kind):
    def make(*args, **kwargs):
        return Editor(kind, args, kwargs)

    return make


globals().update({kind: factory(kind) for kind in KINDS})
