"""Queue identity and immutable snapshot creation shared by the two desktops."""

from dataclasses import dataclass
import hashlib
from pathlib import Path


@dataclass
class QueueEntry:
    id: int
    source: Path | str
    snapshot: Path | str
    digest: str
    status: str = "Pending"
    error: str = ""


def snapshot_file(source, directory, entry_id, data=None):
    source = Path(source).resolve()
    if data is None:
        data = source.read_bytes()
    target = Path(directory) / f"{entry_id:04d}-{source.name}"
    target.write_bytes(data)
    return QueueEntry(entry_id, source, target, hashlib.sha256(data).hexdigest())
