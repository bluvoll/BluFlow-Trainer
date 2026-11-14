import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IS_WINDOWS = os.name == "nt"


def platform_path(value):
    """Normalize a user/config path for the host OS without resolving symlinks."""
    text = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not text:
        return ""
    if not IS_WINDOWS:
        text = text.replace("\\", "/")
    return os.path.normpath(text)


_STEM_EXTRA = set("-_.")
# Windows resolves these as DOS devices whatever the extension, so `CON.toml` is not a file --
# opening it succeeds and writes to the console. Case-insensitive, and still reserved with a
# suffix. ext4 does not care, but a config is meant to be portable between the two.
_WIN_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def safe_stem(name, fallback="gui_run"):
    """`train.run_name` -> a filename stem that is legal on both platforms.

    `run_name` is free text and reaches the filesystem three ways -- the config file, the
    checkpoint stem and the state directory -- so a name with a slash in it does not fail at save
    time, it fails several hours later at the first checkpoint. Windows additionally rejects
    `<>:"/\\|?*`, trailing dots and trailing spaces.

    Accented and non-Latin letters are KEPT: `str.isalnum()` is unicode-aware and both NTFS and
    ext4 store them fine, so folding them to ASCII would mangle perfectly good names (`sesión` ->
    `sesi-n`) for no safety gain. Only genuinely illegal characters collapse, to a single `-`, so
    the common case stays readable: `my run 2` -> `my-run-2`.
    """
    text = str(name or "").strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in _STEM_EXTRA:
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    # A leading dot hides the file on unix; a trailing dot or dash is legal but reads as a typo,
    # and Windows silently strips trailing dots, which would make two names collide.
    stem = "".join(out).strip("-.")
    if stem.upper() in _WIN_RESERVED:
        stem += "-run"
    return stem or fallback
