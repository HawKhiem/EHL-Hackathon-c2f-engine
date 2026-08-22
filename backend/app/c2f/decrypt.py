"""Decrypt a case archive.

The archives are AES-256 zips, which `zipfile` cannot read. The starter script
shells out to 7z; we use `pyzipper` instead, for three reasons:

* No external install. 7z is a per-machine dependency that will bite exactly one
  teammate at exactly the wrong moment - it is not present on this machine at
  all, so the 7z path fails here today.
* No subprocess in the 60-second window - process spawn plus disk round-trip,
  gone.
* We can extract straight to memory. Nothing needs to touch the filesystem
  before the model sees it, though we also write it out for the audit trail.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyzipper

log = logging.getLogger(__name__)

#: A case is three or four small files; anything larger is not a case.
MAX_MEMBER_BYTES: int = 32 * 1024 * 1024


class DecryptError(RuntimeError):
    """The archive could not be opened with the given key."""


def archive_path(cases_dir: str | Path, game_id: int) -> Path:
    return Path(cases_dir) / f"case_{game_id:02d}.zip"


def read_members(archive: str | Path, key: str) -> dict[str, bytes]:
    """Every member of the archive, in memory.

    Raises `DecryptError` rather than leaking pyzipper's exception types, so the
    orchestrator has one thing to catch.
    """
    path = Path(archive)
    if not path.exists():
        raise DecryptError(f"missing {path} - stage the archives before the round")

    try:
        with pyzipper.AESZipFile(path) as zip_file:
            zip_file.setpassword(key.encode())
            out: dict[str, bytes] = {}
            for info in zip_file.infolist():
                if info.is_dir():
                    continue
                if info.file_size > MAX_MEMBER_BYTES:
                    log.warning("c2f skipping oversized member %s", info.filename)
                    continue
                # Flatten any directory nesting; case layouts are flat.
                out[Path(info.filename).name] = zip_file.read(info)
    except Exception as error:  # noqa: BLE001 - one failure type for the caller
        raise DecryptError(f"could not decrypt {path.name}: {error}") from error

    if not out:
        raise DecryptError(f"{path.name} decrypted to nothing")
    return out


def extract_case(archive: str | Path, key: str, out_dir: str | Path) -> Path:
    """Decrypt and write the files out. Returns the directory written to.

    We still write to disk even though parsing could work from memory: a round
    that scores oddly is much easier to explain when the exact inputs are
    sitting on disk next to the decision log.
    """
    members = read_members(archive, key)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in members.items():
        (directory / name).write_bytes(data)
    return directory
