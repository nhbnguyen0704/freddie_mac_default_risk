"""HDFS files for the pandas / pyarrow side of the jobs, through the `hdfs dfs` CLI.

pyarrow cannot open hdfs:// here (no native libhdfs on macOS), so a directory
pandas has to read is copied to a local temp directory for the read and
removed after it; files pandas writes are put to HDFS the same way. Spark
reads and writes hdfs:// itself and needs none of this.
"""
import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

log = logging.getLogger(__name__)

HDFS_CMD = os.getenv('HDFS_CMD', 'hdfs')

def _dfs(*args: str, check: bool = True, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run `hdfs dfs` with the arguments; with check, raise with the tail of stderr when it fails."""
    r = subprocess.run([HDFS_CMD, 'dfs', *args], input=stdin, capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"hdfs dfs {' '.join(args)} failed: {r.stderr.strip()[-500:]}")
    return r

def exists(uri: str) -> bool:
    return _dfs('-test', '-e', uri, check=False).returncode == 0

def glob(pattern: str) -> List[str]:
    """The paths matching a glob ([] when none match)."""
    r = _dfs('-ls', '-C', pattern, check=False)
    return [x for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else []

def put(files: Sequence[Path], uri_dir: str) -> None:
    """Copy local files into an HDFS directory (created if missing), overwriting."""
    _dfs('-mkdir', '-p', uri_dir)
    _dfs('-put', '-f', *map(str, files), uri_dir)

def write_text(uri: str, text: str) -> None:
    _dfs('-put', '-f', '-', uri, stdin=text)

def read_jsons(pattern: str) -> List[dict]:
    """Every JSON object in the files matching a glob, read with one -cat."""
    text = _dfs('-cat', pattern).stdout
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        while i < len(text) and text[i].isspace():
            i += 1
        if i == len(text):
            return out
        obj, i = dec.raw_decode(text, i)
        out.append(obj)

@contextmanager
def local_copy(uri: str) -> Iterator[Path]:
    """A local copy of an HDFS directory or file, deleted when the block ends."""
    tmp = Path(tempfile.mkdtemp(prefix='hdfs_'))
    try:
        _dfs('-get', uri, str(tmp))
        local = tmp / uri.rstrip('/').rsplit('/', 1)[-1]
        log.info('Copied %s to %s for the read', uri, local)
        yield local
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
