import io
import os
import re
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional
from helpers.layout import quarter_from_filename, is_performance_file

SPILL_THRESHOLD_BYTES = int(os.getenv("ZIP_SPILL_THRESHOLD_BYTES", 400_000_000))
TEXT_RE = re.compile(r"\.txt$", re.IGNORECASE)
QUARTER_ZIP_RE = re.compile(r"(?:^|/)[^/]*?((?:19|20)\d{2}Q[1-4])[^/]*\.zip$", re.I)

GRAINS = ('origination', 'performance')

def _classify(names: List[str]) -> Dict[str, str]:
    """Each grain mapped to its .txt member among the archive names (performance files start with perf_)."""
    out: Dict[str, str] = {}
    for n in names:
        if not TEXT_RE.search(n) or n.endswith('/'):
            continue
        base = n.rsplit('/', 1)[-1]
        out['performance' if is_performance_file(base) else 'origination'] = n
    return out

@dataclass(frozen=True)
class QuarterSource:
    """Where one quarter's files are: a zip holding them, a quarter zip inside a larger zip, or loose .txt files."""
    quarter: str
    container: Path
    inner_zip: Optional[str] = None
    loose: Dict[str, Path] = None

    @property
    def label(self) -> str:
        if self.loose:
            return f"{self.quarter} (loose files)"
        inner = f" :: {self.inner_zip}" if self.inner_zip else ""
        return f"{self.quarter} ({self.container.name}{inner})"

class QuarterArchive:
    """Context manager that opens a quarter's source and serves its origination and performance files as binary
    streams. An inner zip larger than SPILL_THRESHOLD_BYTES is spilled to a temp file rather than read into
    memory."""
    def __init__(self, src: QuarterSource):
        self.src = src
        self._outer: Optional[zipfile.ZipFile] = None
        self._inner: Optional[zipfile.ZipFile] = None
        self._spill: Optional[str] = None
        self._members: Dict[str, str] = {}

    def __enter__(self) -> "QuarterArchive":
        if self.src.loose:
            return self
        self._outer = zipfile.ZipFile(self.src.container)
        if self.src.inner_zip is None:
            self._inner = self._outer
        else:
            info = self._outer.getinfo(self.src.inner_zip)
            if info.file_size > SPILL_THRESHOLD_BYTES:
                fd, path = tempfile.mkstemp(suffix=".zip")
                os.close(fd)
                self._spill = path
                with self._outer.open(self.src.inner_zip) as rd, open(path, 'wb') as wr:
                    shutil.copyfileobj(rd, wr, length=8 * 1024 * 1024)
                self._inner = zipfile.ZipFile(path)
            else:
                self._inner = zipfile.ZipFile(io.BytesIO(self._outer.read(self.src.inner_zip)))
        self._members = _classify(self._inner.namelist())
        return self

    def __exit__(self, *exc) -> None:
        for z in (self._inner, self._outer):
            if z is not None:
                try:
                    z.close()
                except Exception:
                    pass
        if self._spill:
            os.unlink(self._spill)
        self._inner = self._outer = None
        self._spill = None

    def grains(self) -> List[str]:
        if self.src.loose:
            return sorted(self.src.loose)
        return sorted(self._members)

    @contextmanager
    def open(self, grain: str) -> Iterator[BinaryIO]:
        """The raw file of one grain, as a binary stream."""
        if grain not in GRAINS:
            raise ValueError(f"Unknown grain {grain}")
        if self.src.loose:
            path = self.src.loose.get(grain)
            if path is None:
                raise KeyError(f"{self.src.quarter}: no {grain} file")
            with open(path, 'rb') as rd:
                yield rd
            return
        if self._inner is None:
            raise RuntimeError("QuarterArchive must be used as a context manager")
        member = self._members.get(grain)
        if member is None:
            raise KeyError(f"{self.src.quarter}: no {grain} member in "
                           f"{self.src.inner_zip or self.src.container.name}")
        with self._inner.open(member) as rd:
            yield rd

def discover(input_dir: Path) -> Dict[str, QuarterSource]:
    """Every quarter found under input_dir: zips of quarter zips, zips holding the .txt files, and loose .txt
    files. The first source found for a quarter wins."""
    found: Dict[str, QuarterSource] = {}
    loose: Dict[str, Dict[str, Path]] = {}

    for path in sorted(input_dir.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()

        if suffix == '.txt':
            q = quarter_from_filename(path.name)
            if q:
                grain = 'performance' if is_performance_file(path) else 'origination'
                loose.setdefault(q, {})[grain] = path
            continue

        if suffix != '.zip':
            continue
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
        except zipfile.BadZipfile:
            continue

        inner_zips = [n for n in names if QUARTER_ZIP_RE.search(n)]
        if inner_zips:
            for member in inner_zips:
                q = quarter_from_filename(member)
                if q and q not in found:
                    found[q] = QuarterSource(quarter=q, container=path, inner_zip=member)
            continue

        if _classify(names):
            q = quarter_from_filename(path) or next(
                (quarter_from_filename(n) for n in names if quarter_from_filename(n)), None
            )
            if q and q not in found:
                found[q] = QuarterSource(quarter=q, container=path, inner_zip=None)

    for q, grains in loose.items():
        if q not in found:
            found[q] = QuarterSource(quarter=q, container=input_dir, inner_zip=grains)

    return found

def check_quarters(sources: Dict[str, QuarterSource], expect: Optional[List[str]] = None) -> List[str]:
    """A message for each expected quarter missing from the download."""
    return [f"{q}: not present in the download" for q in sorted(set(expect or []) - set(sources))]