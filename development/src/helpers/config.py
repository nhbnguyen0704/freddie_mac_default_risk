import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]

def _path(key: str, default: str) -> Path:
    p = Path(os.getenv(key, default))
    return p if p.is_absolute() else ROOT / p

@dataclass(frozen=True)
class Settings:
    """Paths, HDFS location and observation period from .env; relative paths resolve against development/."""
    data_raw : Path = _path("DATA_RAW", 'data/raw')
    data_curated: Path = _path("DATA_CURATED", 'data/curated')
    model_dir: Path = _path("MODEL_DIR", 'artifacts')

    hdfs_uri: str = os.getenv("HDFS_URI", 'hdfs://localhost:9000')
    hdfs_curated_dir: str = os.getenv("HDFS_CURATED_DIR", f'/user/{os.getenv('USER', 'hadoop')}/fre/curated')
    hdfs_enabled: bool = os.getenv("HDFS_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off"}

    observation_start: date = date.fromisoformat(os.getenv("OBSERVATION_START", '2016-01-01'))
    observation_end: date = date.fromisoformat(os.getenv("OBSERVATION_END", '2025-12-31'))

    @property
    def hdfs_curated_url(self) -> str:
        """The curated directory as a full hdfs:// URL."""
        d = self.hdfs_curated_dir
        if "://" in d:
            return d.rstrip("/")
        return f"{self.hdfs_uri.rstrip('/')}/{d.lstrip('/')}".rstrip("/")

    def ensure_dirs(self) -> None:
        for p in (self.data_raw, self.data_curated, self.model_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()

