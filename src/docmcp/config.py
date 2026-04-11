from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Config:
    host: str = "0.0.0.0"
    port: int = 8808
    doc_dirs: list[Path] = field(default_factory=lambda: [Path("/data/docs")])
    index_dir: Path = Path("/data/index")
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Config:
        doc_dirs_raw = os.environ.get("DOCMCP_DOC_DIRS", "/data/docs")
        doc_dirs = [Path(d.strip()) for d in doc_dirs_raw.split(",") if d.strip()]

        config = cls(
            host=os.environ.get("DOCMCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("DOCMCP_PORT", "8808")),
            doc_dirs=doc_dirs,
            index_dir=Path(os.environ.get("DOCMCP_INDEX_DIR", "/data/index")),
            embedding_model=os.environ.get("DOCMCP_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.environ.get("DOCMCP_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.environ.get("DOCMCP_CHUNK_OVERLAP", "200")),
            log_level=os.environ.get("DOCMCP_LOG_LEVEL", "INFO"),
        )

        for d in config.doc_dirs:
            if not d.exists():
                logger.warning("Document directory does not exist: %s", d)

        return config

    def setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.log_level.upper(), logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
