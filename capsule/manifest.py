from __future__ import annotations
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional
import yaml
import structlog

from capsule.models import (
    CapsuleManifest, DeploymentRecord, DeploymentStatus, PackageResult
)

logger = structlog.get_logger(__name__)

DB_PATH = Path(
    __import__("os").environ.get("CAPSULE_DB_PATH", "")
) if __import__("os").environ.get("CAPSULE_DB_PATH") else Path.home() / ".capsule" / "capsule.db"

# ── YAML manifest loader ────────────────────────────────────────────────────


def load_manifest(path: str = "capsule.yaml") -> CapsuleManifest:
    """Load and validate a capsule.yaml file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"capsule.yaml not found at {path}")
    with open(p) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"capsule.yaml at {path} is not a YAML mapping")
    manifest = CapsuleManifest(**raw)
    logger.info(
        "manifest.loaded",
        name=manifest.name,
        version=manifest.version,
        framework=manifest.framework,
    )
    return manifest


# ── SQLite store ────────────────────────────────────────────────────────────

class ManifestStore:
    """
    SQLite-backed store for deployment history and events.

    Production hardening applied:
    - WAL journal mode: allows concurrent reads alongside writes
    - busy_timeout 5 000 ms: retry on lock rather than raising immediately
    - Thread-local connections: one connection per OS thread, never shared
    - Explicit RETURNING-safe primary-key handling for idempotent upserts
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    # ── connection management ──────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating it on first access."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=5.0,
            )
            conn.row_factory = sqlite3.Row
            # WAL mode: readers don't block writers and vice-versa
            conn.execute("PRAGMA journal_mode=WAL")
            # Wait up to 5s before raising OperationalError on lock contention
            conn.execute("PRAGMA busy_timeout=5000")
            # Enforce foreign key constraints
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that commits on success and rolls back on error."""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self):
        """Close the thread-local connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None

    # ── schema ─────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._tx() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS packages (
                    name          TEXT    NOT NULL,
                    version       TEXT    NOT NULL,
                    framework     TEXT    NOT NULL,
                    image_tag     TEXT    NOT NULL,
                    image_digest  TEXT,
                    registry_path TEXT,
                    original_size_mb  REAL,
                    optimised_size_mb REAL,
                    onnx_optimised    INTEGER NOT NULL DEFAULT 0,
                    build_seconds     REAL,
                    packaged_at       TEXT    NOT NULL,
                    PRIMARY KEY (name, version)
                );

                CREATE TABLE IF NOT EXISTS deployments (
                    deployment_id TEXT    PRIMARY KEY,
                    name          TEXT    NOT NULL,
                    version       TEXT    NOT NULL,
                    image_tag     TEXT    NOT NULL,
                    image_digest  TEXT,
                    namespace     TEXT    NOT NULL,
                    status        TEXT    NOT NULL,
                    canary_weight INTEGER NOT NULL DEFAULT 0,
                    stable_version TEXT,
                    deployed_by   TEXT,
                    deployed_at   TEXT    NOT NULL,
                    updated_at    TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_deployments_name_ts
                    ON deployments (name, deployed_at DESC);

                CREATE TABLE IF NOT EXISTS events (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    deployment_name  TEXT    NOT NULL,
                    event_type       TEXT    NOT NULL,
                    canary_weight    INTEGER NOT NULL DEFAULT 0,
                    error_rate       REAL,
                    message          TEXT    NOT NULL,
                    timestamp        TEXT    NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_events_name_ts
                    ON events (deployment_name, timestamp DESC);

                CREATE TABLE IF NOT EXISTS audit_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts        TEXT    NOT NULL,
                    actor     TEXT    NOT NULL DEFAULT 'cli',
                    action    TEXT    NOT NULL,
                    target    TEXT    NOT NULL,
                    detail    TEXT
                );
            """)

    # ── packages ────────────────────────────────────────────────────────────

    def save_package(self, result: PackageResult):
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO packages
                  (name, version, framework, image_tag, image_digest,
                   registry_path, original_size_mb, optimised_size_mb,
                   onnx_optimised, build_seconds, packaged_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    result.name, result.version, result.framework.value,
                    result.image_tag, result.image_digest,
                    result.registry_path,
                    result.original_size_mb, result.optimised_size_mb,
                    int(result.onnx_optimised), result.build_seconds,
                    result.packaged_at.isoformat(),
                ),
            )
        self._audit("package", f"{result.name}:{result.version}", f"framework={result.framework.value}")
        logger.info("store.package_saved", name=result.name, version=result.version)

    def get_package(self, name: str, version: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM packages WHERE name=? AND version=?",
            (name, version),
        ).fetchone()
        return dict(row) if row else None

    def list_packages(self, name: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[dict]:
        if name:
            rows = self._conn().execute(
                "SELECT * FROM packages WHERE name=? ORDER BY packaged_at DESC LIMIT ? OFFSET ?",
                (name, limit, offset),
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT * FROM packages ORDER BY packaged_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── deployments ─────────────────────────────────────────────────────────

    def save_deployment(self, record: DeploymentRecord):
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO deployments
                  (deployment_id, name, version, image_tag, image_digest,
                   namespace, status, canary_weight, stable_version,
                   deployed_by, deployed_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    record.deployment_id, record.name, record.version,
                    record.image_tag, record.image_digest,
                    record.namespace, record.status.value,
                    record.canary_weight, record.stable_version,
                    record.deployed_by,
                    record.deployed_at.isoformat(),
                    record.updated_at.isoformat(),
                ),
            )
        self._audit("deploy", f"{record.name}:{record.version}", f"status={record.status.value}")

    def get_latest_deployment(self, name: str) -> Optional[DeploymentRecord]:
        row = self._conn().execute(
            """
            SELECT * FROM deployments WHERE name=?
            ORDER BY deployed_at DESC LIMIT 1
            """,
            (name,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_deployment(dict(row))

    def get_deployment_by_id(self, deployment_id: str) -> Optional[DeploymentRecord]:
        row = self._conn().execute(
            "SELECT * FROM deployments WHERE deployment_id=?",
            (deployment_id,),
        ).fetchone()
        return self._row_to_deployment(dict(row)) if row else None

    def get_stable_version(self, name: str) -> Optional[str]:
        """
        Return the last version whose status was RUNNING (not CANARY/ROLLED_BACK).
        This prevents rollback from targeting a previously-broken canary version.
        """
        row = self._conn().execute(
            """
            SELECT version FROM deployments
            WHERE name=? AND status=?
            ORDER BY deployed_at DESC LIMIT 1
            """,
            (name, DeploymentStatus.RUNNING.value),
        ).fetchone()
        return row["version"] if row else None

    def get_previous_version(self, name: str, current_version: str) -> Optional[str]:
        """
        Return the last RUNNING version that is not the current version.
        Falls back to any different version if no RUNNING record exists.
        """
        # Prefer an explicitly RUNNING version
        row = self._conn().execute(
            """
            SELECT version FROM deployments
            WHERE name=? AND version != ? AND status=?
            ORDER BY deployed_at DESC LIMIT 1
            """,
            (name, current_version, DeploymentStatus.RUNNING.value),
        ).fetchone()
        if row:
            return row["version"]
        # Fallback: any different version (covers initial deploys)
        row = self._conn().execute(
            """
            SELECT version FROM deployments
            WHERE name=? AND version != ?
            ORDER BY deployed_at DESC LIMIT 1
            """,
            (name, current_version),
        ).fetchone()
        return row["version"] if row else None

    @staticmethod
    def _row_to_deployment(d: dict) -> DeploymentRecord:
        d["status"] = DeploymentStatus(d["status"])
        d["deployed_at"] = datetime.fromisoformat(d["deployed_at"])
        d["updated_at"] = datetime.fromisoformat(d["updated_at"])
        return DeploymentRecord(**d)

    # ── events ───────────────────────────────────────────────────────────────

    def log_event(self, event) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO events
                  (deployment_name, event_type, canary_weight,
                   error_rate, message, timestamp)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    event.deployment_name, event.event_type,
                    event.canary_weight, event.error_rate,
                    event.message, event.timestamp.isoformat(),
                ),
            )

    def get_events(self, name: str, limit: int = 20) -> List[dict]:
        rows = self._conn().execute(
            """
            SELECT * FROM events WHERE deployment_name=?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (name, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── audit ─────────────────────────────────────────────────────────────

    def _audit(self, action: str, target: str, detail: str = "") -> None:
        try:
            with self._tx() as conn:
                conn.execute(
                    "INSERT INTO audit_log (ts, action, target, detail) VALUES (?,?,?,?)",
                    (datetime.utcnow().isoformat(), action, target, detail),
                )
        except Exception as exc:
            # Audit failures must never crash the main path
            logger.warning("store.audit_failed", error=str(exc))

    def get_audit_log(self, limit: int = 50) -> List[dict]:
        rows = self._conn().execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
