"""Repository for job rows in the SQLite state database."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .db import transaction


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: str
    created_at: datetime
    updated_at: datetime
    status: JobStatus
    current_stage: str | None
    source_pdf: str
    data_dir: str
    config: Mapping[str, Any]
    error_message: str | None = None

    @property
    def is_resumable(self) -> bool:
        return self.status in (JobStatus.FAILED, JobStatus.RUNNING)


class JobRepository:
    """Read/write access to the ``jobs`` table.

    Methods are synchronous; the orchestrator wraps writes in
    :func:`asyncio.to_thread` when invoked from the event loop.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        # The connection is opened with check_same_thread=False so the
        # orchestrator (or asyncio.to_thread callers) can safely hit it from
        # any thread. We serialize all access through this lock so concurrent
        # writes don't corrupt SQLite's internal state.
        self._lock = threading.Lock()

    # --------------------------------------------------------------- inserts

    def create_job(
        self,
        *,
        job_id: str,
        source_pdf: Path,
        data_dir: Path,
        config: Mapping[str, Any],
    ) -> JobRecord:
        now = datetime.now(UTC)
        record = JobRecord(
            job_id=job_id,
            created_at=now,
            updated_at=now,
            status=JobStatus.PENDING,
            current_stage=None,
            source_pdf=str(source_pdf),
            data_dir=str(data_dir),
            config=dict(config),
        )
        with self._lock, transaction(self._conn):
            self._conn.execute(
                """
                INSERT INTO jobs (
                    job_id, created_at, updated_at, status,
                    current_stage, source_pdf, data_dir, config_json, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.job_id,
                    _serialize_dt(record.created_at),
                    _serialize_dt(record.updated_at),
                    record.status.value,
                    record.current_stage,
                    record.source_pdf,
                    record.data_dir,
                    json.dumps(record.config, default=str),
                    record.error_message,
                ),
            )
        return record

    # --------------------------------------------------------------- reads

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def list(self, *, limit: int = 50) -> list[JobRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_row_to_record(row) for row in rows]

    # --------------------------------------------------------------- updates

    def update_status(
        self,
        job_id: str,
        *,
        status: JobStatus,
        current_stage: str | None = None,
        error_message: str | None = None,
    ) -> None:
        with self._lock, transaction(self._conn):
            self._conn.execute(
                """
                UPDATE jobs
                SET status = ?, current_stage = ?, updated_at = ?, error_message = ?
                WHERE job_id = ?
                """,
                (
                    status.value,
                    current_stage,
                    _serialize_dt(datetime.now(UTC)),
                    error_message,
                    job_id,
                ),
            )


# --------------------------------------------------------------- helpers


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        created_at=_parse_dt(row["created_at"]),
        updated_at=_parse_dt(row["updated_at"]),
        status=JobStatus(row["status"]),
        current_stage=row["current_stage"],
        source_pdf=row["source_pdf"],
        data_dir=row["data_dir"],
        config=json.loads(row["config_json"]),
        error_message=row["error_message"],
    )


def _serialize_dt(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)
