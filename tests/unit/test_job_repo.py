"""Unit tests for ``JobRepository.update_status`` semantics."""

from __future__ import annotations

from pathlib import Path

import pytest

from booktoanime.state import JobRepository, JobStatus, open_database


@pytest.fixture
def repo(tmp_path: Path) -> JobRepository:
    db = open_database(tmp_path / "state.db")
    return JobRepository(db)


def _seed(repo: JobRepository) -> str:
    record = repo.create_job(
        job_id="JOB1",
        source_pdf=Path("/tmp/x.pdf"),
        data_dir=Path("/tmp"),
        config={"providers": {}},
    )
    return record.job_id


def test_update_status_running_preserves_stage_on_partial_call(repo: JobRepository) -> None:
    job_id = _seed(repo)
    repo.update_status(job_id, status=JobStatus.RUNNING, current_stage="parsing")
    repo.update_status(job_id, status=JobStatus.RUNNING)  # status-only update
    record = repo.get(job_id)
    assert record is not None
    assert record.current_stage == "parsing"


def test_update_status_failed_records_error_and_stage(repo: JobRepository) -> None:
    job_id = _seed(repo)
    repo.update_status(
        job_id,
        status=JobStatus.FAILED,
        current_stage="audio",
        error_message="boom",
    )
    record = repo.get(job_id)
    assert record is not None
    assert record.status == JobStatus.FAILED
    assert record.current_stage == "audio"
    assert record.error_message == "boom"


def test_update_status_completed_clears_prior_error_message(repo: JobRepository) -> None:
    """After a failed→resume→completed cycle, the row should not carry the
    stale error_message from the prior failure."""

    job_id = _seed(repo)
    repo.update_status(
        job_id,
        status=JobStatus.FAILED,
        current_stage="assembly",
        error_message="ffmpeg crashed",
    )
    repo.update_status(job_id, status=JobStatus.COMPLETED)
    record = repo.get(job_id)
    assert record is not None
    assert record.status == JobStatus.COMPLETED
    # Prior failure text must not bleed through.
    assert (record.error_message or "") == ""
    # current_stage preserved (last stage seen, useful for "completed at" UI).
    assert record.current_stage == "assembly"


def test_update_status_completed_with_explicit_message_keeps_it(repo: JobRepository) -> None:
    """Explicit error_message on a COMPLETED transition still writes through —
    the auto-clear only kicks in when the caller passed None."""

    job_id = _seed(repo)
    repo.update_status(
        job_id,
        status=JobStatus.COMPLETED,
        error_message="completed with warnings",
    )
    record = repo.get(job_id)
    assert record is not None
    assert record.error_message == "completed with warnings"
