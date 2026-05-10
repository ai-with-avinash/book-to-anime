-- BookToAnime job-state schema. Minimal by design: rich per-shot data lives
-- in `events.log` and `<stage>/index.json` files inside each job directory.
-- This DB is the source of truth for job listing + resume eligibility.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS jobs (
  job_id      TEXT PRIMARY KEY,
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL,
  status      TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed','cancelled')),
  current_stage TEXT,
  source_pdf  TEXT NOT NULL,
  data_dir    TEXT NOT NULL,
  config_json TEXT NOT NULL,
  error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
