-- PPE Compliance Monitor — PostgreSQL schema
-- Runs automatically on first docker-compose up

CREATE TABLE IF NOT EXISTS violation_events (
    id              BIGSERIAL PRIMARY KEY,
    session         TEXT             NOT NULL,
    camera_id       TEXT             NOT NULL DEFAULT 'default',
    track_id        INTEGER          NOT NULL,
    violation_class TEXT             NOT NULL,
    start_time      DOUBLE PRECISION NOT NULL,
    end_time        DOUBLE PRECISION,
    duration_secs   DOUBLE PRECISION,
    frame_count     INTEGER          DEFAULT 1,
    created_at      TIMESTAMPTZ      DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ve_session    ON violation_events (session);
CREATE INDEX IF NOT EXISTS idx_ve_camera     ON violation_events (camera_id);
CREATE INDEX IF NOT EXISTS idx_ve_class      ON violation_events (violation_class);
CREATE INDEX IF NOT EXISTS idx_ve_created_at ON violation_events (created_at DESC);

-- Useful view: per-session summary
CREATE OR REPLACE VIEW session_summary AS
SELECT
    session,
    camera_id,
    COUNT(*)                                        AS total_events,
    COUNT(DISTINCT track_id)                        AS distinct_violators,
    ROUND(SUM(COALESCE(duration_secs, 0))::numeric, 1) AS total_violation_secs,
    MIN(created_at)                                 AS first_event,
    MAX(created_at)                                 AS last_event
FROM violation_events
GROUP BY session, camera_id;
