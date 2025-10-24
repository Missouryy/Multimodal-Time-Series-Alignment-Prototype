-- Multimodal Time-Series Alignment - PostgreSQL/TimescaleDB schema
-- Note: Generate UUIDs in application layer to avoid DB extension dependencies

CREATE TABLE IF NOT EXISTS streams (
  id UUID PRIMARY KEY,
  name TEXT UNIQUE NOT NULL,
  modality TEXT NOT NULL,
  sample_rate_hz DOUBLE PRECISION,
  metadata JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS timeseries_values (
  stream_id UUID NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
  t DOUBLE PRECISION NOT NULL,
  v DOUBLE PRECISION[] NOT NULL,
  PRIMARY KEY (stream_id, t)
);

CREATE INDEX IF NOT EXISTS idx_timeseries_values_stream_time ON timeseries_values(stream_id, t);

CREATE TABLE IF NOT EXISTS alignments (
  id UUID PRIMARY KEY,
  ref_stream_id UUID NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
  tgt_stream_id UUID NOT NULL REFERENCES streams(id) ON DELETE CASCADE,
  method TEXT NOT NULL,
  params JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alignment_overlays (
  alignment_id UUID NOT NULL REFERENCES alignments(id) ON DELETE CASCADE,
  t DOUBLE PRECISION NOT NULL,
  values JSONB NOT NULL,
  PRIMARY KEY (alignment_id, t)
);

CREATE INDEX IF NOT EXISTS idx_alignment_overlays_alignment_time ON alignment_overlays(alignment_id, t);

-- Optional TimescaleDB hypertables (if TimescaleDB is installed)
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
    PERFORM public.create_hypertable('timeseries_values', 't', if_not_exists => TRUE);
    PERFORM public.create_hypertable('alignment_overlays', 't', if_not_exists => TRUE);
  END IF;
END$$;


