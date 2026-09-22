-- PostgreSQL initialization SQL
-- Auto-runs on first docker-compose up

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- The actual tables are managed by SQLAlchemy / direct SQL in Python code
-- See: src/ragagent/memory/l2_episodic.py

SELECT 'MemoryAgent PostgreSQL init done.' AS status;
