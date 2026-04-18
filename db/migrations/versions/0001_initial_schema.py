"""Initial schema: all tables + TimescaleDB hypertables + pgvector extension.

Revision ID: 0001
Revises:
Create Date: 2026-04-17

Why TimescaleDB hypertables for change_scores and agent_reports:
  - change_scores is queried almost exclusively by time range
    ("show drift scores for the last 30 days"). Hypertable chunks by time,
    so these queries scan 1 chunk instead of the full table.
  - agent_reports: same pattern. Dashboard shows "latest report per regulation",
    which benefits from time-ordered chunks.

Why NOT a hypertable for regulations:
  - Regulations are queried by agency, type, and document_number — not by time.
    A standard btree index outperforms TimescaleDB here.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Enable extensions ───────────────────────────────────────────────────
    # TimescaleDB: optional — available in the full dev stack (timescale image)
    # but not in plain postgres (e.g. HF single-container deploy). Tables work
    # without it; hypertables are a performance optimisation, not a correctness
    # requirement. Both create_hypertable calls below are also wrapped.
    op.execute("""
        DO $$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'timescaledb not available — skipping (tables will work without it): %', SQLERRM;
        END;
        $$;
    """)
    # pgvector: optional for the same reason
    op.execute("""
        DO $$
        BEGIN
            CREATE EXTENSION IF NOT EXISTS vector;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'pgvector not available — skipping: %', SQLERRM;
        END;
        $$;
    """)

    # ── regulations ──────────────────────────────────────────────────────────
    op.create_table(
        "regulations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("document_number", sa.String(64), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("agency", sa.String(128), nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("abstract", sa.Text, nullable=True),
        sa.Column("full_text", sa.Text, nullable=True),
        sa.Column("publication_date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("regulation_type", sa.String(64), nullable=True),
        sa.Column("cfr_references", JSONB, nullable=True),
        sa.Column("raw_metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("document_number", "source", name="uq_regulation_source_number"),
    )
    op.create_index("ix_regulations_agency", "regulations", ["agency"])
    op.create_index("ix_regulations_publication_date", "regulations", ["publication_date"])
    op.create_index("ix_regulations_type", "regulations", ["regulation_type"])
    op.execute(
        "CREATE INDEX ix_regulations_metadata_gin ON regulations USING gin (raw_metadata);"
    )

    # ── regulation_versions ──────────────────────────────────────────────────
    op.create_table(
        "regulation_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("regulation_id", UUID(as_uuid=True), sa.ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_number", sa.Integer, nullable=False),
        sa.Column("full_text", sa.Text, nullable=False),
        sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("word_count", sa.Integer, nullable=True),
        sa.Column("fetched_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("regulation_id", "version_number", name="uq_version_per_regulation"),
    )
    op.create_index("ix_regulation_versions_regulation_id", "regulation_versions", ["regulation_id"])

    # ── change_scores (TimescaleDB hypertable) ───────────────────────────────
    # TimescaleDB requires that all unique indexes (including PK) include the
    # partitioning column. So the PK is composite (id, computed_at).
    # Application code selects by id alone — the extra column in the PK is
    # transparent because id is still globally unique (UUID).
    op.create_table(
        "change_scores",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("regulation_id", UUID(as_uuid=True), sa.ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version_old", sa.Integer, nullable=False),
        sa.Column("version_new", sa.Integer, nullable=False),
        sa.Column("drift_score", sa.Float, nullable=True),
        sa.Column("drift_ci_low", sa.Float, nullable=True),
        sa.Column("drift_ci_high", sa.Float, nullable=True),
        sa.Column("jsd_score", sa.Float, nullable=True),
        sa.Column("jsd_p_value", sa.Float, nullable=True),
        sa.Column("wasserstein_score", sa.Float, nullable=True),
        sa.Column("is_significant", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("flagged_for_analysis", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", "computed_at", name="pk_change_scores"),
    )
    # Convert to hypertable — chunk by 7-day intervals (one week of scores per chunk)
    op.execute("""
        DO $$
        BEGIN
            PERFORM create_hypertable('change_scores', 'computed_at',
                chunk_time_interval => INTERVAL '7 days');
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'create_hypertable(change_scores) skipped: %', SQLERRM;
        END;
        $$;
    """)
    op.create_index("ix_change_scores_regulation_id", "change_scores", ["regulation_id"])
    op.create_index("ix_change_scores_flagged", "change_scores", ["flagged_for_analysis"])

    # ── causal_estimates ──────────────────────────────────────────────────────
    op.create_table(
        "causal_estimates",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("regulation_id", UUID(as_uuid=True), sa.ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("method", sa.String(32), nullable=False),
        sa.Column("att_estimate", sa.Float, nullable=True),
        sa.Column("standard_error", sa.Float, nullable=True),
        sa.Column("p_value", sa.Float, nullable=True),
        sa.Column("ci_low_95", sa.Float, nullable=True),
        sa.Column("ci_high_95", sa.Float, nullable=True),
        sa.Column("outcome_variable", sa.String(64), nullable=True),
        sa.Column("treatment_threshold", sa.Float, nullable=True),
        sa.Column("parallel_trends_p_value", sa.Float, nullable=True),
        sa.Column("details", JSONB, nullable=False, server_default="{}"),
        sa.Column("computed_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("regulation_id", "method", name="uq_causal_per_method"),
    )
    op.create_index("ix_causal_estimates_regulation_id", "causal_estimates", ["regulation_id"])

    # ── agent_reports (TimescaleDB hypertable) ───────────────────────────────
    # Same composite-PK pattern: (id, created_at) satisfies TimescaleDB's
    # requirement that unique indexes include the partitioning column.
    op.create_table(
        "agent_reports",
        sa.Column("id", UUID(as_uuid=True), nullable=False),
        sa.Column("regulation_id", UUID(as_uuid=True), sa.ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("impact_score_low", sa.Float, nullable=True),
        sa.Column("impact_score_medium", sa.Float, nullable=True),
        sa.Column("impact_score_high", sa.Float, nullable=True),
        sa.Column("delta_rwa_median_m", sa.Float, nullable=True),
        sa.Column("delta_rwa_ci_low_m", sa.Float, nullable=True),
        sa.Column("delta_rwa_ci_high_m", sa.Float, nullable=True),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("affected_business_lines", JSONB, nullable=True),
        sa.Column("key_citations", JSONB, nullable=True),
        sa.Column("agent_reasoning_trace", JSONB, nullable=False, server_default="[]"),
        sa.Column("full_report", JSONB, nullable=False, server_default="{}"),
        sa.Column("alert_dispatched", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id", "created_at", name="pk_agent_reports"),
    )
    op.execute("""
        DO $$
        BEGIN
            PERFORM create_hypertable('agent_reports', 'created_at',
                chunk_time_interval => INTERVAL '30 days');
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'create_hypertable(agent_reports) skipped: %', SQLERRM;
        END;
        $$;
    """)
    op.create_index("ix_agent_reports_regulation_id", "agent_reports", ["regulation_id"])
    op.create_index("ix_agent_reports_impact_high", "agent_reports", ["impact_score_high"])

    # ── graph_nodes ──────────────────────────────────────────────────────────
    op.create_table(
        "graph_nodes",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("node_type", sa.String(32), nullable=False),
        sa.Column("label", sa.String(256), nullable=False),
        sa.Column("regulation_id", UUID(as_uuid=True), sa.ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("pagerank_score", sa.Float, nullable=True),
        sa.Column("louvain_community", sa.Integer, nullable=True),
        sa.Column("properties", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_graph_nodes_type", "graph_nodes", ["node_type"])
    op.create_index("ix_graph_nodes_regulation_id", "graph_nodes", ["regulation_id"])

    # ── graph_edges ──────────────────────────────────────────────────────────
    op.create_table(
        "graph_edges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", UUID(as_uuid=True), sa.ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_id", UUID(as_uuid=True), sa.ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("edge_type", sa.String(32), nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("properties", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_graph_edges_source", "graph_edges", ["source_id"])
    op.create_index("ix_graph_edges_target", "graph_edges", ["target_id"])
    op.create_index("ix_graph_edges_valid_from", "graph_edges", ["valid_from"])


def downgrade() -> None:
    op.drop_table("graph_edges")
    op.drop_table("graph_nodes")
    op.drop_table("agent_reports")
    op.drop_table("causal_estimates")
    op.drop_table("change_scores")
    op.drop_table("regulation_versions")
    op.drop_table("regulations")
