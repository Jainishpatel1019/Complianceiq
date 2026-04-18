"""Add ragas_test_set and labelled_change_pairs tables for Phase 5 evaluation.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-18

Why separate migration:
  These tables are populated offline by domain experts and have no FK
  dependencies on the core pipeline tables (they are standalone label stores).
  Keeping them in a separate migration makes rollback safe — you can drop them
  without affecting the ingestion or change-detection tables.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── ragas_test_set ────────────────────────────────────────────────────────
    op.create_table(
        "ragas_test_set",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("ground_truth_answer", sa.Text, nullable=False),
        sa.Column("gold_chunk_ids", postgresql.JSONB, nullable=True),
        sa.Column(
            "regulation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("regulations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_ragas_test_set_regulation_id", "ragas_test_set", ["regulation_id"])

    # ── labelled_change_pairs ─────────────────────────────────────────────────
    op.create_table(
        "labelled_change_pairs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "regulation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("regulations.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("text_old", sa.Text, nullable=False),
        sa.Column("text_new", sa.Text, nullable=False),
        sa.Column("label", sa.Integer, nullable=False),
        sa.Column("labeller_notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_labelled_pairs_label", "labelled_change_pairs", ["label"])
    op.create_index(
        "ix_labelled_pairs_regulation_id", "labelled_change_pairs", ["regulation_id"]
    )


def downgrade() -> None:
    op.drop_table("labelled_change_pairs")
    op.drop_table("ragas_test_set")
