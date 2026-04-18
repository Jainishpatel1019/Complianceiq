"""SQLAlchemy 2.0 ORM models for ComplianceIQ.

Design decisions:
- TimescaleDB hypertables on change_scores and agent_reports (time-series).
  Hypertables chunk by time, making range queries 10-100x faster than
  standard btree indexes on timestamp columns.
- JSONB for metadata: regulatory documents have heterogeneous fields
  (capital_thresholds, effective_dates, affected_agencies). JSONB lets us
  store this without schema migrations every time a new field appears, while
  still being indexable (GIN index).
- Partitioned regulations table by agency: allows parallel scans when
  filtering by agency (which is the most common query pattern).
- pgvector stored in PostgreSQL (not just ChromaDB): enables hybrid
  keyword + semantic search via a single SQL query using cosine distance.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Regulation(Base):
    """A regulatory document from any source (Federal Register, SEC, CFPB).

    This is the canonical record for a regulation. Version history lives
    in RegulationVersion. The embedding of the full document is stored in
    PostgreSQL via pgvector so hybrid search works in a single query.
    """

    __tablename__ = "regulations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_number: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # federal_register | sec | fdic | cfpb
    agency: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    publication_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    regulation_type: Mapped[str | None] = mapped_column(String(64), nullable=True)  # capital | aml | consumer_protection | ...
    cfr_references: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)  # ["12 CFR 3", "12 CFR 6"]
    raw_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    # Relationships
    versions: Mapped[list[RegulationVersion]] = relationship(
        "RegulationVersion", back_populates="regulation", cascade="all, delete-orphan"
    )
    change_scores: Mapped[list[ChangeScore]] = relationship(
        "ChangeScore", back_populates="regulation", cascade="all, delete-orphan"
    )
    agent_reports: Mapped[list[AgentReport]] = relationship(
        "AgentReport", back_populates="regulation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_number", "source", name="uq_regulation_source_number"),
        Index("ix_regulations_agency", "agency"),
        Index("ix_regulations_publication_date", "publication_date"),
        Index("ix_regulations_type", "regulation_type"),
        # GIN index for JSONB metadata — enables fast containment queries
        # e.g. WHERE raw_metadata @> '{"topic": "capital"}'
        Index("ix_regulations_metadata_gin", "raw_metadata", postgresql_using="gin"),
    )


class RegulationVersion(Base):
    """Tracks each fetched version of a regulation document.

    Why version history: regulations are amended. We need to compare
    version N-1 vs version N to compute semantic drift, JSD, and
    Wasserstein distance. Without version tracking, we cannot do
    change detection.
    """

    __tablename__ = "regulation_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    regulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    full_text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # SHA-256 of full_text
    word_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    regulation: Mapped[Regulation] = relationship("Regulation", back_populates="versions")

    __table_args__ = (
        UniqueConstraint("regulation_id", "version_number", name="uq_version_per_regulation"),
        Index("ix_regulation_versions_regulation_id", "regulation_id"),
    )


class ChangeScore(Base):
    """Stores all three change-detection measures between consecutive versions.

    This table is a TimescaleDB hypertable (partitioned by computed_at).
    This enables fast queries like:
      'Show me the 10 regulations with highest drift in the last 30 days'
    which scan only the relevant time chunk rather than the full table.

    NOTE: hypertable conversion happens in the Alembic migration, not here.
    """

    __tablename__ = "change_scores"

    # Composite PK (id, computed_at) required by TimescaleDB: hypertable
    # partitioning column must be part of every unique index, including PK.
    # id remains globally unique (UUID4); computed_at is the partition key.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    regulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False
    )
    version_old: Mapped[int] = mapped_column(Integer, nullable=False)
    version_new: Mapped[int] = mapped_column(Integer, nullable=False)

    # Pillar 1 — three complementary measures (see docs/math_explainer.md)
    drift_score: Mapped[float | None] = mapped_column(Float, nullable=True)       # cosine semantic drift
    drift_ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)      # 95% CI lower bound
    drift_ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)     # 95% CI upper bound
    jsd_score: Mapped[float | None] = mapped_column(Float, nullable=True)         # Jensen-Shannon divergence
    jsd_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)       # permutation test p-value
    wasserstein_score: Mapped[float | None] = mapped_column(Float, nullable=True) # W2 distance
    is_significant: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    flagged_for_analysis: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship
    regulation: Mapped[Regulation] = relationship("Regulation", back_populates="change_scores")

    __table_args__ = (
        PrimaryKeyConstraint("id", "computed_at", name="pk_change_scores"),
        Index("ix_change_scores_regulation_id", "regulation_id"),
        Index("ix_change_scores_computed_at", "computed_at"),
        Index("ix_change_scores_flagged", "flagged_for_analysis"),
    )


class CausalEstimate(Base):
    """Pre-computed DiD / synthetic control estimates from the causal DAG.

    The LangGraph agent calls run_causal_estimate(regulation_id) which
    looks up this table. Pre-computing avoids running econml at query time
    (which takes 30-60s per regulation).
    """

    __tablename__ = "causal_estimates"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    regulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False
    )
    method: Mapped[str] = mapped_column(String(32), nullable=False)  # did | synthetic_control | rdd
    att_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)   # Average Treatment Effect
    standard_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_low_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    outcome_variable: Mapped[str | None] = mapped_column(String(64), nullable=True)  # tier1_capital_ratio | rwa | ...
    treatment_threshold: Mapped[float | None] = mapped_column(Float, nullable=True)   # e.g. 10B asset threshold
    parallel_trends_p_value: Mapped[float | None] = mapped_column(Float, nullable=True)  # pre-test
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    regulation: Mapped[Regulation] = relationship("Regulation")

    __table_args__ = (
        UniqueConstraint("regulation_id", "method", name="uq_causal_per_method"),
        Index("ix_causal_estimates_regulation_id", "regulation_id"),
    )


class AgentReport(Base):
    """Structured JSON report produced by the LangGraph impact agent.

    This table is a TimescaleDB hypertable (partitioned by created_at).
    The dashboard queries 'latest report per regulation' frequently —
    the time index makes this a single chunk scan.
    """

    __tablename__ = "agent_reports"

    # Composite PK (id, created_at) required by TimescaleDB — same pattern
    # as change_scores. id is globally unique; created_at is the partition key.
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), default=uuid.uuid4, nullable=False)
    regulation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="CASCADE"), nullable=False
    )

    # Impact classification
    impact_score_low: Mapped[float | None] = mapped_column(Float, nullable=True)     # P(Low)
    impact_score_medium: Mapped[float | None] = mapped_column(Float, nullable=True)  # P(Medium)
    impact_score_high: Mapped[float | None] = mapped_column(Float, nullable=True)    # P(High)

    # Financial impact (Basel III RWA formula output)
    delta_rwa_median_m: Mapped[float | None] = mapped_column(Float, nullable=True)   # $M median
    delta_rwa_ci_low_m: Mapped[float | None] = mapped_column(Float, nullable=True)   # $M 90% CI low
    delta_rwa_ci_high_m: Mapped[float | None] = mapped_column(Float, nullable=True)  # $M 90% CI high

    # Full structured output from mistral:7b
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    affected_business_lines: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    key_citations: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    agent_reasoning_trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    full_report: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    alert_dispatched: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    regulation: Mapped[Regulation] = relationship("Regulation", back_populates="agent_reports")

    __table_args__ = (
        PrimaryKeyConstraint("id", "created_at", name="pk_agent_reports"),
        Index("ix_agent_reports_regulation_id", "regulation_id"),
        Index("ix_agent_reports_created_at", "created_at"),
        Index("ix_agent_reports_impact_high", "impact_score_high"),
    )


class GraphNode(Base):
    """Node in the regulatory knowledge graph.

    Node types: Regulation | Agency | Institution | CFR_Section | Concept
    The GAT (Graph Attention Network) trains on this structure to produce
    embeddings that capture both document content AND graph position.
    """

    __tablename__ = "graph_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_type: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(256), nullable=False)
    regulation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True
    )
    pagerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    louvain_community: Mapped[int | None] = mapped_column(Integer, nullable=True)
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_graph_nodes_type", "node_type"),
        Index("ix_graph_nodes_regulation_id", "regulation_id"),
    )


class GraphEdge(Base):
    """Temporal edge in the regulatory knowledge graph.

    Every edge carries a timestamp — the graph is queryable at any point in
    time. This lets us answer 'What did the regulatory graph look like in 2008?'
    by filtering edges WHERE valid_from <= '2008-01-01'.
    """

    __tablename__ = "graph_edges"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("graph_nodes.id", ondelete="CASCADE"), nullable=False
    )
    edge_type: Mapped[str] = mapped_column(String(32), nullable=False)  # amends | supersedes | references | applies_to | enforces
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    properties: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        Index("ix_graph_edges_source", "source_id"),
        Index("ix_graph_edges_target", "target_id"),
        Index("ix_graph_edges_type", "edge_type"),
        Index("ix_graph_edges_valid_from", "valid_from"),
    )


class RagasTestSet(Base):
    """Human-labelled RAG queries for RAGAS evaluation (Phase 5).

    Populated offline by domain experts. Each row is one evaluation query
    with a ground-truth answer and the set of gold chunk IDs that should
    be retrieved to answer it correctly.
    """

    __tablename__ = "ragas_test_set"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    ground_truth_answer: Mapped[str] = mapped_column(Text, nullable=False)
    gold_chunk_ids: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    regulation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_ragas_test_set_regulation_id", "regulation_id"),
    )


class LabelledChangePair(Base):
    """Human-labelled document pairs for change detection ablation (Phase 5).

    300 pairs labelled by domain experts: label=1 means a meaningful
    regulatory change occurred, label=0 means the change is cosmetic.
    Used as the dev set for ablation and calibration experiments.
    """

    __tablename__ = "labelled_change_pairs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    regulation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("regulations.id", ondelete="SET NULL"), nullable=True
    )
    text_old: Mapped[str] = mapped_column(Text, nullable=False)
    text_new: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[int] = mapped_column(Integer, nullable=False)  # 1 = meaningful change, 0 = cosmetic
    labeller_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_labelled_pairs_label", "label"),
        Index("ix_labelled_pairs_regulation_id", "regulation_id"),
    )
