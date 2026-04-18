"""
Seed script — loads sample regulatory data for demo / HF Space.

Usage (via Makefile):
    docker compose run --rm api python -m backend.pipelines.seed

What it does:
  1. Inserts 50 sample Regulation rows (Jan–Jun 2024).
  2. Inserts 2 RegulationVersion rows per regulation.
  3. Inserts ChangeScore rows for version-pair comparisons.
  4. Inserts CausalEstimate rows for 5 key regulations.
  5. Upserts 200 ChromaDB chunks (collection: "regulations").

Runs fully synchronously via a dedicated sync engine so this script
does not require an asyncio event loop (avoids conflict with Airflow
and plain CLI invocation).
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import uuid
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Sample data ───────────────────────────────────────────────────────────────

_SAMPLE_REGS = [
    ("2024-00123", "Capital Requirements for Large Banking Organisations", "OCC", "capital"),
    ("2024-00456", "Liquidity Coverage Ratio: Treatment of Operational Deposits", "FRB", "liquidity"),
    ("2024-00789", "Stress Testing Requirements: 2024 Cycle Amendments", "FDIC", "stress_testing"),
    ("2024-01011", "Community Reinvestment Act Modernisation Final Rule", "OCC", "consumer_protection"),
    ("2024-01213", "Fair Debt Collection Practices: Digital Communication", "CFPB", "consumer_protection"),
    ("2024-01415", "Volcker Rule: Covered Fund Definition Clarification", "FRB", "trading"),
    ("2024-01617", "Basel III Endgame: Market Risk Capital Requirements", "OCC", "capital"),
    ("2024-01819", "Safeguards Rule: Information Security Program Requirements", "FDIC", "aml"),
    ("2024-02021", "Anti-Money Laundering: Beneficial Ownership Reporting", "FinCEN", "aml"),
    ("2024-02223", "Interchange Fee Standards: Card Transaction Amendments", "FRB", "consumer_protection"),
]

_BODY = (
    "This final rule amends the regulatory capital rules applicable to banking "
    "organisations supervised by the agencies. The amendments implement the revised "
    "standardised approach for credit risk, the revised internal ratings-based "
    "approaches for credit risk, the revised minimum capital requirements for market "
    "risk, and revised requirements for operational risk. The rule also introduces "
    "changes to the leverage ratio framework and the countercyclical capital buffer. "
    "Institutions subject to the advanced approaches must comply by January 1, 2025. "
    "All other covered institutions must comply by January 1, 2026."
)


def _make_text(title: str, version: int) -> str:
    amendment = f"\n\n[Amendment {version} — revised text follows]\n" if version > 1 else ""
    sections = " ".join(
        f"Section {i}: {_BODY[i * 40: i * 40 + 120]}." for i in range(1, 6)
    )
    return f"# {title}{amendment}\n\n{_BODY}\n\n{sections}"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Synchronous session factory ───────────────────────────────────────────────

def _make_sync_engine():
    """Create a synchronous SQLAlchemy engine for the seed script."""
    from sqlalchemy import create_engine

    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "complianceiq")
    user = os.environ.get("POSTGRES_USER", "complianceiq")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, echo=False, pool_pre_ping=True)


# ── Main seed function ────────────────────────────────────────────────────────

def run_seed() -> None:
    """Insert sample data into PostgreSQL and ChromaDB.

    Idempotent: if PostgreSQL already contains seed regulations the INSERT
    phase is skipped and we go straight to ChromaDB upsert (which is always
    safe to re-run).
    """
    from sqlalchemy import select, text
    from sqlalchemy.orm import Session

    from db.models import (
        CausalEstimate,
        ChangeScore,
        Regulation,
        RegulationVersion,
    )

    engine = _make_sync_engine()
    rng = random.Random(42)

    with Session(engine) as session:
        # ── Check for existing seed data ──────────────────────────────────────
        existing_count = session.execute(
            text("SELECT COUNT(*) FROM regulations WHERE raw_metadata->>'demo' = 'true'")
        ).scalar_one()

        if existing_count > 0:
            log.info(
                "PostgreSQL already has %d seed regulations — skipping INSERT, "
                "collecting existing IDs for ChromaDB seed.",
                existing_count,
            )
            reg_ids = [
                row[0]
                for row in session.execute(
                    select(Regulation.id).where(
                        Regulation.raw_metadata["demo"].as_boolean() == True  # noqa: E712
                    ).order_by(Regulation.created_at)
                ).all()
            ]
            _seed_chromadb(rng, reg_ids)
            return

        # ── 1. Regulations ────────────────────────────────────────────────────
        reg_ids: list[uuid.UUID] = []
        ver_ids: list[tuple[uuid.UUID, uuid.UUID, uuid.UUID]] = []  # (reg_id, v1_id, v2_id)

        for i, (doc_num, title, agency, reg_type) in enumerate(_SAMPLE_REGS * 5):
            pub_dt = datetime(2024, 1 + (i % 6), 1 + (i % 28), tzinfo=timezone.utc)
            reg_id = uuid.uuid4()
            reg = Regulation(
                id=reg_id,
                document_number=f"{doc_num}-{i:02d}",
                source="federal_register",
                agency=agency,
                title=title,
                abstract=_BODY[:200],
                full_text=_make_text(title, 1),
                publication_date=pub_dt,
                regulation_type=reg_type,
                cfr_references=["12 CFR 3"],
                raw_metadata={"demo": True, "seed_index": i},
            )
            session.add(reg)
            session.flush()

            text_v1 = _make_text(title, 1)
            text_v2 = _make_text(title, 2)

            v1 = RegulationVersion(
                id=uuid.uuid4(),
                regulation_id=reg_id,
                version_number=1,
                full_text=text_v1,
                text_hash=_sha256(text_v1),
                word_count=len(text_v1.split()),
            )
            v2 = RegulationVersion(
                id=uuid.uuid4(),
                regulation_id=reg_id,
                version_number=2,
                full_text=text_v2,
                text_hash=_sha256(text_v2),
                word_count=len(text_v2.split()),
            )
            session.add(v1)
            session.add(v2)
            session.flush()

            reg_ids.append(reg_id)
            ver_ids.append((reg_id, v1.id, v2.id))

        log.info("Inserted %d regulations with 2 versions each", len(reg_ids))

        # ── 2. ChangeScores ───────────────────────────────────────────────────
        for reg_id, _v1_id, _v2_id in ver_ids[:20]:
            drift = round(rng.uniform(0.05, 0.45), 4)
            flagged = drift > 0.15
            cs = ChangeScore(
                id=uuid.uuid4(),
                regulation_id=reg_id,
                version_old=1,
                version_new=2,
                drift_score=drift,
                drift_ci_low=round(drift - 0.03, 4),
                drift_ci_high=round(drift + 0.03, 4),
                jsd_score=round(rng.uniform(0.01, 0.30), 4),
                jsd_p_value=round(rng.uniform(0.001, 0.15), 4),
                wasserstein_score=round(rng.uniform(0.02, 0.25), 4),
                is_significant=flagged,
                flagged_for_analysis=flagged,
            )
            session.add(cs)

        log.info("Inserted 20 change scores")

        # ── 3. CausalEstimates ────────────────────────────────────────────────
        causal_specs = [
            (0, "did", 340.0, 25.0, 1e-4, 290.0, 390.0, "tier1_capital_ratio", None),
            (1, "synthetic_control", 210.0, 18.0, 0.002, 175.0, 245.0, "liquidity_coverage_ratio", None),
            (6, "rdd", 480.0, 42.0, 1e-5, 398.0, 562.0, "market_risk_rwa", 10_000_000_000.0),
            (2, "did", 95.0, 12.0, 0.008, 71.0, 119.0, "tier1_capital_ratio", None),
            (5, "synthetic_control", 60.0, 9.0, 0.031, 42.0, 78.0, "trading_book_rwa", None),
        ]
        for idx, method, att, se, pv, ci_lo, ci_hi, outcome, threshold in causal_specs:
            if idx < len(reg_ids):
                ce = CausalEstimate(
                    id=uuid.uuid4(),
                    regulation_id=reg_ids[idx],
                    method=method,
                    att_estimate=att,
                    standard_error=se,
                    p_value=pv,
                    ci_low_95=ci_lo,
                    ci_high_95=ci_hi,
                    outcome_variable=outcome,
                    treatment_threshold=threshold,
                    parallel_trends_p_value=round(rng.uniform(0.15, 0.85), 4),
                    details={"n_treated": rng.randint(50, 200), "n_control": rng.randint(100, 400)},
                )
                session.add(ce)

        session.commit()
        log.info("Inserted 5 causal estimates — PostgreSQL seed complete")

    # ── 4. ChromaDB ───────────────────────────────────────────────────────────
    _seed_chromadb(rng, reg_ids)


def _seed_chromadb(rng: random.Random, reg_ids: list[uuid.UUID]) -> None:
    """Upsert 200 sample chunks into ChromaDB collection 'regulations'."""
    try:
        import numpy as np
        import chromadb

        client = chromadb.HttpClient(
            host=os.environ.get("CHROMADB_HOST", "chromadb"),
            port=int(os.environ.get("CHROMADB_PORT", "8000")),
        )
        collection = client.get_or_create_collection(
            name="regulations",
            metadata={"hnsw:space": "cosine"},
        )

        ids, docs, metas = [], [], []
        for i in range(200):
            spec = _SAMPLE_REGS[i % len(_SAMPLE_REGS)]
            chunk = f"{spec[1]}: {_BODY[rng.randint(0, 80): rng.randint(200, 400)]}"
            reg_id = str(reg_ids[i % len(reg_ids)])
            ids.append(f"seed-{i:04d}")
            docs.append(chunk)
            metas.append({
                "regulation_id": reg_id,
                "document_number": spec[0],
                "agency": spec[2],
                "chunk_index": i,
                "version_number": 1,
            })

        embeddings = np.random.default_rng(42).normal(0, 1, (200, 768))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).tolist()

        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        log.info("Upserted 200 ChromaDB chunks into 'regulations' collection")
    except Exception as exc:
        log.warning("ChromaDB seed skipped (service unavailable): %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_seed()
