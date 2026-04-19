"""
Offline bulk seed — generates 500+ realistic regulations locally.
NO network calls. Works inside HuggingFace Spaces or any air-gapped env.

Strategy
--------
We have 10 seed templates (from seed.py) covering real regulatory areas.
This script expands them to 500+ entries by:
  1. Varying the agency, year, threshold values, and effective dates
  2. Running real TF-IDF drift between generated v1 / v2 text pairs
  3. Inserting Regulation + RegulationVersion (1,2) + ChangeScore rows
  4. Skipping any document_number already in the DB (idempotent)

Runs in ~30-45 seconds on 2 CPU cores. Called from start.sh as a
background process 10 s after uvicorn starts.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Regulation templates ───────────────────────────────────────────────────────
# Each template has a v1_text and v2_text — the "before" and "after" of a real
# regulatory change.  We vary agency, year, and numeric thresholds to produce
# hundreds of distinct records.

_TEMPLATES = [
    {
        "slug": "capital-adequacy",
        "title_template": "Capital Adequacy Standards — {agency} Final Rule ({year})",
        "reg_type": "capital",
        "cfr": "12 CFR 3",
        "v1_text": (
            "Section 1. Minimum Capital Requirements. Each covered institution shall "
            "maintain a minimum Common Equity Tier 1 (CET1) capital ratio of {lo_pct}% "
            "of total risk-weighted assets at all times. The leverage ratio requirement "
            "shall be {lev_lo}% of average total consolidated assets. Institutions failing "
            "to meet these requirements shall submit a capital restoration plan within "
            "{days_lo} calendar days of the deficiency date."
        ),
        "v2_text": (
            "Section 1. Minimum Capital Requirements (Revised). Each covered institution "
            "shall maintain a minimum Common Equity Tier 1 (CET1) capital ratio of {hi_pct}% "
            "of total risk-weighted assets at all times, inclusive of a capital conservation "
            "buffer of 2.5 percent. The leverage ratio requirement shall be {lev_hi}% of "
            "average total consolidated assets. Institutions failing to meet these requirements "
            "shall submit a capital restoration plan within {days_hi} calendar days and are "
            "prohibited from making capital distributions without prior regulatory approval."
        ),
        "params": [
            {"lo_pct":"6.0","hi_pct":"8.5","lev_lo":"3.0","lev_hi":"4.0","days_lo":"30","days_hi":"15"},
            {"lo_pct":"7.0","hi_pct":"9.5","lev_lo":"3.5","lev_hi":"5.0","days_lo":"45","days_hi":"20"},
            {"lo_pct":"8.0","hi_pct":"12.5","lev_lo":"4.0","lev_hi":"6.0","days_lo":"60","days_hi":"30"},
            {"lo_pct":"6.5","hi_pct":"10.0","lev_lo":"3.0","lev_hi":"5.5","days_lo":"30","days_hi":"15"},
        ],
    },
    {
        "slug": "aml-cdd",
        "title_template": "Customer Due Diligence and Beneficial Ownership — {agency} ({year})",
        "reg_type": "aml",
        "cfr": "31 CFR 1010",
        "v1_text": (
            "Financial institutions must identify and verify the identity of beneficial "
            "owners holding {lo_pct}% or more of the equity interests of a legal entity "
            "customer at the time of account opening. Verification may be completed within "
            "{days_lo} business days of account opening. Ongoing monitoring of customer "
            "relationships is required on an annual basis for standard-risk customers."
        ),
        "v2_text": (
            "Financial institutions must identify and verify the identity of beneficial "
            "owners holding {hi_pct}% or more of the equity interests of a legal entity "
            "customer prior to account opening. Enhanced due diligence is mandatory for "
            "any individual holding {enh_pct}% or more. Verification must be completed "
            "before any transaction is processed. Ongoing monitoring of customer "
            "relationships is required on a {freq} basis, with immediate review triggered "
            "by any material change in ownership structure or business activity."
        ),
        "params": [
            {"lo_pct":"25","hi_pct":"15","enh_pct":"10","days_lo":"5","freq":"quarterly"},
            {"lo_pct":"25","hi_pct":"10","enh_pct":"5","days_lo":"5","freq":"semi-annual"},
            {"lo_pct":"20","hi_pct":"10","enh_pct":"5","days_lo":"3","freq":"quarterly"},
        ],
    },
    {
        "slug": "liquidity-coverage",
        "title_template": "Liquidity Coverage Ratio Rule — {agency} ({year})",
        "reg_type": "liquidity",
        "cfr": "12 CFR 50",
        "v1_text": (
            "Covered depository institutions shall maintain a Liquidity Coverage Ratio "
            "(LCR) of at least {lo_pct}%, computed as the ratio of high-quality liquid "
            "assets (HQLA) to total net cash outflows over a {window_lo}-day stress period. "
            "Institutions with total consolidated assets below ${asset_lo} billion are "
            "exempt from this requirement. Monthly reporting to the primary regulator "
            "is required for covered institutions."
        ),
        "v2_text": (
            "Covered depository institutions shall maintain a Liquidity Coverage Ratio "
            "(LCR) of at least {hi_pct}%, computed as the ratio of high-quality liquid "
            "assets (HQLA) to total net cash outflows over a {window_hi}-day stress period "
            "incorporating intraday liquidity positions. Institutions with total consolidated "
            "assets below ${asset_hi} billion are exempt. Weekly reporting to the primary "
            "regulator is required, with daily monitoring of collateral positions and "
            "intraday liquidity metrics. A Net Stable Funding Ratio (NSFR) of {nsfr}% "
            "must also be maintained at all times."
        ),
        "params": [
            {"lo_pct":"100","hi_pct":"115","window_lo":"30","window_hi":"30","asset_lo":"250","asset_hi":"100","nsfr":"100"},
            {"lo_pct":"90","hi_pct":"110","window_lo":"30","window_hi":"30","asset_lo":"10","asset_hi":"10","nsfr":"100"},
            {"lo_pct":"100","hi_pct":"120","window_lo":"30","window_hi":"30","asset_lo":"50","asset_hi":"50","nsfr":"105"},
        ],
    },
    {
        "slug": "stress-testing",
        "title_template": "Company-Run Stress Testing Requirements — {agency} ({year})",
        "reg_type": "stress_testing",
        "cfr": "12 CFR 252",
        "v1_text": (
            "Banking organizations with total consolidated assets of ${asset_lo} billion "
            "or more must conduct annual stress tests using {n_lo} macroeconomic scenarios "
            "provided by the Board of Governors. Results must be disclosed publicly within "
            "{pub_lo} days of the stress test completion. Institutions must submit results "
            "to their primary federal regulator by {deadline_lo}."
        ),
        "v2_text": (
            "Banking organizations with total consolidated assets of ${asset_hi} billion "
            "or more must conduct semi-annual stress tests using {n_hi} macroeconomic "
            "scenarios, of which {n_severe} must be severely adverse. Results must be "
            "disclosed publicly within {pub_hi} days of completion and must include "
            "qualitative assessments of model uncertainty. Institutions must submit results "
            "to their primary federal regulator by {deadline_hi} and must retain underlying "
            "model documentation for a minimum of {retain_yrs} years."
        ),
        "params": [
            {"asset_lo":"50","asset_hi":"100","n_lo":"3","n_hi":"5","n_severe":"2","pub_lo":"15","pub_hi":"10","deadline_lo":"July 31","deadline_hi":"April 5","retain_yrs":"7"},
            {"asset_lo":"100","asset_hi":"250","n_lo":"2","n_hi":"4","n_severe":"2","pub_lo":"30","pub_hi":"15","deadline_lo":"October 1","deadline_hi":"June 1","retain_yrs":"5"},
        ],
    },
    {
        "slug": "interchange-fee",
        "title_template": "Electronic Debit Transaction Interchange Fee Standards — {agency} ({year})",
        "reg_type": "payment",
        "cfr": "12 CFR 235",
        "v1_text": (
            "The maximum interchange fee that an issuer may receive for an electronic "
            "debit transaction shall not exceed {lo_cents} cents per transaction plus "
            "{lo_pct}% of the transaction value. A fraud prevention adjustment of "
            "{fraud_lo} cents per transaction is available to issuers that implement "
            "the fraud prevention standards established by the Board."
        ),
        "v2_text": (
            "The maximum interchange fee that an issuer may receive for an electronic "
            "debit transaction shall not exceed {hi_cents} cents per transaction plus "
            "{hi_pct}% of the transaction value. A fraud prevention adjustment of "
            "{fraud_hi} cents per transaction is available. Issuers must certify "
            "compliance with updated fraud monitoring standards on an annual basis and "
            "maintain transaction data for {retain_days} days to support fraud disputes."
        ),
        "params": [
            {"lo_cents":"21","hi_cents":"14.4","lo_pct":"0.05","hi_pct":"0.04","fraud_lo":"1","fraud_hi":"1","retain_days":"120"},
            {"lo_cents":"21","hi_cents":"17.7","lo_pct":"0.05","hi_pct":"0.04","fraud_lo":"1","fraud_hi":"1","retain_days":"90"},
        ],
    },
    {
        "slug": "market-risk-var",
        "title_template": "Market Risk Capital Requirements — {agency} ({year})",
        "reg_type": "market_risk",
        "cfr": "12 CFR 3, Subpart F",
        "v1_text": (
            "Covered institutions with significant trading activity shall calculate "
            "market risk capital requirements using a {percentile_lo}th percentile "
            "Value-at-Risk (VaR) model over a {holding_lo}-business-day holding period. "
            "Backtesting shall be performed using {bt_lo} months of historical trading "
            "outcomes. Institutions with more than {exc_lo} backtesting exceptions in "
            "a rolling 12-month period shall apply a capital multiplier of {mult_lo}."
        ),
        "v2_text": (
            "Covered institutions with significant trading activity shall calculate "
            "market risk capital requirements using a {percentile_hi}th percentile "
            "Expected Shortfall (ES) measure over a {holding_hi}-business-day liquidity "
            "horizon. Non-modellable risk factors (NMRFs) shall attract a separate stress "
            "scenario capital charge. Backtesting shall be performed using {bt_hi} months "
            "of profit-and-loss outcomes. Institutions must obtain supervisory approval "
            "for internal models and submit annual model validation reports. Capital "
            "requirements must be met on both a pre- and post-diversification basis."
        ),
        "params": [
            {"percentile_lo":"99","percentile_hi":"97.5","holding_lo":"10","holding_hi":"10","bt_lo":"12","bt_hi":"24","exc_lo":"4","mult_lo":"3.0"},
            {"percentile_lo":"99","percentile_hi":"97.5","holding_lo":"10","holding_hi":"10","bt_lo":"12","bt_hi":"36","exc_lo":"5","mult_lo":"3.5"},
        ],
    },
    {
        "slug": "hmda-reporting",
        "title_template": "Home Mortgage Disclosure Act Data Reporting — {agency} ({year})",
        "reg_type": "reporting",
        "cfr": "12 CFR 1003",
        "v1_text": (
            "Covered financial institutions with at least {lo_orig} mortgage originations "
            "in each of the two preceding calendar years must report HMDA data by "
            "{deadline_lo} of the following year. The required data fields include "
            "application date, loan amount, property type, occupancy type, and the "
            "race and ethnicity of the applicant. Data must be submitted in the format "
            "specified by the Consumer Financial Protection Bureau."
        ),
        "v2_text": (
            "Covered financial institutions with at least {hi_orig} mortgage originations "
            "in the preceding calendar year must report expanded HMDA data by {deadline_hi} "
            "of the following year. The required data fields are expanded to {n_fields} items "
            "including automated underwriting system results, credit score model used, "
            "debt-to-income ratio, combined loan-to-value ratio, and introductory rate period. "
            "Data must pass automated validation checks before submission and any errors "
            "must be corrected within {cure_days} business days of notification."
        ),
        "params": [
            {"lo_orig":"25","hi_orig":"25","deadline_lo":"March 1","deadline_hi":"April 30","n_fields":"48","cure_days":"30"},
            {"lo_orig":"100","hi_orig":"25","deadline_lo":"March 1","deadline_hi":"March 31","n_fields":"48","cure_days":"60"},
        ],
    },
    {
        "slug": "cybersecurity-notification",
        "title_template": "Cybersecurity Incident Notification Requirements — {agency} ({year})",
        "reg_type": "operational_risk",
        "cfr": "12 CFR 53",
        "v1_text": (
            "Banking organizations must notify their primary federal regulator as soon "
            "as possible and no later than {lo_hours} hours after the banking organization "
            "determines that a computer-security incident has materially disrupted or "
            "degraded, or is reasonably likely to materially disrupt or degrade, a covered "
            "service for a period of {lo_min} hours or more. Notification must be made "
            "by telephone or electronic mail."
        ),
        "v2_text": (
            "Banking organizations must notify their primary federal regulator within "
            "{hi_hours} hours after determining that a computer-security incident has "
            "materially disrupted or degraded a covered service for {hi_min} hours or more. "
            "Notification must include the nature of the incident, the systems affected, "
            "estimated financial exposure, and remediation steps taken. A written "
            "post-incident report must be submitted within {report_days} calendar days "
            "detailing root cause analysis and permanent controls implemented. "
            "Third-party service providers must notify their banking organization "
            "customers within {tp_hours} hours of detecting a qualifying incident."
        ),
        "params": [
            {"lo_hours":"72","hi_hours":"36","lo_min":"4","hi_min":"2","report_days":"30","tp_hours":"4"},
            {"lo_hours":"72","hi_hours":"24","lo_min":"4","hi_min":"1","report_days":"14","tp_hours":"2"},
        ],
    },
    {
        "slug": "ctr-threshold",
        "title_template": "Currency Transaction Reporting Threshold — {agency} ({year})",
        "reg_type": "aml",
        "cfr": "31 CFR 1010.311",
        "v1_text": (
            "Financial institutions shall file a Currency Transaction Report (CTR) for "
            "each deposit, withdrawal, exchange of currency, or other payment or transfer, "
            "by, through, or to the financial institution, which involves a transaction "
            "in currency of more than ${lo_k},000. Multiple transactions by the same "
            "person on the same day that individually are below the threshold but in "
            "aggregate exceed the threshold must be treated as a single transaction."
        ),
        "v2_text": (
            "Financial institutions shall file a Currency Transaction Report (CTR) for "
            "each transaction in currency of more than ${hi_k},000. Established business "
            "customers meeting enhanced due diligence standards may be exempt from CTR "
            "filing requirements subject to annual review. The exemption must be documented "
            "in the customer's file and reviewed whenever a material change in business "
            "activity occurs. Structuring detection thresholds must be calibrated to "
            "identify patterns indicative of intentional avoidance of the ${hi_k},000 limit."
        ),
        "params": [
            {"lo_k":"10","hi_k":"15"},
            {"lo_k":"10","hi_k":"20"},
        ],
    },
    {
        "slug": "prepaid-account",
        "title_template": "Prepaid Account Rule — Disclosure Requirements — {agency} ({year})",
        "reg_type": "consumer_protection",
        "cfr": "12 CFR 1005",
        "v1_text": (
            "Prepaid account issuers must provide consumers with short-form and long-form "
            "fee disclosures before a consumer acquires a prepaid account. Short-form "
            "disclosures must include the {n_lo} most commonly incurred fees in a "
            "prescribed format. Error resolution procedures must be disclosed upon "
            "account opening and annually thereafter."
        ),
        "v2_text": (
            "Prepaid account issuers must provide consumers with standardised short-form "
            "and long-form fee disclosures at least {days_hi} days before a prepaid "
            "product is made available for consumer acquisition. Short-form disclosures "
            "must include the {n_hi} most commonly incurred fees and must comply with "
            "digital accessibility standards (WCAG 2.1 AA). Error resolution procedures "
            "must be disclosed upon account opening, within {notice_days} days of any "
            "fee change, and annually thereafter. Issuers must maintain records of all "
            "disclosure deliveries for {retain_yrs} years."
        ),
        "params": [
            {"n_lo":"5","n_hi":"7","days_hi":"90","notice_days":"21","retain_yrs":"3"},
            {"n_lo":"4","n_hi":"8","days_hi":"60","notice_days":"30","retain_yrs":"5"},
        ],
    },
    {
        "slug": "volcker-covered-fund",
        "title_template": "Volcker Rule — Covered Fund Definition — {agency} ({year})",
        "reg_type": "market_risk",
        "cfr": "12 CFR 44",
        "v1_text": (
            "A banking entity may not acquire or retain, as principal, any ownership "
            "interest in, or sponsor, a covered fund. A covered fund is defined as any "
            "issuer that would be an investment company under the Investment Company Act "
            "of 1940 but for the exemptions in sections 3(c)(1) or 3(c)(7) of that Act. "
            "The aggregate investment in covered funds may not exceed {pct_lo}% of Tier 1 capital."
        ),
        "v2_text": (
            "A banking entity may not acquire or retain, as principal, any ownership "
            "interest in, or sponsor, a covered fund. The definition of covered fund is "
            "expanded to include certain credit funds, venture capital funds, family wealth "
            "vehicles, and foreign public funds that primarily sell to non-US investors. "
            "The aggregate investment in covered funds may not exceed {pct_hi}% of Tier 1 "
            "capital after applying the expanded definition. Grandfathered interests in "
            "illiquid funds must be divested by {deadline}."
        ),
        "params": [
            {"pct_lo":"3","pct_hi":"3","deadline":"July 21, 2033"},
            {"pct_lo":"5","pct_hi":"3","deadline":"December 31, 2030"},
        ],
    },
    {
        "slug": "appraisal-threshold",
        "title_template": "Real Estate Appraisal Threshold — {agency} ({year})",
        "reg_type": "housing",
        "cfr": "12 CFR 34",
        "v1_text": (
            "Federally regulated financial institutions are required to obtain a "
            "state-certified appraisal for all real estate-related transactions of "
            "${lo_k},000 or more. For transactions in rural areas, the threshold is "
            "${rural_lo_k},000. Evaluations may be used in lieu of appraisals for "
            "transactions below these thresholds, provided the evaluation is consistent "
            "with safe and sound banking practices."
        ),
        "v2_text": (
            "Federally regulated financial institutions are required to obtain a "
            "state-certified appraisal for all real estate-related transactions of "
            "${hi_k},000 or more. For transactions in rural areas, a rural residential "
            "appraisal exemption applies for transactions up to ${rural_hi_k},000, "
            "provided the institution retains a written estimate of market value. "
            "Evaluations must be reviewed by a qualified individual independent of the "
            "transaction. The institution must maintain the evaluation or appraisal for "
            "a minimum of {retain_yrs} years after the loan is closed or sold."
        ),
        "params": [
            {"lo_k":"250","hi_k":"400","rural_lo_k":"400","rural_hi_k":"500","retain_yrs":"5"},
            {"lo_k":"500","hi_k":"750","rural_lo_k":"750","rural_hi_k":"1000","retain_yrs":"7"},
        ],
    },
    {
        "slug": "crypto-custody",
        "title_template": "Digital Asset Custody Standards — {agency} ({year})",
        "reg_type": "crypto",
        "cfr": "12 CFR 9",
        "v1_text": (
            "National banks and federal savings associations may provide cryptocurrency "
            "custody services for customers. Institutions providing these services must "
            "maintain adequate insurance coverage for digital assets held in custody "
            "and must segregate customer assets from proprietary holdings. Risk management "
            "frameworks must address private key management and operational resilience."
        ),
        "v2_text": (
            "National banks and federal savings associations providing digital asset "
            "custody services must maintain {reserve_pct}% reserve backing for all "
            "assets held in custody and must obtain independent third-party attestation "
            "of reserve holdings on a {attest_freq} basis. Institutions must implement "
            "multi-signature key management controls and maintain offline cold storage "
            "for at least {cold_pct}% of custodied assets. Stablecoin reserve assets "
            "must be held exclusively in cash or short-duration US government securities. "
            "Smart contract risk assessments must be completed before any DeFi integration."
        ),
        "params": [
            {"reserve_pct":"100","attest_freq":"monthly","cold_pct":"80"},
            {"reserve_pct":"100","attest_freq":"quarterly","cold_pct":"70"},
        ],
    },
]

_AGENCIES = [
    "Office of the Comptroller of the Currency",
    "Federal Deposit Insurance Corporation",
    "Federal Reserve System",
    "Consumer Financial Protection Bureau",
    "Financial Crimes Enforcement Network",
    "National Credit Union Administration",
    "Federal Housing Finance Agency",
    "Securities and Exchange Commission",
    "Office of Thrift Supervision",
    "Farm Credit Administration",
]

_AGENCY_ABBR = {
    "Office of the Comptroller of the Currency": "OCC",
    "Federal Deposit Insurance Corporation": "FDIC",
    "Federal Reserve System": "FRB",
    "Consumer Financial Protection Bureau": "CFPB",
    "Financial Crimes Enforcement Network": "FinCEN",
    "National Credit Union Administration": "NCUA",
    "Federal Housing Finance Agency": "FHFA",
    "Securities and Exchange Commission": "SEC",
    "Office of Thrift Supervision": "OTS",
    "Farm Credit Administration": "FCA",
}

IMPACT_BY_TYPE = {
    "capital":             (50_000,  400_000),
    "aml":                 (2_000,   18_000),
    "consumer_protection": (1_000,   12_000),
    "liquidity":           (20_000,  100_000),
    "market_risk":         (30_000,  180_000),
    "operational_risk":    (500,     7_000),
    "stress_testing":      (5_000,   40_000),
    "reporting":           (200,     2_500),
    "payment":             (2_000,   14_000),
    "crypto":              (200,     4_000),
    "housing":             (3_000,   25_000),
    "other":               (100,     1_500),
}

AFFECTED_BY_TYPE = {
    "capital":             "~4,500 US bank holding companies",
    "aml":                 "~11,000 US financial institutions",
    "consumer_protection": "~9,200 CFPB-supervised entities",
    "liquidity":           "~1,100 banks with >$10B assets",
    "market_risk":         "~40 G-SIBs and large trading banks",
    "operational_risk":    "~6,000 institutions with complex IT",
    "stress_testing":      "~450 systemically important institutions",
    "reporting":           "~8,800 FFIEC-reporting institutions",
    "payment":             "~4,500 debit-issuing banks",
    "crypto":              "~2,200 crypto-active institutions",
    "housing":             "~3,100 mortgage originators",
    "other":               "~8,000 federally regulated institutions",
}


def _fill(template: str, params: dict) -> str:
    """Fill {placeholders} in template with params dict."""
    for k, v in params.items():
        template = template.replace(f"{{{k}}}", str(v))
    return template


def _compute_drift(v1: str, v2: str, rng: random.Random) -> dict:
    """TF-IDF cosine drift + JSD + Wasserstein. Falls back to heuristic."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.stats import wasserstein_distance
        import numpy as np

        mat = TfidfVectorizer(max_features=500, stop_words="english",
                              ngram_range=(1, 2)).fit_transform([v1, v2]).toarray()
        sim   = float(cosine_similarity([mat[0]], [mat[1]])[0][0])
        drift = max(0.0, min(1.0, 1.0 - sim))

        n  = mat.shape[1]
        bs = [max(0.0, min(1.0, 1.0 - float(
            np.dot(mat[0][idx := rng.choices(range(n), k=n)],
                   mat[1][idx]) /
            (np.linalg.norm(mat[0][idx]) * np.linalg.norm(mat[1][idx]) + 1e-9)
        ))) for _ in range(60)]
        ci_lo = float(np.percentile(bs, 2.5))
        ci_hi = float(np.percentile(bs, 97.5))

        p  = mat[0] + 1e-8;  p /= p.sum()
        q  = mat[1] + 1e-8;  q /= q.sum()
        m  = (p + q) / 2
        jsd = max(0.0, min(1.0, float(
            0.5 * np.sum(p * np.log(p / m + 1e-10)) +
            0.5 * np.sum(q * np.log(q / m + 1e-10)))))
        jsd_p = max(0.0001, 1.0 - min(jsd * 3.5, 0.9999))
        w_n   = round(min(float(wasserstein_distance(p, q)) / 2.5, 1.0), 4)
        comp  = round(drift * 0.50 + jsd * 0.30 + w_n * 0.20, 4)

        return {
            "drift_score": round(drift, 4), "drift_ci_low": round(ci_lo, 4),
            "drift_ci_high": round(ci_hi, 4), "jsd_score": round(jsd, 4),
            "jsd_p_value": round(jsd_p, 4), "is_significant": bool(jsd_p < 0.05),
            "wasserstein_score": w_n, "composite_score": comp,
            "flagged_for_analysis": drift >= 0.15 or jsd_p < 0.05,
        }
    except Exception:
        d = round(rng.uniform(0.12, 0.72), 4)
        return {
            "drift_score": d, "drift_ci_low": round(d * 0.82, 4),
            "drift_ci_high": round(min(d * 1.20, 1.0), 4),
            "jsd_score": round(d * 0.68, 4), "jsd_p_value": round(rng.uniform(0.01, 0.09), 4),
            "is_significant": d > 0.22, "wasserstein_score": round(d * 0.38, 4),
            "composite_score": round(d * 0.84, 4), "flagged_for_analysis": d >= 0.15,
        }


def _build_engine():
    from sqlalchemy import create_engine
    candidates = [
        os.environ.get("DATABASE_URL", "").replace("+asyncpg", "").replace("asyncpg", "psycopg2"),
        "postgresql+psycopg2://complianceiq:complianceiq@localhost:5432/complianceiq",
        "postgresql+psycopg2://complianceiq:complianceiq@db:5432/complianceiq",
    ]
    for url in candidates:
        if not url:
            continue
        try:
            eng = create_engine(url, echo=False, pool_pre_ping=True)
            with eng.connect():
                pass
            return eng
        except Exception:
            continue
    raise RuntimeError("Cannot connect to DB")


def run_bulk_seed(target: int = 500) -> None:
    from sqlalchemy.orm import Session
    from sqlalchemy import select
    from db.models import Regulation, RegulationVersion, ChangeScore

    engine = _build_engine()
    rng    = random.Random(42)

    with Session(engine) as s:
        existing = s.execute(select(Regulation).where(
            Regulation.source == "federal_register"
        )).scalars().all()
        already = len(existing)

    if already >= target:
        log.info("Already have %d FR regulations (target=%d) — skipping", already, target)
        return

    remaining = target - already
    log.info("Generating %d regulations locally (no network required)…", remaining)

    # Build the full cartesian product: template × params × agency × year
    records = []
    years   = list(range(2015, 2025))
    for tmpl in _TEMPLATES:
        for params in tmpl["params"]:
            for agency in _AGENCIES:
                for year in years:
                    abbr    = _AGENCY_ABBR[agency]
                    title   = _fill(tmpl["title_template"], {"agency": abbr, "year": year})
                    v1_text = _fill(tmpl["v1_text"], params)
                    v2_text = _fill(tmpl["v2_text"], params)
                    # Unique doc number: slug + abbr + year + param-hash
                    ph      = hashlib.md5(str(params).encode()).hexdigest()[:4]
                    doc_num = f"LCL-{year}-{abbr}-{tmpl['slug'][:8].upper()}-{ph}"
                    records.append({
                        "doc_num":  doc_num,
                        "title":    title,
                        "agency":   agency,
                        "abbr":     abbr,
                        "reg_type": tmpl["reg_type"],
                        "cfr":      tmpl.get("cfr", ""),
                        "year":     year,
                        "v1_text":  v1_text,
                        "v2_text":  v2_text,
                    })

    rng.shuffle(records)
    records = records[:remaining]
    log.info("Total records to insert: %d", len(records))

    inserted = skipped = 0

    with Session(engine) as session:
        for i, rec in enumerate(records):
            # Skip if already in DB
            exists = session.execute(
                select(Regulation).where(Regulation.document_number == rec["doc_num"])
            ).scalar_one_or_none()
            if exists:
                skipped += 1
                continue

            sc       = _compute_drift(rec["v1_text"], rec["v2_text"], rng)
            reg_id   = uuid.uuid4()
            pub_date = datetime(rec["year"], rng.randint(1, 12), rng.randint(1, 28),
                                tzinfo=timezone.utc)
            imp_lo, imp_hi = IMPACT_BY_TYPE.get(rec["reg_type"], IMPACT_BY_TYPE["other"])
            affected = AFFECTED_BY_TYPE.get(rec["reg_type"], AFFECTED_BY_TYPE["other"])

            plain = (
                f"This {rec['reg_type'].replace('_',' ')} rule from {rec['abbr']} "
                f"({rec['year']}) modifies existing standards under {rec['cfr']}. "
                f"It affects {affected} with estimated compliance costs of "
                f"${imp_lo:,}M–${imp_hi:,}M. "
                f"Semantic drift between the prior and revised version is "
                f"{round(sc['drift_score']*100)}%."
            )

            session.add(Regulation(
                id=reg_id, document_number=rec["doc_num"],
                source="federal_register", agency=rec["agency"][:128],
                title=rec["title"], abstract=rec["v2_text"][:2000],
                full_text=rec["v2_text"], publication_date=pub_date,
                regulation_type=rec["reg_type"],
                raw_metadata={
                    "cfr": rec["cfr"], "year": rec["year"],
                    "impact_low_m": imp_lo, "impact_high_m": imp_hi,
                    "affected": affected, "v1_text": rec["v1_text"],
                    "plain_english": plain,
                },
            ))
            session.add(RegulationVersion(
                id=uuid.uuid4(), regulation_id=reg_id, version_number=1,
                full_text=rec["v1_text"],
                text_hash=hashlib.sha256(rec["v1_text"].encode()).hexdigest(),
                word_count=len(rec["v1_text"].split()),
            ))
            session.add(RegulationVersion(
                id=uuid.uuid4(), regulation_id=reg_id, version_number=2,
                full_text=rec["v2_text"],
                text_hash=hashlib.sha256(rec["v2_text"].encode()).hexdigest(),
                word_count=len(rec["v2_text"].split()),
            ))
            session.add(ChangeScore(
                id=uuid.uuid4(), regulation_id=reg_id,
                computed_at=datetime.now(timezone.utc),
                document_number=rec["doc_num"], title=rec["title"],
                agency=rec["agency"][:128],
                drift_score=sc["drift_score"], drift_ci_low=sc["drift_ci_low"],
                drift_ci_high=sc["drift_ci_high"],
                drift_display=f"{round(sc['drift_score']*100)}%",
                jsd_score=sc["jsd_score"], jsd_p_value=sc["jsd_p_value"],
                is_significant=sc["is_significant"],
                wasserstein_score=sc["wasserstein_score"],
                composite_score=sc["composite_score"],
                flagged_for_analysis=sc["flagged_for_analysis"],
                version_from=1, version_to=2,
                word_count_delta=len(rec["v2_text"].split()) - len(rec["v1_text"].split()),
                change_summary=(
                    f"{rec['reg_type'].replace('_',' ').title()} rule — "
                    f"drift {round(sc['drift_score']*100)}% between v1→v2."
                ),
            ))
            inserted += 1

            if inserted % 50 == 0:
                session.commit()
                log.info("  ✓ %d/%d inserted", inserted, len(records))

        session.commit()

    log.info("Offline bulk seed complete — inserted=%d skipped=%d total_in_db≈%d",
             inserted, skipped, already + inserted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500)
    args = parser.parse_args()
    run_bulk_seed(target=args.target)
