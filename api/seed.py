"""Self-contained async seed for ComplianceIQ.

Generates 3,000+ realistic regulation records with synthetic (but plausible)
drift scores directly into the Postgres DB using the API's own async session.

Design decisions
----------------
- Zero external deps beyond SQLAlchemy (already part of the API stack).
  No sklearn, no ChromaDB, no subprocesses.
- All drift/JSD/Wasserstein scores are computed from random distributions
  seeded deterministically (reproducible across restarts).
- Fully idempotent — skips any document_number already present.
- Called from api/main.py lifespan if COUNT(*) FROM regulations = 0.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# ── Templates ──────────────────────────────────────────────────────────────────

_AGENCIES = [
    ("Federal Reserve", "FED"),
    ("OCC", "OCC"),
    ("CFPB", "CFPB"),
    ("FDIC", "FDIC"),
    ("SEC", "SEC"),
    ("CFTC", "CFTC"),
    ("FinCEN", "FCEN"),
    ("NCUA", "NCUA"),
    ("FHFA", "FHFA"),
    ("OFR", "OFR"),
]

_TEMPLATES = [
    {
        "slug": "capital-adequacy",
        "cfr": "12 CFR 3",
        "reg_type": "final_rule",
        "title": "Capital Adequacy Standards — {agency} Final Rule ({year})",
        "v1": "Each covered institution shall maintain a minimum CET1 capital ratio of {lo}% of total risk-weighted assets. The leverage ratio shall be {lev_lo}% of average total consolidated assets. Institutions failing requirements shall submit a capital restoration plan within {days} calendar days.",
        "v2": "Each covered institution shall maintain a minimum CET1 capital ratio of {hi}% inclusive of a 2.5 percent conservation buffer. The leverage ratio shall be {lev_hi}% of average total consolidated assets. Institutions failing requirements shall submit a capital restoration plan within {days2} calendar days and are prohibited from capital distributions without prior approval.",
        "params": [{"lo":"4.5","hi":"7.0","lev_lo":"3","lev_hi":"5","days":"45","days2":"30"},{"lo":"6.0","hi":"8.5","lev_lo":"4","lev_hi":"6","days":"60","days2":"45"},{"lo":"5.5","hi":"9.0","lev_lo":"3.5","lev_hi":"5.5","days":"30","days2":"21"}],
        "plain_english": "Requires larger capital cushions. Banks must hold {hi}% CET1 — up from {lo}%. Failure means restricted dividends.",
        "impact_lo": 1.2, "impact_hi": 3.8, "affected": "Banks with >$100B assets",
    },
    {
        "slug": "aml-cdd",
        "cfr": "31 CFR 1010",
        "reg_type": "final_rule",
        "title": "AML Customer Due Diligence Requirements — {agency} ({year})",
        "v1": "Covered institutions must identify beneficial owners holding {lo}% or more of equity interest in legal entity customers. Verification must occur at account opening using government-issued identification.",
        "v2": "Covered institutions must identify beneficial owners holding {hi}% or more of equity interest, with enhanced due diligence for politically exposed persons and high-risk jurisdictions. Continuous monitoring required for accounts with annual transactions exceeding ${thresh}M.",
        "params": [{"lo":"25","hi":"10","thresh":"500"},{"lo":"20","hi":"10","thresh":"250"},{"lo":"25","hi":"15","thresh":"1000"}],
        "plain_english": "Tightens beneficial ownership rules — threshold drops from {lo}% to {hi}%. Continuous transaction monitoring now required.",
        "impact_lo": 0.8, "impact_hi": 2.1, "affected": "All FDIC-insured institutions",
    },
    {
        "slug": "liquidity-coverage",
        "cfr": "12 CFR 249",
        "reg_type": "final_rule",
        "title": "Liquidity Coverage Ratio Update — {agency} ({year})",
        "v1": "Each covered company shall maintain an amount of high-quality liquid assets (HQLA) equal to or greater than {lo}% of its projected net cash outflows over a {days}-calendar-day stress period.",
        "v2": "Each covered company shall maintain HQLA equal to or greater than {hi}% of projected net cash outflows over a {days2}-calendar-day stress period. Level 2B assets are capped at {cap}% of total HQLA. Intraday liquidity monitoring required for Category I institutions.",
        "params": [{"lo":"80","hi":"100","days":"30","days2":"30","cap":"15"},{"lo":"90","hi":"110","days":"30","days2":"45","cap":"10"},{"lo":"75","hi":"100","days":"21","days2":"30","cap":"20"}],
        "plain_english": "Raises the liquidity buffer from {lo}% to {hi}%. Banks must hold more cash-equivalent assets to survive a 30-day stress scenario.",
        "impact_lo": 0.5, "impact_hi": 1.9, "affected": "Category I and II banking organizations",
    },
    {
        "slug": "stress-testing",
        "cfr": "12 CFR 252",
        "reg_type": "final_rule",
        "title": "Stress Testing Requirements — {agency} ({year})",
        "v1": "Bank holding companies with total consolidated assets of ${lo}B or more must conduct annual company-run stress tests under baseline, adverse, and severely adverse scenarios. Results must be disclosed publicly within {days} days of submission.",
        "v2": "Bank holding companies with total consolidated assets of ${hi}B or more must conduct semi-annual stress tests. Climate-related financial risk scenarios are added as a mandatory fourth scenario category. Results must be disclosed publicly within {days2} days.",
        "params": [{"lo":"100","hi":"100","days":"45","days2":"30"},{"lo":"50","hi":"50","days":"60","days2":"45"},{"lo":"250","hi":"100","days":"45","days2":"21"}],
        "plain_english": "Expands stress testing to include climate risk scenarios. Publication window shortened to {days2} days.",
        "impact_lo": 0.3, "impact_hi": 1.2, "affected": "BHCs with >$100B in assets",
    },
    {
        "slug": "cybersecurity-notification",
        "cfr": "12 CFR 748",
        "reg_type": "final_rule",
        "title": "Cybersecurity Incident Notification — {agency} ({year})",
        "v1": "Covered institutions must notify the appropriate federal banking regulator as soon as possible, and no later than {lo} hours after a computer-security incident that rises to the level of a notification incident.",
        "v2": "Covered institutions must notify the appropriate federal banking regulator within {hi} hours of discovering a notification incident. Notification must occur prior to any public disclosure. Banking service providers must notify affected institutions within {hrs2} hours.",
        "params": [{"lo":"72","hi":"36","hrs2":"4"},{"lo":"72","hi":"24","hrs2":"2"},{"lo":"96","hi":"36","hrs2":"6"}],
        "plain_english": "Cuts the reporting window from {lo}h to {hi}h. Notice must be sent before going public.",
        "impact_lo": 0.2, "impact_hi": 0.9, "affected": "All banking institutions and service providers",
    },
    {
        "slug": "market-risk-var",
        "cfr": "12 CFR 217",
        "reg_type": "final_rule",
        "title": "Market Risk Capital — Internal Models — {agency} ({year})",
        "v1": "Covered positions subject to market risk capital requirements must be measured using a Value-at-Risk model calibrated to a {lo}-day holding period at a {conf}% confidence level. Backtesting required quarterly.",
        "v2": "Covered positions must use an Expected Shortfall model calibrated to a {hi}-day stressed period at a {conf2}% confidence level, replacing VaR. Backtesting and P&L attribution tests required monthly. Non-modellable risk factors attract a standardised capital surcharge.",
        "params": [{"lo":"10","hi":"10","conf":"99","conf2":"97.5"},{"lo":"10","hi":"20","conf":"99","conf2":"99"},{"lo":"1","hi":"10","conf":"95","conf2":"97.5"}],
        "plain_english": "Replaces VaR with Expected Shortfall — captures tail risk better. Monthly backtesting now mandatory.",
        "impact_lo": 1.5, "impact_hi": 4.2, "affected": "Banks with large trading books (>$1B)",
    },
    {
        "slug": "hmda-reporting",
        "cfr": "12 CFR 1003",
        "reg_type": "final_rule",
        "title": "HMDA Data Reporting Standards — {agency} ({year})",
        "v1": "Financial institutions that originate {lo} or more covered loans in each of the two preceding calendar years must report HMDA data by {deadline} of the following year.",
        "v2": "Financial institutions that originate {hi} or more covered loans in the preceding calendar year must report expanded HMDA data including {fields} new data fields by {deadline2} of the following year. Quarterly reporting required for institutions originating 2,000+ loans.",
        "params": [{"lo":"25","hi":"25","deadline":"March 1","deadline2":"March 1","fields":"48"},{"lo":"100","hi":"25","deadline":"March 1","deadline2":"April 1","fields":"110"},{"lo":"50","hi":"10","deadline":"April 1","deadline2":"March 1","fields":"48"}],
        "plain_english": "Expands HMDA data collection by {fields} new fields including pricing and underwriting data.",
        "impact_lo": 0.1, "impact_hi": 0.5, "affected": "Mortgage lenders",
    },
    {
        "slug": "interchange-fee",
        "cfr": "12 CFR 235",
        "reg_type": "final_rule",
        "title": "Debit Card Interchange Fee Cap — {agency} ({year})",
        "v1": "The maximum permissible interchange fee for an electronic debit transaction is ${lo} plus {pct_lo}% of the value of the transaction. Issuers may also receive a {fraud_lo}-cent fraud prevention adjustment.",
        "v2": "The maximum permissible interchange fee is ${hi} plus {pct_hi}% of the transaction value, effective {eff}. The fraud prevention adjustment is revised to {fraud_hi} cents. Network exclusivity and routing restrictions apply.",
        "params": [{"lo":"0.21","hi":"0.14","pct_lo":"0.05","pct_hi":"0.04","fraud_lo":"1","fraud_hi":"1.3","eff":"2024-07-01"},{"lo":"0.21","hi":"0.21","pct_lo":"0.05","pct_hi":"0.05","fraud_lo":"1","fraud_hi":"1.3","eff":"2025-01-01"},{"lo":"0.22","hi":"0.14","pct_lo":"0.06","pct_hi":"0.04","fraud_lo":"1.3","fraud_hi":"1.5","eff":"2024-10-01"}],
        "plain_english": "Lowers the debit interchange cap from {lo}¢ to {hi}¢ per transaction — directly reduces bank revenue on debit cards.",
        "impact_lo": 2.0, "impact_hi": 5.5, "affected": "Debit card issuers with >$10B in assets",
    },
    {
        "slug": "crypto-custody",
        "cfr": "12 CFR 7",
        "reg_type": "proposed_rule",
        "title": "Digital Asset Custody Standards — {agency} ({year})",
        "v1": "National banks may provide cryptocurrency custody services provided they maintain adequate internal controls. Customer assets must be segregated from proprietary assets. No specific capital requirement for digital asset exposure.",
        "v2": "National banks providing digital asset custody must hold capital equal to {pct}% of total digital asset assets under custody. Assets must be held in cold storage except for {hot}% maintained in hot wallets for operational purposes. Independent third-party audit required annually.",
        "params": [{"pct":"1","hot":"5"},{"pct":"2","hot":"10"},{"pct":"0.5","hot":"3"}],
        "plain_english": "First binding capital rule for crypto custody — banks must hold {pct}% of AUC in reserve capital.",
        "impact_lo": 0.4, "impact_hi": 2.8, "affected": "Banks with digital asset custody services",
    },
    {
        "slug": "climate-risk",
        "cfr": "12 CFR 30",
        "reg_type": "proposed_rule",
        "title": "Climate-Related Financial Risk Management — {agency} ({year})",
        "v1": "Large financial institutions are encouraged to consider climate-related risks in their risk management frameworks on a voluntary basis. No specific supervisory expectations established.",
        "v2": "Financial institutions with total assets of ${thresh}B or more must develop and implement climate-related financial risk management principles including scenario analysis under 1.5°C and 2°C pathways. Transition plans required by {deadline}.",
        "params": [{"thresh":"100","deadline":"2026-01-01"},{"thresh":"50","deadline":"2025-07-01"},{"thresh":"250","deadline":"2027-01-01"}],
        "plain_english": "Makes climate stress testing mandatory for large banks. Both physical and transition risk scenarios required.",
        "impact_lo": 0.6, "impact_hi": 3.1, "affected": f"Banks with >$100B assets",
    },
    {
        "slug": "open-banking",
        "cfr": "12 CFR 1033",
        "reg_type": "final_rule",
        "title": "Personal Financial Data Rights — Open Banking — {agency} ({year})",
        "v1": "Financial institutions must provide consumers with access to their own account data upon request. No specific technical standard or timeline required.",
        "v2": "Financial institutions with more than {thresh} accounts must provide consumer-authorized third parties with access to covered data via standardized APIs within {days} days of a consumer authorization. Data must be provided in machine-readable format at no charge.",
        "params": [{"thresh":"500,000","days":"30"},{"thresh":"250,000","days":"60"},{"thresh":"1,000,000","days":"45"}],
        "plain_english": "Mandates open banking APIs. Consumers can authorize fintechs to access their banking data instantly.",
        "impact_lo": 0.3, "impact_hi": 1.8, "affected": "Banks with >500K deposit accounts",
    },
    {
        "slug": "volcker-covered-fund",
        "cfr": "12 CFR 248",
        "reg_type": "final_rule",
        "title": "Volcker Rule — Covered Fund Exclusions — {agency} ({year})",
        "v1": "Banking entities are prohibited from acquiring or retaining an ownership interest in or sponsoring a covered fund. Credit funds, venture capital funds, and family wealth management vehicles are not excluded from the definition.",
        "v2": "The definition of covered fund is revised to exclude: credit funds that do not issue asset-backed securities, venture capital funds meeting SEC criteria, family wealth management vehicles, and customer facilitation vehicles. Existing fund interests grandfathered for {yrs} years.",
        "params": [{"yrs":"5"},{"yrs":"3"},{"yrs":"7"}],
        "plain_english": "Loosens Volcker Rule to permit banks to participate in credit and VC funds — targeted reversal of the 2013 restrictions.",
        "impact_lo": 0.8, "impact_hi": 2.4, "affected": "Bank holding companies with proprietary trading",
    },
    {
        "slug": "prepaid-account",
        "cfr": "12 CFR 1005",
        "reg_type": "final_rule",
        "title": "Prepaid Account Disclosure Requirements — {agency} ({year})",
        "v1": "Issuers of prepaid accounts must provide a short form disclosure listing fees before account opening. Electronic delivery permitted for online acquisitions.",
        "v2": "Issuers must provide a pre-acquisition short-form and long-form disclosure for all prepaid accounts. Error resolution and limited liability protections extended to all prepaid accounts. Hybrid prepaid-credit features trigger credit card protections. Overdraft fees capped at ${cap} per month.",
        "params": [{"cap":"15"},{"cap":"10"},{"cap":"25"}],
        "plain_english": "Extends full consumer protections to prepaid cards. Overdraft fees capped at ${cap}/month.",
        "impact_lo": 0.1, "impact_hi": 0.6, "affected": "Prepaid card issuers",
    },
]


# ── Main entry point ──────────────────────────────────────────────────────────

async def seed_db(session: AsyncSession, target: int = 3300) -> int:
    """Generate and insert regulations. Returns number inserted."""
    rng = random.Random(42)
    years = list(range(2015, 2027))
    inserted = 0

    for tmpl in _TEMPLATES:
        for agency_name, agency_abbr in _AGENCIES:
            for year in years:
                for params in tmpl["params"]:
                    # Unique doc number
                    ph = hashlib.md5(
                        f"{tmpl['slug']}{agency_abbr}{year}{json.dumps(params, sort_keys=True)}".encode()
                    ).hexdigest()[:6].upper()
                    doc_num = f"LCL-{year}-{agency_abbr}-{tmpl['slug'][:8].upper()}-{ph}"

                    # Skip if exists
                    existing = (await session.execute(
                        text("SELECT 1 FROM regulations WHERE document_number = :dn"),
                        {"dn": doc_num},
                    )).fetchone()
                    if existing:
                        continue

                    # Build texts
                    try:
                        v1 = tmpl["v1"].format(**params)
                        v2 = tmpl["v2"].format(**params)
                        title = tmpl["title"].format(agency=agency_name, year=year, **params)
                        plain = tmpl.get("plain_english", "").format(**params)
                    except KeyError:
                        v1 = tmpl["v1"]
                        v2 = tmpl["v2"]
                        title = tmpl["title"].format(agency=agency_name, year=year)
                        plain = tmpl.get("plain_english", "")

                    # Synthetic drift scores (deterministic per record)
                    seed_val = int(hashlib.md5(doc_num.encode()).hexdigest()[:8], 16)
                    drng = random.Random(seed_val)
                    drift = round(drng.uniform(0.15, 0.88), 4)
                    jsd = round(drng.uniform(0.05, 0.60), 4)
                    wass = round(drng.uniform(0.02, 0.45), 4)
                    composite = round(0.5 * drift + 0.3 * jsd + 0.2 * wass, 4)
                    ci_half = round(drng.uniform(0.02, 0.06), 4)
                    jsd_p = round(1 - min(jsd * 3.5, 0.9999), 4)
                    is_sig = jsd_p < 0.05
                    flagged = composite > 0.65

                    # Publication date
                    pub_month = rng.randint(1, 12)
                    pub_day = rng.randint(1, 28)
                    pub_date = date(year, pub_month, pub_day)

                    meta = {
                        "plain_english": plain,
                        "v1_text": v1[:300],
                        "v2_text": v2[:300],
                        "impact_low_m": tmpl.get("impact_lo", 0),
                        "impact_high_m": tmpl.get("impact_hi", 0),
                        "affected": tmpl.get("affected", ""),
                        "cfr": tmpl["cfr"],
                    }

                    reg_id = uuid.uuid4()

                    try:
                        # Insert regulation
                        await session.execute(text("""
                            INSERT INTO regulations
                                (id, document_number, title, agency, abstract,
                                 publication_date, regulation_type, source,
                                 full_text, raw_metadata, created_at, updated_at)
                            VALUES
                                (:id, :dn, :title, :agency, :abstract,
                                 :pub_date, :reg_type, 'offline_seed',
                                 :full_text, CAST(:meta AS jsonb), NOW(), NOW())
                        """), {
                            "id": str(reg_id),
                            "dn": doc_num,
                            "title": title[:500],
                            "agency": agency_name,
                            "abstract": v2[:1000],
                            "pub_date": pub_date.isoformat(),
                            "reg_type": tmpl["reg_type"],
                            "full_text": v2,
                            "meta": json.dumps(meta),
                        })

                        # Insert v1 version — text_hash required (NOT NULL)
                        v1_hash = hashlib.sha256(v1.encode()).hexdigest()
                        await session.execute(text("""
                            INSERT INTO regulation_versions
                                (id, regulation_id, version_number, full_text,
                                 text_hash, word_count, fetched_at)
                            VALUES (:vid, :rid, 1, :text, :hash, :wc, NOW())
                            ON CONFLICT (regulation_id, version_number) DO NOTHING
                        """), {"vid": str(uuid.uuid4()), "rid": str(reg_id), "text": v1,
                               "hash": v1_hash, "wc": len(v1.split())})

                        # Insert v2 version
                        v2_hash = hashlib.sha256(v2.encode()).hexdigest()
                        await session.execute(text("""
                            INSERT INTO regulation_versions
                                (id, regulation_id, version_number, full_text,
                                 text_hash, word_count, fetched_at)
                            VALUES (:vid, :rid, 2, :text, :hash, :wc, NOW())
                            ON CONFLICT (regulation_id, version_number) DO NOTHING
                        """), {"vid": str(uuid.uuid4()), "rid": str(reg_id), "text": v2,
                               "hash": v2_hash, "wc": len(v2.split())})

                        # Insert change score
                        await session.execute(text("""
                            INSERT INTO change_scores
                                (id, regulation_id, version_old, version_new,
                                 drift_score, drift_ci_low, drift_ci_high,
                                 jsd_score, jsd_p_value, wasserstein_score,
                                 is_significant, flagged_for_analysis, computed_at)
                            VALUES
                                (:id, :rid, 1, 2,
                                 :drift, :ci_lo, :ci_hi,
                                 :jsd, :jsd_p, :wass,
                                 :is_sig, :flagged, NOW())
                            ON CONFLICT DO NOTHING
                        """), {
                            "id": str(uuid.uuid4()),
                            "rid": str(reg_id),
                            "drift": drift,
                            "ci_lo": max(0.0, drift - ci_half),
                            "ci_hi": min(1.0, drift + ci_half),
                            "jsd": jsd,
                            "jsd_p": jsd_p,
                            "wass": wass,
                            "is_sig": is_sig,
                            "flagged": flagged,
                        })

                        inserted += 1

                        if inserted >= target:
                            await session.commit()
                            return inserted

                        # Commit every 10 records — ensures partial runs persist
                        # even if the task is interrupted mid-way
                        if inserted % 10 == 0:
                            await session.commit()

                    except Exception as exc:
                        await session.rollback()
                        # Skip this record and continue
                        continue

    return inserted
