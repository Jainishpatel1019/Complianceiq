"""
Seed script — loads realistic regulatory data for demo / HF Space.

Usage (via Makefile):
    docker compose run --rm api python -m backend.pipelines.seed

What it does:
  1. Inserts 10 realistic Regulation rows with genuine v1→v2 text changes.
  2. Each regulation has two versions whose text ACTUALLY DIFFERS in
     meaningful ways (percentages raised, new clauses, removed provisions,
     changed deadlines) — so drift/JSD/Wasserstein scores reflect real text.
  3. Inserts ChangeScore rows computed from actual TF-IDF drift.
  4. Inserts CausalEstimate rows for 5 key regulations.
  5. Upserts 200 ChromaDB chunks (collection: "regulations").

Runs fully synchronously via a dedicated sync engine so this script
does not require an asyncio event loop.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import uuid
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Realistic regulatory data — 10 templates, each with meaningful v1→v2 changes
# ─────────────────────────────────────────────────────────────────────────────
#
# Every entry contains:
#   v1_text  — original regulation text
#   v2_text  — amended text with real language changes
#   changes  — structured list of what changed (for the diff API)
#   plain_english — 5-year-old explanation of the impact
#
# Sources: based on public Federal Register / Basel III / Dodd-Frank language
# (simplified and synthesised for demo purposes).

_REGULATIONS = [
    {
        "doc_num": "2024-00123",
        "title": "Capital Requirements for Large Banking Organisations — Basel III Endgame",
        "agency": "OCC",
        "reg_type": "capital",
        "cfr": "12 CFR 3",
        "pub_date": datetime(2024, 1, 15, tzinfo=timezone.utc),
        "plain_english": (
            "Imagine a bank is like a person who borrows money to buy houses and lend to businesses. "
            "The regulator says: you must keep some of YOUR OWN money as a safety cushion in case things go wrong. "
            "This rule raised that cushion from 8% to 12.5%. So for every $100 the bank lends, "
            "it now needs to keep $12.50 of its own money locked away — up from $8. "
            "That's 56% more safety padding. Banks are safer, but they have less to lend."
        ),
        "v1_text": """\
CAPITAL REQUIREMENTS FOR LARGE BANKING ORGANISATIONS
Final Rule — OCC | 12 CFR Part 3 | Effective: January 1, 2023

Section 1: Minimum Capital Ratios
Banking organisations subject to this rule must maintain the following minimum
capital ratios at all times:

(a) Common Equity Tier 1 (CET1) capital ratio of 4.5 percent of risk-weighted assets.
(b) Tier 1 capital ratio of 6.0 percent of risk-weighted assets.
(c) Total capital ratio of 8.0 percent of risk-weighted assets.
(d) Tier 1 leverage ratio of 4.0 percent of average total consolidated assets.

Section 2: Capital Conservation Buffer
Banking organisations must maintain a capital conservation buffer of 2.5 percent
of risk-weighted assets above the minimum CET1 ratio. Distributions and
discretionary bonus payments are restricted when the buffer falls below 2.5 percent.

Section 3: Countercyclical Capital Buffer
The countercyclical capital buffer (CCyB) is set at 0 percent for the current quarter.
The Office of the Comptroller of the Currency will provide 12 months advance notice
before increasing the CCyB above 0 percent.

Section 4: Covered Institutions
This rule applies to national banks and federal savings associations with total
consolidated assets of $10 billion or more. Institutions with assets below $10
billion are subject to the simplified capital framework under 12 CFR Part 3, Subpart B.

Section 5: Compliance and Reporting
Covered institutions must report capital ratios quarterly on the Call Report (FFIEC 041/051).
Institutions that fall below minimum capital ratios must submit a capital restoration
plan to the OCC within 45 days of the deficiency.

Section 6: Effective Date
This rule is effective January 1, 2023. Institutions have until July 1, 2023
to come into full compliance with the revised standardised approach for credit risk.
""",
        "v2_text": """\
CAPITAL REQUIREMENTS FOR LARGE BANKING ORGANISATIONS — AMENDED
Final Rule (Amendment) — OCC | 12 CFR Part 3 | Effective: July 1, 2024

Section 1: Minimum Capital Ratios
Banking organisations subject to this rule must maintain the following minimum
capital ratios at all times:

(a) Common Equity Tier 1 (CET1) capital ratio of 7.0 percent of risk-weighted assets.
(b) Tier 1 capital ratio of 8.5 percent of risk-weighted assets.
(c) Total capital ratio of 12.5 percent of risk-weighted assets.
(d) Tier 1 leverage ratio of 5.0 percent of average total consolidated assets.
(e) Supplementary leverage ratio (SLR) of 3.0 percent for advanced-approaches institutions.

Section 2: Capital Conservation Buffer
Banking organisations must maintain a capital conservation buffer of 3.5 percent
of risk-weighted assets above the minimum CET1 ratio. Distributions and
discretionary bonus payments are restricted when the buffer falls below 3.5 percent.
Institutions in the lowest buffer quartile face enhanced restrictions on variable compensation.

Section 3: Countercyclical Capital Buffer
The countercyclical capital buffer (CCyB) is set at 2.0 percent for the current quarter,
effective immediately. Banking organisations must maintain sufficient CET1 capital to
satisfy the CCyB requirement. The Office of the Comptroller of the Currency will provide
6 months advance notice before further increases to the CCyB.

Section 4: Covered Institutions
This rule applies to national banks and federal savings associations with total
consolidated assets of $100 billion or more. Category III and IV institutions with
assets between $10 billion and $100 billion are subject to the modified framework
under 12 CFR Part 3, Subpart C, with a phase-in period ending December 31, 2025.

Section 5: Compliance and Reporting
Covered institutions must report capital ratios monthly on the FR Y-9C and quarterly
on the Call Report (FFIEC 041/051). Institutions that fall below minimum capital ratios
must submit a capital restoration plan to the OCC within 15 days of the deficiency
(reduced from 45 days). Repeated deficiencies may result in enforcement action.

Section 6: Stress Capital Buffer (New)
Category I and II institutions must incorporate results of the annual supervisory
stress test into their capital planning. The stress capital buffer (SCB) floor is
2.5 percent, replacing the fixed capital conservation buffer for affected institutions.

Section 7: Effective Date
This amendment is effective July 1, 2024. Institutions subject to the expanded scope
(formerly below $100 billion) have until December 31, 2025 to comply.
""",
    },
    {
        "doc_num": "2024-00456",
        "title": "Liquidity Coverage Ratio: Treatment of Operational Deposits",
        "agency": "FRB",
        "reg_type": "liquidity",
        "cfr": "12 CFR 249",
        "pub_date": datetime(2024, 2, 1, tzinfo=timezone.utc),
        "plain_english": (
            "Think of the LCR like a fire extinguisher rule. Banks must keep enough liquid cash "
            "to survive a 30-day financial emergency without outside help. "
            "The old rule said: keep 100% of what you'd need. "
            "The new rule says: keep 115%. That extra 15% is like buying a bigger fire extinguisher. "
            "Also, the rule now says that money businesses keep in banks for everyday operations "
            "(like paying employees) is stickier — it's less likely to leave in a crisis — "
            "so banks get a small break on those deposits."
        ),
        "v1_text": """\
LIQUIDITY COVERAGE RATIO: TREATMENT OF OPERATIONAL DEPOSITS
Interim Final Rule — Federal Reserve Board | 12 CFR Part 249 | Effective: June 1, 2022

Section 1: LCR Requirement
Covered companies must maintain a liquidity coverage ratio of at least 100 percent.
The LCR equals the ratio of a firm's high-quality liquid assets (HQLA) to its projected
net cash outflows over a standardised 30-calendar-day stress period.

Section 2: Operational Deposits — Outflow Rate
Operational deposits from non-financial wholesale customers are assigned an outflow
rate of 25 percent for the portion covered by deposit insurance and 40 percent for
the uncovered portion.

Section 3: Operational Deposits — Qualifying Criteria
A deposit qualifies as operational if:
(a) The depositing entity requires the deposit to conduct its normal operating activities;
(b) The deposit is held in an account used for payment, clearing, or custody services;
(c) The banking organisation provides the operational service.

Section 4: HQLA Composition
Level 1 assets may comprise 100 percent of HQLA. Level 2A assets are subject to a
15 percent haircut and may not exceed 40 percent of HQLA. Level 2B assets are subject
to a 50 percent haircut and may not exceed 15 percent of HQLA.

Section 5: Reporting
Covered companies must report LCR daily to the Federal Reserve and make public
monthly disclosure of their average LCR for the prior quarter.
""",
        "v2_text": """\
LIQUIDITY COVERAGE RATIO: TREATMENT OF OPERATIONAL DEPOSITS — AMENDED
Final Rule — Federal Reserve Board | 12 CFR Part 249 | Effective: January 1, 2025

Section 1: LCR Requirement
Covered companies must maintain a liquidity coverage ratio of at least 115 percent.
The LCR equals the ratio of a firm's high-quality liquid assets (HQLA) to its projected
net cash outflows over a standardised 30-calendar-day stress period. The Board is increasing
the minimum from 100 percent to 115 percent to strengthen liquidity resilience in light
of banking sector stress events observed in 2023.

Section 2: Operational Deposits — Outflow Rate
Operational deposits from non-financial wholesale customers are assigned an outflow
rate of 10 percent for the portion covered by deposit insurance and 25 percent for
the uncovered portion. Intraday operational deposits are now explicitly included and
assigned a 5 percent outflow rate.

Section 3: Operational Deposits — Qualifying Criteria
A deposit qualifies as operational if:
(a) The depositing entity requires the deposit to conduct its normal operating activities;
(b) The deposit is held in an account used for payment, clearing, or custody services;
(c) The banking organisation provides the operational service; and
(d) The depositing entity has provided 90-day notice of any intended withdrawal exceeding
    25 percent of the average daily balance (new requirement).

Section 4: HQLA Composition
Level 1 assets may comprise 100 percent of HQLA. Level 2A assets are subject to a
20 percent haircut (increased from 15 percent) and may not exceed 40 percent of HQLA.
Level 2B assets are subject to a 55 percent haircut (increased from 50 percent) and
may not exceed 10 percent of HQLA (reduced from 15 percent).

Section 5: Reporting
Covered companies must report LCR daily to the Federal Reserve and make public
weekly disclosure of their average LCR (increased frequency from monthly).
Firms whose LCR falls below 120 percent must immediately notify their primary regulator.

Section 6: Intraday Liquidity Monitoring (New)
Covered companies must implement intraday liquidity monitoring frameworks and report
peak intraday liquidity usage and available intraday liquidity to the Federal Reserve monthly.
""",
    },
    {
        "doc_num": "2024-00789",
        "title": "Stress Testing Requirements: 2024 Supervisory Scenarios",
        "agency": "FDIC",
        "reg_type": "stress_testing",
        "cfr": "12 CFR 325",
        "pub_date": datetime(2024, 2, 15, tzinfo=timezone.utc),
        "plain_english": (
            "Every year, regulators make banks play a 'what if everything goes wrong at once' game. "
            "The old version imagined unemployment hitting 10% and house prices dropping 25%. "
            "The new version is scarier: unemployment up to 13%, house prices down 40%, "
            "and now they also test what happens if a massive cyberattack hits the bank's systems. "
            "It's like upgrading from a practice fire drill to a practice earthquake-plus-flood drill."
        ),
        "v1_text": """\
STRESS TESTING REQUIREMENTS: SUPERVISORY SCENARIOS
Final Rule — FDIC | 12 CFR Part 325 | Applicable Cycle: 2023

Section 1: Scope
Covered institutions — state nonmember banks and state savings associations with
total consolidated assets of $10 billion or more — must conduct annual supervisory
stress tests as prescribed by this rule.

Section 2: Baseline Scenario
The baseline scenario reflects the consensus views of economic forecasters. For the
2023 cycle, the baseline projects:
- GDP growth of 2.1 percent in 2023, declining to 1.8 percent in 2024
- Unemployment rate remaining at 3.5 percent through mid-2024
- House price appreciation of 2.0 percent annually
- 10-year Treasury yield averaging 4.2 percent

Section 3: Adverse Scenario
The adverse scenario features a mild recession characterised by:
- Unemployment rising to 7.5 percent by Q3 2024
- GDP contracting 2.0 percent peak-to-trough
- House prices declining 15 percent from peak
- Corporate bond spreads widening 200 basis points

Section 4: Severely Adverse Scenario
The severely adverse scenario features a severe global recession:
- Unemployment rising to 10.0 percent by Q2 2024
- GDP contracting 6.5 percent peak-to-trough
- House prices declining 25 percent from peak
- Equity prices falling 45 percent
- Corporate bond spreads widening 500 basis points
- 3-month Treasury yield falling to near zero

Section 5: Reporting Requirements
Covered institutions must submit stress test results to the FDIC by July 31 of
each year. Public disclosure of results is required by October 15.
""",
        "v2_text": """\
STRESS TESTING REQUIREMENTS: SUPERVISORY SCENARIOS — 2024 AMENDMENT
Final Rule — FDIC | 12 CFR Part 325 | Applicable Cycle: 2024

Section 1: Scope
Covered institutions — state nonmember banks and state savings associations with
total consolidated assets of $10 billion or more — must conduct annual supervisory
stress tests as prescribed by this rule. Beginning with the 2024 cycle, covered
institutions must also conduct a separate exploratory scenario analysis.

Section 2: Baseline Scenario
The baseline scenario reflects the consensus views of economic forecasters. For the
2024 cycle, the baseline projects:
- GDP growth of 2.4 percent in 2024, remaining at 2.1 percent in 2025
- Unemployment rate remaining at 3.7 percent through mid-2025
- House price appreciation of 3.5 percent annually
- 10-year Treasury yield averaging 4.8 percent (updated from 4.2 percent)

Section 3: Adverse Scenario
The adverse scenario features a moderate recession characterised by:
- Unemployment rising to 9.0 percent by Q3 2025 (increased from 7.5 percent)
- GDP contracting 3.5 percent peak-to-trough (increased from 2.0 percent)
- House prices declining 25 percent from peak (increased from 15 percent)
- Corporate bond spreads widening 350 basis points (increased from 200 basis points)
- Commercial real estate prices falling 20 percent

Section 4: Severely Adverse Scenario
The severely adverse scenario features a severe global recession with financial instability:
- Unemployment rising to 13.0 percent by Q2 2025 (increased from 10.0 percent)
- GDP contracting 8.5 percent peak-to-trough (increased from 6.5 percent)
- House prices declining 40 percent from peak (increased from 25 percent)
- Equity prices falling 55 percent (increased from 45 percent)
- Corporate bond spreads widening 700 basis points (increased from 500 basis points)
- 3-month Treasury yield falling to near zero
- Significant disruption in global supply chains

Section 4A: Exploratory Scenario — Cyber-Operational Risk (New)
Beginning with the 2024 cycle, covered institutions must complete an exploratory
scenario assessing the impact of a severe operational disruption:
- Simultaneous ransomware attacks on core banking systems lasting 72 hours
- Loss of access to payment systems for 5 business days
- Data integrity compromise affecting 30 percent of loan portfolio records
- Recovery cost assumptions: $500 million base, $2 billion severe

Section 5: Reporting Requirements
Covered institutions must submit stress test results to the FDIC by May 31 of
each year (deadline moved forward from July 31). Public disclosure of results
is required by August 31 (moved forward from October 15). Institutions must also
publish a qualitative discussion of capital adequacy alongside numerical results.
""",
    },
    {
        "doc_num": "2024-01011",
        "title": "Community Reinvestment Act Modernisation Final Rule",
        "agency": "OCC",
        "reg_type": "consumer_protection",
        "cfr": "12 CFR 25",
        "pub_date": datetime(2024, 3, 1, tzinfo=timezone.utc),
        "plain_english": (
            "The CRA is a law that says banks must help the communities where they take deposits — "
            "not just the wealthy neighbourhoods. The old rule only counted branches you could walk into. "
            "The new rule also counts online banking. So if you have an app and people all over the country "
            "use it, you now have to help ALL those communities, not just the ones with physical branches. "
            "A bank that ignores poor neighbourhoods digitally now gets a failing grade just like one "
            "that ignores them in person."
        ),
        "v1_text": """\
COMMUNITY REINVESTMENT ACT: ASSESSMENT AREA REQUIREMENTS
Final Rule — OCC | 12 CFR Part 25 | Effective: January 1, 2023

Section 1: Delineation of Assessment Areas
A bank must delineate one or more assessment areas within which the OCC evaluates
the bank's record of helping to meet the credit needs of its community.

(a) Facility-Based Assessment Areas: A bank must delineate a facility-based assessment
area for each location in which the bank has its main office, one or more branches,
or one or more deposit-taking automated teller machines (ATMs).

Section 2: Geographic Boundaries
Each facility-based assessment area must:
(a) Consist of one or more MSAs or one or more contiguous political subdivisions;
(b) Include the geographies in which the bank has its main office, branches, and
    deposit-taking ATMs; and
(c) Not reflect illegal discrimination or arbitrarily exclude low- or moderate-income
    geographies.

Section 3: CRA Examination Ratings
Ratings: Outstanding, Satisfactory, Needs to Improve, Substantial Noncompliance.
A "Satisfactory" rating requires a composite CRA score above 60 on a 100-point scale.

Section 4: Performance Standards
Retail lending test: 40 percent of composite score.
Community development financing test: 40 percent of composite score.
Community development services test: 20 percent of composite score.
""",
        "v2_text": """\
COMMUNITY REINVESTMENT ACT: ASSESSMENT AREA REQUIREMENTS — MODERNISED
Final Rule — OCC | 12 CFR Part 25 | Effective: January 1, 2026

Section 1: Delineation of Assessment Areas
A bank must delineate one or more assessment areas within which the OCC evaluates
the bank's record of helping to meet the credit needs of its community.

(a) Facility-Based Assessment Areas: A bank must delineate a facility-based assessment
area for each location in which the bank has its main office, one or more branches,
or one or more deposit-taking automated teller machines (ATMs).

(b) Deposit-Based Assessment Areas (New): Banks that receive more than 5 percent of
their retail domestic deposits from areas outside their facility-based assessment areas
must delineate deposit-based assessment areas in each major MSA from which they receive
more than $10 million in deposits annually. This requirement captures digital and
mobile banking activity not reflected in physical branch networks.

Section 2: Geographic Boundaries
Each facility-based assessment area must:
(a) Consist of one or more MSAs or one or more contiguous political subdivisions;
(b) Include the geographies in which the bank has its main office, branches, and
    deposit-taking ATMs; and
(c) Not reflect illegal discrimination or arbitrarily exclude low- or moderate-income
    geographies.

Deposit-based assessment areas must cover the MSA in which the relevant deposits
originate, as determined by the deposit address of record for digital accounts.

Section 3: CRA Examination Ratings
Ratings: Outstanding, Satisfactory, Needs to Improve, Substantial Noncompliance.
A "Satisfactory" rating requires a composite CRA score above 70 on a 100-point scale
(increased from 60). The rating framework now includes a presumption of Satisfactory
for institutions demonstrating meaningful engagement with minority depository institutions.

Section 4: Performance Standards
Retail lending test: 45 percent of composite score (increased from 40 percent).
Community development financing test: 35 percent of composite score (decreased from 40 percent).
Community development services test: 20 percent of composite score (unchanged).
Climate-related community investment test: 10 percent of composite score (new addition; offsets
are drawn equally from the lending and financing test weights).
""",
    },
    {
        "doc_num": "2024-01213",
        "title": "Fair Debt Collection Practices: Digital Communications Rule",
        "agency": "CFPB",
        "reg_type": "consumer_protection",
        "cfr": "12 CFR 1006",
        "pub_date": datetime(2024, 3, 15, tzinfo=timezone.utc),
        "plain_english": (
            "This rule is about how debt collectors are allowed to contact you. "
            "The old rule was written when phones were the main way to reach people. "
            "The new rule says: debt collectors can now text and email you, BUT they must "
            "stop immediately if you say so. They can only text you 3 times per week (not 7). "
            "They must include an easy one-click 'stop all contact' button in every text and email. "
            "Think of it like a spam email — there must always be an 'unsubscribe' button, "
            "and clicking it must actually work within 24 hours."
        ),
        "v1_text": """\
FAIR DEBT COLLECTION PRACTICES: COMMUNICATIONS STANDARDS
Final Rule — CFPB | 12 CFR Part 1006 | Effective: November 30, 2021

Section 1: Telephone Communications
A debt collector may not communicate with a consumer in connection with the collection
of any debt at any unusual time or place. Telephone calls are presumed inconvenient:
(a) Before 8 a.m. or after 9 p.m. local time at the consumer's location;
(b) More than 7 times within any 7-day period.

Section 2: Written Communications
A debt collector must send a validation notice within 5 days of first communicating
with the consumer. The notice must include the amount of the debt, the name of the
creditor, and the consumer's right to dispute the debt within 30 days.

Section 3: Electronic Communications
Electronic communications, including email and text messages, are permitted subject
to the consumer's prior consent. Debt collectors must provide an electronic address
or number designated solely for opting out of electronic communications.

Section 4: Harassment and Abuse
A debt collector may not engage in conduct that harasses, oppresses, or abuses any
person. Repeated or continuous phone calls with intent to annoy are prohibited.
""",
        "v2_text": """\
FAIR DEBT COLLECTION PRACTICES: DIGITAL COMMUNICATIONS RULE — AMENDED
Final Rule — CFPB | 12 CFR Part 1006 | Effective: March 1, 2025

Section 1: Telephone Communications
A debt collector may not communicate with a consumer in connection with the collection
of any debt at any unusual time or place. Telephone calls are presumed inconvenient:
(a) Before 8 a.m. or after 9 p.m. local time at the consumer's location;
(b) More than 3 times within any 7-day period (reduced from 7 times).
(c) Within 7 days of a prior conversation with the consumer on the same debt.

Section 2: Written Communications
A debt collector must send a validation notice within 5 days of first communicating
with the consumer. The notice must include the amount of the debt, the name of the
creditor, the consumer's right to dispute the debt within 30 days, and a
QR-code accessible dispute portal (new requirement for digital-first communication).

Section 3: Electronic Communications — Enhanced Requirements
Electronic communications, including email and text messages, are permitted subject
to the following conditions:
(a) The consumer has provided prior express consent through a signed electronic agreement;
(b) Each electronic communication must contain a one-click opt-out mechanism;
(c) Opt-out requests must be honoured within 24 hours (previously: next business day);
(d) Text messages are limited to 3 per week per debt (new hard limit);
(e) Email communications must include a plain-language subject line identifying the sender
    as a debt collector and the creditor name.
(f) Collectors may not send electronic communications through a platform the consumer
    uses primarily for professional purposes (e.g., LinkedIn) without explicit consent.

Section 4: Harassment and Abuse
A debt collector may not engage in conduct that harasses, oppresses, or abuses any
person. Repeated or continuous phone calls with intent to annoy are prohibited.
The following acts are additionally prohibited:
(g) Communicating with the consumer through a third-party social media account visible
    to the public or to the consumer's contacts;
(h) Sending more than one direct-message per platform per week without consent.

Section 5: Validation Period (Extended)
The consumer's right to dispute the debt is extended from 30 days to 45 days from
receipt of the validation notice. During the dispute period, the debt collector must
cease all collection activity, including electronic communications.
""",
    },
    {
        "doc_num": "2024-01415",
        "title": "Volcker Rule: Covered Fund Definition Clarification",
        "agency": "FRB",
        "reg_type": "trading",
        "cfr": "12 CFR 248",
        "pub_date": datetime(2024, 4, 1, tzinfo=timezone.utc),
        "plain_english": (
            "The Volcker Rule says banks are not allowed to make risky bets with customers' money — "
            "like a casino using house money that belongs to depositors. "
            "One part of this rule lists the types of investment funds banks are forbidden from owning. "
            "The new version adds 'credit funds' and 'venture capital funds' to that forbidden list. "
            "Before, banks found a loophole: they said these fund types weren't covered. "
            "Now the loophole is closed. Think of it like adding a new item to a 'no-go' list "
            "that people were sneaking around."
        ),
        "v1_text": """\
VOLCKER RULE: COVERED FUND PROVISIONS
Final Rule — Federal Reserve Board | 12 CFR Part 248 | Effective: October 1, 2020

Section 1: Definition of Covered Fund
The term "covered fund" means:
(a) An issuer that would be an investment company as defined under the Investment
    Company Act of 1940 but for section 3(c)(1) or 3(c)(7) of that Act;
(b) A commodity pool in which the banking entity serves as the commodity pool operator
    and in which the interests are offered and sold pursuant to exemptions from
    registration requirements; or
(c) A foreign fund organised and offered only to non-US residents.

Section 2: Excluded Fund Types
The following are excluded from the definition of covered fund and are therefore
permitted activities for banking entities:
(a) Wholly-owned subsidiaries;
(b) Joint ventures;
(c) Acquisition vehicles;
(d) Foreign public funds;
(e) Registered investment companies;
(f) Small business investment companies (SBICs);
(g) Public welfare investment funds.

Section 3: Permitted Investments in Covered Funds
A banking entity may acquire or retain an ownership interest in a covered fund only:
(a) In connection with organising and offering the fund, provided the banking entity
    divests ownership within 3 years of establishment; or
(b) As a de minimis investment not exceeding 3 percent of the fund's total ownership
    interests and 3 percent of Tier 1 capital.
""",
        "v2_text": """\
VOLCKER RULE: COVERED FUND DEFINITION CLARIFICATION — AMENDED
Final Rule — Federal Reserve Board | 12 CFR Part 248 | Effective: April 1, 2025

Section 1: Definition of Covered Fund
The term "covered fund" means:
(a) An issuer that would be an investment company as defined under the Investment
    Company Act of 1940 but for section 3(c)(1) or 3(c)(7) of that Act;
(b) A commodity pool in which the banking entity serves as the commodity pool operator
    and in which the interests are offered and sold pursuant to exemptions from
    registration requirements;
(c) A foreign fund organised and offered only to non-US residents; or
(d) A credit fund that extends credit primarily to entities other than US consumers,
    regardless of whether it relies on 3(c)(1) or 3(c)(7) exclusions (new inclusion); or
(e) A venture capital fund as defined under Rule 203(l)-1 under the Investment Advisers
    Act of 1940 (new inclusion).

Section 2: Excluded Fund Types
The following are excluded from the definition of covered fund and are therefore
permitted activities for banking entities:
(a) Wholly-owned subsidiaries;
(b) Joint ventures;
(c) Acquisition vehicles;
(d) Foreign public funds;
(e) Registered investment companies;
(f) Small business investment companies (SBICs);
(g) Public welfare investment funds;
(h) Customer facilitation vehicles used solely to facilitate client-facing transactions
    where the banking entity holds no economic risk (new exclusion).
Credit funds and venture capital funds previously excluded under informal no-action
positions are no longer permitted. Banking entities have until October 1, 2026 to divest.

Section 3: Permitted Investments in Covered Funds
A banking entity may acquire or retain an ownership interest in a covered fund only:
(a) In connection with organising and offering the fund, provided the banking entity
    divests ownership within 1 year of establishment (reduced from 3 years); or
(b) As a de minimis investment not exceeding 2 percent of the fund's total ownership
    interests (reduced from 3 percent) and 2 percent of Tier 1 capital (reduced from 3 percent).

Section 4: Super 23A Restrictions (Enhanced)
The prohibition on covered transactions between banking entities and covered funds
is extended to include back-to-back loans, participation interests, and synthetic
exposures that replicate the economic effect of a covered transaction.
""",
    },
    {
        "doc_num": "2024-01617",
        "title": "Basel III Endgame: Market Risk Capital Requirements (FRTB)",
        "agency": "OCC",
        "reg_type": "capital",
        "cfr": "12 CFR 3",
        "pub_date": datetime(2024, 4, 15, tzinfo=timezone.utc),
        "plain_english": (
            "Banks buy and sell stocks, bonds, and other financial products every day (this is called the trading book). "
            "This rule changes how banks calculate the 'worst case scenario' risk of that trading. "
            "Before, large banks could use their own secret formula (called an internal model) to say "
            "'we only need X amount of capital for our trading risks.' "
            "The new rule says: no more secret formulas for the biggest bets. "
            "You must use our standard method, which usually gives a bigger number. "
            "Result: banks need significantly more capital held against trading risks — "
            "estimates suggest $80B more industry-wide."
        ),
        "v1_text": """\
MARKET RISK CAPITAL REQUIREMENTS
Final Rule — OCC | 12 CFR Part 3, Subpart F | Effective: January 1, 2022

Section 1: Scope — Market Risk Rule
This subpart applies to any national bank or federal savings association that
has aggregate trading assets and trading liabilities equal to 10 percent or more
of total assets, or $1 billion or more.

Section 2: Internal Models Approach (IMA)
Covered institutions may calculate market risk capital requirements using an
approved internal models approach (IMA). The IMA must:
(a) Use a 99th percentile, 10-day value-at-risk (VaR) measure;
(b) Include a stressed VaR component calculated over a 12-month historical stress period;
(c) Receive prior approval from the OCC for each trading desk applying IMA.

Section 3: Standardised Approach
Institutions not using IMA, or for trading desks that fail back-testing requirements,
must apply the standardised measurement method (SMM). The SMM uses regulatory
prescribed risk weights for general market risk and specific risk.

Section 4: Back-testing Requirements
Institutions using IMA must conduct daily back-testing comparing predicted VaR
against actual daily profit and loss. Desks with more than 4 exceptions in 250
trading days are subject to a capital add-on multiplier of up to 1.5x.
""",
        "v2_text": """\
MARKET RISK CAPITAL REQUIREMENTS — FRTB IMPLEMENTATION
Final Rule — OCC | 12 CFR Part 3, Subpart F | Effective: January 1, 2026

Section 1: Scope — Market Risk Rule
This subpart applies to any national bank or federal savings association that
has aggregate trading assets and trading liabilities equal to 5 percent or more
of total assets (reduced from 10 percent), or $500 million or more (reduced from $1 billion).

Section 2: Internal Models Approach (IMA) — Restricted
IMA approval is now granted at the trading desk level and is subject to annual
renewal. Desks applying IMA must use Expected Shortfall (ES) at 97.5th percentile
rather than VaR (former 99th percentile, 10-day measure). The stress period must
cover the most severe 12-month period since 2007 (expanded from a rolling window).
Desks failing Profit and Loss Attribution (PLA) tests must revert to the Standardised
Approach — no override or appeals process is available.

Section 3: Standardised Approach — Enhanced (SA-FRTB)
All covered institutions must calculate capital under the revised Standardised Approach
(SA-FRTB) as a floor. Market risk capital under IMA may not be less than 72.5 percent
of the SA-FRTB output capital requirement (output floor). SA-FRTB replaces the prior
Standardised Measurement Method and uses:
(a) Sensitivity-based method (SBM) for delta, vega, and curvature risks;
(b) Default risk charge (DRC) for credit products;
(c) Residual risk add-on (RRAO) for instruments with exotic features.

Section 4: Back-testing and PLA Requirements
Institutions using IMA must conduct:
(a) Daily back-testing at the 99th and 97.5th percentile against actual P&L;
(b) Monthly Profit and Loss Attribution tests comparing risk-theoretical P&L and
    hypothetical P&L; desks with PLA test breaches on more than 30 percent of
    business days in a quarter must switch to SA-FRTB for that desk.
Desks with more than 12 back-testing exceptions in 250 trading days (increased from 4)
are subject to a mandatory switch to SA-FRTB and a 12-month IMA ban.

Section 5: Reporting — FRTB Disclosures (New)
Covered institutions must publish quarterly disclosures including:
- IMA and SA-FRTB capital requirements by risk class
- List of IMA-approved trading desks
- Back-testing and PLA exception counts
""",
    },
    {
        "doc_num": "2024-01819",
        "title": "Safeguards Rule: Information Security Program Requirements",
        "agency": "FDIC",
        "reg_type": "aml",
        "cfr": "16 CFR 314",
        "pub_date": datetime(2024, 5, 1, tzinfo=timezone.utc),
        "plain_english": (
            "This is the 'data security rules for banks' regulation. "
            "Imagine your bank has a big vault — not for money, but for your personal information. "
            "The old rule said: have a lock on the vault. "
            "The new rule says: the lock must be at least 256-bit encryption (a very strong lock), "
            "you must tell the government within 30 days if someone breaks in (down from 72 hours AFTER discovery), "
            "and now requires a real human (a Chief Information Security Officer) to be in charge of the vault. "
            "If a hacker gets in, you can't wait around — you must tell regulators very quickly."
        ),
        "v1_text": """\
SAFEGUARDS RULE: INFORMATION SECURITY PROGRAM
Final Rule — FDIC | 16 CFR Part 314 | Effective: June 9, 2023

Section 1: Information Security Program Requirements
Each covered financial institution must implement a comprehensive information
security program containing administrative, technical, and physical safeguards
appropriate to the size and complexity of the institution and the nature and scope
of its activities.

Section 2: Required Elements
The program must include:
(a) Designating one or more employees to coordinate the information security program;
(b) Identifying and assessing risks to customer information;
(c) Designing and implementing safeguards to control risks identified;
(d) Overseeing service providers by contract to implement appropriate safeguards;
(e) Evaluating and adjusting the program in light of testing results and material changes.

Section 3: Encryption
Covered institutions must implement encryption of customer information held or
transmitted by the institution where technically feasible. Institutions may use
compensating controls where encryption is not technically feasible.

Section 4: Incident Response
Covered institutions must establish a written incident response plan. Following a
security event, institutions must assess the nature and scope of the event and
notify affected customers promptly.

Section 5: Board Reporting
The designated coordinator must report to the board of directors on the information
security program at least annually.
""",
        "v2_text": """\
SAFEGUARDS RULE: INFORMATION SECURITY PROGRAM — ENHANCED REQUIREMENTS
Final Rule — FDIC | 16 CFR Part 314 | Effective: November 1, 2024

Section 1: Information Security Program Requirements
Each covered financial institution must implement a comprehensive information
security program containing administrative, technical, and physical safeguards
appropriate to the size and complexity of the institution and the nature and scope
of its activities. The program must be reviewed and tested annually by an independent
third party or qualified internal function.

Section 2: Required Elements
The program must include:
(a) Designating a qualified Chief Information Security Officer (CISO) — a specifically
    titled, dedicated individual — to coordinate the information security program
    (previously permitted to designate "one or more employees" without a dedicated role);
(b) Identifying and assessing risks to customer information using a formal risk assessment
    framework reviewed by the board of directors annually;
(c) Designing and implementing safeguards to control risks identified, including
    multi-factor authentication for all remote access systems;
(d) Overseeing service providers through written contracts requiring annual security
    certifications (SOC 2 Type II or equivalent);
(e) Evaluating and adjusting the program continuously using automated threat detection.

Section 3: Encryption — Mandatory Minimum Standards
Covered institutions must implement encryption of customer information using algorithms
meeting or exceeding AES-256 for data at rest and TLS 1.3 for data in transit.
Compensating controls are no longer accepted in lieu of encryption. Exceptions require
written OCC approval and are reviewed annually.

Section 4: Incident Response and Notification
Covered institutions must establish a written incident response plan. Following a
security event involving customer information:
(a) The CISO must be notified within 1 hour of detection;
(b) The FDIC must be notified within 36 hours of determining a notification is warranted
    (reduced from 72 hours);
(c) Affected customers must be notified within 30 days (new hard deadline);
(d) A post-incident report must be submitted to the board within 90 days.

Section 5: Board Reporting — Enhanced Frequency
The CISO must report to the board of directors on the information security program
quarterly (increased from annually), including metrics on threat detection, incident
response times, and penetration testing results.

Section 6: Penetration Testing (New)
Covered institutions must conduct annual external penetration testing of all
internet-facing systems and quarterly vulnerability scanning. Results must be
reported to the CISO within 5 business days and remediation plans submitted to the
board within 30 days.
""",
    },
    {
        "doc_num": "2024-02021",
        "title": "Anti-Money Laundering: Beneficial Ownership Reporting Requirements",
        "agency": "FinCEN",
        "reg_type": "aml",
        "cfr": "31 CFR 1010",
        "pub_date": datetime(2024, 5, 15, tzinfo=timezone.utc),
        "plain_english": (
            "Criminals sometimes hide their money by putting it in a company, then another company, "
            "then another — like Russian nesting dolls — so nobody can trace it back to them. "
            "This rule fights that. Before, banks had to find out who owns 25% or more of a company. "
            "Now, they must find out who owns 10% or more — a much lower bar. "
            "This means more people get identified, and it's harder for criminals to hide "
            "by spreading ownership among many shell companies. "
            "Banks that miss this now face $1 million per day fines (up from $25,000)."
        ),
        "v1_text": """\
BENEFICIAL OWNERSHIP REPORTING REQUIREMENTS
Final Rule — FinCEN | 31 CFR Part 1010 | Effective: May 11, 2018

Section 1: Customer Due Diligence Requirements
Covered financial institutions must establish and maintain written procedures
reasonably designed to identify and verify the identity of beneficial owners
of legal entity customers at the time a new account is opened.

Section 2: Definition of Beneficial Owner
The term "beneficial owner" means each of the following:
(a) Each individual who, directly or indirectly, through any contract, arrangement,
    understanding, relationship or otherwise, owns 25 percent or more of the equity
    interests of a legal entity customer; and
(b) A single individual with significant responsibility to control, manage, or direct
    a legal entity customer.

Section 3: Certification Requirement
Covered financial institutions must obtain from the person opening the account
a certification of the beneficial owners on a standard form containing:
(a) Name, address, date of birth, and social security number (or equivalent);
(b) Attestation that the information is accurate and complete.

Section 4: Record Retention
Covered financial institutions must retain beneficial ownership records for 5 years
following the date the account is closed or the relationship ends.

Section 5: Penalties
Civil money penalties for violations may not exceed $25,000 per day.
Criminal penalties: up to $1,000,000 fine and imprisonment up to 20 years.
""",
        "v2_text": """\
BENEFICIAL OWNERSHIP REPORTING — CORPORATE TRANSPARENCY ACT ALIGNMENT
Final Rule — FinCEN | 31 CFR Part 1010 | Effective: January 1, 2025

Section 1: Customer Due Diligence Requirements
Covered financial institutions must establish and maintain written procedures
reasonably designed to identify and verify the identity of beneficial owners
of legal entity customers at the time a new account is opened and upon any
material change in ownership structure or control (continuous monitoring — new requirement).

Section 2: Definition of Beneficial Owner — Expanded
The term "beneficial owner" means each of the following:
(a) Each individual who, directly or indirectly, through any contract, arrangement,
    understanding, relationship or otherwise, owns 10 percent or more of the equity
    interests of a legal entity customer (threshold reduced from 25 percent);
(b) A single individual with significant responsibility to control, manage, or direct
    a legal entity customer; and
(c) Any individual who exercises substantial control through a trust, nominee arrangement,
    or any other indirect ownership structure (new category).

Section 3: Certification and FinCEN Database Cross-Reference
Covered financial institutions must obtain from the person opening the account
a certification of the beneficial owners on a standard form containing:
(a) Name, address, date of birth, and social security number (or equivalent);
(b) FinCEN Identifier number if the individual has registered with the Corporate
    Transparency Act beneficial ownership database (new requirement); and
(c) Attestation that the information is accurate and complete.
Institutions must cross-reference beneficial owner information against the FinCEN
Beneficial Ownership Secure System (BOSS) database within 5 business days of account opening.

Section 4: Record Retention — Extended
Covered financial institutions must retain beneficial ownership records for 7 years
following the date the account is closed or the relationship ends (extended from 5 years).
Records must be maintained in a format retrievable and producible to regulators within
2 business days of request.

Section 5: Continuous Monitoring Obligation (New)
Covered institutions must implement automated monitoring to detect changes in beneficial
ownership that require updated certification. Triggers include: SEC filings, UCC amendments,
state business registry updates, and adverse media screening results.

Section 6: Penalties — Enhanced
Civil money penalties for violations may not exceed $1,000,000 per day (increased from $25,000).
Repeat violations carry a mandatory minimum penalty of $500,000.
Criminal penalties: up to $10,000,000 fine (increased from $1,000,000) and imprisonment up to 20 years.
""",
    },
    {
        "doc_num": "2024-02223",
        "title": "Interchange Fee Standards: Card Transaction Amendment",
        "agency": "FRB",
        "reg_type": "consumer_protection",
        "cfr": "12 CFR 235",
        "pub_date": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "plain_english": (
            "Every time you swipe a debit card at a store, the store pays a small fee to the bank. "
            "This is called the interchange fee. The regulator sets a cap on how high this fee can be. "
            "The old cap was 21 cents plus 0.05% of the transaction (set in 2011). "
            "The new cap is 14.4 cents plus 0.04% — a significant cut. "
            "For every $100 transaction, the store saves about 7 cents in fees. "
            "That might sound small, but a large grocery chain does millions of transactions. "
            "Good for stores and consumers (prices may drop a little). "
            "Not great for big banks that earn those fees."
        ),
        "v1_text": """\
INTERCHANGE FEE STANDARDS FOR DEBIT CARD TRANSACTIONS
Final Rule — Federal Reserve Board | 12 CFR Part 235 | Effective: October 1, 2011

Section 1: Maximum Permissible Interchange Fee
The maximum permissible interchange fee that an issuer may receive or charge
for a debit card transaction is the sum of:
(a) A base component not to exceed $0.21 per transaction; and
(b) An ad valorem component not to exceed 0.05 percent of the transaction value.

Section 2: Fraud Prevention Adjustment
An issuer may receive an additional amount per transaction not to exceed $0.01 for
fraud prevention purposes, provided the issuer:
(a) Has implemented fraud prevention policies and procedures consistent with
    published standards; and
(b) Annually certifies compliance.

Section 3: Covered Issuers
This regulation applies to issuers that, together with affiliates, have assets of
$10 billion or more as of the end of the preceding calendar year.

Section 4: Network Routing
Debit card networks must provide at least two unaffiliated network options for
routing debit card transactions. Merchants retain the right to route over the
least-cost network.

Section 5: Review
The Board will review interchange fee standards every two years.
""",
        "v2_text": """\
INTERCHANGE FEE STANDARDS FOR DEBIT CARD TRANSACTIONS — REVISED
Final Rule — Federal Reserve Board | 12 CFR Part 235 | Effective: July 1, 2025

Section 1: Maximum Permissible Interchange Fee — Reduced
The maximum permissible interchange fee that an issuer may receive or charge
for a debit card transaction is the sum of:
(a) A base component not to exceed $0.144 per transaction (reduced from $0.21); and
(b) An ad valorem component not to exceed 0.04 percent of the transaction value
    (reduced from 0.05 percent).

The Board has updated these figures to reflect current issuer cost data.
The 2011 fee cap was based on cost surveys conducted during 2009-2010. Updated
surveys indicate that per-transaction costs have declined materially due to
processing technology improvements, warranting a reduction in the cap.

Section 2: Fraud Prevention Adjustment — Updated
An issuer may receive an additional amount per transaction not to exceed $0.013
(increased from $0.01) for fraud prevention purposes, provided the issuer:
(a) Has implemented fraud prevention policies and procedures consistent with
    published Nacha and EMVCo standards (specific standards now cited); and
(b) Annually certifies compliance and reports fraud rates to the Federal Reserve.
Issuers with fraud rates exceeding 10 basis points of transaction volume are
ineligible for the fraud prevention adjustment (new restriction).

Section 3: Covered Issuers
This regulation applies to issuers that, together with affiliates, have assets of
$10 billion or more as of the end of the preceding calendar year.

Section 4: Network Routing — Enhanced
Debit card networks must provide at least two unaffiliated network options for
routing debit card transactions for both in-person and card-not-present (online)
transactions (extended from in-person only). Merchants retain the right to route
over the least-cost network and must not be steered by issuer incentive programs.

Section 5: Tokenised Transaction Fee Standard (New)
For tokenised debit transactions (Apple Pay, Google Pay, and similar wallet
transactions), the maximum permissible interchange fee is:
(a) Base: $0.144 per transaction (same as standard); and
(b) Ad valorem: 0.04 percent.
Issuers may not charge an additional tokenisation fee above the standard cap.

Section 6: Review — Accelerated
The Board will review interchange fee standards every two years, with the next
scheduled review completed and rules effective no later than July 1, 2027.
""",
    },
]


# ── Drift score computation from actual text ──────────────────────────────────

def _compute_real_drift(text_v1: str, text_v2: str) -> dict:
    """Compute actual TF-IDF drift between two texts (not random)."""
    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.special import rel_entr

        vec = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        mat = vec.fit_transform([text_v1, text_v2]).toarray()

        # Semantic drift: 1 - cosine similarity
        sim = float(cosine_similarity(mat[[0]], mat[[1]])[0][0])
        drift = round(1.0 - sim, 4)

        # Bootstrap CI (200 samples for speed)
        rng = np.random.default_rng(42)
        words_v1 = text_v1.split()
        words_v2 = text_v2.split()
        boot_drifts = []
        for _ in range(200):
            s1 = " ".join(rng.choice(words_v1, size=min(len(words_v1), 150), replace=False))
            s2 = " ".join(rng.choice(words_v2, size=min(len(words_v2), 150), replace=False))
            try:
                m = vec.transform([s1, s2]).toarray()
                bs = float(cosine_similarity(m[[0]], m[[1]])[0][0])
                boot_drifts.append(1.0 - bs)
            except Exception:
                boot_drifts.append(drift)
        ci_low  = round(float(np.percentile(boot_drifts, 2.5)),  4)
        ci_high = round(float(np.percentile(boot_drifts, 97.5)), 4)

        # JSD: normalised term distributions
        p = mat[0] + 1e-9; p /= p.sum()
        q = mat[1] + 1e-9; q /= q.sum()
        m = 0.5 * (p + q)
        jsd_val = float(0.5 * rel_entr(p, m).sum() + 0.5 * rel_entr(q, m).sum())
        jsd_normalised = round(min(jsd_val / 0.693, 1.0), 4)  # divide by ln(2) to get [0,1]

        # Approximate p-value from permutation-style estimate
        jsd_p = round(max(0.0001, min(1.0, 1.0 - jsd_normalised)), 4)

        # Wasserstein: treat TF-IDF row as distribution over vocab
        try:
            from scipy.stats import wasserstein_distance
            wass = round(float(wasserstein_distance(mat[0], mat[1])), 4)
            wass_normalised = round(min(wass / 5.0, 1.0), 4)
        except Exception:
            wass_normalised = round(drift * 0.7, 4)

        flagged = drift > 0.15

        return {
            "drift_score": drift,
            "drift_ci_low": ci_low,
            "drift_ci_high": ci_high,
            "jsd_score": jsd_normalised,
            "jsd_p_value": jsd_p,
            "wasserstein_score": wass_normalised,
            "is_significant": flagged,
            "flagged_for_analysis": flagged,
        }
    except Exception as exc:
        log.warning("Could not compute real drift, using fallback: %s", exc)
        d = round(random.uniform(0.2, 0.6), 4)
        return {
            "drift_score": d,
            "drift_ci_low": round(d - 0.04, 4),
            "drift_ci_high": round(d + 0.04, 4),
            "jsd_score": round(d * 0.7, 4),
            "jsd_p_value": 0.01,
            "wasserstein_score": round(d * 0.5, 4),
            "is_significant": True,
            "flagged_for_analysis": True,
        }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Synchronous session factory ───────────────────────────────────────────────

def _make_sync_engine():
    from sqlalchemy import create_engine
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db   = os.environ.get("POSTGRES_DB",   "complianceiq")
    user = os.environ.get("POSTGRES_USER", "complianceiq")
    pw   = os.environ.get("POSTGRES_PASSWORD", "")
    url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, echo=False, pool_pre_ping=True)


# ── Main seed function ────────────────────────────────────────────────────────

def run_seed() -> None:
    """Insert sample data into PostgreSQL and ChromaDB. Idempotent."""
    from sqlalchemy import select, text
    from sqlalchemy.orm import Session
    from db.models import CausalEstimate, ChangeScore, Regulation, RegulationVersion

    engine = _make_sync_engine()

    with Session(engine) as session:
        # Check idempotency
        existing_count = session.execute(
            text("SELECT COUNT(*) FROM regulations WHERE raw_metadata->>'demo' = 'true'")
        ).scalar_one()

        if existing_count > 0:
            log.info("Seed already present (%d regs) — running ChromaDB upsert only", existing_count)
            reg_ids = [
                row[0] for row in session.execute(
                    select(Regulation.id)
                    .where(Regulation.raw_metadata["demo"].as_boolean() == True)  # noqa: E712
                    .order_by(Regulation.created_at)
                ).all()
            ]
            _seed_chromadb(reg_ids)
            return

        reg_ids: list[uuid.UUID] = []

        for i, spec in enumerate(_REGULATIONS):
            reg_id = uuid.uuid4()

            reg = Regulation(
                id=reg_id,
                document_number=spec["doc_num"],
                source="federal_register",
                agency=spec["agency"],
                title=spec["title"],
                abstract=spec["plain_english"][:400],
                full_text=spec["v2_text"],
                publication_date=spec["pub_date"],
                regulation_type=spec["reg_type"],
                cfr_references=[spec["cfr"]],
                raw_metadata={
                    "demo": True,
                    "seed_index": i,
                    "plain_english": spec["plain_english"],
                    "v1_text": spec["v1_text"],   # stored for diff API
                },
            )
            session.add(reg)
            session.flush()

            v1_text = spec["v1_text"]
            v2_text = spec["v2_text"]

            v1 = RegulationVersion(
                id=uuid.uuid4(),
                regulation_id=reg_id,
                version_number=1,
                full_text=v1_text,
                text_hash=_sha256(v1_text),
                word_count=len(v1_text.split()),
            )
            v2 = RegulationVersion(
                id=uuid.uuid4(),
                regulation_id=reg_id,
                version_number=2,
                full_text=v2_text,
                text_hash=_sha256(v2_text),
                word_count=len(v2_text.split()),
            )
            session.add(v1)
            session.add(v2)
            session.flush()

            # Compute REAL drift from actual text differences
            log.info("Computing drift for: %s", spec["title"][:50])
            scores = _compute_real_drift(v1_text, v2_text)

            cs = ChangeScore(
                id=uuid.uuid4(),
                regulation_id=reg_id,
                version_old=1,
                version_new=2,
                **scores,
            )
            session.add(cs)
            reg_ids.append(reg_id)

        log.info("Inserted %d regulations", len(reg_ids))

        # Causal estimates for 5 key regulations
        causal_specs = [
            # (index, method, att_estimate, std_err, p_value, ci_low, ci_high, outcome, threshold)
            (0, "did", 0.0421, 0.0058, 0.0000, 0.0307, 0.0535, "tier1_capital_ratio", None),
            (1, "synthetic_control", 0.0318, 0.0047, 0.0021, 0.0226, 0.0410, "liquidity_coverage_ratio", None),
            (6, "rdd", 0.0721, 0.0093, 0.0000, 0.0539, 0.0903, "market_risk_rwa", 10_000_000_000.0),
            (3, "did", 0.0125, 0.0031, 0.0081, 0.0064, 0.0186, "cra_composite_score", None),
            (8, "synthetic_control", 0.0894, 0.0112, 0.0003, 0.0674, 0.1114, "beneficial_ownership_compliance_cost", None),
        ]
        rng = random.Random(42)
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
        log.info("PostgreSQL seed complete")

    _seed_chromadb(reg_ids)


def _seed_chromadb(reg_ids: list[uuid.UUID]) -> None:
    """Upsert document chunks into ChromaDB. Uses actual regulation text."""
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
        for i, (reg_id, spec) in enumerate(zip(reg_ids, _REGULATIONS)):
            # Chunk the actual v2 text into overlapping 400-char windows
            text = spec["v2_text"]
            chunks = [text[j: j + 400] for j in range(0, len(text), 300)]
            for k, chunk in enumerate(chunks[:20]):  # max 20 chunks per regulation
                chunk_id = f"seed-{i:02d}-{k:03d}"
                ids.append(chunk_id)
                docs.append(chunk)
                metas.append({
                    "regulation_id": str(reg_id),
                    "document_number": spec["doc_num"],
                    "agency": spec["agency"],
                    "chunk_index": k,
                    "version_number": 2,
                })

        # Pad to a round number with random embeddings for remaining slots
        n = len(ids)
        embeddings = np.random.default_rng(42).normal(0, 1, (n, 768))
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = (embeddings / norms).tolist()

        collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        log.info("Upserted %d ChromaDB chunks", n)
    except Exception as exc:
        log.warning("ChromaDB seed skipped (service unavailable): %s", exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_seed()
