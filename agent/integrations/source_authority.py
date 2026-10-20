"""
OpenClaw/Moltbot — Source Authority Scoring

4-tier domain classification system to filter low-quality sources
from the learning engine pipeline. Scores URLs 0-10 based on
domain credibility for financial/economic research.

Tiers:
  1 (9-10): Government, central banks, premier financial press, academic
  2 (6-8):  .gov/.edu TLDs, quality financial blogs, major news
  3 (2-4):  Wikipedia, Investopedia, Medium, generic education
  4 (0):    Block entirely — corporate marketing, SEO farms
"""

import logging
from urllib.parse import urlparse

logger = logging.getLogger("openclaw.source_authority")

# Tier 1: Highest authority — government, central banks, premier press, academic
# Score: 10
TIER_1_DOMAINS = frozenset({
    # US Government / central bank
    "fed.gov", "federalreserve.gov", "newyorkfed.org", "treasury.gov",
    "bls.gov", "bea.gov", "census.gov", "eia.gov", "sec.gov",
    "cftc.gov", "fdic.gov", "occ.gov", "whitehouse.gov",
    # International organizations
    "imf.org", "worldbank.org", "iea.org", "bis.org", "oecd.org",
    "ecb.europa.eu", "bankofengland.co.uk",
    # Premier financial data / research
    "bloomberg.com", "reuters.com", "ft.com", "wsj.com",
    "economist.com",
    # Academic
    "arxiv.org", "ssrn.com", "nber.org", "repec.org",
})

# Tier 2: High authority — specific quality domains with individual scores
TIER_2_DOMAINS = {
    # Research / data portals
    "fred.stlouisfed.org": 9,
    "data.worldbank.org": 8,
    "scholar.google.com": 8,
    # Quality financial analysis
    "bondeconomics.com": 7, "pragcap.com": 7,
    "macrovoices.com": 7, "thefelderreport.com": 7,
    "adventuresincapitalism.com": 7,
    # Major financial news
    "cnbc.com": 7, "marketwatch.com": 7, "barrons.com": 7,
    "seekingalpha.com": 6, "investing.com": 6,
    "morningstar.com": 7, "spglobal.com": 8,
    "fitchratings.com": 8, "moodys.com": 8,
}

# Tier 3: Low authority — deprioritize but don't block
TIER_3_DOMAINS = {
    "wikipedia.org": 3, "investopedia.com": 4,
    "medium.com": 3, "substack.com": 4,
    "hubspot.com": 2, "salesforce.com": 2,
    "quora.com": 2, "reddit.com": 3,
}

# Tier 4: Block entirely — SEO farms, corporate marketing, retail education slop
TIER_4_BLOCK = frozenset({
    "6sigma.us", "transglobeacademy.com", "emlii.com",
    "gatekeeperhq.com", "pocketful.in", "corporatefinanceinstitute.com",
    "daytrading.com", "kalkine.com", "tradingtechnologies.com",
    "avatrade.com", "ig.com", "etoro.com", "fxstreet.com",
    "babypips.com", "thebalancemoney.com",
})


def get_source_authority(url: str) -> int:
    """Score a URL's source authority (0-10). 0 = block entirely."""
    try:
        domain = (urlparse(url).hostname or "").lower().removeprefix("www.")
    except Exception as e:
        logger.debug(f"URL parse failed in authority check: {e}")
        return 5  # Can't parse — treat as unknown

    if not domain:
        return 5

    # Check tier 1 — exact domain or subdomain match
    if domain in TIER_1_DOMAINS:
        return 10
    for t1 in TIER_1_DOMAINS:
        if domain.endswith("." + t1):
            return 10

    # Check tier 4 blocks
    if domain in TIER_4_BLOCK:
        return 0

    # Check tier 2 — exact domain or subdomain match
    for pattern, score in TIER_2_DOMAINS.items():
        if domain == pattern or domain.endswith("." + pattern):
            return score

    # Check tier 3 — exact domain or subdomain match
    for pattern, score in TIER_3_DOMAINS.items():
        if domain == pattern or domain.endswith("." + pattern):
            return score

    # TLD-based scoring for domains not in any list
    if domain.endswith(".gov") or domain.endswith(".mil"):
        return 8
    if domain.endswith(".edu"):
        return 8
    if domain.endswith(".org"):
        return 6

    # Default: unknown commercial domain — neutral score
    return 5


def filter_and_rank_results(results: list[dict]) -> list[dict]:
    """
    Filter out blocked sources and re-rank search results by authority.
    Expects list of dicts with 'url' key. Returns filtered, sorted list.
    """
    scored = []
    for result in results:
        url = result.get("url", "")
        authority = get_source_authority(url)
        if authority == 0:
            logger.info(f"Source authority: blocked {url}")
            continue
        scored.append((result, authority))

    # Sort by authority descending (stable sort preserves relevance order for ties)
    scored.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in scored]


def compute_authority_weighted_quality(quality: int, source_urls: list[str]) -> float:
    """
    Compute effective quality score weighted by source authority.

    Formula: quality * (0.6 + 0.04 * avg_authority)
    - Tier 1 (authority 10): multiplier = 1.0 (full quality)
    - Unknown (authority 5):  multiplier = 0.80
    - Tier 3 (authority 3):   multiplier = 0.72
    - Tier 3 (authority 2):   multiplier = 0.68

    Threshold: effective_quality >= 4.5 to save.
    """
    if not source_urls:
        return float(quality) * 0.80  # No URLs — treat as unknown authority

    total_authority = sum(get_source_authority(u) for u in source_urls)
    avg_authority = total_authority / len(source_urls)
    multiplier = 0.6 + 0.04 * avg_authority
    return float(quality) * multiplier
