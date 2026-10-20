"""
Operation Lighthouse v2.0 — Global News Feed Parser & Registry

Self-hosted RSSHub + 55 international RSS feeds across 11 regions.
Parses headlines into structured objects with temporal, geographic,
and reliability metadata for CLIO learning engine consumption.

CC-LH-2 | Atlas Strategies — March 2026
"""

import hashlib
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger("openclaw.lighthouse")

# S233 CR-3: Sanitize all external RSS content before downstream use
try:
    from agent.security import sanitize_ingested_content as _sanitize_rss
except ImportError:
    def _sanitize_rss(c, source=""):
        return f"<ingested_data source=\"{source}\">{c}</ingested_data>"

_NY = ZoneInfo("America/New_York")
_UTC = timezone.utc

# ============================================================
# Safe env var parsing
# ============================================================

def _safe_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.debug("Lighthouse: bad float env %s=%r, using default %s", name, raw, default)
        return default

# ============================================================
# Feature Flags
# ============================================================

# S148 H-1: All feature flags moved to runtime os.environ.get() at call site

def _get_t3_discount() -> float:
    return _safe_float_env("CLIO_RSS_T3_DISCOUNT", 0.50)

def _get_t2_discount() -> float:
    return _safe_float_env("CLIO_RSS_T2_DISCOUNT", 0.85)

# ============================================================
# Constants
# ============================================================

_RSSHUB_BASE = "http://localhost:1201"  # S162: v2 container (healthy)
_FETCH_TIMEOUT = 20
_MAX_HEADLINES_PER_FEED = 15
_MAX_HEADLINE_AGE_HOURS = 24
_DEDUP_WINDOW_HOURS = 12
_MIN_RELEVANCE_KEYWORDS = 1

_US_OPEN_HOUR = 9
_US_OPEN_MINUTE = 30

# ============================================================
# Feed Registry (55 feeds, 11 regions)
# ============================================================

T1 = "T1_VERIFIED"
T2 = "T2_SEMI_INDEPENDENT"
T3 = "T3_STATE_AFFILIATED"

FEED_REGISTRY = [
    # --- ASIA-PACIFIC (18) ---
    {"id": "nhk_world",       "name": "NHK World-Japan",       "region": "japan",     "tier": T1, "url": f"{_RSSHUB_BASE}/nhk/world",                            "type": "rsshub", "focus": ["defense", "geopolitical", "central_bank", "trade_tariff", "earnings_corporate"], "disabled": True},  # S135: direct dead + RSSHub plugin missing
    {"id": "nippon",          "name": "Nippon.com",            "region": "japan",     "tier": T2, "url": "https://www.nippon.com/en/feed/",                      "type": "native", "focus": ["geopolitical", "trade_tariff"]},  # S135: URL fixed
    {"id": "japan_times",     "name": "Japan Times",           "region": "japan",     "tier": T2, "url": "https://www.japantimes.co.jp/feed/",                   "type": "native", "focus": ["geopolitical", "earnings_corporate"]},
    {"id": "nikkei_asia",     "name": "Nikkei Asia",           "region": "japan",     "tier": T2, "url": f"{_RSSHUB_BASE}/nikkei/asia",                          "type": "rsshub", "focus": ["semiconductors", "trade_tariff", "earnings_corporate"]},
    {"id": "yonhap",          "name": "Yonhap News",           "region": "korea",     "tier": T1, "url": "https://en.yna.co.kr/RSS/news.xml",                   "type": "native", "focus": ["semiconductors", "defense", "geopolitical"]},
    {"id": "korea_herald",    "name": "Korea Herald",          "region": "korea",     "tier": T2, "url": "http://www.koreaherald.com/common/rss.php",            "type": "native", "focus": ["semiconductors", "tech_ai", "earnings_corporate"], "disabled": True},  # S135: RSS redirects to HTML
    {"id": "korea_joongang",  "name": "Korea JoongAng",        "region": "korea",     "tier": T2, "url": "https://koreajoongangdaily.joins.com/xmls/joins_rss.xml", "type": "native", "focus": ["geopolitical", "earnings_corporate"], "disabled": True},  # S135: RSS discontinued
    {"id": "taiwan_news",     "name": "Taiwan News",           "region": "taiwan",    "tier": T1, "url": "https://www.taiwannews.com.tw/en/rss",                "type": "native", "focus": ["semiconductors", "defense", "geopolitical"], "disabled": True},  # S135: RSS discontinued (Next.js rewrite)
    {"id": "focus_taiwan",    "name": "Focus Taiwan (CNA)",    "region": "taiwan",    "tier": T1, "url": "https://focustaiwan.tw/rss",                          "type": "native", "focus": ["semiconductors", "geopolitical", "trade_tariff"], "disabled": True},  # S135: RSS discontinued
    {"id": "cna_singapore",   "name": "Channel NewsAsia",      "region": "singapore", "tier": T2, "url": "https://www.channelnewsasia.com/api/v1/rss-outbound-feed?_format=xml", "type": "native", "focus": ["trade_tariff", "tech_ai", "geopolitical"]},
    {"id": "abc_au",          "name": "ABC Australia",          "region": "australia", "tier": T1, "url": "https://www.abc.net.au/news/feed/2942460/rss.xml",    "type": "native", "focus": ["commodities", "geopolitical", "defense"]},
    {"id": "abc_au_biz",      "name": "ABC AU Business",       "region": "australia", "tier": T1, "url": "https://www.abc.net.au/news/feed/51120/rss.xml",      "type": "native", "focus": ["commodities", "energy_oil", "earnings_corporate"]},
    {"id": "xinhua",          "name": "Xinhua",                "region": "china",     "tier": T3, "url": "https://english.news.cn/rss/main.xml",                "type": "native", "focus": ["trade_tariff", "geopolitical", "tech_ai"], "state_media": True, "disabled": True},  # S135: RSS discontinued
    {"id": "global_times",    "name": "Global Times",          "region": "china",     "tier": T3, "url": "https://www.globaltimes.cn/rss/outbrain.xml",           "type": "native", "focus": ["geopolitical", "defense", "trade_tariff"], "state_media": True},  # S135: RSSHub→native (route missing)
    {"id": "china_daily",     "name": "China Daily",           "region": "china",     "tier": T3, "url": "https://www.chinadaily.com.cn/rss/world_rss.xml",     "type": "native", "focus": ["trade_tariff", "tech_ai"], "state_media": True},
    {"id": "cgtn",            "name": "CGTN",                  "region": "china",     "tier": T3, "url": "https://www.cgtn.com/subscribe/rss/section/world.xml", "type": "native", "focus": ["geopolitical", "trade_tariff"], "state_media": True},
    {"id": "scmp",            "name": "SCMP",                  "region": "hongkong",  "tier": T2, "url": "https://www.scmp.com/rss/91/feed",                     "type": "native", "focus": ["geopolitical", "trade_tariff", "tech_ai", "semiconductors"]},  # S135: RSSHub→native (route broken)
    # --- EUROPE (10) ---
    {"id": "dw_all",          "name": "DW English",            "region": "eu",        "tier": T1, "url": "https://rss.dw.com/xml/rss-en-all",                   "type": "native", "focus": ["geopolitical", "defense", "energy_oil"]},
    {"id": "dw_business",     "name": "DW Business",           "region": "eu",        "tier": T1, "url": "https://rss.dw.com/xml/rss-en-bus",                   "type": "native", "focus": ["trade_tariff", "central_bank", "earnings_corporate"]},
    {"id": "france24",        "name": "France24",              "region": "eu",        "tier": T1, "url": "https://www.france24.com/en/rss",                     "type": "native", "focus": ["geopolitical", "defense", "energy_oil"]},
    {"id": "france24_econ",   "name": "France24 Economy",      "region": "eu",        "tier": T1, "url": "https://www.france24.com/en/tag/economy/rss",         "type": "native", "focus": ["central_bank", "trade_tariff", "currency_fx"]},  # S135: URL fixed
    {"id": "euronews",        "name": "Euronews",              "region": "eu",        "tier": T2, "url": "https://www.euronews.com/rss",                        "type": "native", "focus": ["geopolitical", "energy_oil"]},
    {"id": "politico_eu",     "name": "Politico EU",           "region": "eu",        "tier": T2, "url": "https://www.politico.eu/feed/",                       "type": "native", "focus": ["trade_tariff", "geopolitical"]},
    {"id": "ecb_press",       "name": "ECB Press",             "region": "eu",        "tier": T1, "url": "https://www.ecb.europa.eu/rss/press.html",            "type": "native", "focus": ["central_bank", "currency_fx"]},
    {"id": "ecb_bulletin",    "name": "ECB Blog",              "region": "eu",        "tier": T1, "url": "https://www.ecb.europa.eu/rss/blog.html",             "type": "native", "focus": ["central_bank"]},  # S135: EB discontinued, using Blog RSS
    {"id": "rferl",           "name": "RFE/RL",                "region": "eu",        "tier": T2, "url": "https://www.rferl.org/api/",                          "type": "native", "focus": ["geopolitical", "energy_oil"]},
    {"id": "euobserver",      "name": "EUobserver",            "region": "eu",        "tier": T2, "url": "https://euobserver.com/rss",                          "type": "native", "focus": ["trade_tariff", "geopolitical"]},  # S135: URL fixed
    # --- UK (5) ---
    {"id": "bbc_world",       "name": "BBC World",             "region": "uk",        "tier": T1, "url": "https://feeds.bbci.co.uk/news/world/rss.xml",         "type": "native", "focus": ["geopolitical", "defense"]},
    {"id": "bbc_business",    "name": "BBC Business",          "region": "uk",        "tier": T1, "url": "https://feeds.bbci.co.uk/news/business/rss.xml",      "type": "native", "focus": ["central_bank", "earnings_corporate", "energy_oil"]},
    {"id": "guardian_world",  "name": "Guardian World",        "region": "uk",        "tier": T2, "url": "https://www.theguardian.com/world/rss",               "type": "native", "focus": ["geopolitical", "defense"]},  # S135: works with User-Agent header
    {"id": "guardian_biz",    "name": "Guardian Business",     "region": "uk",        "tier": T2, "url": "https://www.theguardian.com/uk/business/rss",         "type": "native", "focus": ["central_bank", "energy_oil", "earnings_corporate"]},  # S135: works with User-Agent header
    {"id": "reuters_world",   "name": "Reuters World",         "region": "uk",        "tier": T1, "url": f"{_RSSHUB_BASE}/reuters/world",                       "type": "rsshub", "focus": ["geopolitical", "defense", "energy_oil"]},
    # --- MIDDLE EAST (2) + INDIA (1) ---
    {"id": "aljazeera",       "name": "Al Jazeera",            "region": "me",        "tier": T2, "url": "https://www.aljazeera.com/xml/rss/all.xml",           "type": "native", "focus": ["geopolitical", "energy_oil", "defense"]},
    {"id": "dw_asia",         "name": "DW Asia",               "region": "asia",      "tier": T1, "url": "https://rss.dw.com/xml/rss-en-asia",                  "type": "native", "focus": ["geopolitical", "tech_ai", "trade_tariff"]},  # S135: replaces AJ duplicate, fills India/Asia gap
    {"id": "times_israel",    "name": "Times of Israel",       "region": "me",        "tier": T2, "url": "https://www.timesofisrael.com/feed/",                 "type": "native", "focus": ["geopolitical", "defense"]},  # S135: works with User-Agent header
    # --- RUSSIA (3 — ALL STATE_MEDIA) ---
    {"id": "tass",            "name": "TASS",                  "region": "russia",    "tier": T3, "url": "https://tass.com/rss/v2.xml",                         "type": "native", "focus": ["geopolitical", "energy_oil", "defense"], "state_media": True},
    {"id": "rt",              "name": "RT",                    "region": "russia",    "tier": T3, "url": "https://www.rt.com/rss/news/",                        "type": "native", "focus": ["geopolitical", "defense"], "state_media": True},
    {"id": "sputnik",         "name": "Sputnik",               "region": "russia",    "tier": T3, "url": f"{_RSSHUB_BASE}/sputniknews",                         "type": "rsshub", "focus": ["geopolitical", "defense"], "state_media": True},
    # --- AMERICAS (6) ---
    {"id": "cbc_world",       "name": "CBC World",             "region": "americas",  "tier": T1, "url": "https://www.cbc.ca/cmlink/rss-world",                "type": "native", "focus": ["geopolitical", "energy_oil"]},
    {"id": "cbc_business",    "name": "CBC Business",          "region": "americas",  "tier": T1, "url": "https://www.cbc.ca/cmlink/rss-business",             "type": "native", "focus": ["energy_oil", "commodities", "trade_tariff"]},
    {"id": "globe_mail",      "name": "Globe and Mail",        "region": "americas",  "tier": T2, "url": "https://www.theglobeandmail.com/arc/outboundfeeds/rss/category/business/", "type": "native", "focus": ["commodities", "energy_oil"]},
    {"id": "folha",           "name": "Folha de São Paulo",    "region": "americas",  "tier": T2, "url": "https://feeds.folha.uol.com.br/mundo/rss091.xml",    "type": "native", "focus": ["commodities", "geopolitical"]},  # S135: RSSHub→native (route missing)
    {"id": "reuters_latam",   "name": "Reuters LatAm",         "region": "americas",  "tier": T1, "url": f"{_RSSHUB_BASE}/reuters/world/americas",             "type": "rsshub", "focus": ["commodities", "trade_tariff"]},
    {"id": "ap_world",        "name": "AP World",              "region": "americas",  "tier": T1, "url": f"{_RSSHUB_BASE}/apnews/topics/apf-topnews",          "type": "rsshub", "focus": ["geopolitical", "defense"]},
    # --- WIRE + INSTITUTIONAL (4) ---
    {"id": "reuters_biz",     "name": "Reuters Business",      "region": "global",    "tier": T1, "url": f"{_RSSHUB_BASE}/reuters/business",                   "type": "rsshub", "focus": ["earnings_corporate", "trade_tariff", "central_bank"]},
    {"id": "ap_business",     "name": "AP Business",           "region": "global",    "tier": T1, "url": f"{_RSSHUB_BASE}/apnews/topics/business",             "type": "rsshub", "focus": ["earnings_corporate", "trade_tariff"]},  # S135: apf-business→business (AP changed slug)
    {"id": "boj",             "name": "Bank of Japan",         "region": "japan",     "tier": T1, "url": "https://www.boj.or.jp/en/rss/whatsnew.xml",           "type": "native", "focus": ["central_bank", "currency_fx"]},
    {"id": "boe",             "name": "Bank of England",       "region": "uk",        "tier": T1, "url": "https://www.bankofengland.co.uk/rss/news",            "type": "native", "focus": ["central_bank", "currency_fx"]},
    # --- SPECIALIST (6) ---
    {"id": "defense_news",    "name": "Defense News",          "region": "global",    "tier": T2, "url": "https://www.defensenews.com/arc/outboundfeeds/rss/",  "type": "native", "focus": ["defense"]},
    {"id": "japan_industry",  "name": "Japan Industry News",   "region": "japan",     "tier": T2, "url": "https://www.japanindustrynews.com/feed/",             "type": "native", "focus": ["semiconductors", "tech_ai"]},
    {"id": "diplomat",        "name": "The Diplomat",          "region": "global",    "tier": T2, "url": "https://thediplomat.com/feed/",                       "type": "native", "focus": ["geopolitical", "defense"]},
    {"id": "kyodo",           "name": "Kyodo News",            "region": "japan",     "tier": T1, "url": f"{_RSSHUB_BASE}/kyodonews/en",                        "type": "rsshub", "focus": ["geopolitical", "central_bank"], "disabled": True},  # S135: RSSHub plugin config broken
    {"id": "taipei_times",    "name": "Taipei Times",          "region": "taiwan",    "tier": T2, "url": "https://www.taipeitimes.com/xml/index.rss",           "type": "native", "focus": ["geopolitical", "semiconductors"]},
    {"id": "india_et",        "name": "Economic Times India",  "region": "india",     "tier": T2, "url": "https://economictimes.indiatimes.com/rssfeedstopstories.cms", "type": "native", "focus": ["tech_ai", "earnings_corporate"], "disabled": True},  # S135: 403 anti-bot
]

# ============================================================
# Relevance Keywords
# ============================================================

RELEVANCE_KEYWORDS = {
    "semiconductors": ["chip", "semiconductor", "tsmc", "samsung", "hynix", "wafer", "fab", "foundry", "asml", "tokyo electron", "chip export"],
    "defense": ["military", "defense", "defence", "missile", "navy", "aircraft", "procurement", "nato", "pentagon", "arms deal", "weapons"],
    "energy_oil": ["oil", "crude", "brent", "opec", "lng", "natural gas", "pipeline", "hormuz", "tanker", "refinery", "energy crisis"],
    "central_bank": ["central bank", "interest rate", "monetary policy", "inflation", "ecb", "boj", "boe", "fed ", "rate decision", "rate cut", "rate hike", "quantitative"],
    "trade_tariff": ["tariff", "trade war", "sanctions", "export control", "import duty", "trade deficit", "trade deal", "trade restriction"],
    "geopolitical": ["iran", "taiwan strait", "ukraine", "russia", "china", "conflict", "escalation", "ceasefire", "invasion", "strait of hormuz", "territorial"],
    "tech_ai": ["artificial intelligence", "ai chip", "data center", "cloud computing", "nvidia", "amd", "tech regulation", "ai model"],
    "commodities": ["iron ore", "copper", "gold", "silver", "wheat", "coal", "mining", "commodity", "lithium", "rare earth"],
    "currency_fx": ["dollar", "euro", "yen", "yuan", "forex", "exchange rate", "currency", "devaluation"],
    "earnings_corporate": ["earnings", "revenue", "profit", "quarterly", "guidance", "outlook", "ipo", "merger", "acquisition"],
}

# ============================================================
# Scan Windows (timezone-aware scheduling)
# ============================================================

_ASIA_PACIFIC_IDS = {f["id"] for f in FEED_REGISTRY if f["region"] in ("japan", "korea", "taiwan", "singapore", "australia", "china", "hongkong", "india", "asia")}
_EUROPE_ME_RUSSIA_IDS = {f["id"] for f in FEED_REGISTRY if f["region"] in ("eu", "uk", "me", "russia")}
# S159 M-11: Hardcoded because "global" region includes non-wire feeds (e.g., RSSHub aggregators).
# Update this set when adding/removing wire-service feeds from FEED_REGISTRY.
_WIRE_IDS = {"bbc_world", "bbc_business", "reuters_world", "reuters_biz", "aljazeera", "cbc_world", "cbc_business", "ap_world", "ap_business"}
_ALL_IDS = {f["id"] for f in FEED_REGISTRY}

SCAN_WINDOWS = {
    "asian":      {"start_hour_et": 19, "start_min_et": 0,  "end_hour_et": 2,  "end_min_et": 0,  "feed_ids": _ASIA_PACIFIC_IDS},
    "european":   {"start_hour_et": 2,  "start_min_et": 0,  "end_hour_et": 6,  "end_min_et": 0,  "feed_ids": _EUROPE_ME_RUSSIA_IDS},
    "premarket":  {"start_hour_et": 6,  "start_min_et": 0,  "end_hour_et": 9,  "end_min_et": 30, "feed_ids": _ALL_IDS},
    "intraday":   {"start_hour_et": 9,  "start_min_et": 30, "end_hour_et": 16, "end_min_et": 0,  "feed_ids": _WIRE_IDS},
    "afterhours": {"start_hour_et": 16, "start_min_et": 0,  "end_hour_et": 19, "end_min_et": 0,  "feed_ids": _ALL_IDS},
}

# ============================================================
# Data Model
# ============================================================

@dataclass
class ParsedHeadline:
    title: str
    summary: str
    source_name: str
    source_id: str
    source_region: str
    reliability_tier: str
    published_utc: datetime
    hours_until_us_open: float
    feed_url: str
    link: str
    is_state_media: bool
    relevance_tags: list = field(default_factory=list)

    @property
    def reliability_discount(self) -> float:
        if self.reliability_tier == T3:
            return _get_t3_discount()
        if self.reliability_tier == T2:
            return _get_t2_discount()
        return 1.0

    @property
    def dedup_key(self) -> str:
        # Strip <ingested_data> wrapper (CR-3 sanitization) before hashing
        # so that the same headline from different feeds produces the same key.
        raw = re.sub(r'</?ingested_data[^>]*>', '', self.title)
        normalized = re.sub(r'\s+', ' ', raw.lower().strip())
        return hashlib.md5(normalized.encode()).hexdigest()[:12]

# ============================================================
# RSS Feed Parser
# ============================================================

class RSSFeedParser:
    _NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "dc": "http://purl.org/dc/elements/1.1/",
        "content": "http://purl.org/rss/1.0/modules/content/",
    }

    def __init__(self, timeout: int = _FETCH_TIMEOUT):
        self._timeout = timeout
        # S233 CR-2: httpx.Client replaces requests.Session (CLIO convention, no event loop blocking via executor)
        self._session = httpx.Client(
            timeout=self._timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "CLIO-Lighthouse/2.0 (RSS Reader; +https://atlasstrategies.com)",
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            },
        )

    def fetch_feed(self, feed_meta: dict) -> list[ParsedHeadline]:
        url = feed_meta["url"]
        try:
            resp = self._session.get(url)  # S233 CR-2: timeout on client, not per-request
            resp.raise_for_status()
            return self._parse_xml(resp.text, feed_meta)
        except httpx.HTTPError as e:  # S233 CR-2: httpx exception hierarchy
            logger.debug("Lighthouse: feed fetch failed [%s]: %s", feed_meta["id"], e)
            return []
        except ET.ParseError as e:
            logger.debug("Lighthouse: XML parse failed [%s]: %s", feed_meta["id"], e)
            return []

    def _parse_xml(self, xml_text: str, feed_meta: dict) -> list[ParsedHeadline]:
        if len(xml_text) > 5_000_000:  # 5MB safety cap
            logger.debug("Lighthouse: XML too large [%s]: %d bytes", feed_meta["id"], len(xml_text))
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        tag = root.tag.split("}")[-1] if "}" in root.tag else root.tag
        if tag == "feed":
            items = self._parse_atom(root, feed_meta)
        elif tag == "rss":
            channel = root.find("channel")
            items = self._parse_rss2(channel, feed_meta) if channel is not None else []
        elif tag == "RDF":
            items = self._parse_rdf(root, feed_meta)
        else:
            channel = root.find("channel")
            items = self._parse_rss2(channel, feed_meta) if channel is not None else []
        return items[:_MAX_HEADLINES_PER_FEED]

    def _parse_rss2(self, channel, feed_meta: dict) -> list[ParsedHeadline]:
        headlines = []
        for item in channel.findall("item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            summary = re.sub(r"<[^>]+>", "", (item.findtext("description") or ""))[:500].strip()
            link = (item.findtext("link") or "").strip()
            pub_date = self._parse_date(item.findtext("pubDate") or item.findtext(f"{{{self._NS['dc']}}}date") or "")
            headlines.append(self._build_headline(title, summary, link, pub_date, feed_meta))
        return headlines

    def _parse_atom(self, root, feed_meta: dict) -> list[ParsedHeadline]:
        ns = self._NS["atom"]
        headlines = []
        for entry in root.findall(f"{{{ns}}}entry"):
            title = (entry.findtext(f"{{{ns}}}title") or "").strip()
            if not title:
                continue
            summary = re.sub(r"<[^>]+>", "", (entry.findtext(f"{{{ns}}}summary") or entry.findtext(f"{{{ns}}}content") or ""))[:500].strip()
            link_el = entry.find(f"{{{ns}}}link[@rel='alternate']") or entry.find(f"{{{ns}}}link")
            link = (link_el.get("href", "") if link_el is not None else "").strip()
            pub_date = self._parse_date(entry.findtext(f"{{{ns}}}published") or entry.findtext(f"{{{ns}}}updated") or "")
            headlines.append(self._build_headline(title, summary, link, pub_date, feed_meta))
        return headlines

    def _parse_rdf(self, root, feed_meta: dict) -> list[ParsedHeadline]:
        """Parse RSS 1.0 / RDF format (used by Taipei Times, some older feeds)."""
        ns_rss = "http://purl.org/rss/1.0/"
        headlines = []
        for item in root.findall(f"{{{ns_rss}}}item"):
            title = (item.findtext(f"{{{ns_rss}}}title") or "").strip()
            if not title:
                continue
            summary = re.sub(r"<[^>]+>", "", (item.findtext(f"{{{ns_rss}}}description") or ""))[:500].strip()
            link = (item.findtext(f"{{{ns_rss}}}link") or "").strip()
            pub_date = self._parse_date(item.findtext(f"{{{self._NS['dc']}}}date") or "")
            headlines.append(self._build_headline(title, summary, link, pub_date, feed_meta))
        return headlines

    def _build_headline(self, title, summary, link, pub_date, feed_meta):
        now_utc = datetime.now(_UTC)
        if pub_date is None:
            pub_date = now_utc
        # S233 CR-3: Sanitize external RSS content before passing to consumers
        _src = f"rss:{feed_meta['id']}"
        title = _sanitize_rss(title, source=_src) if title else ""
        summary = _sanitize_rss(summary, source=_src) if summary else ""
        return ParsedHeadline(
            title=title, summary=summary,
            source_name=feed_meta["name"], source_id=feed_meta["id"],
            source_region=feed_meta["region"], reliability_tier=feed_meta["tier"],
            published_utc=pub_date,
            hours_until_us_open=self._compute_hours_to_open(pub_date),
            feed_url=feed_meta["url"], link=link,
            is_state_media=feed_meta.get("state_media", False),
            relevance_tags=self._tag_relevance(title, summary),
        )

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        date_str = date_str.strip()
        try:
            return parsedate_to_datetime(date_str).astimezone(_UTC)
        except (ValueError, TypeError):
            pass
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f%z"):
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=_UTC)
                return dt.astimezone(_UTC)
            except ValueError:
                continue
        return None

    @staticmethod
    def _compute_hours_to_open(pub_utc: datetime) -> float:
        pub_et = pub_utc.astimezone(_NY)
        today_open = pub_et.replace(hour=_US_OPEN_HOUR, minute=_US_OPEN_MINUTE, second=0, microsecond=0)
        if pub_et >= today_open:
            next_open = today_open + timedelta(days=1)
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
        else:
            next_open = today_open
            while next_open.weekday() >= 5:
                next_open += timedelta(days=1)
        return round(max(0.0, (next_open - pub_et).total_seconds() / 3600), 2)

    @staticmethod
    def _tag_relevance(title: str, summary: str) -> list:
        combined = f"{title} {summary}".lower()
        # S159 M-10: Phrase-aware keyword matching — multi-word keywords use \s+ instead of literal space
        def _kw_match(kw: str, text: str) -> bool:
            _escaped = re.escape(kw.strip())
            _pattern = r'\b' + _escaped.replace(r'\ ', r'\s+') + r'\b'
            return bool(re.search(_pattern, text, re.IGNORECASE))
        return [tag for tag, kws in RELEVANCE_KEYWORDS.items() if any(_kw_match(kw, combined) for kw in kws)]

# ============================================================
# Scan Scheduler
# ============================================================

class ScanScheduler:
    @staticmethod
    def get_active_window() -> Optional[str]:
        if os.environ.get("CLIO_RSS_TZ_SCHEDULING_ENABLED", "1") != "1":
            return "premarket"
        now = datetime.now(_NY)
        current_mins = now.hour * 60 + now.minute
        for name, win in SCAN_WINDOWS.items():
            s = win["start_hour_et"] * 60 + win["start_min_et"]
            e = win["end_hour_et"] * 60 + win["end_min_et"]
            if s > e:  # crosses midnight
                if current_mins >= s or current_mins < e:
                    return name
            else:
                if s <= current_mins < e:
                    return name
        return None

    @staticmethod
    def get_feeds_for_window(window_name: Optional[str]) -> list[dict]:
        if window_name is None:
            return []
        win = SCAN_WINDOWS.get(window_name)
        if win is None:
            return []
        return [f for f in FEED_REGISTRY if f["id"] in win["feed_ids"] and not f.get("disabled")]

# ============================================================
# Headline Manager
# ============================================================

class HeadlineManager:
    def __init__(self):
        self._seen_keys: dict[str, float] = {}
        self._last_cleanup = 0.0
        self._lock = threading.Lock()

    def process(self, raw_headlines: list[ParsedHeadline]) -> list[ParsedHeadline]:
        self._maybe_cleanup()
        cutoff = datetime.now(_UTC) - timedelta(hours=_MAX_HEADLINE_AGE_HOURS)
        result = []
        with self._lock:
            for hl in raw_headlines:
                if hl.published_utc < cutoff:
                    continue
                if len(hl.relevance_tags) < _MIN_RELEVANCE_KEYWORDS:
                    continue
                if hl.dedup_key in self._seen_keys:
                    continue
                self._seen_keys[hl.dedup_key] = time.time()
                result.append(hl)
        result.sort(key=lambda h: h.published_utc, reverse=True)
        return result

    def _maybe_cleanup(self):
        now = time.time()
        with self._lock:
            if now - self._last_cleanup < 3600:
                return
            cutoff = now - (_DEDUP_WINDOW_HOURS * 3600)
            self._seen_keys = {k: v for k, v in self._seen_keys.items() if v > cutoff}
            self._last_cleanup = now

# ============================================================
# Context Generator
# ============================================================

class GlobalNewsContextGenerator:
    @staticmethod
    def format_for_analysis(headlines: list[ParsedHeadline], window_name: str) -> str:
        if not headlines:
            return ""
        regions = sorted({h.source_region for h in headlines})
        t1 = sum(1 for h in headlines if h.reliability_tier == T1)
        t3 = sum(1 for h in headlines if h.reliability_tier == T3)
        lines = [
            f"GLOBAL NEWS INTELLIGENCE — {window_name.upper()} WINDOW",
            f"Headlines: {len(headlines)} | Regions: {', '.join(regions)} | T1: {t1} | T3(STATE): {t3}",
            "",
        ]
        for hl in headlines:
            state_tag = " [STATE_MEDIA]" if hl.is_state_media else ""
            time_tag = f" [T-{hl.hours_until_us_open:.1f}h to US open]" if hl.hours_until_us_open > 0 else " [MARKET HOURS]"
            lines.append(f"[{hl.reliability_tier}]{state_tag} {hl.source_name} ({hl.source_region}){time_tag}")
            lines.append(f"  {hl.title}")
            if hl.summary:
                lines.append(f"  > {hl.summary[:200]}{'...' if len(hl.summary) > 200 else ''}")
            if hl.relevance_tags:
                lines.append(f"  Tags: {', '.join(hl.relevance_tags)}")
            lines.append("")
        lines.extend([
            "SOURCE RELIABILITY GUIDE:",
            "- T1_VERIFIED: Treat as reliable factual reporting",
            "- T2_SEMI_INDEPENDENT: Generally reliable, may have editorial slant",
            "- T3_STATE_AFFILIATED: [STATE_MEDIA] — Interpret as regime messaging, NOT factual reporting",
        ])
        return "\n".join(lines)

    @staticmethod
    def format_enrichment_context(headlines: list[ParsedHeadline], tag_filter: list[str]) -> str:
        filtered = [h for h in headlines if any(t in h.relevance_tags for t in tag_filter)][:20]
        if not filtered:
            return ""
        lines = [f"GLOBAL NEWS CONTEXT ({len(filtered)} headlines, tags: {', '.join(tag_filter)}):"]
        for hl in filtered:
            state = " [STATE_MEDIA]" if hl.is_state_media else ""
            lines.append(f"  [{hl.reliability_tier}]{state} {hl.source_name}: {hl.title}")
        return "\n".join(lines)

# ============================================================
# Module Singletons (thread-safe)
# ============================================================

_singleton_lock = threading.Lock()
_parser: Optional[RSSFeedParser] = None
_scheduler = ScanScheduler()
_headline_mgr: Optional[HeadlineManager] = None
_context_gen = GlobalNewsContextGenerator()

# S186 W3.1: per-feed circuit breaker (anti-pattern: stuck-loop on chronically-broken feeds)
# Module-level dedup is acceptable here per S184-followup precedent: 24h quarantine window
# dominates restart rate (~2/day), so persistence not justified.
_FEED_QUARANTINE_THRESHOLD = 3       # fails before quarantine
_FEED_QUARANTINE_HOURS = 24.0        # quarantine duration
_feed_health: dict[str, dict] = {}   # feed_id -> {consecutive_failures, quarantined_until_ts, last_attempt_ts}
_feed_health_lock = threading.Lock()

# ============================================================
# Recent-headlines buffer (multi-consumer starvation fix — ORCH follow-up)
# ============================================================
#
# HeadlineManager._seen_keys (above) is a process-global "not yet returned to ANY
# caller" filter: scan_current_window()'s RETURN VALUE only ever contains headlines
# no earlier caller has already been handed. That's fine with one consumer, but
# learning_engine (15-min cycle), market_events.check_rss_breaking_news (5-min,
# market hours) and DPOF's Jev headline refresher (10-min) already race each other
# for the same finite "not yet seen" set — whichever calls scan_current_window()
# first "steals" a headline from the others, which then never see it again.
# newsflash_lite.py would be a 4th such racer if it called scan_current_window()
# itself. The buffer below is a SEPARATE, non-consuming, ADDITIVE view: every
# scan_current_window() call appends whatever it just parsed (post relevance-tag
# filter, same objects) into this buffer, deduped by dedup_key (not by
# _seen_keys), aged out at _MAX_HEADLINE_AGE_HOURS, capped at
# _RECENT_BUFFER_CAP. get_recent_headlines() reads it without marking anything
# consumed, so any number of readers can see the same headline. This is a pure
# side effect of scan_current_window() — its own return value and behavior for
# existing callers are unchanged (see tests/test_global_news_feeds_buffer.py::
# test_scan_current_window_return_value_unchanged).
#
# ORCH follow-up #2 item 4: even a caller that only wants the SIDE EFFECT (drive a
# scan / populate the buffer), and deliberately ignores scan_current_window()'s
# return value, still "steals" from learning_engine by calling scan_current_window()
# at all — the RETURN VALUE is computed via HeadlineManager.process(), which marks
# every item "seen" regardless of whether the caller looks at what came back.
# refresh_recent_headlines() below is a side-effect-only driver that fetches the
# same feeds and applies the same age/relevance filtering WITHOUT ever touching
# _seen_keys, for exactly that "I only want the buffer updated" use case
# (market_events.check_rss_breaking_news uses this now; DPOF's Jev refresher is
# next, per the orchestrator).
_RECENT_BUFFER_CAP = 1000
_recent_buffer: list["ParsedHeadline"] = []
_recent_buffer_lock = threading.Lock()
_last_scan_time: Optional[float] = None


def _add_to_recent_buffer(headlines: list) -> int:
    """Returns the count of genuinely NEW (not-already-buffered) headlines added."""
    global _last_scan_time
    cutoff = datetime.now(_UTC) - timedelta(hours=_MAX_HEADLINE_AGE_HOURS)
    added = 0
    with _recent_buffer_lock:
        if headlines:
            existing_keys = {h.dedup_key for h in _recent_buffer}
            for hl in headlines:
                if hl.published_utc < cutoff:
                    continue
                if hl.dedup_key in existing_keys:
                    continue
                existing_keys.add(hl.dedup_key)
                _recent_buffer.append(hl)
                added += 1
            _recent_buffer[:] = [h for h in _recent_buffer if h.published_utc >= cutoff]
            _recent_buffer.sort(key=lambda h: h.published_utc, reverse=True)
            if len(_recent_buffer) > _RECENT_BUFFER_CAP:
                del _recent_buffer[_RECENT_BUFFER_CAP:]
        _last_scan_time = time.time()
    return added


def get_recent_headlines(max_age_seconds: int = 3600) -> list:
    """Sync, never raises, no network. Returns buffer items with published_utc within
    max_age_seconds, newest first. A non-consuming read (unlike scan_current_window()'s
    return value) — safe for any number of independent callers. Populated as a side
    effect of scan_current_window(), whichever caller's cadence happens to trigger it."""
    try:
        cutoff = datetime.now(_UTC) - timedelta(seconds=max_age_seconds)
        with _recent_buffer_lock:
            snapshot = list(_recent_buffer)
        return [h for h in snapshot if h.published_utc >= cutoff]
    except Exception as e:  # noqa: BLE001 — reader contract: never raise
        logger.debug("Lighthouse: get_recent_headlines failed: %s", e)
        return []


def get_last_scan_time() -> Optional[float]:
    """Epoch seconds of the last time ANY caller's scan_current_window() actually
    performed a scan (i.e. was not short-circuited by disabled flags / no active
    window / no feeds). None if no scan has happened yet this process."""
    with _recent_buffer_lock:
        return _last_scan_time


def _feed_is_quarantined(feed_id: str) -> bool:
    """Return True if feed_id is currently in quarantine window."""
    with _feed_health_lock:
        rec = _feed_health.get(feed_id)
        if not rec:
            return False
        until = rec.get("quarantined_until_ts", 0)
        return time.time() < until


def _record_feed_outcome(feed_id: str, feed_name: str, success: bool) -> None:
    """Track success/failure; quarantine if 3 consecutive failures."""
    now = time.time()
    with _feed_health_lock:
        rec = _feed_health.setdefault(
            feed_id, {"consecutive_failures": 0, "quarantined_until_ts": 0, "last_attempt_ts": 0}
        )
        rec["last_attempt_ts"] = now
        if success:
            if rec["consecutive_failures"] > 0 and rec.get("quarantined_until_ts", 0) > 0:
                logger.info("Lighthouse: feed %s [%s] recovered after quarantine", feed_id, feed_name)
            # S190 M12: reset BOTH counters atomically (held under _feed_health_lock).
            # Resetting only consecutive_failures while leaving quarantined_until_ts in
            # the future would cause new failures to recompute quarantine off a stale
            # window. Pair must stay locked together.
            rec["consecutive_failures"] = 0
            rec["quarantined_until_ts"] = 0
        else:
            rec["consecutive_failures"] += 1
            if (
                rec["consecutive_failures"] >= _FEED_QUARANTINE_THRESHOLD
                and rec.get("quarantined_until_ts", 0) < now
            ):
                rec["quarantined_until_ts"] = now + (_FEED_QUARANTINE_HOURS * 3600)
                logger.warning(
                    "Lighthouse: quarantining feed %s [%s] for %.0fh after %d consecutive failures",
                    feed_id, feed_name, _FEED_QUARANTINE_HOURS, rec["consecutive_failures"],
                )

def _get_parser() -> RSSFeedParser:
    global _parser
    if _parser is None:
        with _singleton_lock:
            if _parser is None:
                _parser = RSSFeedParser()
    return _parser

def _get_headline_mgr() -> HeadlineManager:
    global _headline_mgr
    if _headline_mgr is None:
        with _singleton_lock:
            if _headline_mgr is None:
                _headline_mgr = HeadlineManager()
    return _headline_mgr

# ============================================================
# Public API
# ============================================================

def _fetch_raw_for_feeds(feeds: list[dict]) -> tuple[list, int, int]:
    """Shared feed-fetch loop (quarantine handling + _record_feed_outcome), factored out
    so both scan_current_window() and refresh_recent_headlines() (ORCH follow-up #2 item
    4) drive the same underlying fetch without duplicating the quarantine bookkeeping.
    Returns (raw_headlines, error_count, skipped_quarantined_count)."""
    parser = _get_parser()
    raw = []
    errors = 0
    skipped_quarantined = 0
    for feed in feeds:
        # S186 W3.1: skip chronically-broken feeds within quarantine window
        if _feed_is_quarantined(feed["id"]):
            skipped_quarantined += 1
            continue
        hl_list = parser.fetch_feed(feed)
        success = bool(hl_list)
        _record_feed_outcome(feed["id"], feed.get("name", ""), success)
        if not success:
            errors += 1
        raw.extend(hl_list)
    return raw, errors, skipped_quarantined


def scan_current_window() -> list[ParsedHeadline]:
    if os.environ.get("CLIO_LIGHTHOUSE_ENABLED", "1") != "1" or os.environ.get("CLIO_RSS_FEED_PARSER_ENABLED", "1") != "1":
        return []
    window = _scheduler.get_active_window()
    if window is None:
        return []
    feeds = _scheduler.get_feeds_for_window(window)
    if not feeds:
        return []
    mgr = _get_headline_mgr()
    raw, errors, skipped_quarantined = _fetch_raw_for_feeds(feeds)
    processed = mgr.process(raw)
    _add_to_recent_buffer(processed)  # side effect only — see "Recent-headlines buffer" above
    logger.info(
        "LIGHTHOUSE%s window=%s feeds=%d raw=%d processed=%d errors=%d quarantined=%d",
        " [LOG_ONLY]" if os.environ.get("CLIO_LIGHTHOUSE_LOG_ONLY", "1") == "1" else "",
        window, len(feeds), len(raw), len(processed), errors, skipped_quarantined,
    )
    return processed


def refresh_recent_headlines() -> int:
    """ORCH follow-up #2 item 4 ("residual starvation for learning_engine"):
    market_events.check_rss_breaking_news() (and, later, DPOF's Jev refresher) drove
    the scan by calling scan_current_window(), whose HeadlineManager.process() marks
    every returned headline "seen" in _seen_keys — even though those callers only
    wanted the SIDE EFFECT (populate the buffer / trigger a live fetch), not the return
    value. That silently stole headlines from scan_current_window()'s one remaining
    RETURN-VALUE consumer, learning_engine. This function drives the identical
    feed-fetch (_fetch_raw_for_feeds(), including quarantine handling) and applies the
    SAME age (_MAX_HEADLINE_AGE_HOURS) + relevance-tag (_MIN_RELEVANCE_KEYWORDS)
    filtering HeadlineManager.process() does, but WITHOUT ever touching
    HeadlineManager._seen_keys — so scan_current_window() called afterwards (by
    learning_engine or anyone else) still returns those same headlines, unconsumed.
    Sync, never raises, logs like scan_current_window(). Returns the number of
    genuinely new headlines added to the shared buffer (0 if disabled / no active
    window / no feeds / nothing new)."""
    try:
        if os.environ.get("CLIO_LIGHTHOUSE_ENABLED", "1") != "1" or os.environ.get("CLIO_RSS_FEED_PARSER_ENABLED", "1") != "1":
            return 0
        window = _scheduler.get_active_window()
        if window is None:
            return 0
        feeds = _scheduler.get_feeds_for_window(window)
        if not feeds:
            return 0
        raw, errors, skipped_quarantined = _fetch_raw_for_feeds(feeds)
        cutoff = datetime.now(_UTC) - timedelta(hours=_MAX_HEADLINE_AGE_HOURS)
        filtered = [
            hl for hl in raw
            if hl.published_utc >= cutoff and len(hl.relevance_tags) >= _MIN_RELEVANCE_KEYWORDS
        ]
        added = _add_to_recent_buffer(filtered)
        logger.info(
            "LIGHTHOUSE-REFRESH%s window=%s feeds=%d raw=%d filtered=%d added=%d errors=%d quarantined=%d",
            " [LOG_ONLY]" if os.environ.get("CLIO_LIGHTHOUSE_LOG_ONLY", "1") == "1" else "",
            window, len(feeds), len(raw), len(filtered), added, errors, skipped_quarantined,
        )
        return added
    except Exception as e:  # noqa: BLE001 — reader/driver contract: never raise
        logger.debug("Lighthouse: refresh_recent_headlines failed: %s", e)
        return 0

def get_analysis_context(headlines: Optional[list[ParsedHeadline]] = None) -> str:
    if headlines is None:
        headlines = scan_current_window()
    if not headlines:
        return ""
    window = _scheduler.get_active_window() or "unknown"
    return _context_gen.format_for_analysis(headlines, window)

def get_enrichment_context(headlines: Optional[list[ParsedHeadline]] = None,
                           tag_filter: Optional[list[str]] = None) -> str:
    if headlines is None:
        headlines = scan_current_window()
    if not headlines or not tag_filter:
        return ""
    return _context_gen.format_enrichment_context(headlines, tag_filter)

def get_active_window_name() -> Optional[str]:
    return _scheduler.get_active_window()

def get_feed_count() -> int:
    return sum(1 for f in FEED_REGISTRY if not f.get("disabled"))

def get_feed_stats() -> dict:
    active = [f for f in FEED_REGISTRY if not f.get("disabled")]
    return {
        "total_feeds": len(FEED_REGISTRY),
        "active_feeds": len(active),
        "disabled_feeds": len(FEED_REGISTRY) - len(active),
        "native_rss": sum(1 for f in active if f["type"] == "native"),
        "rsshub": sum(1 for f in active if f["type"] == "rsshub"),
        "t1": sum(1 for f in active if f["tier"] == T1),
        "t2": sum(1 for f in active if f["tier"] == T2),
        "t3": sum(1 for f in active if f["tier"] == T3),
        "state_media": sum(1 for f in active if f.get("state_media")),
        "regions": sorted({f["region"] for f in active}),
        "active_window": _scheduler.get_active_window(),
        "enabled": os.environ.get("CLIO_LIGHTHOUSE_ENABLED", "1") == "1",
        "log_only": os.environ.get("CLIO_LIGHTHOUSE_LOG_ONLY", "1") == "1",
    }
