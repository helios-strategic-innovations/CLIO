"""
OpenClaw/Moltbot Configuration
Loads all API keys and credentials from .env
"""

import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from urllib.parse import urlparse


# --- Anthropic routing (OneProvider gateway, OP-01) ---
_GATEWAY_ALLOWED_HOSTS = {"api.oneprovider.dev"}  # the gateway key may only ever go here
_DEFAULT_GATEWAY_URL = "https://api.oneprovider.dev"
_GATEWAY_MODES = ("auto", "on", "off")


def _clean_secret(value) -> str:
    """Strip whitespace and one pair of surrounding quotes from a secret. Never raises."""
    try:
        s = str(value or "").strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        return s.strip()
    except Exception:
        return ""


def _resolve_anthropic_route(env: Mapping) -> tuple[str, str | None, bool]:
    """Return (api_key, base_url_or_None, gateway) for the Anthropic client.

    gateway=True means traffic goes through OneProvider. An invalid or missing
    gateway configuration fails closed: the key is "" and no key is sent anywhere.
    Messages never include key material.
    """
    try:
        direct_key = _clean_secret(env.get("ANTHROPIC_API_KEY"))
        op_key = _clean_secret(env.get("ONEPROVIDER_API_KEY"))
        mode = str(env.get("CLIO_ANTHROPIC_GATEWAY") or "").strip().lower() or "auto"
        if mode not in _GATEWAY_MODES:
            print("[config] WARNING: CLIO_ANTHROPIC_GATEWAY is not auto|on|off; treating as auto", file=sys.stderr)
            mode = "auto"
        if mode == "off":
            return (direct_key, None, False)
        if mode == "auto" and not op_key:
            return (direct_key, None, False)
        if not op_key:
            return ("", None, True)
        url = str(env.get("ONEPROVIDER_BASE_URL") or "").strip() or _DEFAULT_GATEWAY_URL
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        hostport = parsed.netloc.rpartition("@")[2]
        reason = None
        if parsed.scheme != "https":
            reason = "scheme must be https"
        elif host not in _GATEWAY_ALLOWED_HOSTS:
            reason = "host not in allowlist"
        elif "@" in parsed.netloc:
            reason = "userinfo not allowed"
        elif parsed.query or parsed.fragment:
            reason = "query or fragment not allowed"
        elif ":" in hostport:
            reason = "explicit port not allowed"
        elif parsed.path not in ("", "/"):
            reason = "path must be empty"
        if reason is not None:
            print(f"[config] ERROR: ONEPROVIDER_BASE_URL rejected ({reason}); gateway disabled, no key sent", file=sys.stderr)
            return ("", None, True)
        return (op_key, f"https://{host}", True)
    except Exception:
        print("[config] WARNING: Anthropic route resolution failed; using direct key", file=sys.stderr)
        try:
            return (_clean_secret(env.get("ANTHROPIC_API_KEY")), None, False)
        except Exception:
            return ("", None, False)


def _gateway_mode_value(env: Mapping) -> str:
    """Normalized CLIO_ANTHROPIC_GATEWAY value (silent; invalid -> auto)."""
    try:
        raw = str(env.get("CLIO_ANTHROPIC_GATEWAY") or "").strip().lower()
    except Exception:
        return "auto"
    return raw if raw in _GATEWAY_MODES else "auto"


# --- Model ID normalization (OP-02) ---
_DOTTED_MODEL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_NORM_WARNED: set[str] = set()


def _norm_model_id(value) -> str:
    """Rewrite dotted Claude IDs (claude-sonnet-5.5) to hyphenated canonical form."""
    v = str(value or "").strip()
    if v.startswith("claude-") and _DOTTED_MODEL_RE.search(v):
        normalized = _DOTTED_MODEL_RE.sub("-", v)
        if v not in _NORM_WARNED:
            _NORM_WARNED.add(v)
            print(f"[config] WARNING: dotted model id {v!r} normalized to {normalized!r}", file=sys.stderr)
        return normalized
    return v


def _model_from_env(name: str, default: str) -> str:
    return _norm_model_id(os.getenv(name) or "") or default


def _safe_int(value: str, default: int, label: str) -> int:
    """Parse an int from a string, returning default on failure with a warning."""
    try:
        return int(value)
    except (ValueError, TypeError):
        print(f"[config] WARNING: invalid integer for {label}: {value!r}, using default {default}", file=sys.stderr)
        return default


@dataclass
class EmailAccount:
    name: str = ""
    imap_host: str = ""
    imap_port: int = 993
    smtp_host: str = ""
    smtp_port: int = 587
    address: str = ""
    password: str = ""
    can_send: bool = False


@dataclass
class Config:
    # --- Anthropic ---
    anthropic_api_key: str = ""
    oneprovider_api_key: str = ""
    oneprovider_base_url: str = "https://api.oneprovider.dev"
    anthropic_gateway_mode: str = "auto"
    anthropic_gateway: bool = False   # derived in from_env (True = route via OneProvider)
    anthropic_base_url: str = ""      # derived in from_env; "" = SDK default endpoint
    default_model: str = "claude-opus-5-5"
    fast_model: str = "claude-sonnet-5-5"
    judgment_model: str = "claude-opus-5-5"  # S201 per-role split: judgment tier (synthesis/consolidation + bridge-facing reports). Override via JUDGMENT_MODEL.
    sampling_model: str = "claude-sonnet-4-6"  # SEAL multigen diversity -- MUST accept sampling params (temperature). Sonnet 5.5 / Opus 5.5 return 400 for non-default temperature (Anthropic docs 2026-10-11); claude-sonnet-4-6 is still available. Do NOT repoint. Override via SAMPLING_MODEL.
    max_tokens: int = 8192

    # --- Telegram ---
    telegram_bot_token: str = ""
    telegram_authorized_users: list[int] = field(default_factory=list)

    # --- Alpaca (Paper Trading) ---
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_paper: bool = True
    alpaca_stock_feed: str = "iex"              # "iex" (free) or "sip" (paid SIP data)
    alpaca_options_feed: str = "indicative"     # "indicative" (free) or "opra" (paid OPRA data)

    # --- Polygon ---
    polygon_api_key: str = ""
    polygon_base_url: str = "https://api.polygon.io"

    # --- Unusual Whales ---
    uw_api_key: str = ""
    uw_base_url: str = "https://api.unusualwhales.com"

    # --- News & Data ---
    news_api_key: str = ""
    finnhub_api_key: str = ""
    fred_api_key: str = ""

    # --- SEC-API ---
    sec_api_key: str = ""
    sec_base_url: str = "https://api.sec-api.io"

    # --- Kalshi ---
    kalshi_api_key: str = ""
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_private_key_path: str = ""

    # --- SAM.gov ---
    sam_api_key: str = ""
    sam_contract_awards_base_url: str = ""
    sam_opportunities_base_url: str = ""

    # --- Reddit ---
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "OpenClaw/1.0"

    # --- Twitter/X via Apify ---
    apify_api_token: str = ""
    apify_twitter_actor_id: str = "apidojo/tweet-scraper"
    twitter_handles: dict[str, list[str]] = field(default_factory=dict)
    twitter_all_handles: list[str] = field(default_factory=list)

    # --- Email Accounts ---
    email_accounts: dict[str, EmailAccount] = field(default_factory=dict)

    # --- Canvas LMS ---
    canvas_api_url: str = ""
    canvas_api_token: str = ""

    # --- BigShort ---
    bigshort_api_key: str = ""

    # --- Financial Datasets (Bloomberg-grade fundamentals) ---
    financial_datasets_api_key: str = ""
    financial_datasets_base_url: str = "https://api.financialdatasets.ai"

    # --- OpenAI (Whisper voice transcription) ---
    openai_api_key: str = ""

    # --- ElevenLabs (Text-to-Speech) ---
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = ""

    # --- Twilio (Phone Call Integration) ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""
    voice_server_port: int = 8080
    voice_server_public_url: str = ""

    # --- Override Directives ---
    override_directives_enabled: bool = True
    override_voice_pin: str = ""  # Numeric PIN (4-12 digits) for Twilio voice call auth

    # --- News Intelligence ---
    news_intel_enabled: bool = True

    # --- Anthropic API Optimization ---
    adaptive_thinking: bool = False  # thinking={"type": "adaptive"} on core.py Opus calls
    server_compaction: bool = False  # context_management compaction on context overflow

    # --- Telegram Notifications (scheduled/monitoring alerts → Telegram; e.g.
    # error scans, book pipeline, code sentinel — not Hydrinia-specific) ---
    telegram_notify_chat_ids: list[str] = field(default_factory=list)

    # --- Database ---
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5432/openclaw"

    # --- Paths ---
    tradingbot_dir: str = "/home/agent/tradingbot"
    log_dir: str = "/home/agent/openclaw/logs"

    # --- Budget Caps (overridable via BUDGET_* env vars) ---
    budget_sec_api: int = 500              # monthly
    # trading_economics retired S187 (econ_calendar migrated to UW); slot intentionally absent
    budget_unusual_whales: int = 16000     # daily (CLIO runaway guard; ~10% of UW Startup 160K/day)
    budget_sam_gov: int = 500              # monthly
    budget_apify: int = 1000              # daily
    budget_whisper: int = 20              # daily
    budget_email_send: int = 50           # daily
    budget_email_read: int = 200          # daily
    budget_web_search: int = 1000         # daily  (generous: learning+news+intel share this)
    budget_web_fetch: int = 500           # daily
    budget_learning_analysis: int = 120   # daily  (raised from 108)
    budget_learning_synthesis: int = 12   # daily
    budget_polygon_learning: int = 1000   # daily
    budget_book_analysis: int = 16        # daily
    budget_book_synthesis: int = 2        # daily
    budget_learning_query_gen: int = 100  # daily  (adaptive query generation)
    budget_elevenlabs_tts: int = 100       # daily
    budget_premarket_scan: int = 5          # daily
    budget_hourly_debrief: int = 10         # daily
    budget_postmarket_debrief: int = 5      # daily
    budget_news_update: int = 30            # daily
    budget_report_web_search: int = 400    # daily  (S129: raised from 250; 6 deep-analysis modules exhaust by 3:45 PM)
    budget_news_intel: int = 50             # daily  (Sonnet analysis for news intelligence, raised from 40)
    budget_macro_research: int = 10         # daily  (raised from 5 for expanded commodity research)
    budget_geopolitical_research: int = 3   # daily  (Opus for crisis analysis)
    budget_fed_analysis: int = 3            # daily  (Opus for rate path analysis)
    budget_earnings_intel: int = 3          # daily  (Opus for bellwether analysis)
    budget_trade_policy: int = 3            # daily  (Opus for impact chain analysis)
    budget_commodity_research: int = 3      # daily  (Opus for supply balance analysis)
    budget_bond_intel: int = 3              # daily  (Opus for auction demand analysis)
    budget_seal_enrichment: int = 200       # daily  (Haiku calls for SEAL knowledge enrichment)
    budget_seal_retention: int = 150        # daily  (Haiku calls for SEAL QA retention testing)
    budget_adversarial_resolver: int = 30  # daily  (Haiku calls for adversarial contradiction resolution)
    budget_financial_datasets: int = 500  # daily (CLIO's share — plan allows 1K req/min)
    budget_social_enrich: int = 120       # daily (Sonnet tweet-signal extraction call attempts; ~40/day typical + sweep retries. Dollar belt: CLIO_ENRICH_DAILY_CAP_USD)
    budget_fred_api: int = 60             # daily (econ_calendar FRED fallback; free tier is ~120/min, this is a self-imposed courtesy cap — ~1 call/day typical)
    budget_econ_surprise: int = 1500      # daily (agent/integrations/econ_surprise.py — self-imposed courtesy cap for the unofficial keyless Nasdaq + TradingView econ-print endpoints; no vendor quota exists. Slow refresh is 5 calls/cycle (4 Nasdaq lookahead-days + 1 TradingView range call), NOT 2: weekdays 04:00-20:00 ET at 30min cadence ~= 32 refreshes/day ~= 160 calls, plus the remaining 8h/day + all-weekend at 3h cadence ~= 3-8 refreshes/day ~= 15-40 calls -- ~175 calls on a typical weekday. Fast-poll bursts around importance>=2 releases add ~11min window / 15s cadence x2 sources ~88 calls/event, up to a handful of same-day events (~500-600 calls on a very busy multi-release day). Worst-case weekday total ~700-750 calls/day — 1500/day leaves ample headroom)
    budget_newsflash_polygon: int = 800   # daily (agent/integrations/newsflash_lite.py — Polygon /v2/reference/news general-market poll, 60s min-interval gate; 800/day is well under Polygon Massive-tier limits for a single 20-item-per-call feed)
    budget_dpof_reflection: int = 50      # daily (agent/dpof/reflection.py — one Opus/GPT-6-Sol call per closed DPOF position reflected on; max_per_run=2/cycle keeps this nowhere near the cap)
    budget_dpof_debate: int = 200         # daily (agent/dpof/debate.py — bull+bear Sonnet/GPT-6-Sol calls on borderline entries/reviews only; 200/day = 100 debates)

    def anthropic_client_kwargs(self, timeout: float = 120, **extra) -> dict:
        """Kwargs for anthropic.Anthropic(...): key + timeout, base_url only when routing via the gateway."""
        kwargs = {"api_key": self.anthropic_api_key, "timeout": timeout}
        if self.anthropic_gateway and self.anthropic_base_url:
            kwargs["base_url"] = self.anthropic_base_url
            if os.environ.get("CLIO_ANTHROPIC_GATEWAY_TRANSPORT", "") != "0":
                try:
                    from agent.anthropic_gateway import get_shared_http_client
                    kwargs["http_client"] = get_shared_http_client()
                except Exception:
                    pass  # guard is best-effort; the plain SDK client still works
        kwargs.update(extra)
        return kwargs

    @classmethod
    def from_env(cls) -> "Config":
        _anthropic_key, _anthropic_base, _anthropic_gw = _resolve_anthropic_route(os.environ)
        authorized_users = []
        for env_var in ("TELEGRAM_AUTHORIZED_USERS", "TELEGRAM_AUTHORIZED_USERS_2"):
            raw = os.getenv(env_var, "")
            for uid in raw.split(","):
                uid = uid.strip()
                if uid:
                    parsed = _safe_int(uid, 0, env_var)
                    if parsed != 0:
                        authorized_users.append(parsed)

        # Notification chat IDs (send proactive alerts to all configured users)
        notify_chat_ids = []
        for env_key in ("TELEGRAM_NOTIFY_CHAT_ID", "TELEGRAM_NOTIFY_CHAT_ID_2"):
            cid = os.getenv(env_key, "").strip()
            if cid:
                notify_chat_ids.append(cid)

        # Parse Twitter handles by sector
        twitter_sectors = [
            "DEFENSE", "GEOPOLITICS", "HEALTHCARE", "BIOTECH", "POLITICS",
            "TRUMP", "CRYPTO", "COMMODITIES", "TECH", "AI", "REALESTATE",
            "ECONOMICS", "FOMC", "JAPAN", "BONDS", "FLOWS", "INDUSTRIALS", "LEGAL",
        ]
        twitter_handles = {}
        for sector in twitter_sectors:
            raw = os.getenv(f"TWITTER_HANDLES_{sector}", "")
            if raw:
                twitter_handles[sector.lower()] = [h.strip() for h in raw.split(",") if h.strip()]

        all_handles_raw = os.getenv("TWITTER_ALL_HANDLES", "")
        twitter_all_handles = [h.strip() for h in all_handles_raw.split(",") if h.strip()]

        # Parse email accounts
        email_accounts = {}
        # Personal Gmail (read-only)
        if os.getenv("EMAIL_PERSONAL_GMAIL_ADDRESS"):
            email_accounts["personal_gmail"] = EmailAccount(
                name="personal_gmail",
                imap_host=os.getenv("EMAIL_PERSONAL_GMAIL_IMAP_HOST", "imap.gmail.com"),
                imap_port=_safe_int(os.getenv("EMAIL_PERSONAL_GMAIL_IMAP_PORT", "993"), 993, "EMAIL_PERSONAL_GMAIL_IMAP_PORT"),
                address=os.getenv("EMAIL_PERSONAL_GMAIL_ADDRESS", ""),
                password=os.getenv("EMAIL_PERSONAL_GMAIL_PASSWORD", ""),
                can_send=False,
            )
        # Personal Atlas (read-only)
        if os.getenv("EMAIL_PERSONAL_ATLAS_ADDRESS"):
            email_accounts["personal_atlas"] = EmailAccount(
                name="personal_atlas",
                imap_host=os.getenv("EMAIL_PERSONAL_ATLAS_IMAP_HOST", "imap.gmail.com"),
                imap_port=_safe_int(os.getenv("EMAIL_PERSONAL_ATLAS_IMAP_PORT", "993"), 993, "EMAIL_PERSONAL_ATLAS_IMAP_PORT"),
                address=os.getenv("EMAIL_PERSONAL_ATLAS_ADDRESS", ""),
                password=os.getenv("EMAIL_PERSONAL_ATLAS_PASSWORD", ""),
                can_send=False,
            )
        # Bot Atlas (read + send)
        if os.getenv("EMAIL_BOT_ATLAS_ADDRESS"):
            email_accounts["bot_atlas"] = EmailAccount(
                name="bot_atlas",
                imap_host=os.getenv("EMAIL_BOT_ATLAS_IMAP_HOST", "imap.gmail.com"),
                imap_port=_safe_int(os.getenv("EMAIL_BOT_ATLAS_IMAP_PORT", "993"), 993, "EMAIL_BOT_ATLAS_IMAP_PORT"),
                smtp_host=os.getenv("EMAIL_BOT_ATLAS_SMTP_HOST", "smtp.gmail.com"),
                smtp_port=_safe_int(os.getenv("EMAIL_BOT_ATLAS_SMTP_PORT", "587"), 587, "EMAIL_BOT_ATLAS_SMTP_PORT"),
                address=os.getenv("EMAIL_BOT_ATLAS_ADDRESS", ""),
                password=os.getenv("EMAIL_BOT_ATLAS_PASSWORD", ""),
                can_send=True,
            )
        # Bot Gmail (read + send)
        if os.getenv("EMAIL_BOT_GMAIL_ADDRESS"):
            email_accounts["bot_gmail"] = EmailAccount(
                name="bot_gmail",
                imap_host=os.getenv("EMAIL_BOT_GMAIL_IMAP_HOST", "imap.gmail.com"),
                imap_port=_safe_int(os.getenv("EMAIL_BOT_GMAIL_IMAP_PORT", "993"), 993, "EMAIL_BOT_GMAIL_IMAP_PORT"),
                smtp_host=os.getenv("EMAIL_BOT_GMAIL_SMTP_HOST", "smtp.gmail.com"),
                smtp_port=_safe_int(os.getenv("EMAIL_BOT_GMAIL_SMTP_PORT", "587"), 587, "EMAIL_BOT_GMAIL_SMTP_PORT"),
                address=os.getenv("EMAIL_BOT_GMAIL_ADDRESS", ""),
                password=os.getenv("EMAIL_BOT_GMAIL_PASSWORD", ""),
                can_send=True,
            )

        # Determine paper/live mode first so key selection is correct
        _is_paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        if _is_paper:
            _alpaca_key = os.getenv("ALPACA_PAPER_KEY_ID", "")
            _alpaca_secret = os.getenv("ALPACA_PAPER_SECRET_KEY", "")
        else:
            _alpaca_key = os.getenv("ALPACA_KEY_ID") or os.getenv("ALPACA_PAPER_KEY_ID", "")
            _alpaca_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("ALPACA_PAPER_SECRET_KEY", "")
            if not os.getenv("ALPACA_KEY_ID"):
                import logging as _cfg_log
                _cfg_log.getLogger("openclaw.config").critical(
                    "ALPACA_KEY_ID not set in LIVE mode — falling back to PAPER keys. THIS IS NOT LIVE TRADING."
                )  # S233 ME-14: Elevated from warning to critical

        instance = cls(
            anthropic_api_key=_anthropic_key,
            oneprovider_api_key=_clean_secret(os.getenv("ONEPROVIDER_API_KEY")),
            oneprovider_base_url=(os.getenv("ONEPROVIDER_BASE_URL") or "").strip() or _DEFAULT_GATEWAY_URL,
            anthropic_gateway_mode=_gateway_mode_value(os.environ),
            anthropic_gateway=_anthropic_gw,
            anthropic_base_url=_anthropic_base or "",
            default_model=_model_from_env("DEFAULT_MODEL", "claude-opus-5-5"),
            fast_model=_model_from_env("FAST_MODEL", "claude-sonnet-5-5"),
            judgment_model=_model_from_env("JUDGMENT_MODEL", "claude-opus-5-5"),
            sampling_model=_model_from_env("SAMPLING_MODEL", "claude-sonnet-4-6"),
            max_tokens=_safe_int(os.getenv("MAX_TOKENS", "8192"), 8192, "MAX_TOKENS"),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
            telegram_authorized_users=authorized_users,
            # Alpaca — paper keys in paper mode, live keys (with paper fallback) in live mode
            alpaca_api_key=_alpaca_key,
            alpaca_secret_key=_alpaca_secret,
            alpaca_base_url=(
                os.getenv("ALPACA_PAPER_BASE_URL", "https://paper-api.alpaca.markets")
                if _is_paper
                else os.getenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
            ),
            alpaca_paper=_is_paper,
            alpaca_stock_feed=os.getenv("ALPACA_STOCK_FEED", "iex"),
            alpaca_options_feed=os.getenv("ALPACA_OPTIONS_FEED", "indicative"),
            # Polygon
            polygon_api_key=os.getenv("POLYGON_API_KEY", ""),
            polygon_base_url=os.getenv("POLYGON_BASE_URL", "https://api.polygon.io"),
            # Unusual Whales
            uw_api_key=os.getenv("UW_API_KEY", ""),
            uw_base_url=os.getenv("UW_BASE_URL", "https://api.unusualwhales.com"),
            # News & Data
            news_api_key=os.getenv("NEWS_API_KEY", ""),
            finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
            fred_api_key=os.getenv("FRED_API_KEY", ""),
            # SEC
            sec_api_key=os.getenv("SEC_API_KEY", ""),
            sec_base_url=os.getenv("SEC_BASE_URL", "https://api.sec-api.io"),
            # Kalshi
            kalshi_api_key=os.getenv("KALSHI_API_KEY", ""),
            kalshi_base_url=os.getenv("KALSHI_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"),
            kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH", ""),
            # SAM.gov
            sam_api_key=os.getenv("SAM_API_KEY", ""),
            sam_contract_awards_base_url=os.getenv("SAM_CONTRACT_AWARDS_BASE_URL", ""),
            sam_opportunities_base_url=os.getenv("SAM_OPPORTUNITIES_BASE_URL", ""),
            # Reddit
            reddit_client_id=os.getenv("REDDIT_CLIENT_ID", ""),
            reddit_client_secret=os.getenv("REDDIT_CLIENT_SECRET", ""),
            # Twitter/X via Apify
            apify_api_token=os.getenv("APIFY_API_TOKEN", ""),
            apify_twitter_actor_id=os.getenv("APIFY_TWITTER_ACTOR_ID", "apidojo/tweet-scraper"),
            twitter_handles=twitter_handles,
            twitter_all_handles=twitter_all_handles,
            # Email
            email_accounts=email_accounts,
            # Canvas
            canvas_api_url=os.getenv("CANVAS_API_URL", ""),
            canvas_api_token=os.getenv("CANVAS_API_TOKEN", ""),
            # BigShort
            bigshort_api_key=os.getenv("BIGSHORT_API_KEY", ""),
            # Financial Datasets
            financial_datasets_api_key=os.getenv("FINANCIAL_DATASETS_API_KEY", ""),
            financial_datasets_base_url=os.getenv("FINANCIAL_DATASETS_BASE_URL", "https://api.financialdatasets.ai"),
            # OpenAI (Whisper)
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            # ElevenLabs (TTS)
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
            # Twilio (Voice calls)
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            twilio_phone_number=os.getenv("TWILIO_PHONE_NUMBER", ""),
            voice_server_port=_safe_int(os.getenv("VOICE_SERVER_PORT", "8080"), 8080, "VOICE_SERVER_PORT"),
            voice_server_public_url=os.getenv("VOICE_SERVER_PUBLIC_URL", ""),
            # Override Directives
            override_directives_enabled=os.getenv("OVERRIDE_DIRECTIVES_ENABLED", "true").lower() in ("true", "yes", "1"),
            override_voice_pin=os.getenv("OVERRIDE_VOICE_PIN", ""),
            # News Intelligence
            news_intel_enabled=os.getenv("NEWS_INTEL_ENABLED", "true").lower() in ("true", "1", "yes"),
            # Anthropic API Optimization
            adaptive_thinking=os.getenv("MOLTBOT_ADAPTIVE_THINKING", "0") in ("1", "true", "yes"),
            server_compaction=os.getenv("MOLTBOT_SERVER_COMPACTION", "0") in ("1", "true", "yes"),
            # Telegram Notifications
            telegram_notify_chat_ids=notify_chat_ids,
            # Database
            database_url=os.getenv("DATABASE_URL", "postgresql+asyncpg://agent:agent@localhost:5432/openclaw"),
            # Budget caps (overridable via .env)
            budget_sec_api=_safe_int(os.getenv("BUDGET_SEC_API", "500"), 500, "BUDGET_SEC_API"),
            budget_unusual_whales=_safe_int(os.getenv("BUDGET_UNUSUAL_WHALES", "16000"), 16000, "BUDGET_UNUSUAL_WHALES"),
            budget_sam_gov=_safe_int(os.getenv("BUDGET_SAM_GOV", "500"), 500, "BUDGET_SAM_GOV"),
            budget_apify=_safe_int(os.getenv("BUDGET_APIFY", "1000"), 1000, "BUDGET_APIFY"),
            budget_whisper=_safe_int(os.getenv("BUDGET_WHISPER", "20"), 20, "BUDGET_WHISPER"),
            budget_email_send=_safe_int(os.getenv("BUDGET_EMAIL_SEND", "50"), 50, "BUDGET_EMAIL_SEND"),
            budget_email_read=_safe_int(os.getenv("BUDGET_EMAIL_READ", "200"), 200, "BUDGET_EMAIL_READ"),
            budget_web_search=_safe_int(os.getenv("BUDGET_WEB_SEARCH", "1000"), 1000, "BUDGET_WEB_SEARCH"),
            budget_web_fetch=_safe_int(os.getenv("BUDGET_WEB_FETCH", "500"), 500, "BUDGET_WEB_FETCH"),
            budget_learning_analysis=_safe_int(os.getenv("BUDGET_LEARNING_ANALYSIS", "120"), 120, "BUDGET_LEARNING_ANALYSIS"),
            budget_learning_synthesis=_safe_int(os.getenv("BUDGET_LEARNING_SYNTHESIS", "12"), 12, "BUDGET_LEARNING_SYNTHESIS"),
            budget_polygon_learning=_safe_int(os.getenv("BUDGET_POLYGON_LEARNING", "1000"), 1000, "BUDGET_POLYGON_LEARNING"),
            budget_book_analysis=_safe_int(os.getenv("BUDGET_BOOK_ANALYSIS", "16"), 16, "BUDGET_BOOK_ANALYSIS"),
            budget_book_synthesis=_safe_int(os.getenv("BUDGET_BOOK_SYNTHESIS", "2"), 2, "BUDGET_BOOK_SYNTHESIS"),
            budget_learning_query_gen=_safe_int(os.getenv("BUDGET_LEARNING_QUERY_GEN", "100"), 100, "BUDGET_LEARNING_QUERY_GEN"),
            budget_elevenlabs_tts=_safe_int(os.getenv("BUDGET_ELEVENLABS_TTS", "100"), 100, "BUDGET_ELEVENLABS_TTS"),
            budget_premarket_scan=_safe_int(os.getenv("BUDGET_PREMARKET_SCAN", "5"), 5, "BUDGET_PREMARKET_SCAN"),
            budget_hourly_debrief=_safe_int(os.getenv("BUDGET_HOURLY_DEBRIEF", "10"), 10, "BUDGET_HOURLY_DEBRIEF"),
            budget_postmarket_debrief=_safe_int(os.getenv("BUDGET_POSTMARKET_DEBRIEF", "5"), 5, "BUDGET_POSTMARKET_DEBRIEF"),
            budget_news_update=_safe_int(os.getenv("BUDGET_NEWS_UPDATE", "30"), 30, "BUDGET_NEWS_UPDATE"),
            budget_report_web_search=_safe_int(os.getenv("BUDGET_REPORT_WEB_SEARCH", "400"), 400, "BUDGET_REPORT_WEB_SEARCH"),
            budget_news_intel=_safe_int(os.getenv("BUDGET_NEWS_INTEL", "50"), 50, "BUDGET_NEWS_INTEL"),
            budget_macro_research=_safe_int(os.getenv("BUDGET_MACRO_RESEARCH", "10"), 10, "BUDGET_MACRO_RESEARCH"),
            budget_geopolitical_research=_safe_int(os.getenv("BUDGET_GEOPOLITICAL_RESEARCH", "3"), 3, "BUDGET_GEOPOLITICAL_RESEARCH"),
            budget_fed_analysis=_safe_int(os.getenv("BUDGET_FED_ANALYSIS", "3"), 3, "BUDGET_FED_ANALYSIS"),
            budget_earnings_intel=_safe_int(os.getenv("BUDGET_EARNINGS_INTEL", "3"), 3, "BUDGET_EARNINGS_INTEL"),
            budget_trade_policy=_safe_int(os.getenv("BUDGET_TRADE_POLICY", "3"), 3, "BUDGET_TRADE_POLICY"),
            budget_commodity_research=_safe_int(os.getenv("BUDGET_COMMODITY_RESEARCH", "3"), 3, "BUDGET_COMMODITY_RESEARCH"),
            budget_bond_intel=_safe_int(os.getenv("BUDGET_BOND_INTEL", "3"), 3, "BUDGET_BOND_INTEL"),
            budget_seal_enrichment=_safe_int(os.getenv("BUDGET_SEAL_ENRICHMENT", "200"), 200, "BUDGET_SEAL_ENRICHMENT"),
            budget_seal_retention=_safe_int(os.getenv("BUDGET_SEAL_RETENTION", "150"), 150, "BUDGET_SEAL_RETENTION"),
            budget_adversarial_resolver=_safe_int(os.getenv("BUDGET_ADVERSARIAL_RESOLVER", "30"), 30, "BUDGET_ADVERSARIAL_RESOLVER"),
            budget_financial_datasets=_safe_int(os.getenv("BUDGET_FINANCIAL_DATASETS", "500"), 500, "BUDGET_FINANCIAL_DATASETS"),
            budget_social_enrich=_safe_int(os.getenv("BUDGET_SOCIAL_ENRICH", "120"), 120, "BUDGET_SOCIAL_ENRICH"),
            budget_fred_api=_safe_int(os.getenv("BUDGET_FRED_API", "60"), 60, "BUDGET_FRED_API"),
            budget_econ_surprise=_safe_int(os.getenv("BUDGET_ECON_SURPRISE", "1500"), 1500, "BUDGET_ECON_SURPRISE"),
            budget_newsflash_polygon=_safe_int(os.getenv("BUDGET_NEWSFLASH_POLYGON", "800"), 800, "BUDGET_NEWSFLASH_POLYGON"),
            budget_dpof_reflection=_safe_int(os.getenv("BUDGET_DPOF_REFLECTION", "50"), 50, "BUDGET_DPOF_REFLECTION"),
            budget_dpof_debate=_safe_int(os.getenv("BUDGET_DPOF_DEBATE", "200"), 200, "BUDGET_DPOF_DEBATE"),
        )

        # Validate OVERRIDE_VOICE_PIN: digits only, length within the accepted range.
        # Was hardcoded to exactly 4 digits, which silently BLANKED (→ disabled voice
        # auth) any rotation to a longer PIN. Range must stay in sync with
        # PIN_MIN_DIGITS/PIN_MAX_DIGITS in agent/override_directive.py.
        _PIN_MIN, _PIN_MAX = 4, 12
        if instance.override_voice_pin and not re.match(
            rf'^\d{{{_PIN_MIN},{_PIN_MAX}}}$', instance.override_voice_pin
        ):
            print(
                f"[config] WARNING: OVERRIDE_VOICE_PIN must be {_PIN_MIN}-{_PIN_MAX} digits, "
                f"got [REDACTED, len={len(instance.override_voice_pin)}] — disabling voice PIN auth",
                file=sys.stderr,
            )
            instance.override_voice_pin = ""

        return instance


config = Config.from_env()
if config.anthropic_gateway:
    _gw_host = urlparse(config.anthropic_base_url).hostname if config.anthropic_base_url else None
    print(f"[config] Anthropic route: gateway host={_gw_host or 'REJECTED'}", file=sys.stderr)
else:
    print("[config] Anthropic route: direct", file=sys.stderr)
