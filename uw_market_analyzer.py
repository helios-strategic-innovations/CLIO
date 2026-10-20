#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ATLAS STRATEGIES — UNUSUAL WHALES MARKET MOVEMENT ANALYZER
  Signal Development Engine v1.0
═══════════════════════════════════════════════════════════════════════════════

  SIGNAL PILLARS:
    1. NOFA (Net Options Flow Activity) — directional institutional bets
    2. Net Premium Flow — call vs put premium imbalance
    3. Dark Pool Accumulation/Distribution — block trade patterns
    4. Gamma Exposure (GEX) — dealer hedging pressure
    5. Congressional/Insider Alignment — smart money confirmation

  USAGE (on VPS):
    export UNUSUAL_WHALES_API_KEY="your_key_here"
    cd ~/tradingbot && python3 uw_market_analyzer.py

  OUTPUT:
    - Console report with ranked signals
    - JSON:     ~/tradingbot/data/uw_signals_YYYY-MM-DD.json
    - Markdown: ~/tradingbot/data/uw_report_YYYY-MM-DD.md
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, json, time, logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests"); sys.exit(1)

try:
    from dotenv import load_dotenv
    for p in [Path.home() / "tradingbot" / ".env", Path(".env")]:
        if p.exists(): load_dotenv(p); break
except ImportError:
    pass

# ─── CONFIG ─────────────────────────────────────────────────────────────────

API_BASE = "https://api.unusualwhales.com/api"
API_KEY = os.environ.get("UNUSUAL_WHALES_API_KEY", "")

if not API_KEY:
    print("ERROR: UNUSUAL_WHALES_API_KEY not set. Export it or add to .env")
    sys.exit(1)

# Per UW skill.md: both headers required
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "UW-CLIENT-API-ID": "100001",
    "Accept": "application/json, text/plain",
}

REQUEST_DELAY = 0.4
TIMEOUT = 15
OUTPUT_DIR = Path.home() / "tradingbot" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TODAY = datetime.now().strftime("%Y-%m-%d")

# Focus universe — Hydrinia-aligned
TIER1 = [
    "AVAV","LMT","RTX","NOC","GD","PLTR","HII","LHX",
    "NVDA","AMD","TSM","AVGO","MRVL","QCOM","MU",
    "AAPL","MSFT","GOOGL","AMZN","META","TSLA",
    "XOM","CVX","COP","HAL","SLB","OXY",
    "NEM","GOLD","AEM",
    "MRNA","LLY","UNH","REGN","VRTX",
    "ZIM","DAC","GOGL","RKLB",
    "BABA","PDD","JD",
]
ETFS = [
    "SPY","QQQ","IWM","DIA","XLE","XLF","XLK","XBI",
    "DFEN","ITA","XAR","GLD","GDX","SLV","USO","UNG",
    "TLT","HYG","FXI","KWEB",
    "TQQQ","SQQQ","SOXL","SOXS","UVIX","SVIX","SPXL",
    "LABU","LABD","BOIL","KOLD",
]
ALL_TICKERS = TIER1 + ETFS

WEIGHTS = {
    "nofa": 0.35, "premium": 0.25, "darkpool": 0.20,
    "gex": 0.10, "congress": 0.10,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("UW")

# ─── API LAYER ──────────────────────────────────────────────────────────────

req_count = 0
usage = {"limit": None, "used": None, "remaining": None}

def uw_get(endpoint, params=None, _retry_depth=0):
    global req_count, usage
    url = f"{API_BASE}{endpoint}"
    time.sleep(REQUEST_DELAY)
    req_count += 1
    try:
        r = requests.get(url, headers=HEADERS, params=params or {}, timeout=TIMEOUT)
        dl = r.headers.get("x-uw-token-req-limit")
        du = r.headers.get("x-uw-daily-req-count")
        mr = r.headers.get("x-uw-req-per-minute-remaining")
        if dl: usage["limit"] = int(dl)
        if du:
            usage["used"] = int(du)
            if usage["limit"]: usage["remaining"] = usage["limit"] - int(du)
        if req_count % 20 == 0:
            log.info(f"API: {usage.get('used','?')}/{usage.get('limit','?')} daily | min-remaining: {mr or '?'}")
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 429:
            # S118 W-068: Cap retry depth to prevent unbounded recursion
            if _retry_depth >= 3:
                log.error(f"Rate limited 3x on {endpoint} — giving up"); return {}
            wait = max(int(r.headers.get("x-uw-req-per-minute-reset", "60000")) / 1000, 5)
            log.warning(f"Rate limited. Waiting {wait:.0f}s..."); time.sleep(wait)
            return uw_get(endpoint, params, _retry_depth=_retry_depth + 1)
        elif r.status_code == 401:
            log.error("AUTH FAILED (401). Check API key."); return {}
        elif r.status_code == 403:
            log.warning(f"403 on {endpoint}. May need higher tier."); return {}
        else:
            log.warning(f"HTTP {r.status_code} on {endpoint}: {r.text[:200]}"); return {}
    except Exception as e:
        log.error(f"Error on {endpoint}: {e}"); return {}


# ═════════════════════════════════════════════════════════════════════════════
#  PILLAR 1: NOFA — Net Options Flow Activity
# ═════════════════════════════════════════════════════════════════════════════

def fetch_flow_alerts():
    log.info("═══ PILLAR 1: Flow Alerts (NOFA) ═══")
    alerts = []

    # High-premium OTM openers (institutional directional bets)
    d = uw_get("/option-trades/flow-alerts", {
        "limit": 200, "min_premium": 50000,
        "size_greater_oi": True, "is_otm": True,
    })
    a = d.get("data", []); log.info(f"  High-prem OTM openers: {len(a)}"); alerts.extend(a)

    # Calls $25K+
    d = uw_get("/option-trades/flow-alerts", {
        "limit": 100, "min_premium": 25000,
        "is_call": True, "size_greater_oi": True,
    })
    a = d.get("data", []); log.info(f"  Call openers $25K+: {len(a)}"); alerts.extend(a)

    # Puts $25K+
    d = uw_get("/option-trades/flow-alerts", {
        "limit": 100, "min_premium": 25000,
        "is_put": True, "size_greater_oi": True,
    })
    a = d.get("data", []); log.info(f"  Put openers $25K+: {len(a)}"); alerts.extend(a)

    return alerts


def compute_nofa(alerts):
    """NOFA = (Call Premium - Put Premium) / Total Premium per ticker. Range [-1, +1]."""
    td = defaultdict(lambda: {
        "call_prem": 0, "put_prem": 0, "total_prem": 0,
        "call_n": 0, "put_n": 0, "total_n": 0,
        "call_vol": 0, "put_vol": 0,
        "openers": 0, "otm_n": 0, "max_trade": 0, "trades": [],
    })
    seen = set()
    for a in alerts:
        tk = a.get("ticker_symbol") or a.get("ticker", "")
        if not tk: continue
        key = f"{tk}_{a.get('option_symbol','')}_{a.get('total_premium','')}"
        if key in seen: continue
        seen.add(key)

        prem = float(a.get("total_premium", 0) or 0)
        vol = int(a.get("total_size", 0) or 0)
        is_call = str(a.get("type", "")).upper() == "CALL" or a.get("is_call")
        is_put = str(a.get("type", "")).upper() == "PUT" or a.get("is_put")

        t = td[tk]
        t["total_prem"] += prem; t["total_n"] += 1
        if prem > t["max_trade"]: t["max_trade"] = prem
        if is_call: t["call_prem"] += prem; t["call_n"] += 1; t["call_vol"] += vol
        elif is_put: t["put_prem"] += prem; t["put_n"] += 1; t["put_vol"] += vol
        if a.get("size_greater_oi"): t["openers"] += 1
        if a.get("is_otm"): t["otm_n"] += 1
        t["trades"].append({
            "type": "CALL" if is_call else "PUT", "premium": prem,
            "volume": vol, "strike": a.get("strike"), "expiry": a.get("expires"),
        })

    nofa = {}
    for tk, t in td.items():
        raw = (t["call_prem"] - t["put_prem"]) / max(t["total_prem"], 1)
        nofa[tk] = {
            "nofa": round(raw, 4),
            "dir": "BULLISH" if raw > 0.15 else ("BEARISH" if raw < -0.15 else "NEUTRAL"),
            "total_prem": t["total_prem"], "call_prem": t["call_prem"], "put_prem": t["put_prem"],
            "call_n": t["call_n"], "put_n": t["put_n"], "total_n": t["total_n"],
            "cp_ratio": round(t["call_n"] / max(t["put_n"], 1), 2),
            "openers_pct": round(t["openers"] / max(t["total_n"], 1), 2),
            "otm_pct": round(t["otm_n"] / max(t["total_n"], 1), 2),
            "max_trade": t["max_trade"],
            "top_trades": sorted(t["trades"], key=lambda x: x["premium"], reverse=True)[:5],
        }
    return nofa


# ═════════════════════════════════════════════════════════════════════════════
#  PILLAR 2: NET PREMIUM FLOW
# ═════════════════════════════════════════════════════════════════════════════

def fetch_market_tide():
    log.info("═══ PILLAR 2: Market Tide (Net Premium) ═══")
    full = uw_get("/market/market-tide", {"interval_5m": False}).get("data", [])
    log.info(f"  Full-day records: {len(full)}")
    intra = uw_get("/market/market-tide", {"interval_5m": True}).get("data", [])
    log.info(f"  5min records: {len(intra)}")
    return {"full_day": full, "intraday": intra}


def fetch_ticker_premiums(tickers):
    log.info(f"  Fetching net premium for {len(tickers)} tickers...")
    out = {}
    for tk in tickers:
        d = uw_get(f"/stock/{tk}/net-prem-ticks").get("data", [])
        if d: out[tk] = d
    log.info(f"  Got premium data for {len(out)} tickers")
    return out


def compute_premium_signals(tide, tk_prems):
    sig = {"market": {}, "tickers": {}}
    full = tide.get("full_day", [])
    if full:
        latest = full[-1] if isinstance(full, list) else full
        nc = float(latest.get("net_call_premium", 0) or 0)
        np_ = float(latest.get("net_put_premium", 0) or 0)
        sig["market"] = {
            "net_call": nc, "net_put": np_, "net_flow": nc + np_,
            "sentiment": "BULLISH" if nc > abs(np_) else "BEARISH",
            "ratio": round(nc / max(abs(np_), 1), 2),
        }
        log.info(f"  Tide: Call ${nc:,.0f} | Put ${np_:,.0f} | {sig['market']['sentiment']}")
    intra = tide.get("intraday", [])
    if len(intra) >= 6:
        rc = sum(float(r.get("net_call_premium", 0) or 0) for r in intra[-6:])
        rp = sum(float(r.get("net_put_premium", 0) or 0) for r in intra[-6:])
        sig["market"]["last_30m"] = "BULLISH" if rc > abs(rp) else "BEARISH"
        sig["market"]["last_30m_net"] = round(rc + rp, 2)
    for tk, ticks in tk_prems.items():
        if not ticks: continue
        net = sum(float(t.get("net_premium", 0) or 0) for t in ticks[-20:])
        cp = sum(float(t.get("call_premium", 0) or 0) for t in ticks[-20:])
        pp = sum(float(t.get("put_premium", 0) or 0) for t in ticks[-20:])
        sig["tickers"][tk] = {
            "net": net, "call": cp, "put": pp,
            "dir": "BULLISH" if net > 0 else "BEARISH",
            "score": min(max(net / max(abs(cp) + abs(pp), 1), -1), 1),
        }
    return sig


# ═════════════════════════════════════════════════════════════════════════════
#  PILLAR 3: DARK POOL
# ═════════════════════════════════════════════════════════════════════════════

def fetch_darkpool():
    log.info("═══ PILLAR 3: Dark Pool ═══")
    mkt = uw_get("/darkpool/recent").get("data", [])
    log.info(f"  Market-wide prints: {len(mkt)}")
    tk_dp = {}
    focus = TIER1[:15]
    log.info(f"  Fetching DP for {len(focus)} tickers...")
    for tk in focus:
        d = uw_get(f"/darkpool/{tk}").get("data", [])
        if d: tk_dp[tk] = d
    log.info(f"  Got DP for {len(tk_dp)} tickers")
    return mkt, tk_dp


def compute_dp_signals(mkt_prints, tk_prints):
    sig = {"market_top": {}, "tickers": {}}

    # Market aggregation
    by_tk = defaultdict(lambda: {"notional": 0, "shares": 0, "n": 0, "prices": []})
    for p in mkt_prints:
        tk = p.get("ticker", "")
        px = float(p.get("price", 0) or 0)
        sz = int(p.get("size", 0) or 0)
        if tk and px > 0 and sz > 0:
            b = by_tk[tk]; b["notional"] += px*sz; b["shares"] += sz; b["n"] += 1; b["prices"].append(px)

    top = sorted(by_tk.items(), key=lambda x: x[1]["notional"], reverse=True)[:30]
    sig["market_top"] = {
        tk: {"notional": round(d["notional"],2), "shares": d["shares"], "prints": d["n"],
             "avg_size": round(d["shares"]/max(d["n"],1)),
             "vwap": round(d["notional"]/max(d["shares"],1),2)}
        for tk, d in top
    }

    # Per-ticker accumulation/distribution
    for tk, prints in tk_prints.items():
        if not prints: continue
        notional = 0; shares = 0; prices = []
        for p in prints:
            px = float(p.get("price",0) or 0); sz = int(p.get("size",0) or 0)
            if px > 0 and sz > 0:
                notional += px*sz; shares += sz; prices.append(px)
        if shares == 0: continue
        vwap = notional / shares
        if prices:
            med = sorted(prices)[len(prices)//2]
            bias = "ACCUMULATION" if vwap >= med else "DISTRIBUTION"
            bs = (vwap - med) / max(med, 0.01)
        else:
            bias = "NEUTRAL"; bs = 0
        sig["tickers"][tk] = {
            "notional": round(notional,2), "shares": shares, "prints": len(prints),
            "avg_size": round(shares/len(prints)), "vwap": round(vwap,2),
            "bias": bias, "bias_score": round(min(max(bs,-1),1),4),
        }
    return sig


# ═════════════════════════════════════════════════════════════════════════════
#  PILLAR 4: GAMMA EXPOSURE (GEX)
# ═════════════════════════════════════════════════════════════════════════════

GEX_TICKERS = ["SPY","QQQ","IWM","NVDA","TSLA","AAPL","MSFT","AMD","AMZN","META","GOOGL","AVAV","PLTR","XLE","GLD","TLT"]

def fetch_gex():
    log.info("═══ PILLAR 4: GEX ═══")
    out = {}
    for tk in GEX_TICKERS:
        d = uw_get(f"/stock/{tk}/spot-exposures/strike").get("data", [])
        if d: out[tk] = d
    log.info(f"  GEX data for {len(out)} tickers")
    return out


def compute_gex(gex_data):
    sig = {}
    for tk, strikes in gex_data.items():
        if not strikes: continue
        tc = 0; tp = 0; mx_strike = None; mx_val = 0
        for s in strikes:
            st = float(s.get("strike",0) or 0)
            cg = float(s.get("call_gamma_oi",0) or s.get("call_gex",0) or 0)
            pg = float(s.get("put_gamma_oi",0) or s.get("put_gex",0) or 0)
            tc += cg; tp += pg
            net = cg + pg
            if abs(net) > abs(mx_val): mx_val = net; mx_strike = st
        ng = tc + tp
        sig[tk] = {
            "net_gex": round(ng,2), "call_gex": round(tc,2), "put_gex": round(tp,2),
            "regime": "STABILIZING" if ng > 0 else "ACCELERATING",
            "pin_strike": mx_strike, "pin_val": round(mx_val,2),
        }
    return sig


# ═════════════════════════════════════════════════════════════════════════════
#  PILLAR 5: CONGRESSIONAL / INSIDER
# ═════════════════════════════════════════════════════════════════════════════

def fetch_congress_insider():
    log.info("═══ PILLAR 5: Congress + Insider ═══")
    cong = uw_get("/congress/recent-trades").get("data", [])
    log.info(f"  Congress trades: {len(cong)}")
    ins = uw_get("/insider/transactions").get("data", [])
    log.info(f"  Insider txns: {len(ins)}")
    return cong, ins


def compute_congress(cong, ins):
    ta = defaultdict(lambda: {"cb":0,"cs":0,"ib":0,"is_":0,"pols":[]})
    for t in cong:
        tk = t.get("ticker",""); tp = str(t.get("type","")).lower()
        if not tk: continue
        a = ta[tk]
        if "purchase" in tp or "buy" in tp: a["cb"] += 1
        elif "sale" in tp or "sell" in tp: a["cs"] += 1
        a["pols"].append(f"{t.get('politician','?')}: {tp} ({t.get('amount','')})")
    for t in ins:
        tk = t.get("ticker",""); tp = str(t.get("transaction_type","") or t.get("type","")).lower()
        if not tk: continue
        a = ta[tk]
        if "purchase" in tp or "buy" in tp: a["ib"] += 1
        elif "sale" in tp or "sell" in tp: a["is_"] += 1

    sig = {}
    for tk, a in ta.items():
        tot = a["cb"]+a["cs"]+a["ib"]+a["is_"]
        if tot == 0: continue
        sc = (a["cb"]*2 - a["cs"]*2 + a["ib"] - a["is_"]) / max(tot, 1)
        sig[tk] = {
            "score": round(min(max(sc,-1),1),4),
            "dir": "BULLISH" if sc > 0.1 else ("BEARISH" if sc < -0.1 else "NEUTRAL"),
            "cb": a["cb"], "cs": a["cs"], "ib": a["ib"], "is_": a["is_"],
            "pols": a["pols"][:10],
        }
    return sig


# ═════════════════════════════════════════════════════════════════════════════
#  BONUS: HOTTEST CHAINS (OPTIONS SCREENER)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_hottest():
    log.info("═══ BONUS: Hottest Chains ═══")
    calls = uw_get("/screener/option-contracts", {
        "limit": 100, "is_otm": True, "issue_types[]": ["Common Stock","ADR"],
        "max_dte": 90, "min_premium": 100000, "type": "Calls",
        "vol_greater_oi": True, "max_multileg_volume_ratio": 0.1,
    }).get("data", [])
    log.info(f"  Bullish chains: {len(calls)}")
    puts = uw_get("/screener/option-contracts", {
        "limit": 100, "is_otm": True, "issue_types[]": ["Common Stock","ADR"],
        "max_dte": 90, "min_premium": 100000, "type": "Puts",
        "vol_greater_oi": True, "max_multileg_volume_ratio": 0.1,
    }).get("data", [])
    log.info(f"  Bearish chains: {len(puts)}")
    return calls + puts


# ═════════════════════════════════════════════════════════════════════════════
#  COMPOSITE SCORING
# ═════════════════════════════════════════════════════════════════════════════

def composite_score(nofa, prem, dp, gex, cong):
    log.info("═══ Computing Composite Signals ═══")
    all_tk = set()
    all_tk.update(nofa.keys())
    all_tk.update(prem.get("tickers",{}).keys())
    all_tk.update(dp.get("tickers",{}).keys())
    all_tk.update(gex.keys())
    all_tk.update(cong.keys())

    out = []
    for tk in all_tk:
        n = nofa.get(tk, {}); nr = n.get("nofa", 0)
        p = prem.get("tickers",{}).get(tk, {}); pr = p.get("score", 0)
        d = dp.get("tickers",{}).get(tk, {}); dr = d.get("bias_score", 0)
        g = gex.get(tk, {})
        gr = 0.5 if g.get("regime") == "ACCELERATING" and nr > 0 else (
            -0.5 if g.get("regime") == "ACCELERATING" and nr < 0 else 0)
        c = cong.get(tk, {}); cr = c.get("score", 0)

        comp = (WEIGHTS["nofa"]*nr + WEIGHTS["premium"]*pr + WEIGHTS["darkpool"]*dr +
                WEIGHTS["gex"]*gr + WEIGHTS["congress"]*cr)
        conf = sum([1 if x else 0 for x in [n, p, d, g, c]])
        dirn = ("STRONG BUY" if comp > 0.4 else "BUY" if comp > 0.15 else
                "LEAN BULL" if comp > 0.05 else "NEUTRAL" if comp > -0.05 else
                "LEAN BEAR" if comp > -0.15 else "SELL" if comp > -0.4 else "STRONG SELL")

        out.append({
            "ticker": tk, "score": round(comp, 4), "dir": dirn, "conf": conf,
            "nofa": round(nr,4), "prem": round(pr,4), "dp": round(dr,4),
            "gex": round(gr,4), "cong": round(cr,4),
            "flow_prem": n.get("total_prem",0), "dp_notional": d.get("notional",0),
            "gex_regime": g.get("regime","N/A"), "gex_pin": g.get("pin_strike"),
            "_n": n, "_p": p, "_d": d, "_g": g, "_c": c,
        })

    out.sort(key=lambda x: abs(x["score"]), reverse=True)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  REPORT
# ═════════════════════════════════════════════════════════════════════════════

def gen_report(sigs, mkt_tide, mkt_dp, hottest):
    now = datetime.now().strftime("%Y-%m-%d %H:%M ET")
    m = mkt_tide.get("market", {})
    L = [
        f"# Atlas Strategies — UW Signal Report",
        f"### {now}\n---\n",
        f"## Market Sentiment (Market Tide)\n",
        f"| Metric | Value |", f"|--------|-------|",
        f"| Net Call Premium | ${m.get('net_call',0):,.0f} |",
        f"| Net Put Premium | ${m.get('net_put',0):,.0f} |",
        f"| Net Flow | ${m.get('net_flow',0):,.0f} |",
        f"| Sentiment | **{m.get('sentiment','N/A')}** |",
        f"| C/P Ratio | {m.get('ratio','N/A')} |",
        f"| Last 30m | {m.get('last_30m','N/A')} |",
        f"\n---\n",
        f"## Top Signals (Composite)\n",
        f"| # | Ticker | Score | Direction | NOFA | Prem | DP | GEX | Cong | Flow$ | Conf |",
        f"|---|--------|-------|-----------|------|------|----|-----|------|-------|------|",
    ]
    for i, s in enumerate(sigs[:40], 1):
        fp = f"${s['flow_prem']:,.0f}" if s['flow_prem'] > 0 else "-"
        L.append(f"| {i} | **{s['ticker']}** | {s['score']:+.4f} | {s['dir']} | "
                 f"{s['nofa']:+.3f} | {s['prem']:+.3f} | {s['dp']:+.3f} | "
                 f"{s['gex']:+.2f} | {s['cong']:+.3f} | {fp} | {s['conf']}/5 |")

    bulls = [s for s in sigs if s['score'] > 0.1]
    bears = [s for s in sigs if s['score'] < -0.1]

    L.append(f"\n---\n## Bullish ({len(bulls)})\n")
    for s in bulls[:15]:
        n = s["_n"]
        tt = n.get("top_trades", [])
        L.append(f"### {s['ticker']} — {s['dir']} ({s['score']:+.4f})")
        L.append(f"- NOFA: {s['nofa']:+.4f} | C/P: {n.get('cp_ratio','N/A')} | "
                 f"Flow: ${n.get('total_prem',0):,.0f}")
        L.append(f"- DP: {s['_d'].get('bias','N/A')} (${s['_d'].get('notional',0):,.0f})")
        L.append(f"- GEX: {s['gex_regime']}" + (f" Pin: ${s['gex_pin']}" if s['gex_pin'] else ""))
        if tt:
            L.append(f"- Top: {tt[0]['type']} ${tt[0].get('strike','?')} "
                     f"exp {tt[0].get('expiry','?')} — ${tt[0]['premium']:,.0f}")
        L.append("")

    L.append(f"\n---\n## Bearish ({len(bears)})\n")
    for s in bears[:15]:
        n = s["_n"]
        L.append(f"### {s['ticker']} — {s['dir']} ({s['score']:+.4f})")
        L.append(f"- NOFA: {s['nofa']:+.4f} | Flow: ${n.get('total_prem',0):,.0f}")
        L.append(f"- DP: {s['_d'].get('bias','N/A')}"); L.append("")

    L.append(f"\n---\n## Top Dark Pool (Market)\n")
    L.append(f"| Ticker | Notional | Shares | Prints | AvgSz | VWAP |")
    L.append(f"|--------|----------|--------|--------|-------|------|")
    for tk, d in list(mkt_dp.get("market_top",{}).items())[:20]:
        L.append(f"| {tk} | ${d['notional']:,.0f} | {d['shares']:,} | "
                 f"{d['prints']} | {d['avg_size']:,} | ${d['vwap']:.2f} |")

    L.append(f"\n---\n## Hottest Chains\n")
    for h in hottest[:20]:
        tk = h.get("ticker_symbol", h.get("ticker", "?"))
        tp = h.get("type", "?")
        pr = float(h.get("total_premium", 0) or h.get("ask_side_volume", 0) or 0)
        vol = h.get("volume", h.get("total_size", "?"))
        L.append(f"- **{tk}** {tp} — Vol: {vol} — Prem: ${pr:,.0f}")

    L.append(f"\n---\n*UW Signal Engine v1.0 | API calls: {req_count} | "
             f"Usage: {usage.get('used','?')}/{usage.get('limit','?')}*")
    return "\n".join(L)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("  ATLAS STRATEGIES — UW MARKET MOVEMENT ANALYZER")
    log.info("=" * 70)

    # Pillar 1
    alerts = fetch_flow_alerts()
    nofa = compute_nofa(alerts)
    log.info(f"  NOFA: {len(nofa)} tickers")

    # Pillar 2
    tide = fetch_market_tide()
    tk_prems = fetch_ticker_premiums(TIER1[:20])
    prem_sig = compute_premium_signals(tide, tk_prems)

    # Pillar 3
    mkt_dp, tk_dp = fetch_darkpool()
    dp_sig = compute_dp_signals(mkt_dp, tk_dp)

    # Pillar 4
    gex_data = fetch_gex()
    gex_sig = compute_gex(gex_data)

    # Pillar 5
    cong, ins = fetch_congress_insider()
    cong_sig = compute_congress(cong, ins)

    # Bonus
    hottest = fetch_hottest()

    # Composite
    sigs = composite_score(nofa, prem_sig, dp_sig, gex_sig, cong_sig)

    # Report
    report = gen_report(sigs, prem_sig, dp_sig, hottest)

    # Save JSON
    jp = OUTPUT_DIR / f"uw_signals_{TODAY}.json"
    with open(jp, "w") as f:
        json.dump({
            "date": TODAY, "generated": datetime.now().isoformat(),
            "api_calls": req_count, "usage": usage,
            "market": prem_sig.get("market", {}),
            "signals": [{k:v for k,v in s.items() if not k.startswith("_")} for s in sigs[:50]],
            "nofa": {k:{kk:vv for kk,vv in v.items() if kk != "top_trades"} for k,v in nofa.items()},
            "dp_top": dp_sig.get("market_top", {}),
            "gex": gex_sig, "congress": cong_sig,
        }, f, indent=2, default=str)
    log.info(f"  JSON: {jp}")

    # Save MD
    mp = OUTPUT_DIR / f"uw_report_{TODAY}.md"
    with open(mp, "w") as f: f.write(report)
    log.info(f"  Report: {mp}")

    # Console
    elapsed = time.time() - t0
    m = prem_sig.get("market", {})
    print("\n" + "=" * 70)
    print("  SIGNAL SUMMARY")
    print("=" * 70)
    print(f"\n  Tide: {m.get('sentiment','N/A')} | Net: ${m.get('net_flow',0):,.0f} | "
          f"30m: {m.get('last_30m','N/A')}")
    print(f"\n  {'TK':<8} {'SCORE':>8} {'DIR':<12} {'NOFA':>7} {'PREM':>7} {'DP':>7} {'FLOW$':>12}")
    print(f"  {'-'*8} {'-'*8} {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*12}")
    for s in sigs[:25]:
        fp = f"${s['flow_prem']:,.0f}" if s['flow_prem'] > 0 else "-"
        print(f"  {s['ticker']:<8} {s['score']:>+8.4f} {s['dir']:<12} "
              f"{s['nofa']:>+7.3f} {s['prem']:>+7.3f} {s['dp']:>+7.3f} {fp:>12}")
    print(f"\n  API: {req_count} calls | {usage.get('used','?')}/{usage.get('limit','?')} daily | {elapsed:.1f}s")
    print(f"  {jp}\n  {mp}")
    print("=" * 70)

if __name__ == "__main__":
    main()
