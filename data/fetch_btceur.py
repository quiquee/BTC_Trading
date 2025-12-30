#!/usr/bin/env python3
"""Download BTC/EUR candles, fetch BTC news sentiment, and store a merged CSV."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import feedparser
except ImportError as exc:  # pragma: no cover
    raise SystemExit("feedparser is required. Install it via 'pip install feedparser'.") from exc

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise SystemExit("yfinance is required. Install it with 'pip install yfinance'.") from exc

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None


SYMBOL = "BTC-EUR"
DEFAULT_INTERVAL = "1h"
DEFAULT_PERIOD = "1y"
DEFAULT_OUTPUT = Path(__file__).with_name("BTC_EUR_latest.csv")
DEFAULT_NEWS_OUTPUT = Path(__file__).with_name("btc_news_sentiment.csv")
DEFAULT_MODEL = "gemini-2.5-pro"
DEFAULT_SOURCES = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/?output=xml",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "BitcoinMagazine": "https://bitcoinmagazine.com/.rss/full/",
    "CryptoSlate": "https://cryptoslate.com/feed/",
    "Decrypt": "https://decrypt.co/feed",
}
MAX_ARTICLES_PER_SOURCE = 400
SLEEP_BETWEEN_CALLS = 1.0
PROMPT = (
    "You are a financial sentiment analyst specializing in Bitcoin. Given the news metadata "
    "below, classify the short-term BTC price sentiment as POSITIVE, NEGATIVE, or NEUTRAL and return "
    "JSON with keys sentiment_label, sentiment_score (-1..1), and rationale (one sentence)."
)


@dataclass
class Article:
    source: str
    title: str
    summary: str
    url: str
    published: datetime

    def prompt_payload(self) -> str:
        timestamp = self.published.strftime("%Y-%m-%d %H:%M UTC")
        return (
            f"Title: {self.title}\nSource: {self.source}\nPublished: {timestamp}\nURL: {self.url}\n"
            f"Summary: {self.summary}"
        )


def article_cache_key(source: str, title: str, url: str) -> str:
    key = (url or "").strip()
    if key:
        return key
    return f"{source.strip()}|{title.strip()}"


def build_article_key(article: Article) -> str:
    return article_cache_key(article.source, article.title, article.url)


def format_timestamp(index) -> list[str]:
    if getattr(index, "tz", None) is None:
        index = index.tz_localize(timezone.utc)
    else:
        index = index.tz_convert(timezone.utc)
    return [ts.strftime("%d.%m.%Y %H:%M:%S.%f")[:-3] for ts in index]


def fetch_dataframe(period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        SYMBOL,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        actions=False,
    )
    if data.empty:
        raise RuntimeError("Yahoo Finance returned no data. Try a shorter period or different interval.")
    ordered = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    ordered.insert(0, "Gmt time", format_timestamp(ordered.index))
    return ordered


def parse_entry(entry, source: str) -> Optional[Article]:
    published_parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if published_parsed is None:
        return None
    published = datetime(*published_parsed[:6], tzinfo=timezone.utc)
    title = getattr(entry, "title", "").strip()
    summary = getattr(entry, "summary", getattr(entry, "description", "")).strip()
    link = getattr(entry, "link", "").strip()
    if not title or not link:
        return None
    cleaned_summary = summary.replace("\n", " ").replace("\r", " ").strip()
    return Article(source=source, title=title, summary=cleaned_summary, url=link, published=published)


def fetch_articles(source_keys: Iterable[str], start: datetime, end: datetime, limit: int) -> List[Article]:
    collected: List[Article] = []
    for key in source_keys:
        feed_url = DEFAULT_SOURCES.get(key)
        if not feed_url:
            print(f"[Skip] Unknown source '{key}'.", file=sys.stderr)
            continue
        print(f"Fetching {key} feed...")
        parsed = feedparser.parse(feed_url)
        if parsed.bozo:
            print(f"  [Warn] Feed parser flagged an issue for {key}: {parsed.bozo_exception}", file=sys.stderr)
        count = 0
        for entry in parsed.entries:
            article = parse_entry(entry, key)
            if article is None or not (start <= article.published <= end):
                continue
            collected.append(article)
            count += 1
            if count >= limit:
                break
        print(f"  Collected {count} articles from {key} within range.")
    collected.sort(key=lambda art: art.published)
    return collected


def configure_gemini(model_name: str):
    if genai is None:
        raise RuntimeError(
            "google-generativeai is not installed. Install it via 'pip install google-generativeai' "
            "or run with --skip-sentiment."
        )
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def analyze_sentiment(model, article: Article) -> dict:
    prompt = f"{PROMPT}\n\n{article.prompt_payload()}\nRespond with JSON only."
    response = model.generate_content(prompt)
    text = response.text.strip()
    if not text:
        raise RuntimeError("Empty response from Gemini.")
    if text.startswith("```"):
        text = text.strip("`\n")
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    payload = json.loads(text)
    payload["raw_response"] = response.text
    return payload


def annotate_articles(
    articles: List[Article],
    model_name: str,
    skip_sentiment: bool,
    cached_records: Dict[str, dict],
) -> None:
    if not articles:
        return

    model = None
    for idx, article in enumerate(articles, start=1):
        key = build_article_key(article)
        if key in cached_records:
            continue

        if skip_sentiment:
            cached_records[key] = asdict(article)
            continue

        if model is None:
            model = configure_gemini(model_name)

        print(f"Scoring sentiment {idx}/{len(articles)} | {article.source} | {article.title[:60]}...")
        retries = 3
        delay = SLEEP_BETWEEN_CALLS
        while retries:
            try:
                sentiment = analyze_sentiment(model, article)
                break
            except Exception as exc:
                retries -= 1
                if retries == 0:
                    sentiment = {
                        "sentiment_label": "ERROR",
                        "sentiment_score": 0.0,
                        "rationale": str(exc),
                    }
                    print(f"  [Error] Failed after retries: {exc}", file=sys.stderr)
                else:
                    print(f"  [Warn] Gemini call failed ({exc}); retrying in {delay:.1f}s...", file=sys.stderr)
                    time.sleep(delay)
                    delay *= 2
        record = asdict(article)
        record.update(sentiment)
        cached_records[key] = record
        time.sleep(SLEEP_BETWEEN_CALLS)


def export_news(records: List[dict], output_path: Path) -> None:
    if not records:
        print("No articles collected; skipping news CSV export.")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df["published"] = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df.sort_values("published", inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} annotated articles to {output_path}")


def load_existing_news(path: Optional[Path]) -> Dict[str, dict]:
    if not path or not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[Sentiment] Could not read existing news file {path}: {exc}")
        return {}
    cache: Dict[str, dict] = {}
    for record in df.to_dict(orient="records"):
        key = article_cache_key(record.get("source", ""), record.get("title", ""), record.get("url", ""))
        cache[key] = record
    return cache


def filter_records_by_range(records: Iterable[dict], start: datetime, end: datetime) -> List[dict]:
    filtered: List[dict] = []
    for record in records:
        ts = pd.to_datetime(record.get("published"), utc=True, errors="coerce")
        if pd.isna(ts) or not (start <= ts <= end):
            continue
        copy = dict(record)
        copy["published"] = ts
        filtered.append(copy)
    return filtered


def compute_daily_sentiment(records: List[dict]) -> Optional[pd.Series]:
    if not records:
        return None
    df = pd.DataFrame(records)
    if "sentiment_score" not in df.columns:
        return None
    published = pd.to_datetime(df["published"], utc=True, errors="coerce")
    df = df.assign(published=published).dropna(subset=["published"])
    if df.empty:
        return None
    df["published_date"] = df["published"].dt.normalize()
    daily_scores = df.groupby("published_date")["sentiment_score"].mean()
    return daily_scores if not daily_scores.empty else None


def attach_sentiment_column(df: pd.DataFrame, daily_scores: Optional[pd.Series]) -> None:
    if daily_scores is None:
        df["news_sentiment"] = 0.0
        return

    daily_series = daily_scores.copy()
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize(timezone.utc)
    else:
        idx = idx.tz_convert(timezone.utc)

    sentiment_index = daily_series.index
    if getattr(sentiment_index, "tz", None) is None:
        sentiment_index = sentiment_index.tz_localize(timezone.utc)
        daily_series.index = sentiment_index

    normalized_idx = idx.normalize()
    mapped = normalized_idx.map(daily_series)
    df["news_sentiment"] = pd.Series(mapped, index=df.index).ffill().fillna(0.0)


def resolve_news_range(
    price_index: pd.Index,
    start_text: Optional[str],
    end_text: Optional[str],
) -> tuple[datetime, datetime]:
    def parse_date(text: str) -> datetime:
        return datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if start_text:
        start_dt = parse_date(start_text)
    else:
        start_dt = None
    if end_text:
        end_dt = parse_date(end_text) + timedelta(hours=23, minutes=59, seconds=59, microseconds=999999)
    else:
        end_dt = None

    if price_index.size:
        idx = price_index
        if getattr(idx, "tz", None) is None:
            idx = idx.tz_localize(timezone.utc)
        else:
            idx = idx.tz_convert(timezone.utc)
        price_start = idx.min().to_pydatetime()
        price_end = idx.max().to_pydatetime()
    else:
        now = datetime.now(timezone.utc)
        price_end = now
        price_start = now - timedelta(days=365)

    start_dt = start_dt or price_start
    end_dt = end_dt or price_end

    if start_dt > end_dt:
        raise ValueError("News start date must be on or before end date.")
    start_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_dt = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    return start_dt, end_dt


def save_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download BTC/EUR candles from Yahoo Finance, fetch BTC news sentiment via Gemini, "
            "and emit a CSV ready for the trading pipeline."
        )
    )
    parser.add_argument("--period", default=DEFAULT_PERIOD, help=f"Price period (default: {DEFAULT_PERIOD}).")
    parser.add_argument("--interval", default=DEFAULT_INTERVAL, help=f"Price interval (default: {DEFAULT_INTERVAL}).")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination price CSV path (default: {DEFAULT_OUTPUT.name}).",
    )
    parser.add_argument(
        "--news-output",
        type=Path,
        default=DEFAULT_NEWS_OUTPUT,
        help=f"Optional CSV path for annotated news (default: {DEFAULT_NEWS_OUTPUT.name}).",
    )
    parser.add_argument(
        "--news-start-date",
        type=str,
        default=None,
        help="News start date (YYYY-MM-DD). Defaults to earliest price bar.",
    )
    parser.add_argument(
        "--news-end-date",
        type=str,
        default=None,
        help="News end date (YYYY-MM-DD). Defaults to latest price bar.",
    )
    parser.add_argument(
        "--news-sources",
        nargs="*",
        default=list(DEFAULT_SOURCES.keys()),
        help="Subset of RSS sources to query.",
    )
    parser.add_argument(
        "--news-max-articles",
        type=int,
        default=MAX_ARTICLES_PER_SOURCE,
        help=f"Max articles per source (default: {MAX_ARTICLES_PER_SOURCE}).",
    )
    parser.add_argument(
        "--gemini-model",
        default=DEFAULT_MODEL,
        help=f"Gemini model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip Gemini calls and use neutral sentiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        price_df = fetch_dataframe(args.period, args.interval)
    except Exception as exc:
        raise SystemExit(f"Failed to download price data: {exc}") from exc

    try:
        news_start, news_end = resolve_news_range(price_df.index, args.news_start_date, args.news_end_date)
    except ValueError as exc:
        raise SystemExit(f"Invalid news date range: {exc}") from exc

    print(f"Collecting articles from {news_start.date()} through {news_end.date()} (UTC)...")
    articles = fetch_articles(args.news_sources, news_start, news_end, args.news_max_articles)
    news_cache = load_existing_news(args.news_output)
    annotate_articles(articles, args.gemini_model, args.skip_sentiment, news_cache)

    all_records = list(news_cache.values())
    if args.news_output:
        export_news(all_records, args.news_output)

    relevant_records = filter_records_by_range(all_records, news_start, news_end)
    daily_scores = compute_daily_sentiment(relevant_records)
    attach_sentiment_column(price_df, daily_scores)

    save_csv(price_df, args.output)
    print(f"Saved {len(price_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
