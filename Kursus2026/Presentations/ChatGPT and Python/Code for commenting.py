import os
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional, Tuple, List

import httpx
import pandas as pd
import requests

with open(r"C:/Users/B279683/Desktop/Python Course/Elsevier_API_Key.txt", "r", encoding="utf-8") as f:
    API_key = f.readline()

ELSEVIER_API_KEY = os.getenv("ELSEVIER_API_KEY", API_key).strip()

QUERY = '(PM2.5 W/15 "heart attack") OR (PM2.5 W/15 "myocardial infarction")'
SHOW = 25
OFFSET = 0
SORT_BY = "relevance"
OPEN_ACCESS_ONLY = False
TIMEOUT_S = 30.0

FETCH_ABSTRACTS = False
N_ABSTRACTS = 10
SLEEP_BETWEEN_CALLS_S = 0.3
ABSTRACT_VIEW = "META_ABS"

CHECK_QUOTA = False

PRINT_HEAD_N = 10
PRINT_KEY_COLUMNS_ONLY = True
KEY_COLUMNS = ["title", "doi", "pii", "publication_date", "open_access", "authors", "abstract"]

EXCLUDE_RETRACTED = True
RETRACTED_PATTERNS = [
    r"\bretract(?:ed|ion)\b",
    r"\bwithdrawn\b",
    r"\bwithdrawal\b",
    r"\barticle withdrawn\b",
]

REQUIRE_PUBLISHED = True
REQUIRE_IDENTIFIER = True
DROP_FUTURE_DATES = True

BASE_SEARCH_URL = "https://api.elsevier.com/content/search/sciencedirect"
ARTICLE_BY_DOI_URL = "https://api.elsevier.com/content/article/doi/"
ARTICLE_BY_PII_URL = "https://api.elsevier.com/content/article/pii/"


def sciencedirect_search_v2_put(
    client: httpx.Client,
    *,
    qs: str,
    api_key: str,
    show: int = 25,
    offset: int = 0,
    sort_by: str = "relevance",
    open_access_only: bool = False,
) -> dict:
    if show not in (10, 25, 50, 100):
        raise ValueError("SHOW must be one of: 10, 25, 50, 100")
    if not (0 <= offset <= 6000):
        raise ValueError("OFFSET must be between 0 and 6000")
    if sort_by not in ("relevance", "date"):
        raise ValueError("SORT_BY must be 'relevance' or 'date'")

    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    body: Dict[str, Any] = {"qs": qs, "display": {"offset": offset, "show": show, "sortBy": sort_by}}
    if open_access_only:
        body["filters"] = {"openAccess": True}

    r = client.put(BASE_SEARCH_URL, headers=headers, json=body)
    if r.status_code >= 400:
        raise RuntimeError(f"Search HTTP {r.status_code}\n{r.text[:1200]}")
    return r.json()


def results_to_df(v2_json: dict) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in (v2_json.get("results") or []):
        authors = item.get("authors") or []
        author_str = "; ".join(
            a.get("name", "")
            for a in authors
            if isinstance(a, dict) and a.get("name")
        )
        rows.append({
            "title": item.get("title"),
            "doi": item.get("doi"),
            "pii": item.get("pii"),
            "source_title": item.get("sourceTitle"),
            "publication_date": item.get("publicationDate"),
            "load_date": item.get("loadDate"),
            "open_access": item.get("openAccess"),
            "authors": author_str,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["publication_date"] = pd.to_datetime(df["publication_date"], errors="coerce", utc=True)
        df["load_date"] = pd.to_datetime(df["load_date"], errors="coerce", utc=True)
    return df


def _normalize_doi(doi: str) -> str:
    doi = (doi or "").strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("doi:", "").strip()
    return doi


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _extract_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if "$" in obj and isinstance(obj["$"], str):
            return obj["$"]
        parts = [_extract_text(v) for v in obj.values()]
        return " ".join(p for p in parts if p)
    if isinstance(obj, list):
        parts = [_extract_text(it) for it in obj]
        return " ".join(p for p in parts if p)
    return ""


def _find_abstract_in_response(ft_resp: Dict[str, Any]) -> Optional[str]:
    core = ft_resp.get("coredata") or {}
    cand = core.get("dc:description") or core.get("prism:teaser")
    text = _extract_text(cand)
    if text and len(text.strip()) > 20:
        return _strip_html(text)

    item = ft_resp.get("item") or {}
    bibrecord = item.get("bibrecord") or {}
    head = bibrecord.get("head") or {}
    cand2 = head.get("abstracts") or head.get("abstract")
    text2 = _extract_text(cand2)
    if text2 and len(text2.strip()) > 20:
        return _strip_html(text2)

    found_texts: List[str] = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if "abstract" in str(k).lower():
                    t = _extract_text(v)
                    if t:
                        found_texts.append(t)
                walk(v)
        elif isinstance(o, list):
            for it in o:
                walk(it)

    walk(ft_resp)

    if found_texts:
        best = max((t.strip() for t in found_texts), key=len, default="")
        if best and len(best) > 20:
            return _strip_html(best)

    return None


def fetch_sciencedirect_abstract(
    session: requests.Session,
    *,
    api_key: str,
    doi: Optional[str] = None,
    pii: Optional[str] = None,
    view: str = "META_ABS",
    timeout_s: float = 30.0,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not api_key:
        raise ValueError("Missing api_key")

    doi_norm = _normalize_doi(doi) if doi else None
    if doi_norm:
        url = f"{ARTICLE_BY_DOI_URL}{doi_norm}"
    elif pii:
        url = f"{ARTICLE_BY_PII_URL}{str(pii).strip()}"
    else:
        return None, {"error": "No doi or pii provided"}

    params = {"APIKey": api_key, "httpAccept": "application/json", "view": view}

    try:
        r = session.get(url, params=params, timeout=timeout_s)
        status = r.status_code
        if status >= 400:
            return None, {
                "error": f"HTTP {status}",
                "url": r.url,
                "response_text": (r.text or "")[:1500],
            }
        j = r.json()
    except Exception as e:
        return None, {"error": f"Request failed: {e}", "url": str(url)}

    ft = j.get("full-text-retrieval-response") or {}
    abstract = _find_abstract_in_response(ft)
    return abstract, {"url": r.url, "status": status, "json": j}


def _pick_first_existing(row: pd.Series, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return None


def attach_abstracts_to_results(
    results_df: pd.DataFrame,
    session: requests.Session,
    *,
    api_key: str,
    n: int = 10,
    sleep_s: float = 0.3,
    view: str = "META_ABS",
    timeout_s: float = 30.0,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if results_df is None or results_df.empty:
        return results_df, []

    df = results_df.copy()
    doi_cols = ["doi", "DOI"]
    pii_cols = ["pii", "PII"]
    raw_payloads: List[Dict[str, Any]] = []

    for col in ["abstract", "article_api_status", "article_api_url"]:
        if col not in df.columns:
            df[col] = None

    for idx in df.index[: min(n, len(df))]:
        row = df.loc[idx]
        doi = _pick_first_existing(row, doi_cols)
        pii = _pick_first_existing(row, pii_cols)

        if not doi and not pii:
            raw_payloads.append({"index": idx, "doi": doi, "pii": pii, "payload": {"error": "No doi or pii"}})
            continue

        abstract, payload = fetch_sciencedirect_abstract(
            session,
            api_key=api_key,
            doi=doi,
            pii=pii,
            view=view,
            timeout_s=timeout_s,
        )

        df.at[idx, "abstract"] = abstract
        df.at[idx, "article_api_status"] = payload.get("status")
        df.at[idx, "article_api_url"] = payload.get("url")
        raw_payloads.append({"index": idx, "doi": doi, "pii": pii, "payload": payload})

        if sleep_s and sleep_s > 0:
            time.sleep(sleep_s)

    return df, raw_payloads


def elsevier_quota_snapshot(
    client: httpx.Client,
    api_key: str,
    *,
    verbose: bool = True,
    qs: str = "test",
    show: int = 10,
    offset: int = 0,
    sort_by: str = "relevance",
    open_access_only: bool = False,
) -> Dict[str, Any]:
    if not api_key or not api_key.strip():
        raise ValueError("api_key is empty")

    if show not in (10, 25, 50, 100):
        raise ValueError("show must be one of: 10, 25, 50, 100")
    if not (0 <= offset <= 6000):
        raise ValueError("offset must be between 0 and 6000")
    if sort_by not in ("relevance", "date"):
        raise ValueError("sort_by must be one of: relevance, date")

    headers = {"X-ELS-APIKey": api_key.strip(), "Accept": "application/json"}
    body: Dict[str, Any] = {"qs": qs, "display": {"offset": offset, "show": show, "sortBy": sort_by}}
    if open_access_only:
        body["filters"] = {"openAccess": True}

    header_candidates = [
        "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset",
        "RateLimit-Limit", "RateLimit-Remaining", "RateLimit-Reset",
        "Retry-After",
        "X-ELS-Status", "X-ELS-Message", "X-Request-Id", "X-Correlation-Id",
    ]

    t0 = time.perf_counter()
    resp = client.put(BASE_SEARCH_URL, headers=headers, json=body)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    interesting_headers = {h: resp.headers[h] for h in header_candidates if h in resp.headers}
    ok = 200 <= resp.status_code < 300
    error_text = None if ok else (resp.text or "")[:1200]

    result = {
        "ok": ok,
        "status_code": resp.status_code,
        "elapsed_ms": round(elapsed_ms, 1),
        "headers": interesting_headers,
        "error_text": error_text,
    }

    if verbose:
        print(f"Elsevier quota snapshot: HTTP {result['status_code']} | {result['elapsed_ms']} ms")
        if interesting_headers:
            for k, v in interesting_headers.items():
                print(f"  {k}: {v}")

            reset_raw = interesting_headers.get("X-RateLimit-Reset") or interesting_headers.get("RateLimit-Reset")
            if reset_raw:
                try:
                    reset_ts = int(float(reset_raw))
                    reset_utc = datetime.fromtimestamp(reset_ts, tz=timezone.utc)
                    reset_cph = reset_utc.astimezone(ZoneInfo("Europe/Copenhagen"))
                    print(f"  Rate limit resets (UTC): {reset_utc.isoformat()}")
                    print(f"  Rate limit resets (Copenhagen): {reset_cph.isoformat()}")
                except Exception:
                    pass
        if error_text:
            print("  Error (truncated):", error_text)

    return result


def _is_retracted_title(title: Any, rx: re.Pattern) -> bool:
    if title is None:
        return False
    t = str(title).strip()
    if not t:
        return False
    return rx.search(t) is not None


def filter_results(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    stats = {"initial": 0, "removed": 0, "final": 0}
    if df is None or df.empty:
        return df, stats

    stats["initial"] = len(df)
    out = df.copy()

    out = out[out["title"].notna() & (out["title"].astype(str).str.strip() != "")]

    if REQUIRE_PUBLISHED:
        out = out[out["publication_date"].notna()]

    if DROP_FUTURE_DATES and "publication_date" in out.columns:
        now_utc = datetime.now(timezone.utc)
        out = out[out["publication_date"] <= now_utc]

    if REQUIRE_IDENTIFIER:
        has_id = out["doi"].notna() & (out["doi"].astype(str).str.strip() != "")
        has_pii = out["pii"].notna() & (out["pii"].astype(str).str.strip() != "")
        out = out[has_id | has_pii]

    if EXCLUDE_RETRACTED:
        rx = re.compile("|".join(RETRACTED_PATTERNS), flags=re.IGNORECASE)
        out = out[~out["title"].apply(lambda x: _is_retracted_title(x, rx))]

    doi_norm = out["doi"].fillna("").astype(str).map(_normalize_doi).str.lower()
    pii_norm = out["pii"].fillna("").astype(str).str.strip().str.lower()
    key = doi_norm.where(doi_norm != "", pii_norm)
    out = out.assign(_dedupe_key=key)
    out = out[out["_dedupe_key"] != ""].drop_duplicates(subset=["_dedupe_key"], keep="first").drop(columns=["_dedupe_key"])

    stats["final"] = len(out)
    stats["removed"] = stats["initial"] - stats["final"]
    return out, stats


def print_summary(raw_response: dict, df: pd.DataFrame, stats: Optional[Dict[str, int]] = None, *, head_n: int = 10) -> None:
    print("Results found:", raw_response.get("resultsFound"))
    print("Returned this page (raw):", len(raw_response.get("results") or []))
    if stats:
        print("After filtering:", stats.get("final", len(df)), "| Removed:", stats.get("removed", 0))
    else:
        print("Returned this page:", len(df))

    if df is None or df.empty:
        return

    if PRINT_KEY_COLUMNS_ONLY:
        cols = [c for c in KEY_COLUMNS if c in df.columns]
        print(df[cols].head(head_n))
    else:
        print(df.head(head_n))


def main() -> Tuple[pd.DataFrame, dict, Optional[dict], Optional[List[Dict[str, Any]]], Dict[str, int]]:
    if not ELSEVIER_API_KEY:
        raise RuntimeError("API key is empty. Set ELSEVIER_API_KEY in your environment or paste your full key.")

    quota_info: Optional[dict] = None
    raw_article_payloads: Optional[List[Dict[str, Any]]] = None

    with httpx.Client(timeout=TIMEOUT_S, follow_redirects=True) as httpx_client, requests.Session() as req_session:
        raw_response = sciencedirect_search_v2_put(
            httpx_client,
            qs=QUERY,
            api_key=ELSEVIER_API_KEY,
            show=SHOW,
            offset=OFFSET,
            sort_by=SORT_BY,
            open_access_only=OPEN_ACCESS_ONLY,
        )
        results_df = results_to_df(raw_response)

        filtered_df, filter_stats = filter_results(results_df)

        final_df = filtered_df
        if FETCH_ABSTRACTS and not filtered_df.empty:
            final_df, raw_article_payloads = attach_abstracts_to_results(
                filtered_df,
                req_session,
                api_key=ELSEVIER_API_KEY,
                n=N_ABSTRACTS,
                sleep_s=SLEEP_BETWEEN_CALLS_S,
                view=ABSTRACT_VIEW,
                timeout_s=TIMEOUT_S,
            )

        print_summary(raw_response, final_df, filter_stats, head_n=PRINT_HEAD_N)

        if CHECK_QUOTA:
            quota_info = elsevier_quota_snapshot(httpx_client, ELSEVIER_API_KEY, verbose=True)

    return final_df, raw_response, quota_info, raw_article_payloads, filter_stats


if __name__ == "__main__":
    final_df, raw_response, quota_info, raw_article_payloads, filter_stats = main()
