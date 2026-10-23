"""
按主题/关键词检索候选文献（区别于 fetch_abstracts.py 的"已知标题回填摘要"场景，
这里是"从一句话论断/概念/方法名出发找论文"的正向检索）。

依次查询 OpenAlex → Semantic Scholar，命中 429 时按与 fetch_abstracts.py 一致的策略退避重试。
输出候选列表（标题、作者、年份、DOI、摘要、被引次数、来源库），交给上层判断是否与待引用处匹配。

用法:
  python search_paper.py "ResNet deep residual learning" --limit 5
  python search_paper.py --doi 10.1109/CVPR.2016.90
  python search_paper.py "contrastive learning" --author "Chen" --year 2020 --json
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import certifi
    _CAFILE = certifi.where()
except ImportError:
    _CAFILE = None


def _load_local_keys():
    """从脚本同目录的 api_keys.local.json 读取用户自备的 key（未配置/文件不存在则为 None，脚本在无 key 时也能正常工作，只是限流阈值更低）。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_keys.local.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None, None
    return data.get("openalex_api_key") or None, data.get("s2_api_key") or None


OPENALEX_API_KEY, S2_API_KEY = _load_local_keys()
_S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else None
_S2_FIELDS = "title,abstract,externalIds,citationCount,authors,year,publicationDate,venue,journal"


def _diagnostic(message):
    # Keep --json machine-readable even when an upstream request fails.
    print(message, file=sys.stderr, flush=True)


def _http_get(url, headers=None, timeout=20, retries=3, label=""):
    ctx = ssl.create_default_context(cafile=_CAFILE)
    hdrs = {"User-Agent": "PaperCitationSearch/1.0 (mailto:research@example.com)"}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = (attempt + 1) * 5
                tag = f"[{label}] " if label else ""
                _diagnostic(f"    {tag}[429] 限流，等待 {wait}s 后重试...")
                time.sleep(wait)
                continue
            raise


def _openalex_get(url, label="OpenAlex"):
    """命中 429 立即改用内置 API Key 重试，Key 请求仍限流则等 5s 再试一次。"""
    try:
        return _http_get(url, label=label, retries=1)
    except urllib.error.HTTPError as e:
        if e.code == 429 and OPENALEX_API_KEY:
            _diagnostic(f"    [{label}] 限流，使用 API Key 重试...")
            return _http_get(f"{url}&api_key={OPENALEX_API_KEY}", label=label, retries=2)
        raise


def _openalex_to_record(work):
    inv = work.get("abstract_inverted_index")
    abstract = None
    if isinstance(inv, dict) and inv:
        positions = sorted(
            ((pos, w) for w, ps in inv.items() for pos in ps), key=lambda x: x[0]
        )
        abstract = " ".join(w for _, w in positions)
    source = (work.get("primary_location") or {}).get("source") or {}
    biblio = work.get("biblio") or {}
    first_page, last_page = biblio.get("first_page"), biblio.get("last_page")
    if first_page and last_page:
        pages = f"{first_page}--{last_page}"
    else:
        pages = first_page or last_page or None
    return {
        "source": "OpenAlex",
        "title": work.get("display_name") or work.get("title") or "",
        "authors": [
            (a.get("author") or {}).get("display_name", "")
            for a in work.get("authorships", [])
        ],
        "year": work.get("publication_year"),
        "year_basis": "OpenAlex.publication_year; may reflect earliest electronic publication, not issue year",
        "publication_date": work.get("publication_date"),
        "source_url": work.get("id"),
        "doi": (work.get("doi") or "").replace("https://doi.org/", "") or None,
        "venue": source.get("display_name"),
        "venue_type": source.get("type"),
        "publisher": source.get("host_organization_name"),
        "volume": biblio.get("volume") or None,
        "issue": biblio.get("issue") or None,
        "pages": pages,
        "cited_by": work.get("cited_by_count"),
        "abstract": abstract,
        "openalex_id": work.get("id"),
    }


def search_openalex(query, limit=5):
    url = (
        "https://api.openalex.org/works"
        f"?search={urllib.parse.quote(query)}&per_page={limit}&mailto=research@example.com"
    )
    try:
        data = _openalex_get(url)
    except Exception as e:
        _diagnostic(f"  [OpenAlex] 请求失败: {e}")
        return []
    return [_openalex_to_record(w) for w in data.get("results", [])]


def by_doi_openalex(doi):
    url = (
        "https://api.openalex.org/works"
        f"?filter=doi:{urllib.parse.quote(doi)}&per_page=1&mailto=research@example.com"
    )
    try:
        data = _openalex_get(url)
    except Exception as e:
        _diagnostic(f"  [OpenAlex] DOI 查询失败: {e}")
        return []
    return [_openalex_to_record(w) for w in data.get("results", [])]


def _s2_to_record(paper):
    journal = paper.get("journal") or {}
    return {
        "source": "Semantic Scholar",
        "title": paper.get("title") or "",
        "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
        "year": paper.get("year"),
        "year_basis": "Semantic Scholar.year; online/issue date type not specified",
        "publication_date": paper.get("publicationDate"),
        "source_url": ("https://www.semanticscholar.org/paper/" + paper["paperId"])
        if paper.get("paperId") else None,
        "doi": (paper.get("externalIds") or {}).get("DOI") or None,
        "venue": paper.get("venue"),
        "venue_type": None,
        "publisher": None,
        "volume": journal.get("volume") or None,
        "issue": None,
        "pages": journal.get("pages") or None,
        "cited_by": paper.get("citationCount"),
        "abstract": paper.get("abstract"),
        "paper_id": paper.get("paperId"),
    }


def search_semantic_scholar(query, limit=5):
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={urllib.parse.quote(query)}&fields={_S2_FIELDS}&limit={limit}"
    )
    try:
        data = _http_get(url, headers=_S2_HEADERS, label="Semantic Scholar")
    except Exception as e:
        _diagnostic(f"  [Semantic Scholar] 请求失败: {e}")
        return []
    return [_s2_to_record(p) for p in data.get("data", [])]


def by_doi_s2(doi):
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields={_S2_FIELDS}"
    try:
        paper = _http_get(url, headers=_S2_HEADERS, label="Semantic Scholar")
    except Exception as e:
        _diagnostic(f"  [Semantic Scholar] DOI 查询失败: {e}")
        return []
    return [_s2_to_record(paper)] if paper else []


def _partial_date(value):
    """Preserve Crossref's date precision; do not invent missing months/days."""
    parts = (value or {}).get("date-parts")
    if not parts or len(parts) != 1 or not parts[0]:
        return None
    date = parts[0]
    if len(date) > 3 or any(type(part) is not int for part in date):
        return None
    if not 1 <= date[0] <= 9999:
        return None
    if len(date) > 1 and not 1 <= date[1] <= 12:
        return None
    if len(date) > 2:
        from datetime import date as calendar_date
        try:
            calendar_date(*date)
        except ValueError:
            return None
    return "-".join([f"{date[0]:04d}"] + [f"{part:02d}" for part in date[1:]])


def _crossref_to_record(work, source_url):
    issue = work.get("journal-issue") or {}
    dates = {
        "published-online": _partial_date(work.get("published-online")),
        "published-print": _partial_date(work.get("published-print")),
        "issued": _partial_date(work.get("issued")),
        "published": _partial_date(work.get("published")),
        "journal-issue.published-online": _partial_date(issue.get("published-online")),
        "journal-issue.published-print": _partial_date(issue.get("published-print")),
    }
    return {
        "source": "Crossref",
        "title": next(iter(work.get("title") or []), ""),
        "authors": [a.get("name") or " ".join(
            p for p in (a.get("given"), a.get("family")) if p
        ) for a in (work.get("author") or [])],
        # Dates are evidence for a version/style decision, not a corrected year.
        "year": None,
        "year_basis": "Not selected; choose online or issue year from publisher/version/style evidence",
        "publication_date": None,
        "publication_dates": dates,
        "source_url": source_url,
        "doi": work.get("DOI"),
        "venue": next(iter(work.get("container-title") or []), None),
        "venue_type": work.get("type"),
        "publisher": work.get("publisher"),
        "volume": work.get("volume"),
        "issue": work.get("issue"),
        "pages": work.get("page") or work.get("article-number"),
        "cited_by": work.get("is-referenced-by-count"),
        "abstract": None,
    }


def by_doi_crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    try:
        data = _http_get(url, label="Crossref")
    except Exception as e:
        _diagnostic(f"  [Crossref] 出版日期查询失败，需另查出版社页面: {e}")
        return []
    work = data.get("message") or {}
    # Never attach dates from a different DOI/version returned upstream.
    if str(work.get("DOI") or "").casefold() != doi.casefold():
        _diagnostic("  [Crossref] 返回的 DOI 不符，未采用其日期")
        return []
    return [_crossref_to_record(work, url)]


def _print_human(records):
    if not records:
        print("(无候选)")
        return
    for i, r in enumerate(records, 1):
        authors = ", ".join(r["authors"][:3]) + (" et al." if len(r["authors"]) > 3 else "")
        print(f"\n[{i}] ({r['source']}) {r['title']}")
        print(f"    作者: {authors}  来源库年份: {r['year']}  被引: {r['cited_by']}  DOI: {r['doi']}")
        if r.get("year_basis"):
            print(f"    年份口径: {r['year_basis']}")
        if r.get("publication_date"):
            print(f"    来源库发表日期（未区分在线/卷期）: {r['publication_date']}")
        for kind, value in (r.get("publication_dates") or {}).items():
            print(f"    {kind}: {value or '未提供'}")
        if r.get("source_url"):
            print(f"    元数据来源: {r['source_url']}")
        if r.get("venue"):
            print(f"    期刊/会议: {r['venue']}")
        biblio_bits = []
        if r.get("publisher"):
            biblio_bits.append(f"出版社: {r['publisher']}")
        if r.get("volume"):
            biblio_bits.append(f"卷: {r['volume']}")
        if r.get("issue"):
            biblio_bits.append(f"期: {r['issue']}")
        if r.get("pages"):
            biblio_bits.append(f"页码: {r['pages']}")
        if biblio_bits:
            print("    " + "  ".join(biblio_bits))
        abs_ = r.get("abstract") or ""
        print(f"    摘要: {abs_[:280]}{'...' if len(abs_) > 280 else ''}")


def main():
    parser = argparse.ArgumentParser(description="按主题/关键词从 OpenAlex + Semantic Scholar 检索候选文献")
    parser.add_argument("query", nargs="?", help="检索关键词/论断/方法名/数据集名")
    parser.add_argument("--doi", help="已知 DOI 时直接精确查询，跳过关键词检索")
    parser.add_argument("--publication-dates", action="store_true",
                        help="与 --doi 配合，追加 Crossref 在线/纸本/卷期日期；不自动选择引用年份")
    parser.add_argument("--author", help="提示性作者姓氏，仅用于人工核对，不参与自动过滤")
    parser.add_argument("--year", type=int, help="提示性年份，仅用于人工核对，不参与自动过滤")
    parser.add_argument("--limit", type=int, default=5, help="每个来源返回的候选数（默认 5）")
    parser.add_argument("--json", action="store_true", help="输出原始 JSON 而非可读文本")
    args = parser.parse_args()
    if args.publication_dates and not args.doi:
        parser.error("--publication-dates 需要 --doi；先确认文献身份，再逐条查询日期")
    if args.doi:
        args.doi = args.doi.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
            if args.doi.lower().startswith(prefix):
                args.doi = args.doi[len(prefix):].strip()
                break

    if args.doi:
        records = by_doi_openalex(args.doi)
        if not records:
            _diagnostic("  [OpenAlex] 未命中该 DOI，改用 Semantic Scholar")
            records = by_doi_s2(args.doi)
        if args.publication_dates:
            records += by_doi_crossref(args.doi)
    else:
        if not args.query:
            parser.error("需要提供 query 或 --doi")
        records = search_openalex(args.query, args.limit)
        time.sleep(1)
        records += search_semantic_scholar(args.query, args.limit)

    if args.author or args.year:
        _diagnostic(f"(提示：期望作者含 \"{args.author}\"，原条目年份 {args.year}；需按版本及日期口径核对，不做自动过滤)")

    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
    else:
        _print_human(records)


if __name__ == "__main__":
    main()
