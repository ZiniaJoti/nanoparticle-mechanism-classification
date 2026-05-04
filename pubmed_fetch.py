"""
pubmed_fetch.py
===============
Fetches PubMed abstracts using the NCBI E-utilities REST API.

Usage:
    python pubmed_fetch.py
    python pubmed_fetch.py --query "nanoparticle endocytosis" --max_results 500
    python pubmed_fetch.py --query "nanoparticle membrane" --max_results 1000 --email you@example.com
"""

import argparse
import csv
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests

import config

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "pubmed_fetch.log")),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# NCBI E-UTILITIES BASE URLS
# ─────────────────────────────────────────────
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def esearch_pmids(
    query: str,
    max_results: int,
    email: str,
    retries: int = config.PUBMED_RETRIES,
    retry_delay: float = config.PUBMED_RETRY_DELAY,
) -> List[str]:
    """
    Run an ESearch query and return a list of PMIDs (up to max_results).

    Args:
        query:       PubMed search query string.
        max_results: Maximum number of PMIDs to retrieve.
        email:       Email address for NCBI rate-limiting compliance.
        retries:     Number of retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        List of PMID strings.
    """
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "email":   email,
        "tool":    "nanoparticle_nlp",
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"ESearch attempt {attempt}/{retries} | max_results={max_results}")
            resp = requests.get(ESEARCH_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            pmids = data["esearchresult"]["idlist"]
            total = int(data["esearchresult"]["count"])
            logger.info(f"Total hits on PubMed: {total} | Retrieved: {len(pmids)} PMIDs")
            return pmids
        except Exception as exc:
            logger.warning(f"ESearch attempt {attempt} failed: {exc}")
            if attempt < retries:
                time.sleep(retry_delay)

    logger.error("ESearch failed after all retries.")
    return []


def efetch_abstracts(
    pmids: List[str],
    email: str,
    batch_size: int = config.PUBMED_BATCH_SIZE,
    retries: int = config.PUBMED_RETRIES,
    retry_delay: float = config.PUBMED_RETRY_DELAY,
) -> List[Dict]:
    """
    Fetch full abstract records for a list of PMIDs using EFetch in batches.

    Args:
        pmids:      List of PubMed IDs to fetch.
        email:      Email for NCBI compliance.
        batch_size: Number of PMIDs per HTTP request.
        retries:    Retry attempts on failure.
        retry_delay: Delay between retries.

    Returns:
        List of dicts with keys: pmid, title, abstract, journal, year, authors.
    """
    records = []
    total_batches = (len(pmids) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch = pmids[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        logger.info(f"Fetching batch {batch_idx + 1}/{total_batches} ({len(batch)} PMIDs)…")

        params = {
            "db":      "pubmed",
            "id":      ",".join(batch),
            "rettype": "xml",
            "retmode": "xml",
            "email":   email,
            "tool":    "nanoparticle_nlp",
        }

        xml_text = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(EFETCH_URL, params=params, timeout=60)
                resp.raise_for_status()
                xml_text = resp.text
                break
            except Exception as exc:
                logger.warning(f"EFetch batch {batch_idx + 1} attempt {attempt} failed: {exc}")
                if attempt < retries:
                    time.sleep(retry_delay)

        if xml_text is None:
            logger.error(f"Batch {batch_idx + 1} failed after all retries — skipping.")
            continue

        batch_records = _parse_pubmed_xml(xml_text)
        records.extend(batch_records)
        logger.info(f"Parsed {len(batch_records)} records from batch {batch_idx + 1}")

        # NCBI rate limit: max 3 requests/sec without API key
        time.sleep(0.4)

    return records


def _parse_pubmed_xml(xml_text: str) -> List[Dict]:
    """
    Parse PubMed EFetch XML and extract fields for each article.

    Args:
        xml_text: Raw XML response from EFetch.

    Returns:
        List of article dicts.
    """
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.error(f"XML parse error: {exc}")
        return records

    for article in root.findall(".//PubmedArticle"):
        record: Dict[str, Optional[str]] = {
            "pmid":     None,
            "title":    None,
            "abstract": None,
            "journal":  None,
            "year":     None,
            "authors":  None,
        }

        # PMID
        pmid_node = article.find(".//PMID")
        if pmid_node is not None:
            record["pmid"] = pmid_node.text

        # Title
        title_node = article.find(".//ArticleTitle")
        if title_node is not None:
            record["title"] = "".join(title_node.itertext()).strip()

        # Abstract (may contain multiple AbstractText nodes)
        abstract_nodes = article.findall(".//AbstractText")
        if abstract_nodes:
            parts = []
            for node in abstract_nodes:
                label = node.get("Label")
                text  = "".join(node.itertext()).strip()
                if label:
                    parts.append(f"{label}: {text}")
                elif text:
                    parts.append(text)
            record["abstract"] = " ".join(parts).strip() or None

        # Journal
        journal_node = article.find(".//Journal/Title")
        if journal_node is None:
            journal_node = article.find(".//MedlineTA")
        if journal_node is not None:
            record["journal"] = journal_node.text

        # Publication year
        year_node = article.find(".//PubDate/Year")
        if year_node is None:
            year_node = article.find(".//PubDate/MedlineDate")
        if year_node is not None:
            record["year"] = year_node.text[:4] if year_node.text else None

        # Authors
        author_nodes = article.findall(".//Author")
        author_list = []
        for auth in author_nodes:
            last  = auth.findtext("LastName", default="")
            first = auth.findtext("ForeName",  default="")
            name  = f"{last} {first}".strip()
            if name:
                author_list.append(name)
        if author_list:
            record["authors"] = "; ".join(author_list[:5])  # cap at 5

        records.append(record)

    return records


def save_to_csv(records: List[Dict], query: str, filepath: str) -> None:
    """Save records to a CSV file, adding the search_query column."""
    fieldnames = ["pmid", "title", "abstract", "journal", "year", "authors", "search_query"]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: rec.get(k, "") for k in fieldnames}
            row["search_query"] = query
            writer.writerow(row)

    logger.info(f"Saved {len(records)} records → {filepath}")


def save_to_jsonl(records: List[Dict], query: str, filepath: str) -> None:
    """Save records to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            rec["search_query"] = query
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(records)} records → {filepath}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch PubMed abstracts for nanoparticle-membrane interaction research."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=config.PUBMED_SEARCH_QUERY,
        help="PubMed search query string.",
    )
    parser.add_argument(
        "--max_results",
        type=int,
        default=config.PUBMED_MAX_RESULTS,
        help="Maximum number of abstracts to fetch.",
    )
    parser.add_argument(
        "--email",
        type=str,
        default=config.PUBMED_EMAIL,
        help="Email address (required by NCBI).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=config.RAW_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default=config.RAW_JSONL,
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("PubMed Abstract Fetcher")
    logger.info(f"Query       : {args.query[:80]}…")
    logger.info(f"Max results : {args.max_results}")
    logger.info(f"Email       : {args.email}")
    logger.info("=" * 60)

    # 1. Get PMIDs
    pmids = esearch_pmids(args.query, args.max_results, args.email)
    if not pmids:
        logger.error("No PMIDs returned. Exiting.")
        return

    # 2. Fetch full records
    records = efetch_abstracts(pmids, args.email)

    # 3. Filter out records with no abstract
    before = len(records)
    records = [r for r in records if r.get("abstract")]
    logger.info(f"Dropped {before - len(records)} records with no abstract. Remaining: {len(records)}")
    
    # 4. Save outputs
    save_to_csv(records, args.query, args.out_csv)
    save_to_jsonl(records, args.query, args.out_jsonl)

    logger.info("Done!")


if __name__ == "__main__":
    main()
