"""
label_abstracts_llm.py
======================
Labels PubMed abstracts using an LLM (OpenAI or Ollama).

For each abstract, the LLM assigns:
  - llm_label       : one of {wrapping, adhesion, penetration, endocytosis, uncertain}
  - llm_confidence  : float 0.0-1.0
  - llm_reason      : brief rationale
  - uncertain       : bool

Supports modular LLM backends. Progress is saved every N abstracts.

Usage:
    python label_abstracts_llm.py
    python label_abstracts_llm.py --input data/raw/abstracts_raw.csv --backend openai
"""

import argparse
import json
import logging
import os
import re
import time
from typing import Dict, Optional
from google import genai

import pandas as pd

import config

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOGS_DIR, "labeling.log")),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a highly skilled biomedical NLP expert specializing in biophysics 
and nanotechnology. Your task is to classify PubMed abstracts about nanoparticle–biological 
membrane interactions into one of four mechanism classes.

CLASSES:
1. wrapping      - Membrane deformation, partial or full wrapping, engulfment, or membrane 
                   bending around the nanoparticle.
2. adhesion      - Attachment, binding, adsorption, or surface association with the membrane 
                   or cell surface, WITHOUT clear internalization.
3. penetration   - Direct insertion, translocation, pore formation, membrane disruption, 
                   rupture, or crossing through the membrane.
4. endocytosis   - Cellular uptake via endocytic pathways: receptor-mediated endocytosis, 
                   clathrin/caveolin-mediated uptake, vesicular internalization, pinocytosis, 
                   macropinocytosis, or similar active uptake mechanisms.

LABELING RULES:
- Assign ONLY the single best label based on the DOMINANT mechanism described.
- If the abstract is ambiguous, mixed (multiple mechanisms equally prominent), too general, 
  theoretical without a clear mechanism, or unclear → assign label "uncertain".
- Be CONSERVATIVE: when in doubt, choose "uncertain" rather than guessing.
- A confidence score of 1.0 = absolutely certain; 0.0 = completely uncertain.
- If confidence < 0.5, you MUST use "uncertain" as the label.
IMPORTANT:
- You MUST choose one of the four labels unless the abstract is completely irrelevant.
- Do NOT default to "uncertain" unless absolutely necessary.
- If evidence is weak, choose the MOST LIKELY label.
RESPONSE FORMAT (respond ONLY with valid JSON, no extra text):
{
  "label": "<wrapping|adhesion|penetration|endocytosis|uncertain>",
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<1-2 sentence explanation>"
}
"""

USER_TEMPLATE = """Please classify the following abstract:

Title: {title}

Abstract: {abstract}

Respond ONLY with a JSON object as specified.
Do not use markdown.
Do not use triple backticks.
Do not add any text before or after the JSON."""


# ─────────────────────────────────────────────
# LLM BACKENDS
# ─────────────────────────────────────────────
def call_gemini(title: str, abstract: str) -> Optional[Dict]:
    """Call Google Gemini API."""
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)

        user_input = USER_TEMPLATE.format(
            title=title or "",
            abstract=abstract or "",
        )

        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"{SYSTEM_PROMPT}\n\n{user_input}",
            config={
                "temperature": config.LLM_TEMPERATURE,
                "max_output_tokens": config.LLM_MAX_TOKENS,
            }
        )

        raw = getattr(response, "text", "")
        if raw:
            return _parse_llm_response(raw.strip())
        return None

    except Exception as exc:
        logger.error(f"Gemini call failed: {exc}")
        return None
def call_openai(title: str, abstract: str) -> Optional[Dict]:
    """Call OpenAI Chat Completions API."""
    try:
        import openai
        client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_TEMPLATE.format(
                    title=title or "", abstract=abstract or ""
                )},
            ],
        )
        raw = response.choices[0].message.content.strip()
        return _parse_llm_response(raw)
    except Exception as exc:
        logger.error(f"OpenAI call failed: {exc}")
        return None

def call_ollama(title: str, abstract: str) -> Optional[Dict]:
    """Call a local Ollama model."""
    try:
        import requests as req
        payload = {
            "model":  config.OLLAMA_MODEL,
            "prompt": SYSTEM_PROMPT + "\n\n" + USER_TEMPLATE.format(
                title=title or "", abstract=abstract or ""
            ),
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE},
        }
        resp = req.post(
            f"{config.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        return _parse_llm_response(raw)
    except Exception as exc:
        logger.error(f"Ollama call failed: {exc}")
        return None


# Map backend name → function
BACKEND_MAP = {
    "gemini":    call_gemini,
    "openai":    call_openai,
    "ollama":    call_ollama,
}


def _parse_llm_response(raw: str) -> Optional[Dict]:
    """
    Parse LLM JSON response into a structured dict.

    Args:
        raw: Raw string from LLM.

    Returns:
        Dict with keys label, confidence, reason — or None on failure.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting JSON object with regex as fallback
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response: {raw[:200]}")
                return None
        else:
            logger.warning(f"No JSON found in LLM response: {raw[:200]}")
            return None

    label      = str(data.get("label", "uncertain")).strip().lower()
    confidence = float(data.get("confidence", 0.0))
    reason     = str(data.get("reason", "")).strip()

    valid_labels = config.LABEL_NAMES + ["uncertain"]
    if label not in valid_labels:
        logger.warning(f"Invalid label '{label}' — setting to uncertain")
        label      = "uncertain"
        confidence = 0.0

    # Enforce conservative rule: confidence < threshold → uncertain
    if confidence < config.UNCERTAIN_CONFIDENCE_THRESHOLD and label != "uncertain":
        logger.debug(f"Low confidence ({confidence:.2f}) on label '{label}' — forcing uncertain")
        label = "uncertain"

    return {
        "llm_label":      label,
        "llm_confidence": round(confidence, 4),
        "llm_reason":     reason,
        "uncertain":      label == "uncertain",
    }

def safe_save(df: pd.DataFrame, out_path: str) -> None:
    dir_name = os.path.dirname(out_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    temp_path = out_path.replace(".csv", ".tmp.csv")
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, out_path)

# ─────────────────────────────────────────────
# MAIN LABELING FUNCTION
# ─────────────────────────────────────────────

def label_dataframe(
    df: pd.DataFrame,
    backend: str,
    resume: bool = False,
    out_path: str = config.LABELED_CSV,
    only_uncertain: bool = False,
) -> pd.DataFrame:
    if backend not in BACKEND_MAP:
        raise ValueError(f"Unknown backend '{backend}'. Choose from {list(BACKEND_MAP.keys())}")

    call_fn = BACKEND_MAP[backend]
    logger.info(f"Using LLM backend: {backend} | Total abstracts: {len(df)}")

    if "pmid" not in df.columns:
        raise ValueError("Input file must contain a 'pmid' column.")

    df["pmid"] = df["pmid"].astype(str).str.strip()

    # Ensure label columns exist
    for col in ["llm_label", "llm_confidence", "llm_reason", "uncertain"]:
        if col not in df.columns:
            df[col] = None

    # Resume safely: merge previous output back in
    if resume and os.path.exists(out_path):
        logger.info(f"Loading existing progress from {out_path}...")
        existing = pd.read_csv(out_path)
        existing["pmid"] = existing["pmid"].astype(str).str.strip()

        keep_cols = ["pmid", "llm_label", "llm_confidence", "llm_reason", "uncertain"]
        existing = existing[[c for c in keep_cols if c in existing.columns]].copy()

        # Drop current labeling columns before merge
        df = df.drop(columns=[c for c in ["llm_label", "llm_confidence", "llm_reason", "uncertain"] if c in df.columns], errors="ignore")
        df = df.merge(existing, on="pmid", how="left")

    # Determine which rows need labeling
    if only_uncertain:
        needs_label = df["llm_label"].astype(str).str.strip().str.lower() == "uncertain"
        logger.info(f"Relabel-uncertain mode ON | Rows to relabel: {needs_label.sum()}")
    else:
        needs_label = (
            df["llm_label"].isna()
            | (df["llm_label"].astype(str).str.strip() == "")
            | (df["llm_label"].astype(str).str.lower() == "nan")
        )
        logger.info(f"Rows still needing labels: {needs_label.sum()}")

    indices_to_label = df.index[needs_label].tolist()
    logger.info(f"Abstracts to label: {len(indices_to_label)}")

    for count, idx in enumerate(indices_to_label, start=1):
        row = df.loc[idx]
        title = str(row.get("title", "") or "")
        abstract = str(row.get("abstract", "") or "")

        if not abstract.strip():
            df.at[idx, "llm_label"] = "uncertain"
            df.at[idx, "llm_confidence"] = 0.0
            df.at[idx, "llm_reason"] = "Empty abstract"
            df.at[idx, "uncertain"] = True
            logger.warning(f"Row {idx} has empty abstract - marked uncertain")
            continue

        result = None
        for attempt in range(1, config.LLM_RETRIES + 1):
            result = call_fn(title, abstract)
            if result is not None:
                break
            logger.warning(f"Retry {attempt}/{config.LLM_RETRIES} for row {idx}")
            time.sleep(config.LLM_RETRY_DELAY * attempt)

        if result is None:
            result = {
                "llm_label": "uncertain",
                "llm_confidence": 0.0,
                "llm_reason": "LLM call failed after all retries",
                "uncertain": True,
            }

        for key, val in result.items():
            df.at[idx, key] = val

        logger.info(
            f"[{count}/{len(indices_to_label)}] PMID={row.get('pmid', '?')} "
            f"-> {result['llm_label']} (conf={result['llm_confidence']:.2f})"
        )

        if count % config.LLM_BATCH_SAVE_EVERY == 0:
            safe_save(df, out_path)
            logger.info(f"Checkpoint saved ({count} rows processed) -> {out_path}")

    safe_save(df, out_path)
    logger.info(f"All labeling complete. Saved -> {out_path}")
    return df
# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Label PubMed abstracts using an LLM."
    )
    parser.add_argument("--input",   type=str, default=config.RAW_CSV,     help="Input CSV path.")
    parser.add_argument("--output",  type=str, default=config.LABELED_CSV, help="Output CSV path.")
    parser.add_argument("--backend", type=str, default=config.LLM_BACKEND,
                        choices=["gemini","openai", "ollama"],
                        help="LLM backend to use.")
    parser.add_argument("--limit", type=int, default=None,
                        help = "Limit number of abstracts to label for testing")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip already-labeled rows if output file exists.")
    parser.add_argument("--only_uncertain", action="store_true",
                        help="Only relabel rows where llm_label == 'uncertain'")
    
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        return

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit)
        logger.info(f"Limiting to first {args.limit} abstracts for testing")
    logger.info(f"Loaded {len(df)} abstracts from {args.input}")

    df = label_dataframe(
        df,
        args.backend,
        resume=args.resume,
        out_path=args.output,
        only_uncertain=args.only_uncertain,
    )

    # Summary
    label_counts = df["llm_label"].value_counts()
    logger.info("Label distribution:")
    for label, count in label_counts.items():
        logger.info(f"  {label:<15} : {count}")


if __name__ == "__main__":
    main()
