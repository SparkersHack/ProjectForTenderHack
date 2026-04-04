from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .dataset_layout import resolve_synonyms_path
from .semantic import DEFAULT_FASTTEXT_MODEL_PATH, SemanticExpander
from .text import normalize_text, normalize_tokens, stem_tokens, tokenize, unique_preserve_order


DEFAULT_SYNONYMS_PATH = resolve_synonyms_path()

MEASUREMENT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:мг|г|кг|мл|л|мм|см|м|шт|гб|мб|тб|мкг|ед|dpi)\b",
    re.IGNORECASE,
)
ALPHANUMERIC_CODE_RE = re.compile(r"^(?=.*[a-zа-я])(?=.*\d)[0-9a-zа-я-]{2,}$", re.IGNORECASE)
FORMAT_CODE_HINTS = {"a0", "a1", "a2", "a3", "a4", "a5", "usb", "ssd", "hdd", "wifi", "bluetooth"}


def _edit_distance(left: str, right: str, max_distance: int = 2) -> int:
    if left == right:
        return 0
    if abs(len(left) - len(right)) > max_distance:
        return max_distance + 1
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        best = current[0]
        for j, right_char in enumerate(right, start=1):
            cost = 0 if left_char == right_char else 1
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            current.append(value)
            best = min(best, value)
        if best > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


@dataclass
class QueryEntity:
    kind: str
    value: str
    normalized_value: str
    tokens: List[str]

    def to_mapping(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "tokens": list(self.tokens),
        }


@dataclass
class QueryAnalysis:
    original_query: str
    normalized_query: str
    corrected_query: str
    original_tokens: List[str]
    corrected_tokens: List[str]
    stemmed_tokens: List[str]
    expanded_tokens: List[str]
    expanded_stems: List[str]
    semantic_expansions: List[str]
    applied_corrections: List[Dict[str, str]]
    applied_synonyms: List[Dict[str, List[str]]]
    applied_semantic_neighbors: List[Dict[str, object]]
    entities: List[QueryEntity]
    entity_tokens: List[str]
    entity_stems: List[str]


class TypoCorrector:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def _candidate_tokens(self, token: str) -> List[sqlite3.Row]:
        first_char = token[0]
        length = len(token)
        candidates: Dict[str, sqlite3.Row] = {}

        prefix = token[: min(4, len(token))]
        for row in self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE token LIKE ?
              AND token_length BETWEEN ? AND ?
            ORDER BY frequency DESC
            LIMIT 150
            """,
            (f"{prefix}%", max(1, length - 2), length + 2),
        ).fetchall():
            candidates[row["token"]] = row

        for row in self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE first_char = ?
              AND token_length BETWEEN ? AND ?
            ORDER BY frequency DESC
            LIMIT 250
            """,
            (first_char, max(1, length - 2), length + 2),
        ).fetchall():
            candidates.setdefault(row["token"], row)

        if candidates:
            return sorted(candidates.values(), key=lambda row: row["frequency"], reverse=True)

        return self.conn.execute(
            """
            SELECT token, frequency
            FROM token_frequency
            WHERE token_length BETWEEN ? AND ?
            ORDER BY frequency DESC
            LIMIT 250
            """,
            (max(1, length - 1), length + 1),
        ).fetchall()

    def correct_tokens(self, tokens: Iterable[str]) -> tuple[List[str], List[Dict[str, str]]]:
        corrected: List[str] = []
        applied: List[Dict[str, str]] = []
        for token in tokens:
            if len(token) <= 2 or token.isdigit():
                corrected.append(token)
                continue
            exists = self.conn.execute(
                "SELECT 1 FROM token_frequency WHERE token = ? LIMIT 1",
                (token,),
            ).fetchone()
            if exists:
                corrected.append(token)
                continue
            best_token = token
            best_distance = 99
            best_frequency = -1
            for row in self._candidate_tokens(token):
                candidate = row["token"]
                distance = _edit_distance(token, candidate, max_distance=2)
                if distance > 2:
                    continue
                if distance < best_distance or (distance == best_distance and row["frequency"] > best_frequency):
                    best_token = candidate
                    best_distance = distance
                    best_frequency = row["frequency"]
            corrected.append(best_token)
            if best_token != token:
                applied.append({"source": token, "target": best_token})
        return corrected, applied


class QueryUnderstandingService:
    def __init__(
        self,
        conn: sqlite3.Connection,
        synonyms_path: Path | str = DEFAULT_SYNONYMS_PATH,
        semantic_top_n: int = 4,
        semantic_backend: str = "auto",
        fasttext_model_path: Path | str = DEFAULT_FASTTEXT_MODEL_PATH,
        fasttext_similarity_threshold: float = 0.55,
    ) -> None:
        self.conn = conn
        self.synonyms_path = Path(synonyms_path)
        self.corrector = TypoCorrector(conn)
        self.synonyms = self._load_synonyms()
        self.semantic_expander = SemanticExpander(
            conn,
            top_n=semantic_top_n,
            backend=semantic_backend,
            fasttext_model_path=fasttext_model_path,
            fasttext_similarity_threshold=fasttext_similarity_threshold,
        )

    def _load_synonyms(self) -> Dict[str, Dict[str, List[str]]]:
        payload = json.loads(self.synonyms_path.read_text(encoding="utf-8"))
        phrase_synonyms = {
            normalize_text(key): [normalize_text(value) for value in values]
            for key, values in payload.get("phrase_synonyms", {}).items()
        }
        token_synonyms = {
            normalize_text(key): [normalize_text(value) for value in values]
            for key, values in payload.get("token_synonyms", {}).items()
        }
        return {
            "phrase_synonyms": phrase_synonyms,
            "token_synonyms": token_synonyms,
        }

    def _apply_synonyms(self, normalized_query: str, corrected_tokens: List[str]) -> tuple[List[str], List[Dict[str, List[str]]]]:
        expanded: List[str] = list(corrected_tokens)
        applied: List[Dict[str, List[str]]] = []
        phrase_synonyms = self.synonyms["phrase_synonyms"]
        token_synonyms = self.synonyms["token_synonyms"]

        for phrase, replacements in phrase_synonyms.items():
            if phrase and phrase in normalized_query:
                expanded.extend(replacements)
                applied.append({"source": phrase, "targets": replacements})

        for token in corrected_tokens:
            replacements = token_synonyms.get(token)
            if replacements:
                expanded.extend(replacements)
                applied.append({"source": token, "targets": replacements})

        expanded_tokens: List[str] = []
        for item in expanded:
            expanded_tokens.extend(tokenize(item))
        return unique_preserve_order(expanded_tokens), applied

    @staticmethod
    def _dedupe_applied_targets(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
        result: List[Dict[str, object]] = []
        seen = set()
        for item in items:
            source = item.get("source")
            targets = tuple(item.get("targets", []))
            key = (source, targets)
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _extract_entities(self, normalized_query: str, corrected_tokens: List[str]) -> List[QueryEntity]:
        entities: List[QueryEntity] = []
        seen = set()

        for match in MEASUREMENT_RE.finditer(normalized_query):
            value = normalize_text(match.group(0))
            if not value:
                continue
            key = ("measurement", value)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                QueryEntity(
                    kind="measurement",
                    value=match.group(0),
                    normalized_value=value,
                    tokens=tokenize(value),
                )
            )

        for token in corrected_tokens:
            normalized_token = normalize_text(token)
            if not normalized_token:
                continue
            if normalized_token in FORMAT_CODE_HINTS or ALPHANUMERIC_CODE_RE.fullmatch(normalized_token):
                key = ("code", normalized_token)
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    QueryEntity(
                        kind="code",
                        value=token,
                        normalized_value=normalized_token,
                        tokens=[normalized_token],
                    )
                )

        return entities

    def analyze_query(self, query: str) -> QueryAnalysis:
        normalized_query = normalize_text(query)
        original_tokens = normalize_tokens(tokenize(normalized_query))

        original_synonym_expansions, original_applied_synonyms = self._apply_synonyms(normalized_query, original_tokens)

        corrected_tokens: List[str] = []
        corrections: List[Dict[str, str]] = []
        synonym_keys = set(self.synonyms["token_synonyms"].keys())
        for token in original_tokens:
            if token in synonym_keys:
                corrected_tokens.append(token)
                continue
            corrected_part, applied_part = self.corrector.correct_tokens([token])
            corrected_tokens.extend(corrected_part)
            corrections.extend(applied_part)

        corrected_tokens = normalize_tokens(corrected_tokens)
        corrected_query = " ".join(corrected_tokens)

        corrected_synonym_expansions, corrected_applied_synonyms = self._apply_synonyms(
            corrected_query or normalized_query,
            corrected_tokens,
        )
        synonym_expansions = unique_preserve_order(original_synonym_expansions + corrected_synonym_expansions)
        applied_synonyms = self._dedupe_applied_targets(original_applied_synonyms + corrected_applied_synonyms)
        semantic_expansions, applied_semantic_neighbors = self.semantic_expander.expand_tokens(corrected_tokens)
        merged_tokens = unique_preserve_order(corrected_tokens + synonym_expansions + semantic_expansions)
        stemmed_tokens = stem_tokens(corrected_tokens)
        expanded_stems = stem_tokens(merged_tokens)
        entities = self._extract_entities(normalized_query or corrected_query, corrected_tokens)
        entity_tokens = unique_preserve_order(
            token
            for entity in entities
            for token in entity.tokens
        )
        entity_stems = stem_tokens(entity_tokens)
        return QueryAnalysis(
            original_query=query,
            normalized_query=normalized_query,
            corrected_query=corrected_query,
            original_tokens=original_tokens,
            corrected_tokens=corrected_tokens,
            stemmed_tokens=stemmed_tokens,
            expanded_tokens=merged_tokens,
            expanded_stems=expanded_stems,
            semantic_expansions=semantic_expansions,
            applied_corrections=corrections,
            applied_synonyms=applied_synonyms,
            applied_semantic_neighbors=applied_semantic_neighbors,
            entities=entities,
            entity_tokens=entity_tokens,
            entity_stems=entity_stems,
        )
