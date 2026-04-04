#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tenderhack.text import STOPWORDS, normalize_text, stem_tokens, tokenize, unique_preserve_order


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "reference" / "search_synonyms.json"
RAW_CATALOG_GLOBS = (
    "СТЕ*.csv",
    "data/raw/СТЕ*.csv",
    "data/raw/*ste*.csv",
    "data/reference/*ste*.csv",
)

PAREN_ALIAS_RE = re.compile(r"(?P<base>[^()]{3,}?)\s*\((?P<alias>[^()]{2,80})\)")
BAD_ALIAS_TOKENS = {
    "вариант",
    "варианта",
    "комплектации",
    "исполнения",
    "упаковке",
    "упаковка",
    "форма",
    "цвет",
    "размер",
    "тип",
    "модель",
    "серия",
    "код",
    "позиция",
    "набор",
    "комплект",
}

# Синонимы в этом проекте используются как additive query expansion. Поэтому
# здесь допустимы только очень точные alias -> canonical связи:
# аббревиатуры, общепринятый жаргон и устойчивые технические варианты записи.
# Нельзя смешивать синонимы с гиперонимами, смежными товарами, атрибутами или
# просто "похожими" словами: это резко ухудшает precision.
MANUAL_TOKEN_SYNONYMS: dict[str, tuple[str, ...]] = {
    # IT / office hardware
    "мобильник": ("мобильный телефон",),
    "сотовый": ("мобильный телефон",),
    "смартфон": ("мобильный телефон",),
    "ноут": ("ноутбук",),
    "лэптоп": ("ноутбук",),
    "laptop": ("ноутбук",),
    "пк": ("персональный компьютер",),
    "системник": ("системный блок",),
    "дисплей": ("монитор",),
    "клава": ("клавиатура",),
    "мышка": ("мышь",),
    "вебкамера": ("веб камера",),
    "мфу": ("многофункциональное устройство",),
    "флешка": ("флеш накопитель", "usb накопитель"),
    "флешдрайв": ("флеш накопитель", "usb накопитель"),
    "hdd": ("жесткий диск",),
    "ssd": ("твердотельный накопитель",),
    "роутер": ("маршрутизатор",),
    "router": ("маршрутизатор",),
    "switch": ("коммутатор",),
    "ибп": ("источник бесперебойного питания",),
    "бесперебойник": ("источник бесперебойного питания",),
    "софт": ("программное обеспечение",),
    "software": ("программное обеспечение",),
    "эцп": ("электронная подпись", "квалифицированная электронная подпись"),
    "кэп": ("квалифицированная электронная подпись", "электронная подпись"),
    "эдо": ("электронный документооборот",),
    "cctv": ("видеонаблюдение",),
    "акб": ("аккумулятор",),
    "led": ("светодиодный",),
    # Office supplies / services
    "мультифора": ("файл",),
    "тетрадка": ("тетрадь",),
    "скобосшиватель": ("степлер",),
    "стерка": ("ластик",),
    "клининг": ("уборка",),
    # Medicine / pharma
    "лекарство": ("лекарственный препарат",),
    "медикамент": ("лекарственный препарат",),
    "анальгетик": ("обезболивающее",),
    "анальгетики": ("обезболивающее",),
    "антипиретик": ("жаропонижающее",),
    "антипиретики": ("жаропонижающее",),
    "антибиотик": ("антибактериальный препарат",),
    "дезсредство": ("дезинфицирующее средство",),
    "антисептик": ("дезинфицирующее средство",),
    "капельница": ("инфузионная система",),
    "укол": ("инъекция",),
    "табл": ("таблетка",),
    "капс": ("капсула",),
    "амп": ("ампула",),
    "фл": ("флакон",),
    "сусп": ("суспензия",),
    "вв": ("внутривенно",),
    "вм": ("внутримышечно",),
    # Transport / facilities / tools
    "дизтопливо": ("дизельное топливо",),
    "ушм": ("углошлифовальная машина",),
    "болгарка": ("углошлифовальная машина",),
    "винтоверт": ("шуруповерт",),
    "умывальник": ("раковина",),
    # Stable abbreviations and units
    "мг": ("миллиграмм",),
    "мл": ("миллилитр",),
    "кг": ("килограмм",),
    "мм": ("миллиметр",),
    "см": ("сантиметр",),
    "гб": ("гигабайт",),
    "мб": ("мегабайт",),
    "тб": ("терабайт",),
    "вт": ("ватт",),
}

MANUAL_PHRASE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "мобильный телефон": ("сотовый телефон", "телефон сотовой связи", "смартфон"),
    "веб камера": ("вебкамера", "web camera"),
    "многофункциональное устройство": (
        "мфу",
        "принтер сканер копир",
        "сканер принтер копир",
        "сканер копир принтер",
    ),
    "флеш накопитель": ("usb накопитель", "usb флешка", "флеш драйв"),
    "usb накопитель": ("флеш накопитель", "usb флешка", "флеш драйв"),
    "источник бесперебойного питания": ("ибп", "бесперебойник"),
    "программное обеспечение": ("софт", "software"),
    "электронная подпись": ("эцп", "квалифицированная электронная подпись", "кэп"),
    "квалифицированная электронная подпись": ("кэп", "эцп", "электронная подпись"),
    "электронный документооборот": ("эдо",),
    "камера видеонаблюдения": ("камера наблюдения", "видеокамера"),
    "аккумуляторная батарея": ("акб",),
    "ручка канцелярская": ("канц ручка", "шариковая ручка"),
    "шариковая ручка": ("ручка канцелярская", "канц ручка"),
    "лекарственный препарат": ("лекарство", "медикамент"),
    "дезинфицирующее средство": ("дезсредство", "антисептик"),
    "инфузионная система": ("капельница",),
    "дизельное топливо": ("дизтопливо",),
    "сплит система": ("сплитсистема",),
    "углошлифовальная машина": ("ушм", "болгарка"),
    "лакокрасочные материалы": ("лкм",),
}


@dataclass
class CatalogRecord:
    name: str
    category: str
    attributes: str


def _find_default_catalog() -> Path | None:
    for pattern in RAW_CATALOG_GLOBS:
        matches = sorted(PROJECT_ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def _resolve_catalog_path(catalog_path: str | None) -> Path:
    if catalog_path:
        path = Path(catalog_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл датасета: {path}")
        return path
    detected = _find_default_catalog()
    if detected is None:
        raise FileNotFoundError(
            "Не нашел файл СТЕ для генерации синонимов. "
            "Передай путь через --catalog-path."
        )
    return detected


def _row_has_header(row: list[str]) -> bool:
    normalized = {normalize_text(cell) for cell in row}
    return bool({"ste_id", "clean_name", "category"} & normalized) or any(
        marker in normalized for marker in {"наименование сте", "категория сте"}
    )


def _detect_csv_dialect(catalog_path: Path) -> csv.Dialect:
    sample = catalog_path.read_text(encoding="utf-8-sig", errors="ignore").replace("\x00", "")[:8192]
    try:
        return csv.Sniffer().sniff(sample, delimiters=";,")
    except csv.Error:
        dialect = csv.excel()
        dialect.delimiter = ";"
        return dialect


def _iter_clean_csv_lines(handle: Iterable[str]) -> Iterable[str]:
    for line in handle:
        yield line.replace("\x00", "")


def iter_catalog_records(catalog_path: Path) -> Iterable[CatalogRecord]:
    dialect = _detect_csv_dialect(catalog_path)
    with catalog_path.open("r", encoding="utf-8-sig", newline="") as handle:
        first_line = handle.readline()
        if not first_line:
            return
        reader = csv.reader(
            itertools.chain([first_line.replace("\x00", "")], _iter_clean_csv_lines(handle)),
            dialect,
        )
        first_row = next(reader, None)
        if not first_row:
            return
        if _row_has_header(first_row):
            handle.seek(0)
            dict_reader = csv.DictReader(_iter_clean_csv_lines(handle), dialect=dialect)
            for row in dict_reader:
                name = (
                    row.get("clean_name")
                    or row.get("normalized_name")
                    or row.get("Наименование СТЕ")
                    or row.get("name")
                    or ""
                )
                category = (
                    row.get("category")
                    or row.get("normalized_category")
                    or row.get("Категория СТЕ")
                    or ""
                )
                attributes = (
                    row.get("attribute_keys")
                    or row.get("key_tokens")
                    or row.get("Характеристики СТЕ")
                    or ""
                )
                yield CatalogRecord(name=str(name), category=str(category), attributes=str(attributes))
            return

        current_row = first_row
        while current_row:
            name = current_row[1] if len(current_row) > 1 else ""
            category = current_row[2] if len(current_row) > 2 else ""
            attributes = current_row[3] if len(current_row) > 3 else ""
            yield CatalogRecord(name=name, category=category, attributes=attributes)
            current_row = next(reader, None)


def _looks_like_bad_alias(normalized: str) -> bool:
    tokens = tokenize(normalized)
    if not tokens:
        return True
    if len(tokens) > 5:
        return True
    if STOPWORDS & set(tokens):
        return True
    if any(not token.isalpha() for token in tokens):
        return True
    if BAD_ALIAS_TOKENS & set(tokens):
        return True
    if all(len(token) <= 2 for token in tokens):
        return True
    if len("".join(tokens)) > 40:
        return True
    return False


def _is_acronym_like_alias(alias_raw: str, normalized_alias: str) -> bool:
    compact_raw = re.sub(r"[\s().,_/-]+", "", alias_raw)
    compact_normalized = normalized_alias.replace(" ", "")
    if not compact_raw or not compact_normalized:
        return False
    if len(compact_normalized) < 2 or len(compact_normalized) > 16:
        return False
    if re.fullmatch(r"[IVXLCDMivxlcdm]+", compact_raw):
        return False
    if compact_raw.isupper():
        return True
    if re.fullmatch(r"[A-Za-z0-9]{2,16}", compact_raw):
        return True
    return False


def _extract_parenthetical_alias_pairs(text: str) -> list[tuple[str, str, bool]]:
    pairs: list[tuple[str, str, bool]] = []
    if not text:
        return pairs
    for match in PAREN_ALIAS_RE.finditer(text):
        base_raw = match.group("base").strip(" ,;-")
        alias_raw = match.group("alias").strip(" ,;-")
        base = normalize_text(base_raw)
        alias = normalize_text(alias_raw)
        if not base or not alias or base == alias:
            continue
        alias_tokens = tokenize(alias)
        base_tokens = tokenize(base)
        alias_compact = alias.replace(" ", "")
        base_compact = base.replace(" ", "")
        if len(alias_tokens) > 2 or len(base_tokens) > 6:
            continue
        if len(alias_tokens) == 2 and max(len(token) for token in alias_tokens) > 7:
            continue
        if len(alias_compact) > 18:
            continue
        if len(alias_compact) >= len(base_compact):
            continue
        if not _is_acronym_like_alias(alias_raw, alias):
            continue
        if _looks_like_bad_alias(base) or _looks_like_bad_alias(alias):
            continue
        alias_is_short = len(alias_tokens) == 1 and 2 <= len(alias_compact) <= 8
        pairs.append((alias, base, alias_is_short))
    return pairs


def _target_tokens_supported(value: str, token_counter: Counter[str]) -> bool:
    tokens = tokenize(value)
    if not tokens:
        return False
    return all(token_counter.get(token, 0) > 0 for token in tokens)


def _score_target(value: str, token_counter: Counter[str]) -> tuple[int, int, str]:
    tokens = tokenize(value)
    token_frequency_sum = sum(token_counter.get(token, 0) for token in tokens)
    return (len(tokens), -token_frequency_sum, value)


def _append_mapping(
    token_synonyms: dict[str, list[str]],
    phrase_synonyms: dict[str, list[str]],
    source: str,
    target: str,
) -> None:
    source_normalized = normalize_text(source)
    target_normalized = normalize_text(target)
    if not source_normalized or not target_normalized or source_normalized == target_normalized:
        return
    if stem_tokens(tokenize(source_normalized)) == stem_tokens(tokenize(target_normalized)):
        return
    if len(tokenize(source_normalized)) == 1:
        token_synonyms.setdefault(source_normalized, []).append(target_normalized)
        return
    phrase_synonyms.setdefault(source_normalized, []).append(target_normalized)


def generate_synonyms_payload(
    catalog_path: Path,
    min_auto_pair_count: int = 2,
    max_targets_per_source: int = 8,
) -> dict[str, object]:
    token_counter: Counter[str] = Counter()
    auto_pairs: Counter[tuple[str, str]] = Counter()
    auto_short_aliases: set[tuple[str, str]] = set()
    row_count = 0

    for record in iter_catalog_records(catalog_path):
        row_count += 1
        combined_text = " ".join(part for part in [record.name, record.category, record.attributes] if part)
        for token in tokenize(combined_text):
            if len(token) <= 1:
                continue
            token_counter[token] += 1
        for source, target, is_short_alias in _extract_parenthetical_alias_pairs(record.category):
            auto_pairs[(source, target)] += 1
            if is_short_alias:
                auto_short_aliases.add((source, target))

    token_synonyms: dict[str, list[str]] = {}
    phrase_synonyms: dict[str, list[str]] = {}

    for source, targets in MANUAL_TOKEN_SYNONYMS.items():
        for target in targets:
            if not _target_tokens_supported(target, token_counter):
                continue
            _append_mapping(token_synonyms, phrase_synonyms, source, target)

    for source, targets in MANUAL_PHRASE_SYNONYMS.items():
        for target in targets:
            if not _target_tokens_supported(target, token_counter):
                continue
            _append_mapping(token_synonyms, phrase_synonyms, source, target)

    for (source, target), count in auto_pairs.items():
        if count < min_auto_pair_count and (source, target) not in auto_short_aliases:
            continue
        if not _target_tokens_supported(target, token_counter):
            continue
        _append_mapping(token_synonyms, phrase_synonyms, source, target)
        _append_mapping(token_synonyms, phrase_synonyms, target, source)

    def finalize(items: dict[str, list[str]]) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for source, targets in sorted(items.items()):
            unique_targets = unique_preserve_order(target for target in targets if target and target != source)
            supported_targets = [target for target in unique_targets if _target_tokens_supported(target, token_counter)]
            supported_targets.sort(key=lambda value: _score_target(value, token_counter))
            if max_targets_per_source > 0:
                supported_targets = supported_targets[:max_targets_per_source]
            if supported_targets:
                normalized[source] = supported_targets
        return normalized

    finalized_token_synonyms = finalize(token_synonyms)
    finalized_phrase_synonyms = finalize(phrase_synonyms)
    return {
        "metadata": {
            "catalog_path": str(catalog_path),
            "rows_scanned": row_count,
            "dataset_token_vocab_size": len(token_counter),
            "manual_token_rule_count": len(MANUAL_TOKEN_SYNONYMS),
            "manual_phrase_rule_count": len(MANUAL_PHRASE_SYNONYMS),
            "auto_parenthetical_pair_count": sum(1 for count in auto_pairs.values() if count >= min_auto_pair_count),
        },
        "phrase_synonyms": finalized_phrase_synonyms,
        "token_synonyms": finalized_token_synonyms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a large search synonym dictionary from the STE dataset.")
    parser.add_argument("--catalog-path", help="Path to raw or preprocessed STE catalog CSV.")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="Where to save the generated JSON.")
    parser.add_argument("--min-auto-pair-count", type=int, default=2, help="How many times an auto alias pair must appear before it is kept.")
    parser.add_argument("--max-targets-per-source", type=int, default=8, help="Hard cap for how many targets to keep per source term.")
    args = parser.parse_args()

    catalog_path = _resolve_catalog_path(args.catalog_path)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = generate_synonyms_payload(
        catalog_path=catalog_path,
        min_auto_pair_count=args.min_auto_pair_count,
        max_targets_per_source=args.max_targets_per_source,
    )
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "metadata": payload["metadata"],
                "phrase_synonyms": len(payload["phrase_synonyms"]),
                "token_synonyms": len(payload["token_synonyms"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
