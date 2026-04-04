# TenderHack Search Platform

Локальный прототип поиска по СТЕ для Портала поставщиков Москвы.

Ключевой принцип репозитория: проект должен подниматься локально из исходных датасетов, без внешних API и без синтетики в критическом поисковом контуре.

## Что внутри

- `backend/` - FastAPI backend
- `frontend/` - React/Vite веб-интерфейс
- `src/tenderhack/` - search, personalization, offers, query understanding
- `scripts/` - preprocessing, asset build, training, bootstrap, dataset normalization
- `datasets/raw/` - исходные CSV в нормализованной структуре
- `datasets/reference/` - словари синонимов и тестовые запросы
- `datasets/processed/` - локально собираемые артефакты поиска и персонализации

## Требования

- Linux
- Python 3.11 или 3.12 предпочтительно
- Node.js 20+
- npm 10+

## Входные данные

Каноническая структура входных файлов:

- `datasets/raw/ste_catalog.csv`
- `datasets/raw/contracts.csv`

Если у вас raw CSV пока лежат в legacy-местах, сначала нормализуйте layout:

```bash
cd ProjectForTenderHack
python3 scripts/normalize_datasets.py --copy
```

Скрипт подхватит:

- `СТЕ_*.csv` из корня репозитория
- `Контракты_*.csv` из корня репозитория
- legacy `data/reference/*`
- legacy `data/processed/*`

## Быстрый старт

```bash
cd ProjectForTenderHack
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
make install
make install-frontend
make normalize-datasets
make bootstrap
```

После сборки артефактов у вас есть два режима запуска.

Режим разработки, если нужно быстро итерироваться по UI:

```bash
cd ProjectForTenderHack
source .venv/bin/activate
make backend
```

```bash
cd ProjectForTenderHack
make frontend
```

Адреса в dev-режиме:

- backend: `http://127.0.0.1:8000`
- frontend: `http://127.0.0.1:5173`

Production-like запуск без frontend dev-server:

```bash
cd ProjectForTenderHack
source .venv/bin/activate
make frontend-build
make backend-prod
```

В этом режиме backend раздаёт собранный `frontend/dist` и API из одного процесса:

- приложение: `http://127.0.0.1:8000`
- API health: `http://127.0.0.1:8000/api/health`

## Bootstrap pipeline

Единая точка сборки:

```bash
python3 scripts/bootstrap_local.py \
  --ste-path ./datasets/raw/ste_catalog.csv \
  --contracts-path ./datasets/raw/contracts.csv \
  --train-personalization
```

Что делает bootstrap:

1. Прогоняет `scripts/preprocess_data.py`
2. Собирает `datasets/processed/tenderhack_search.sqlite`
3. Собирает `datasets/processed/tenderhack_preprocessed.sqlite`
4. Строит `ste_offer_lookup` и `ste_offer_candidates` из контрактов
5. Обучает локальную fastText-модель
6. Опционально обучает personalization ranker

Упрощённый режим без fastText:

```bash
make bootstrap-lite
```

## Runtime данные

Проект использует только локальные артефакты:

- `datasets/processed/tenderhack_search.sqlite`
- `datasets/processed/tenderhack_preprocessed.sqlite`
- `datasets/processed/tenderhack_fasttext.bin`
- `artifacts/personalization_model.cbm` опционально

Если `personalization_model.cbm` отсутствует, runtime использует rule-based personalization baseline. Это штатный fallback, а не mock API.

## API

Основные endpoint'ы:

- `POST /api/auth/login`
- `POST /api/search`
- `POST /api/event`
- `GET /api/items/{ste_id}`
- `GET /api/items/{ste_id}/offers`
- `POST /api/cart/add`
- `GET /api/cart`
- `POST /api/cart/create-procurement`

## Оферты

Оферты в выдаче не генерируются искусственно. Они строятся из исторических контрактов:

- `ste_offer_lookup` - агрегированная сводка по СТЕ
- `ste_offer_candidates` - кандидаты поставщиков по `(ste_id, supplier_inn, supplier_region)`

Если таблица кандидатов отсутствует, backend возвращает единственный исторический агрегат по СТЕ, а не синтетические предложения.

## Проверка

```bash
cd ProjectForTenderHack
source .venv/bin/activate
make test
make frontend-build
```

Важно: `vite preview` не используется как production server. Собранный `dist` обслуживается backend'ом после `make frontend-build`.

## Принцип разработки

- никакие внешние API не используются
- retrieval, personalization и explainability разделены по слоям
- backend должен запускаться из локальных артефактов без ручных правок кода
- любые fallback-механизмы должны быть детерминированными и основанными на локальных данных
