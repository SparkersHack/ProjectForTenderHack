# ProjectForTenderHack

Прототип персонализированного поиска СТЕ для Tender Hack и Портала поставщиков.

Проект решает задачу поиска по каталогу СТЕ с учетом:

- опечаток в запросе
- синонимов и словоформ
- семантической близости
- истории закупок заказчика
- поведения пользователя в текущей сессии
- объяснимого ранжирования результатов

В репозитории есть рабочий backend на `FastAPI`, frontend на `React + Vite`, офлайн-пайплайн подготовки данных и артефакты для персонализации.

## Короткий технический отчёт

### Что уже реализовано
Система умеет:
- искать товары по названию, категории, ключевым токенам и атрибутам;
- исправлять опечатки и учитывать синонимы;
- использовать semantic expansion через `fastText`;
- персонализировать выдачу по истории закупок пользователя;
- учитывать динамические сигналы сессии: клики, открытия карточек, корзину;
- возвращать explanations и подсказки для autocomplete.

### Архитектура проекта
Проект разделён на несколько понятных слоёв:

1. `data/` и `scripts/`  
   Очистка исходных CSV, агрегация контрактов, построение поисковых и аналитических артефактов.

2. `src/tenderhack/`  
   Основная runtime-логика:
   - поиск;
   - semantic expansion;
   - personalization runtime;
   - online session state;
   - cache;
   - offer lookup;
   - generation of explanations.

3. `src/features/`, `src/training/`, `src/eval/`  
   Offline-слой для feature engineering, обучения и оценки personalization ranking.

4. `backend/`  
   FastAPI API для логина, поиска, suggestions и событий.

5. `frontend/`  
   React + TypeScript интерфейс с поисковой строкой, карточками, профилем и personalized autocomplete.

### Главные достоинства архитектуры

#### 1. Модульность
Поиск, персонализация, кэш, online state, обучение и UI разделены на независимые модули. Это упрощает развитие проекта и снижает риск регрессий.

#### 2. Гибридный поиск
В проекте сочетаются:
- lexical search;
- typo correction;
- synonym expansion;
- semantic expansion через `fastText`.

Это делает выдачу устойчивой к опечаткам, сокращениям и неточным формулировкам.

#### 3. Персонализация по истории закупок
Выдача учитывает:
- ранее купленные СТЕ;
- часто покупаемые категории;
- регион пользователя;
- поставщиков из истории;
- сигналы текущей сессии.

За счёт этого поиск работает как персональный инструмент заказчика, а не как общий каталог.

#### 4. Динамический rerank
Система умеет менять порядок выдачи прямо во время сессии. Сейчас приоритет выстраивается так:
- `cart` и `click` сигналы;
- история прошлых покупок;
- базовая текстовая релевантность.

#### 5. Объяснимость
Backend возвращает reason codes и человекочитаемые причины ранжирования. Это важно и для демо, и для доверия к персонализированной выдаче.

#### 6. Offline + Online слой
Подготовка данных и обучение отделены от online inference. Это сильная архитектурная сторона проекта: модель и runtime можно развивать независимо.

#### 7. Готовность к масштабированию
Текущую архитектуру можно эволюционировать:
- от `SQLite` к `Postgres`;
- от локального cache к `Redis`;
- от текущего retrieval к `OpenSearch`;
- от fallback scoring к полной обученной personalization model.

#### 8. Типизированный контракт frontend/backend
Frontend и backend связаны через понятные контракты данных. Это позволяет быстро менять UI и backend-логику без хаотичной связности.

#### 9. Наличие тестов
В проекте есть unit/API-тесты на:
- поиск;
- персонализацию;
- suggestions;
- исправление опечаток;
- динамическую выдачу.

Это особенно важно при быстрой разработке в формате хакатона.

### Что можно доработать
- Подключить полноценную обученную personalization model вместо fallback scoring там, где она ещё не включена в runtime.
- Вынести долговременное хранение событий и профилей в `Postgres/Redis`.
- Перенести retrieval-слой в `OpenSearch` для production-ready hybrid search.
- Добавить отдельные endpoint’ы `/item/{id}` и `/explain`.
- Улучшить extraction атрибутов и фильтров из пользовательского запроса.
- Добавить online-метрики и A/B-сравнение качества.
- Ускорить preload за счёт более лёгкого runtime-path и предрассчитанных preview.

## Что уже умеет система

- искать СТЕ по названию, категории и ключевым токенам
- исправлять типовые опечатки
- расширять запрос синонимами
- использовать семантический слой для сокращений и близких формулировок
- персонализировать выдачу по ИНН заказчика, региону и истории закупок
- учитывать онлайн-сигналы из текущей сессии
- понижать категории после быстрого отказа
- возвращать причины, почему товар показан выше
- отдавать search suggestions и corrected query

## Архитектура

Система состоит из четырех основных слоев.

### 1. Data pipeline

Скрипты из `scripts/` подготавливают данные и поисковые артефакты:

- `scripts/preprocess_data.py`
- `scripts/build_search_assets.py`
- `scripts/build_offer_assets.py`
- `scripts/train_fasttext.py`
- `scripts/run_personalization_pipeline.py`

На выходе формируются:

- очищенные и агрегированные таблицы в `data/processed/`
- SQLite-базы для поиска и персонализации
- справочники и артефакты в `artifacts/`
- офлайн-отчеты в `reports/`

### 2. Search layer

Основной поиск реализован в `src/tenderhack/search.py`.

Pipeline запроса:

1. нормализация текста
2. typo correction
3. расширение синонимами
4. семантическое расширение
5. retrieval через SQLite FTS / BM25
6. дополнительный lexical + semantic scoring

### Пересборка большого словаря синонимов

Поиск использует файл `data/reference/search_synonyms.json`. Теперь его можно
пересобирать автоматически из вашего каталога СТЕ:

```bash
python3 scripts/generate_search_synonyms.py --catalog-path ./СТЕ_20260403.csv
```

Скрипт:

- вытаскивает словарь токенов из датасета;
- добавляет только высокоточные доменные alias -> canonical синонимы;
- автоматически подхватывает только acronym-like алиасы и сокращения из категорий;
- не смешивает синонимы с гиперонимами, атрибутами и просто "похожими" словами;
- сохраняет итоговый JSON в `data/reference/search_synonyms.json`.

### 3. Personalization layer

Персонализация реализована в:

- `src/tenderhack/personalization.py`
- `src/tenderhack/personalization_runtime.py`
- `src/tenderhack/personalization_model.py`
- `src/training/`

Используемые сигналы:

- история закупок конкретного заказчика
- любимые категории и ранее купленные СТЕ
- региональные предпочтения
- популярность у похожих заказчиков
- действия в текущей сессии: открытие карточки, клик

### 4. Product layer

- backend: `backend/main.py`
- frontend: `frontend/`

Frontend работает с живым API и умеет отправлять пользовательские события обратно в backend, чтобы демонстрировать динамическое изменение выдачи.

## Структура репозитория

```text
backend/                 FastAPI API
frontend/                React + Vite клиент
scripts/                 подготовка данных и сборка артефактов
src/tenderhack/          поиск, персонализация, online state, cache
src/training/            offline personalization pipeline
data/reference/          синонимы и тестовые запросы
artifacts/               feature spec, explain rules, offline metrics
reports/                 data contract и offline evaluation
tests/                   unit и API тесты
```

## Технологии

- Python
- FastAPI
- SQLite
- CatBoost
- fastText
- Redis или in-memory cache
- React
- TypeScript
- Zustand
- Vite

## Быстрый запуск

### 1. Установить backend-зависимости

```bash
python3 -m pip install --user -r requirements-backend.txt
python3 -m pip install --user -r requirements-semantic.txt
python3 -m pip install --user -r requirements-personalization.txt
```

Примечание:

- для `catboost` рекомендуется `Python 3.11` или `3.12`
- в `requirements-personalization.txt` CatBoost помечен как optional runtime dependency для финального ranker

### 2. Подготовить данные

Если `data/processed/` уже собрана, шаг можно пропустить.

```bash
python3 scripts/preprocess_data.py
python3 scripts/build_search_assets.py
python3 scripts/build_offer_assets.py
python3 scripts/train_fasttext.py
```

Для офлайн-персонализации:

```bash
python3 scripts/run_personalization_pipeline.py
```

### 3. Запустить backend

```bash
python3 -m uvicorn backend.main:app --reload
```

Проверка:

```bash
curl http://127.0.0.1:8000/api/health
```

### 4. Запустить frontend

```bash
cd frontend
npm install
npm run dev
```

По умолчанию frontend ходит в:

```text
http://127.0.0.1:8000
```

Если backend поднят на другом адресе:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## API

Сейчас в backend доступны:

- `GET /api/health`
- `POST /api/auth/login`
- `POST /api/search`
- `GET /api/search/suggestions`
- `POST /api/event`

### Что возвращает поиск

Поиск отдает:

- список товаров
- общее количество кандидатов
- `correctedQuery`, если запрос был исправлен
- персонализированные причины показа через `reasonToShow`

## Демо-сценарии

Для быстрой демонстрации удобно использовать:

- ИНН: `7714338609`
- запрос `парацетомол 500 мг`
- запрос `флешка`
- запрос `мфу`
- запрос `канцелярские ручки`

Хороший live demo выглядит так:

1. пользователь логинится по ИНН
2. вводит запрос с опечаткой или синонимом
3. получает исправленную и персонализированную выдачу
4. открывает карточку или добавляет товар в корзину
5. повторно ищет и видит, что ранжирование изменилось

## Тестирование

В репозитории есть unit- и API-тесты:

- `tests/test_search.py`
- `tests/test_personalization.py`
- `tests/test_api.py`

Запуск:

```bash
python3 -m unittest tests.test_search tests.test_personalization tests.test_api
```

## Офлайн-персонализация

В проекте есть отдельный офлайн-контур ранжирования:

- data contract: `reports/data_contract.md`
- offline evaluation: `reports/offline_eval.md`
- feature spec: `artifacts/feature_spec.json`
- explain rules: `artifacts/explain_rules.json`
- defaults: `artifacts/feature_defaults.json`
- training config: `artifacts/train_config.yaml`

Stable inference entrypoint:

```python
predict_personalization(candidates, user_profile, query_features)
```

## Ограничения текущего состояния

- исходные большие датасеты не закоммичены в репозиторий
- без реальных входных CSV офлайн-оценка в `reports/offline_eval.md` остается в статусе `missing_input`
- если таблица `ste_offer_lookup` не собрана, поиск все равно работает, но цена и поставщик могут быть оценочными
- часть качества демонстрации зависит от наличия подготовленных `data/processed/` артефактов

## Текущее позиционирование проекта

Текущий MVP показывает:

- интеллектуальный поиск по СТЕ
- динамическую персонализацию
- объяснимость ранжирования
- рабочую связку backend + frontend
- основу для дальнейшего обучения ML-ранкера на офлайн-истории закупок
