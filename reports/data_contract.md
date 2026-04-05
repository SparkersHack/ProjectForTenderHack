# Personalization Data Contract

## Expected Inputs

- STE catalog: `data/processed/ste_catalog_clean.csv` or raw `СТЕ_*.csv`.
- Contracts: `data/processed/contracts_clean.csv` or raw `Контракты_*.csv`.
- Primary join key between contracts and STE catalog: `ste_id`.

## Required STE Columns

- `ste_id`
- `clean_name`
- `normalized_name`
- `category`
- `normalized_category`
- `attribute_keys`
- `attribute_count`
- `key_tokens`

## Required Contract Columns

- `contract_item_name`
- `contract_id`
- `ste_id`
- `contract_datetime`
- `contract_amount`
- `customer_inn`
- `customer_name`
- `customer_region`
- `supplier_inn`
- `supplier_name`
- `supplier_region`

## Validation Status

- Status: `ready`
- STE path: `/Users/nikolaj/ProjectForTenderHack/data/processed/ste_catalog_clean.csv`
- Contracts path: `/Users/nikolaj/ProjectForTenderHack/Контракты_20260403.csv`

## Observed STE Dataset

- Rows: 542,993
- Unique `ste_id`: 537,314
- Duplicate `ste_id`: 5,679
- Missing `ste_id`: 0
- Missing `clean_name`: 0
- Missing `normalized_name`: 0
- Missing `category`: 0
- Missing `normalized_category`: 0
- Missing `attribute_keys`: 0
- Missing `attribute_count`: 0
- Missing `key_tokens`: 0

## Observed Contracts Dataset

- Rows: 2,010,224
- Unique compound keys (`contract_id`, `ste_id`, `customer_inn`): 2,009,457
- Duplicate compound keys: 767
- Missing `contract_item_name`: 0
- Missing `contract_id`: 0
- Missing `ste_id`: 0
- Missing `contract_datetime`: 0
- Missing `contract_amount`: 0
- Missing `customer_inn`: 0
- Missing `customer_name`: 0
- Missing `customer_region`: 0
- Missing `supplier_inn`: 0
- Missing `supplier_name`: 0
- Missing `supplier_region`: 0

## Join Checks

- Join key: `ste_id`
- Contract rows without STE catalog match: 1561
- Catalog STE without contract history: 40505

## Missing Data Notes

- Empty `customer_inn` is replaced with `UNKNOWN` during loading.
- Empty `customer_region` and `supplier_region` are replaced with `UNKNOWN` during loading.
- Invalid `contract_datetime` rows are skipped because ranking splits are time-based.
- Invalid or empty `contract_amount` is retained as `0.0` and flagged in the contract summary.

## Samples

### STE

- `{'ste_id': '1222958', 'clean_name': 'Флеш накопитель SMARTBUY Glossy USB 2.0 черный 16 Гб', 'category': 'Usb-накопители твердотельные (флеш-драйвы)', 'attribute_count': 12}`
- `{'ste_id': '1223536', 'clean_name': 'Флеш накопитель SMARTBUY Crown USB 2.0 черный 4 Гб', 'category': 'Usb-накопители твердотельные (флеш-драйвы)', 'attribute_count': 12}`
- `{'ste_id': '36025516', 'clean_name': 'Набор реагентов для определения активности аланинаминотрансферазы в сыворотке и плазме крови кинетическим УФ-методом (АЛТ IFCC), (вариант комплектации 1) , №8079', 'category': 'Аланинаминотрансфераза (алт) ивд, набор, ферментный спектрофотометрический анализ', 'attribute_count': 4}`
- `{'ste_id': '17466132', 'clean_name': 'Парацетамол табл. 500 мг бл N 10x1 Еврофарм Россия', 'category': 'АНАЛЬГЕТИКИ,N02', 'attribute_count': 8}`
- `{'ste_id': '17517067', 'clean_name': 'Трамадол табл. 100 мг бан N 20x1 Органика Россия', 'category': 'АНАЛЬГЕТИКИ,N02', 'attribute_count': 8}`

### Contracts

- `{'contract_id': '203255114', 'ste_id': '36047183', 'customer_inn': '7417005438', 'contract_date': '2024-06-05', 'contract_amount': 1200.0}`
- `{'contract_id': '194452173', 'ste_id': '35778463', 'customer_inn': '5911021660', 'contract_date': '2022-05-26', 'contract_amount': 16800.0}`
- `{'contract_id': '214274949', 'ste_id': '46358579', 'customer_inn': '5944020101', 'contract_date': '2026-02-11', 'contract_amount': 6825.0}`
- `{'contract_id': '199009522', 'ste_id': '24120140', 'customer_inn': '8907002558', 'contract_date': '2023-06-17', 'contract_amount': 205000.0}`
- `{'contract_id': '193681593', 'ste_id': '35754688', 'customer_inn': '8901007006', 'contract_date': '2022-04-01', 'contract_amount': 73130.0}`
