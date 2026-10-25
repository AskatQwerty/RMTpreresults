"""Создать читаемый notebook; запуск выполняется отдельно execute_notebook.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# 01. Подготовка данных Home Credit Default Risk

**Единица анализа — одна текущая заявка `SK_ID_CURR`.** Этот notebook выполняет аудит, технический прогон и полную детерминированную подготовку данных. Модели здесь не обучаются.

Основание: локальный `Aubakirov_Dagystanov_A2_Methods_v2.pdf`, Methods and Materials — Assignment 2, особенно стр. 2. Принятый план: **7 моделей × 4 уровня × 5 повторов = 140 основных обучений**. Для каждой из 28 пар «модель + уровень» отдельно будет выполнен Randomized Search: 20 конфигураций × stratified 3-fold CV, выбор по log-loss; затем параметры фиксируются на пять повторов. Это 1 680 CV-обучений кандидатов сверх 140 основных обучений (без учёта внутренних обучений ансамблей, refit и калибровки).

Преобразования, требующие оценки по данным — медианы, масштабирование, one-hot/иное обучаемое кодирование, отбор признаков — **остаются внутри Pipeline** следующего этапа. На Home Credit здесь не вызывается `.fit()`, не выполняются ресемплинг классов и подбор признаков по TARGET. Проверка поведения будущего Pipeline использует только искусственные данные.

Исходники реализации находятся в `src/data_audit.py` и `src/data_preparation.py`; notebook явно запускает каждый этап и сохраняет реальные результаты. CSV читаются без изменений. Окружение зафиксировано в `requirements.txt` (`python -m pip install -r requirements.txt`); из корня проекта можно выполнить `python scripts/execute_notebook.py` или Jupyter Run All.""")

code(r"""from pathlib import Path
import sys, os, time, json, gc, shutil
import pandas as pd
import psutil
from IPython.display import display

ROOT = Path.cwd()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'src/data_preparation.py').is_file(), 'Запускайте notebook из корня проекта или notebooks/'
sys.path.insert(0, str(ROOT))
from src.data_audit import audit_sources
from src.data_preparation import (source_manifest, connect, run_preparation,
                                  compare_pilot_to_full, dump_json)

DATA_DIR = Path(os.environ.get('HOME_CREDIT_DATA_DIR', r'C:\Users\askat\Study\RMT\home-credit-default-risk'))
REPORTS = ROOT / 'reports'
PILOT_DIR = ROOT / 'data/processed/pilot'
FULL_DIR = ROOT / 'data/processed/full'
PILOT_SIZE = 3000
RUN_FULL = True
RUN_START = time.perf_counter()
REPORTS.mkdir(exist_ok=True)
print('Исходные CSV:', DATA_DIR)
print('RAM всего/доступно, GiB:', round(psutil.virtual_memory().total/1024**3,2),
      round(psutil.virtual_memory().available/1024**3,2))
print('Свободный диск, GiB:', round(shutil.disk_usage(ROOT).free/1024**3,2))
print('DuckDB: memory_limit=2GB; threads=2; последовательные источники, spill на диск.')""")

md("""## 1. Наличие файлов и неизменность источников

Проверяются все восемь обязательных файлов. При отсутствии хотя бы одного выполнение останавливается с его именем. Для каждого сохраняется SHA-256; после подготовки хэши будут вычислены повторно. `application_test.csv` и `sample_submission.csv` не входят в исследуемую размеченную популяцию и не используются.

Сырые CSV, Parquet и временные файлы исключены через `.gitignore`. В Git допустимы только код, notebook и небольшие обезличенные сводки.""")
code("""manifest_before = source_manifest(DATA_DIR)
dump_json(REPORTS / 'source_sha256_before.json', manifest_before)
display(pd.DataFrame(manifest_before)[['file','bytes']])""")

md("""## 2. Полный аудит до построения признаков

Аудит читает **все строки каждого CSV**, даже если последующий технический прогон использует 3 000 клиентов. Типы сначала определяет DuckDB на 100 000 строках, затем всё содержимое читается строго: `ignore_errors=false`, несовместимый тип вызовет ошибку. Для каждого столбца сохраняются тип, число/доля пропусков, min/max числовых значений; для дат — положительные, нулевые значения и заглушки `365243`.

Проверяются точные повторы полных строк и кандидатных ключей. Для таблицы платежей повтор ключа может означать частичную оплату: автоматического удаления нет. Уникальность application, bureau и previous_application — обязательное условие соединений. История клиентов без размеченной текущей заявки учитывается в аудите, но не включается в X.

Словарь читается в `cp1252`, имена очищаются от пробелов. Документированный `SK_BUREAU_ID` сопоставляется с реальным `SK_ID_BUREAU`. Поля, описанные в словаре, но отсутствующие в CSV, не создаются. Аудит связей различает отсутствующего родителя, неверного владельца договора и клиента вне `application_train`.""")
code("""audit_connection = connect(ROOT / 'tmp/audit_spill')
try:
    audit = audit_sources(audit_connection, DATA_DIR, REPORTS / 'audit')
finally:
    audit_connection.close()
table_audit = pd.read_csv(REPORTS / 'audit/table_summary.csv')
display(table_audit)
display(pd.read_csv(REPORTS / 'audit/target_balance.csv'))
key_tables = table_audit[table_audit.table.isin(['application_train.csv','bureau.csv','previous_application.csv'])]
assert (key_tables.key_duplicate_surplus == 0).all()
assert (key_tables.rows_with_null_key == 0).all()
assert not audit['dictionary_unmatched_columns']
print('Все реальные столбцы сопоставлены со словарём; ключи родительских таблиц уникальны.')""")
code("""column_audit = pd.read_csv(REPORTS / 'audit/column_audit.csv')
display(column_audit.sort_values('missing_fraction', ascending=False)[
    ['table','column','dtype','missing_count','missing_fraction']].head(25))
display(column_audit[column_audit.positive_count.notna()][
    ['table','column','min','max','positive_count','zero_count','placeholder_365243_count']])
relationships = pd.read_csv(REPORTS / 'audit/relationships.csv')
display(relationships)
print('Расхождения словаря:')
display(pd.read_csv(REPORTS / 'audit/dictionary_extra_rows.csv')[['Table','Row','Description']])""")

md("""## 3. Временная доступность и правила признаков

В датасете нет календарных дат и полного журнала изменений. Поэтому правила ниже обеспечивают воспроизводимое отсечение по имеющимся полям, но не доказывают point-in-time происхождение каждого атрибута. Анкета, включая `EXT_SOURCE_1/2/3`, трактуется как снимок при подаче заявки; словарь не раскрывает даты внешних оценок. Финальные состояния, для которых нельзя обосновать дату доступности, не используются.

| Источник | Правило и смысл |
|---|---|
| application | `DAYS_EMPLOYED=365243` и другие положительные фактические даты превращаются в NULL; для неизвестной даты трудоустройства есть индикатор. Возраст и пол удаляются из X до создания отношений. |
| bureau | `DAYS_CREDIT < 0` и `DAYS_CREDIT_UPDATE < 0`: решение и последняя информация строго до дня заявки. `DAYS_CREDIT_ENDDATE` — оставшаяся договорная длительность, поэтому положительные плановые сроки сохраняются. `DAYS_ENDDATE_FACT` допускается только <0. |
| bureau_balance | `MONTHS_BALANCE < 0`, затем связь через допустимые bureau → application. Месяц 0 исключён: точный день снимка неизвестен. |
| POS / credit card | `MONTHS_BALANCE < 0`, прямая связь по `SK_ID_CURR`. Словарь POS разрешает месяц 0 как снимок при заявке, но для единообразного строгого режима он не используется. Количество будущих взносов — известное в историческом снимке плановое обязательство. |
| installments | Фактический `DAYS_ENTRY_PAYMENT < 0`; неизвестные/будущие фактические платежи не используются. Положительный `DAYS_INSTALMENT` не удаляется автоматически: это может быть срок уже совершённой предоплаты. `365243` не является датой. |
| previous | Только `DAYS_DECISION < 0`. `DAYS_FIRST_DUE` и `DAYS_LAST_DUE_1ST_VERSION` — договорные даты, положительные сохраняются, `365243` → NULL. `DAYS_FIRST_DRAWING`, `DAYS_LAST_DUE`, `DAYS_TERMINATION` исключены из X из-за неоднозначности доступности/обновлений. |

Связь `SK_ID_PREV` с previous проверяется аудитом. Отсутствие опубликованной предыдущей заявки не отменяет корректную историю с прямым `SK_ID_CURR` и допустимой собственной датой; Level 3 не зависит от наличия таблицы Level 4. Все исключения считаются и сохраняются в `temporal_filters.csv`; конечный объём — в `history_coverage.csv`.

Полные дубли диагностируются, но без идентификатора события не удаляются вслепую. По взносам сначала суммируются частичные `AMT_PAYMENT` внутри `(SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER)`. Предписанная сумма берётся один раз. Если внутри ключа различаются срок или сумма, этот взнос исключается из агрегатов, число таких случаев сохраняется. Доля просрочки относится **к наблюдаемым оплаченным взносам**; полностью отсутствующие неоплаченные обязательства из этой таблицы не восстанавливаются.

Суммы взносов и частичных оплат агрегируются как **DECIMAL(24,3)**: у реального CSV проверены максимум три десятичных знака. Это исключает ложную недоплату из-за порядка сложения floating-point значений на границе равенства. Перед преобразованием проверяется отсутствие большей точности; финальные числовые признаки сохраняются в обычном числовом формате. Просрочка по дате последнего платежа — наблюдаемое приближение: поздняя переплата может пометить взнос, полностью погашенный вовремя, как поздний. Это не точная дата погашения долга. Нулевой или отрицательный знаменатель отношения даёт NULL, а не бесконечность. Mean/min/max историй описывают типичный уровень и диапазон; средние по месячным таблицам взвешены числом наблюдаемых договор-месяцев. Индикаторы `HAS_*` означают наличие пригодных агрегатов. При отсутствии истории они равны 0, а суммы, доли и количества остаются NULL.""")

md("""## 4. Четыре вложенных уровня и технический прогон

1. **Level 1:** application и пять отношений (credit/income, annuity/income, credit/goods, annuity/credit, income/family), индикатор неизвестной даты трудоустройства.
2. **Level 2:** Level 1 + bureau и bureau_balance: число кредитов, задолженность, просрочки, сроки и доли месячных статусов.
3. **Level 3:** Level 2 + POS и card: история взносов и просрочек, обороты, платежи и использование карточного лимита.
4. **Level 4:** Level 3 + installments и previous: оплата/просрочка наблюдаемых взносов, суммы и исходы предыдущих заявок.

Выбираем 3 000 **клиентов** стратифицированно с seed 2026 и читаем всю связанную с ними историю, включая bureau_balance через bureau. Нельзя брать первые строки каждой таблицы независимо. Технический набор сохраняется отдельно и не заменяет полный результат. Каждый этап сначала сохраняет агрегат с уникальным `SK_ID_CURR`, затем используется left join; размножение строк запрещено assertions.""")
code("""pilot = run_preparation(DATA_DIR, PILOT_DIR, REPORTS / 'pilot', sample_size=PILOT_SIZE)
display(pd.DataFrame(pilot['shapes']))
display(pd.DataFrame(pilot['history_coverage']))
print('Пилот: все проверки пройдены:', pilot['checks'])
print('Время пилота, секунд:', pilot['elapsed_seconds'])""")

md("""## 5. Полная подготовка и сохранённые разбиения

Полная подготовка запускается только после успешного пилота и проверки ресурсов. DuckDB ограничен 2 GB и может выгружать промежуточные операции на диск. В памяти одновременно не удерживаются все исторические таблицы и четыре X. Для старта требуется ≥3 GiB доступной RAM и ≥10 GiB свободного диска; при нехватке notebook явно запишет причину и оставшееся действие.

Разбиение выполняется один раз для полной популяции: `train_test_split(test_size=0.20, stratify=TARGET, random_state=42)`. Development служит для последующего поиска; test сохраняется для итоговой оценки. Пять 90%-ных стратифицированных подвыборок development используют seeds 42…46. Оставшиеся 10% каждого повтора не становятся новым test. Подвыборки могут пересекаться между собой — это ожидается. Дополнительно сохраняется общий 3-fold для 28 будущих поисков. После сохранения следует загружать эти ID, а не генерировать разбиения отдельно по уровням/моделям.

Fairness хранится отдельно: исходные `CODE_GENDER`, `DAYS_BIRTH`, `AGE_YEARS=-DAYS_BIRTH/365.25`, группы `<25`/`>=25`, неизвестный пол отдельной группой. Возраст в годах приближённый: календарных дат нет. Ни эти значения, ни ID, ни TARGET не входят в X. Исключение прямых защищённых атрибутов не гарантирует отсутствия косвенных прокси.""")
code("""available_gib = psutil.virtual_memory().available / 1024**3
disk_gib = shutil.disk_usage(ROOT).free / 1024**3
full = None
full_skip_reason = None
if not RUN_FULL:
    full_skip_reason = 'RUN_FULL=False: для завершения установить True и выполнить этапы 5–7.'
elif available_gib < 3 or disk_gib < 10:
    full_skip_reason = f'Недостаточно ресурсов: RAM={available_gib:.2f} GiB, диск={disk_gib:.2f} GiB. Освободить ресурсы и повторить этапы 5–7.'
else:
    full = run_preparation(DATA_DIR, FULL_DIR, REPORTS / 'full', sample_size=None)
if full is None:
    print('ПОЛНЫЙ ЗАПУСК НЕ ВЫПОЛНЕН:', full_skip_reason)
else:
    print('ПОЛНЫЙ ЗАПУСК ВЫПОЛНЕН:', full['n_applicants'], 'текущих заявок')
    display(pd.DataFrame(full['shapes']))
    display(pd.DataFrame(full['splits']))
    display(pd.DataFrame(full['history_coverage']))
    print('Противоречивых взносов исключено:', full['installment_conflicting_obligations_excluded'])""")

md("""## 6. Проверки результата и формат файлов

`X_level_N.parquet` содержит **только признаки**, без идентификаторов даже в индексе. Строки всех четырёх файлов упорядочены по `SK_ID_CURR`; их точное соответствие хранится в `row_ids.parquet`. `target.parquet` и `fairness.parquet` содержат ID для безопасного совмещения. JSON-списки фиксируют порядок и вложенность признаков, а `feature_catalog.csv` сохраняет смысл и SQL-формулу каждого признака.

Автоматические проверки: уникальность заявки и ключей агрегатов; одинаковые клиенты и количество строк на всех уровнях; равенство общих признаков; отсутствие TARGET/ID/пола/возраста в X; непересечение development/test; принадлежность повторов development; отсутствие бесконечностей; NULL у отсутствующей истории. Ни один клиент не отбрасывается из-за отсутствующей истории.

Дополнительная проверка сравнивает значения пилота и полного результата для **тех же клиентов**, читая только 30 столбцов за раз. Это выявляет ошибочный выбор только части истории и зависимость детерминированных признаков от состава выборки. Допуск `1e-10` нужен для порядка суммирования floating-point агрегатов при параллельной обработке; это не обучение параметров.""")
code("""crosscheck = compare_pilot_to_full(PILOT_DIR, FULL_DIR) if full is not None else None
print('Сравнение пилота с полным набором:', crosscheck)
manifest_after = source_manifest(DATA_DIR)
assert manifest_after == manifest_before, 'Один из исходных CSV изменён!'
dump_json(REPORTS / 'source_sha256_after.json', manifest_after)
print('Все восемь исходных CSV неизменны: SHA-256 совпадают.')
dump_json(REPORTS / 'run_summary.json', {
    'processed_scope': 'full' if full is not None else 'technical_sample_only',
    'audit_seconds': audit['elapsed_seconds'], 'pilot': pilot, 'full': full,
    'full_skip_reason': full_skip_reason, 'pilot_full_comparison': crosscheck,
    'original_csv_sha256_unchanged': True,
    'total_elapsed_seconds': round(time.perf_counter()-RUN_START,3)
})
print('Проверки:', (full or pilot)['checks'])""")

md("""## 7. Итог и передача на этап моделирования

Следующая ячейка показывает фактически выполненный объём, размеры, баланс, время и перечень файлов. Большие данные находятся в `data/processed/full/`; промежуточные агрегаты — в `aggregates/`; ID — в `splits/`. Пилот находится в соседней папке `pilot/`. Небольшие отчёты остаются в `reports/`.

Чтобы восстановить X по ID на следующем этапе:

```python
ids = pd.read_parquet(FULL_DIR / 'row_ids.parquet')['SK_ID_CURR']
X = pd.read_parquet(FULL_DIR / 'X_level_1.parquet')
X.index = pd.Index(ids, name='SK_ID_CURR')  # ID служит индексом, не входным столбцом
y = pd.read_parquet(FULL_DIR / 'target.parquet').set_index('SK_ID_CURR')['TARGET']
dev_ids = pd.read_parquet(FULL_DIR / 'splits/development_ids.parquet')['SK_ID_CURR']
X_dev, y_dev = X.loc[dev_ids], y.loc[dev_ids]
```

Fit медиан, индикаторов пропусков, категории и стандартизация выполняются **внутри каждого CV-train**. После поиска они заново fit на каждой 90%-ной train-подвыборке с уже фиксированными гиперпараметрами. Test не используется для этих решений; fairness-файл доступен только для предусмотренного протокола анализа/коррекции. Для всех моделей внутри уровня используется один и тот же список исходных признаков.""")
code("""result = full or pilot
print('Реально обработано:', result['scope'], '—', result['n_applicants'], 'заявок')
display(pd.DataFrame(result['shapes']).rename(columns={'features':'n_features','rows':'n_rows'}))
display(pd.DataFrame([{'TARGET=0': result['target_0'], 'TARGET=1': result['target_1'],
                       'доля TARGET=1': result['positive_rate']}]))
run_report = json.loads((REPORTS/'run_summary.json').read_text(encoding='utf-8'))
print('Время, секунд: аудит=', audit['elapsed_seconds'], 'пилот=', pilot['elapsed_seconds'],
      'полный=', full['elapsed_seconds'] if full else None,
      'весь notebook=', run_report['total_elapsed_seconds'])
generated = []
for directory in [ROOT/'data/processed', REPORTS, ROOT/'notebooks', ROOT/'src', ROOT/'scripts', ROOT/'tests']:
    for path in sorted(directory.rglob('*')):
        if path.is_file() and 'duckdb_tmp' not in str(path) and '__pycache__' not in str(path):
            generated.append({'file':str(path.relative_to(ROOT)), 'bytes':path.stat().st_size})
for name in ['.gitignore','requirements.txt']:
    path = ROOT/name
    generated.append({'file':name, 'bytes':path.stat().st_size})
inventory = pd.DataFrame(generated)
inventory.to_csv(REPORTS/'generated_files.csv', index=False)
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 120):
    display(inventory)
if full_skip_reason:
    print(full_skip_reason)""")

from quality_notebook_cells import quality_cells
nb['cells'] = cells + quality_cells()
nb['metadata'] = {'kernelspec': {'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
                  'language_info': {'name':'python','version':'3.12'}}
(ROOT/'notebooks').mkdir(exist_ok=True)
nbf.write(nb, ROOT/'notebooks/01_data_preparation.ipynb')
print('Notebook created:',len(cells),'cells')
