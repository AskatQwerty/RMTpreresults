"""Ячейки дополнения к 01: только аудит существующих данных, без их пересоздания."""
import nbformat as nbf


def quality_cells():
    cells=[]
    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source,metadata={'tags':['data_quality_v1']}))
    def code(source):
        cells.append(nbf.v4.new_code_cell(source,metadata={'tags':['data_quality_v1']}))

    md("""## 8. Дополнительная проверка качества перед моделированием

Этот раздел использует **существующие** `development_ids.parquet`, четыре X и списки признаков. Разбиения не создаются повторно, признаки не перезаписываются. SHA-256 всех подготовленных файлов проверяется до и после. Распределения, процентили и смысловые ограничения анализируются только на **development**; test участвует лишь в механических проверках сохранности и в запрошенной разбивке 2 905 строк с неполным ключом. Решения о выбросах по test не принимаются.

Результаты записываются в `reports/quality/`. Число **значений исходной истории**, **клиентов** и **ячеек агрегатов** указывается раздельно: это разные единицы счёта.""")
    code(r"""from pathlib import Path
import sys, os, json, time
import pandas as pd
from IPython.display import display

QA_ROOT=Path.cwd()
if QA_ROOT.name=='notebooks':
    QA_ROOT=QA_ROOT.parent
sys.path.insert(0,str(QA_ROOT))
from src.data_quality import (artifact_manifest, development_profile, semantic_checks, sentinel_checks)
from src.payment_quality import audit_payment_keys
from src.fold_preprocessing import make_numeric_pipeline, numeric_pipeline_smoke_check
from src.data_preparation import dump_json, source_manifest

QA_DATA=Path(os.environ.get('HOME_CREDIT_DATA_DIR',r'C:\Users\askat\Study\RMT\home-credit-default-risk'))
QA_PREP=QA_ROOT/'data/processed/full'
QA_REPORT=QA_ROOT/'reports/quality'
QA_REPORT.mkdir(parents=True,exist_ok=True)
QA_STARTED=time.perf_counter()
qa_manifest_before=artifact_manifest(QA_PREP)
qa_sources_before=source_manifest(QA_DATA)
dump_json(QA_REPORT/'prepared_sha256_before.json',qa_manifest_before)
print('Существующих подготовленных файлов под защитой SHA-256:',len(qa_manifest_before))""")

    md("""### 8.1. Числовые распределения development

Вычисляются точные линейно интерполированные P1, P25, медиана, P75, P99, P99,9 по непустым значениям; отдельно — пропуски, min/max, нули, отрицательные значения и бесконечности. Вычисление медианы для **отчёта** не заполняет данные и не обучает преобразование. Читается по 24 столбца, чтобы не держать в памяти все уровни.

Полная таблица — `numeric_development_summary.csv`; `first_level` и `levels` показывают принадлежность признака уровням. Общие признаки профилируются один раз. Отдельно различаются полностью пустой столбец, один уникальный **наблюдаемый** уровень при наличии пропусков и истинно постоянный столбец без пропусков. Ничего не удаляется по этой глобальной сводке.""")
    code("""qa_profile=development_profile(QA_PREP,QA_REPORT)
display(pd.DataFrame(qa_profile['levels']))
numeric_summary=pd.read_csv(QA_REPORT/'numeric_development_summary.csv')
print('Полная сводка:',QA_REPORT/'numeric_development_summary.csv')
display(numeric_summary.sort_values('missing_fraction',ascending=False)[
    ['feature','first_level','missing','missing_fraction','min','p01','p25','median','p75','p99','p999','max','zeros','negatives']].head(25))
print('Пустые и постоянные столбцы (все типы):')
display(pd.read_csv(QA_REPORT/'empty_and_constant_columns.csv'))
print('Полностью пустые числовые:',qa_profile['all_missing_numeric'])
print('Истинно постоянные числовые без пропусков:',qa_profile['constant_fully_numeric'])""")

    md("""### 8.2. Смысловые ограничения и предлагаемые действия

Строгие правила: положительные текущие доход/кредит; неотрицательные номинальные суммы; целочисленные счётчики семьи и детей; доли `*_SHARE` в `[0,1]`. Пустое значение не считается отрицательным или нулевым. Отсутствие нарушения знака не подтверждает достоверность экстремального значения.

Отношения сумм **не обязаны лежать в `[0,1]`**: переплата, превышение лимита и отличие одобренного кредита от запрошенного допустимы. Словарь не задаёт сопоставимые периоды дохода и аннуитета — `annuity/income > 1` само по себе не основание исправления. Отрицательные остатки и денежные движения могут быть переплатой/сторно, поэтому общее правило «все AMT < 0 удалить» не применяется.

У каждого правила в `semantic_source_checks.csv` указаны SQL-условие, обоснование, действие, статус и число затронутых исходных значений/клиентов. Конкретное предложение для отрицательного первоначального взноса — заменить его NULL **до** агрегации после подтверждения знаковой конвенции; вычисляется точное число изменившихся агрегатов, но замена сейчас не применяется. Для неоднозначных лимитов/движений предлагается уточнение источника, а не неподтверждённое исправление.

Редкие большие значения сохраняются. P99/P99,9 служат описанием хвоста, а не порогами удаления, clipping или winsorization.""")
    code("""qa_semantics=semantic_checks(QA_DATA,QA_PREP,QA_REPORT)
display(pd.read_csv(QA_REPORT/'semantic_source_checks.csv')[
    ['source','field','condition','affected_values','affected_clients','classification','proposed_action_ru','reason_ru']])
display(pd.read_csv(QA_REPORT/'ratio_checks_development.csv'))
focus=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','CNT_FAM_MEMBERS','CNT_CHILDREN',
       'APP_CREDIT_TO_INCOME','APP_ANNUITY_TO_INCOME','APP_CREDIT_TO_GOODS','APP_ANNUITY_TO_CREDIT','APP_INCOME_PER_MEMBER']
display(numeric_summary[numeric_summary.feature.isin(focus)][
    ['feature','missing','min','p01','p25','median','p75','p99','p999','max','zeros','negatives']])
print('Оценка воздействия предлагаемой замены, не применено:')
display(pd.DataFrame([qa_semantics['proposed_fix']]))
print('Хвост дохода, сохранён без изменений:',qa_semantics['income_tail'])""")

    md("""### 8.3. Заглушки: источник → правило → сохранённый результат

Для `DAYS_EMPLOYED=365243` проверяется не только отсутствие числа в X, но и соответствие **каждой заявки** исходнику: значение стало NULL, `APP_EMPLOYMENT_DATE_UNKNOWN=1`, остальные допустимые даты сохранены. Аналогично проверяются остальные фактические даты анкеты.

В previous `365243` заменён NULL в используемых плановых датах `DAYS_FIRST_DUE` и `DAYS_LAST_DUE_1ST_VERSION`. Их AVG/MIN/MAX заново вычисляются из очищенных строк и сверяются с сохранёнными агрегатами для всех подготовленных заявок. `DAYS_FIRST_DRAWING`, `DAYS_LAST_DUE`, `DAYS_TERMINATION` вообще не входят в X: для них «после» означает отсутствие признака, а не число заполненных значений. Положительные плановые сроки сохраняются.

В таблице до/после NULL подсчитаны на одном наборе **исходных строк**, поскольку сравнивать число исходных NULL с числом клиентских агрегатов некорректно. Проверка агрегатов выделена в отдельный файл. `placeholder_before` везде означает только `365243`. Вспомогательный `positive_before` для фактических дат application считает все положительные значения (включая заглушку), а для плановых дат previous/bureau — только положительные значения без заглушки: эти поля проверяются по разным смысловым правилам. Общая механическая проверка итоговых временных признаков не подбирает параметры по test.""")
    code("""qa_sentinels=sentinel_checks(QA_DATA,QA_PREP,QA_REPORT)
display(pd.read_csv(QA_REPORT/'sentinels_before_after.csv')[
    ['source','field','scope','source_rows','placeholder_before','null_before','positive_before',
     'null_after_same_rows','placeholder_after_same_rows','saved_feature_status']])
display(pd.read_csv(QA_REPORT/'saved_date_aggregate_checks.csv'))
print('Проверки сохранённых заглушек:',qa_sentinels)""")

    md("""### 8.4. Неполный ключ 2 905 строк installments_payments

Проверяемый кандидатный ключ строки платежа полностью:
`(SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER, DAYS_ENTRY_PAYMENT)`.
`SK_ID_CURR` дополнительно задаёт владельца и связь с текущей заявкой. Это **не гарантированно уникальный ID платёжного события**: несколько частичных платежей могут иметь одну дату и один кандидатный ключ.

Ключ взноса при подготовке:
`(SK_ID_CURR, SK_ID_PREV, NUM_INSTALMENT_VERSION, NUM_INSTALMENT_NUMBER)`.
Дата фактической оплаты в него не входит. Суммы частичных оплат складываются точно, а предписанная сумма учитывается один раз. Полные дубли диагностируются; повторение кандидатного ключа не приводит к удалению.

Показываются пропуски всех восьми полей, разбивка по уже существующим split, наличие других оплат того же взноса и сохранность клиентов. Строка без даты фактической оплаты не проходит критерий `<0`; неизвестная сумма не заполняется нулём. Из такой строки нельзя достоверно заключить ни факт оплаты, ни факт неоплаты. В конце проверяется сохранение **числа исходных платёжных строк** и **точной суммы оплат** в агрегированных взносах.""")
    code("""qa_payments=audit_payment_keys(QA_DATA,QA_PREP,QA_REPORT)
display(pd.read_csv(QA_REPORT/'installments_incomplete_key_fields.csv'))
display(pd.read_csv(QA_REPORT/'installments_incomplete_key_patterns.csv'))
display(pd.DataFrame(qa_payments['by_split']))
display(pd.DataFrame(qa_payments['obligation_recovery']))
display(pd.DataFrame(qa_payments['client_retention']))
display(pd.DataFrame([qa_payments['partial_payment_preservation']]).T)
print('Неполных строк:',qa_payments['missing_key_rows'],'Пустые поля:',qa_payments['null_fields'])""")

    md("""### 8.5. Полностью пустой столбец внутри train-фолда

Даже если во всём development пустых столбцов нет, отдельный train-фолд или 90%-ный повтор может их получить. Для следующего этапа подготовлена фабрика `src.fold_preprocessing.make_numeric_pipeline(scale=False)`; она возвращает **необученный** объект.

Числовая ветка: `SimpleImputer(strategy='median', keep_empty_features=True)`. Если столбец полностью пуст в **текущем train**, медианы не существует: используется заранее заданный технический fallback **0**, столбец сохраняется. Непустые значения этого столбца в validation/test остаются реальными; заполняются только пропуски. Это не оценка по development/test. Для случая без обучающих наблюдений модель не получает оснований оценивать эффект этой переменной.

Параллельно `MissingIndicator(features='all')` создаёт индикатор исходного пропуска для каждого числового столбца, включая новый пропуск в validation. Выходная числовая ширина стабильна — **2 × p** — и не зависит от того, какие пропуски встретились в train. При `scale=True` StandardScaler обучается только в ветке значений, внутри train; индикаторы не масштабируются. После fallback полностью пустого столбца scaler имеет mean=0, scale=1.

Категориальный кодировщик и возможный отбор признаков остаются в полном Pipeline следующего этапа. Списки исходных признаков четырёх уровней сохраняются. Фабрику нужно включить в ColumnTransformer и Pipeline **до передачи в CV**, а не вызвать fit_transform на полном development заранее. Здесь выполняется только проверка на трёх искусственных столбцах; Home Credit не подвергается fit, заполнению, масштабированию или кодированию.""")
    code("""unfitted_numeric_pipeline=make_numeric_pipeline(scale=False)
qa_pipeline=numeric_pipeline_smoke_check(QA_REPORT/'fold_pipeline_checks.json')
print('Демонстрация только на искусственных данных:')
display(pd.DataFrame([qa_pipeline]))""")

    md("""### 8.6. Итог проверки и оставшиеся вопросы

**Уже исправлено на этапе подготовки:** заглушки дат заменены NULL либо поля исключены; неиспользуемые фактические будущие/недатированные платежи отсечены; деление на неположительный знаменатель даёт NULL; частичные оплаты объединены без потери суммы и без повторения номинала взноса. Десятичная арифметика исключает ложную недоплату на границе равенства.

**Сохранено:** редкие большие доходы/суммы, отношения >1, положительные договорные сроки, нулевые суммы отказанных/отменённых прошлых заявок, отрицательные остатки/движения до уточнения их учётного смысла; два признака с постоянным наблюдаемым нулём и пропусками. Качество не означает автоматическую нормализацию распределений.

**Остаётся внутри Pipeline:** медианы по train-фолду, обработка пустого train-столбца с fallback=0 и индикаторами, при необходимости масштабирование, обучаемое кодирование и отбор. Ни один из этих этапов не fit на полном Home Credit.

**Открытые вопросы:** подтверждение дохода 117 млн и единиц/периодов дохода; знак отрицательного первоначального взноса, снятий, лимитов и остатков; неизбежная неполнота сведений о взносах без даты/суммы оплаты; точное время появления обновляемых исторических значений. При недостаточном словаре неоднозначные значения помечены для проверки, а не молча исправлены.

Последняя ячейка подтверждает неизменность всех X, списков признаков, target, fairness, ID и агрегатов; исходные CSV тоже проверяются по SHA-256.""")
    code("""qa_manifest_after=artifact_manifest(QA_PREP)
assert qa_manifest_before==qa_manifest_after,'Подготовленные данные или разбиения изменились!'
assert qa_sources_before==source_manifest(QA_DATA),'Исходные CSV изменились!'
dump_json(QA_REPORT/'prepared_sha256_after.json',qa_manifest_after)
qa_result={'scope':'development_quality_audit_existing_prepared_data',
           'profile':qa_profile,'semantics':qa_semantics,'sentinels':qa_sentinels,
           'payments':qa_payments,'pipeline':qa_pipeline,
           'prepared_files_sha256_unchanged':True,'source_csv_sha256_unchanged':True,
           'X_and_split_files_modified':0,'global_fit_on_home_credit':False,
           'elapsed_seconds':round(time.perf_counter()-QA_STARTED,3)}
dump_json(QA_REPORT/'quality_summary.json',qa_result)
print('Development:',qa_profile['development_rows'],'числовых признаков:',qa_profile['numeric_features'])
print('Строгих нарушений:',qa_semantics['hard_rule_violations'],'нарушений правил отношений:',qa_semantics['ratio_rule_violations'])
print('X, уровни, разбиения и исходники: SHA-256 неизменны. Значения не исправлялись по результатам этого аудита.')
print('Время дополнительного аудита, секунд:',qa_result['elapsed_seconds'])
quality_files=pd.DataFrame([{'file':p.name,'bytes':p.stat().st_size} for p in sorted(QA_REPORT.iterdir()) if p.is_file()])
display(quality_files)""")
    return cells
