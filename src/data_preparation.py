"""Детерминированные признаки: SQL-агрегаты с ограничением RAM, без fit()."""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_audit import register_sources

SEEDS = [42, 43, 44, 45, 46]
FILES = ["application_train.csv", "bureau.csv", "bureau_balance.csv",
         "POS_CASH_balance.csv", "credit_card_balance.csv", "installments_payments.csv",
         "previous_application.csv", "HomeCredit_columns_description.csv"]
FORBIDDEN = {"TARGET", "SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU", "CODE_GENDER",
             "DAYS_BIRTH", "AGE_YEARS", "AGE_GROUP"}


def dump_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sql_path(path):
    return "'" + str(Path(path).resolve()).replace("\\", "/").replace("'", "''") + "'"


def qi(name):
    return '"' + name.replace('"', '""') + '"'


def source_manifest(data_dir):
    """SHA-256 до/после запуска доказывает неизменность входных CSV."""
    out = []
    missing = [f for f in FILES if not (Path(data_dir) / f).is_file()]
    if missing:
        raise FileNotFoundError("Отсутствуют обязательные файлы: " + ", ".join(missing))
    for name in FILES:
        p = Path(data_dir) / name
        h = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024**2), b""):
                h.update(block)
        out.append({"file": name, "bytes": p.stat().st_size, "sha256": h.hexdigest()})
    return out


def connect(work_dir, memory="2GB"):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    c.execute(f"SET memory_limit='{memory}'")
    c.execute("SET threads=2")
    c.execute("SET preserve_insertion_order=false")
    c.execute(f"SET temp_directory={sql_path(work_dir)}")
    return c


def safe_ratio(numerator, denominator):
    # Нулевой/отрицательный знаменатель не трактуем как осмысленную величину.
    return f"CASE WHEN ({denominator}) > 0 THEN ({numerator})::DOUBLE / ({denominator}) END"


def run_preparation(data_dir, out_dir, report_dir, sample_size=None):
    """Один полный цикл: связанные клиенты -> признаки -> разбиения -> проверки."""
    started = time.perf_counter()
    out_dir, report_dir = Path(out_dir), Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    agg_dir = out_dir / "aggregates"
    agg_dir.mkdir(exist_ok=True)
    c = connect(out_dir / "duckdb_tmp")
    register_sources(c, Path(data_dir))
    for source, key in [("application","SK_ID_CURR"),("bureau","SK_ID_BUREAU")]:
        n, unique_n, nonnull = c.execute(f"SELECT count(*),count(DISTINCT {key}),count({key}) FROM raw_{source}").fetchone()
        assert n == unique_n == nonnull, f"Неуникальный/пустой ключ {source}.{key}"
    ids = c.execute("SELECT SK_ID_CURR, TARGET FROM raw_application ORDER BY SK_ID_CURR").df()
    assert ids.SK_ID_CURR.is_unique and ids.TARGET.isin([0, 1]).all()
    if sample_size is not None:
        chosen, _ = train_test_split(ids, train_size=sample_size, stratify=ids.TARGET, random_state=2026)
        ids = chosen.sort_values("SK_ID_CURR").reset_index(drop=True)
    c.register("selected_ids", ids[["SK_ID_CURR"]])
    c.execute("CREATE VIEW cohort_app AS SELECT a.* FROM raw_application a INNER JOIN selected_ids s USING(SK_ID_CURR)")
    # Отбираем клиентов, а не первые N строк исторических CSV.
    for table in ["bureau", "pos", "cc", "installments", "previous"]:
        c.execute(f"CREATE VIEW cohort_{table} AS SELECT h.* FROM raw_{table} h INNER JOIN selected_ids s USING(SK_ID_CURR)")
    c.execute("CREATE VIEW cohort_bb AS SELECT h.* FROM raw_bb h INNER JOIN cohort_bureau b USING(SK_ID_BUREAU)")

    metadata, time_checks, coverage, stage_times = [], [], [], []
    feature_blocks = {}

    def export(query, path):
        c.execute(f"COPY ({query}) TO {sql_path(path)} (FORMAT PARQUET, COMPRESSION ZSTD)")

    def stage(name, query, description, expressions=None):
        t = time.perf_counter()
        path = agg_dir / f"{name}.parquet"
        export(query, path)
        c.execute(f"CREATE OR REPLACE VIEW agg_{name} AS SELECT * FROM read_parquet({sql_path(path)})")
        schema = c.execute(f"DESCRIBE agg_{name}").df()
        cols = schema.column_name.tolist()
        n, u = c.execute(f"SELECT count(*), count(DISTINCT SK_ID_CURR) FROM agg_{name}").fetchone()
        assert n == u, f"Размножение заявок: {name}"
        feature_blocks[name] = [col for col in cols if col != "SK_ID_CURR"]
        for col in feature_blocks[name]:
            metadata.append({"feature": col, "block": name, "description_ru": description,
                             "sql_expression": (expressions or {}).get(col, "Исходное поле application; см. словарь колонок")})
        stage_times.append({"stage": name, "seconds": round(time.perf_counter() - t, 3), "rows": n})
        print(f"{name}: {n:,} клиентов, {len(cols)-1} признаков; {stage_times[-1]['seconds']:.1f} с", flush=True)

    def record_cutoff(name, source, condition, note):
        n, eligible = c.execute(f"SELECT count(*), count(*) FILTER (WHERE {condition}) FROM {source}").fetchone()
        time_checks.append({"table": name, "cohort_rows": n, "time_eligible_rows": eligible,
                            "excluded_rows": n-eligible, "rule": condition, "meaning_ru": note})

    # Application: сохраняем исходные категориальные строки, без fit кодировщика.
    app_cols = c.execute("DESCRIBE cohort_app").df().column_name.tolist()
    exprs = {}
    for col in app_cols:
        if col in FORBIDDEN:
            continue
        if col == "DAYS_EMPLOYED":
            exprs[col] = "CASE WHEN DAYS_EMPLOYED <= 0 THEN DAYS_EMPLOYED END"
        elif col in {"DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"}:
            exprs[col] = f"CASE WHEN {col} <= 0 THEN {col} END"
        else:
            exprs[col] = qi(col)
    exprs["APP_EMPLOYMENT_DATE_UNKNOWN"] = "CAST(DAYS_EMPLOYED IS NULL OR DAYS_EMPLOYED > 0 AS INTEGER)"
    for label, num, den in [
        ("APP_CREDIT_TO_INCOME", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
        ("APP_ANNUITY_TO_INCOME", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
        ("APP_CREDIT_TO_GOODS", "AMT_CREDIT", "AMT_GOODS_PRICE"),
        ("APP_ANNUITY_TO_CREDIT", "AMT_ANNUITY", "AMT_CREDIT"),
        ("APP_INCOME_PER_MEMBER", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    ]:
        exprs[label] = safe_ratio(num, den)
    stage("application", "SELECT SK_ID_CURR, " + ", ".join(f"{v} AS {qi(k)}" for k,v in exprs.items()) + " FROM cohort_app ORDER BY SK_ID_CURR",
          "Анкета на момент заявки. Отношения описывают долговую нагрузку, стоимость кредита и доход на члена семьи; не используют возраст/пол.", exprs)

    def aggregate(name, source, numeric, extras, note, entity=None):
        """Средние взвешены строками источника; min/max описывают диапазон истории."""
        expressions = {f"{name.upper()}_RECORD_COUNT": "count(*)"}
        if entity:
            expressions[f"{name.upper()}_CONTRACT_COUNT"] = f"count(DISTINCT {entity})"
        for col in numeric:
            for stat in ["avg", "min", "max"]:
                value = f"{stat}({qi(col)})"
                # Decimal остаётся точным внутри обязательств/предикатов;
                # конечные числовые признаки совместимы с sklearn/Arrow.
                if name == "installments" and col in {"AMOUNT_DUE", "AMOUNT_PAID"}:
                    value = f"CAST({value} AS DOUBLE)"
                expressions[f"{name.upper()}_{col}_{stat.upper()}"] = value
        expressions.update({f"{name.upper()}_{key}": value for key,value in extras.items()})
        select = ", ".join(f"{v} AS {qi(k)}" for k,v in expressions.items())
        stage(name, f"SELECT SK_ID_CURR, {select} FROM {source} GROUP BY SK_ID_CURR ORDER BY SK_ID_CURR", note, expressions)

    # Для фактических событий: строго до дня заявки. Для месячных снимков: <0.
    bureau_rule = "DAYS_CREDIT < 0 AND DAYS_CREDIT_UPDATE < 0"
    record_cutoff("bureau", "cohort_bureau", bureau_rule,
                  "Не используем выданный/обновлённый в день заявки или позднее кредит; дата обновления должна быть известна. Плановый срок может быть положительным.")
    c.execute(f"CREATE VIEW eligible_bureau AS SELECT * EXCLUDE(DAYS_ENDDATE_FACT, DAYS_CREDIT_ENDDATE), CASE WHEN DAYS_ENDDATE_FACT < 0 THEN DAYS_ENDDATE_FACT END AS DAYS_ENDDATE_FACT, NULLIF(DAYS_CREDIT_ENDDATE,365243) AS DAYS_CREDIT_ENDDATE FROM cohort_bureau WHERE {bureau_rule}")
    # Маленькая карта прошедших временную проверку bureau для BB.
    export("SELECT SK_ID_CURR, SK_ID_BUREAU FROM eligible_bureau", agg_dir / "bureau_id_map.parquet")
    c.execute(f"CREATE VIEW bureau_id_map AS SELECT * FROM read_parquet({sql_path(agg_dir / 'bureau_id_map.parquet')})")
    aggregate("bureau", "eligible_bureau",
              ["DAYS_CREDIT", "DAYS_CREDIT_UPDATE", "DAYS_CREDIT_ENDDATE", "DAYS_ENDDATE_FACT", "CREDIT_DAY_OVERDUE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_LIMIT", "AMT_CREDIT_SUM_OVERDUE", "AMT_ANNUITY"],
              {"ACTIVE_SHARE": "avg(CASE WHEN CREDIT_ACTIVE IS NOT NULL THEN CAST(CREDIT_ACTIVE='Active' AS INTEGER) END)",
               "CLOSED_SHARE": "avg(CASE WHEN CREDIT_ACTIVE IS NOT NULL THEN CAST(CREDIT_ACTIVE='Closed' AS INTEGER) END)",
               "DEBT_TO_CREDIT": safe_ratio("sum(AMT_CREDIT_SUM_DEBT)", "sum(AMT_CREDIT_SUM)"),
               "CREDIT_TYPE_COUNT": "count(DISTINCT CREDIT_TYPE)"},
              "Одна запись — кредит бюро; средние/экстремумы сумм, задолженности и дат. Доли активных/закрытых кредитов; совокупный долг / кредит.", "SK_ID_BUREAU")

    record_cutoff("bureau_balance", "cohort_bb", "MONTHS_BALANCE < 0",
                  "Месяц 0 исключён: неизвестно, какая часть месяца предшествовала заявке. Дополнительно используем только допустимые по датам bureau.")
    c.execute("CREATE VIEW eligible_bb AS SELECT b.SK_ID_CURR, h.* FROM raw_bb h INNER JOIN bureau_id_map b USING(SK_ID_BUREAU) WHERE MONTHS_BALANCE < 0")
    aggregate("bb", "eligible_bb", ["MONTHS_BALANCE"],
              {"DELINQUENT_SHARE": "avg(CAST(coalesce(STATUS IN ('1','2','3','4','5'),false) AS INTEGER))",
               "SEVERE_DELINQUENT_SHARE": "avg(CAST(coalesce(STATUS IN ('3','4','5'),false) AS INTEGER))",
               "CLOSED_SHARE": "avg(CAST(coalesce(STATUS='C',false) AS INTEGER))",
               "UNKNOWN_SHARE": "avg(CAST(STATUS='X' OR STATUS IS NULL AS INTEGER))",
               "MAX_DPD_STATUS": "max(try_cast(STATUS AS INTEGER))"},
              "Кредит-месяцы бюро через SK_ID_BUREAU: доли статусов 1–5 (просрочка), 3–5, C (закрыт), X (неизвестен). Доли считаются по всем наблюдаемым кредит-месяцам.", "SK_ID_BUREAU")

    for name in ["pos", "cc"]:
        record_cutoff(name, f"cohort_{name}", "MONTHS_BALANCE < 0", "Только полностью предшествующие месяцы; значения будущих плановых обязательств внутри снимка допустимы.")
        # Собственный отрицательный месяц уже подтверждает доступность снимка.
        # Отсутствие previous не означает отсутствие достоверной месячной истории.
        c.execute(f"CREATE VIEW eligible_{name} AS SELECT * FROM cohort_{name} WHERE MONTHS_BALANCE < 0")
    aggregate("pos", "eligible_pos", ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"],
              {"LATE_MONTH_SHARE": "avg(CAST(SK_DPD > 0 AS INTEGER))", "DPD_DEF_MONTH_SHARE": "avg(CAST(SK_DPD_DEF > 0 AS INTEGER))"},
              "POS: договор-месяцы, оставшиеся плановые взносы, просрочка. Средние и доли взвешены числом наблюдаемых месяцев.", "SK_ID_PREV")
    cc_nums = ["MONTHS_BALANCE", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT", "AMT_DRAWINGS_CURRENT", "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT", "AMT_INST_MIN_REGULARITY", "AMT_PAYMENT_CURRENT", "AMT_PAYMENT_TOTAL_CURRENT", "AMT_RECEIVABLE_PRINCIPAL", "AMT_RECIVABLE", "AMT_TOTAL_RECEIVABLE", "CNT_DRAWINGS_ATM_CURRENT", "CNT_DRAWINGS_CURRENT", "CNT_DRAWINGS_OTHER_CURRENT", "CNT_DRAWINGS_POS_CURRENT", "CNT_INSTALMENT_MATURE_CUM", "SK_DPD", "SK_DPD_DEF"]
    aggregate("cc", "eligible_cc", cc_nums,
              {"UTILIZATION_AVG": "avg(" + safe_ratio("AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL") + ")",
               "UTILIZATION_MAX": "max(" + safe_ratio("AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL") + ")",
               "LATE_MONTH_SHARE": "avg(CAST(SK_DPD > 0 AS INTEGER))"},
              "Карточные договор-месяцы: суммы, операции и просрочка; utilization = баланс / положительный лимит, затем среднее/максимум по снимкам.", "SK_ID_PREV")

    payment_rule = "DAYS_ENTRY_PAYMENT < 0 AND DAYS_ENTRY_PAYMENT <> 365243 AND DAYS_INSTALMENT IS NOT NULL AND DAYS_INSTALMENT <> 365243"
    record_cutoff("installments", "cohort_installments", payment_rule,
                  "Берём только датированные фактические платежи до заявки. Будущая договорная дата допустима при уже совершённой предоплате; не трактуем неизвестную дату как платёж.")
    c.execute(f"CREATE VIEW eligible_installments AS SELECT * FROM cohort_installments WHERE {payment_rule}")
    # Реальные CSV имеют не более 3 десятичных знаков в суммах. Проверяем это
    # при каждом запуске: не округляем молча другой вариант исходных данных.
    precision_violations = c.execute("""SELECT count(*) FROM cohort_installments
        WHERE AMT_PAYMENT <> round(AMT_PAYMENT,3)
           OR AMT_INSTALMENT <> round(AMT_INSTALMENT,3)""").fetchone()[0]
    assert precision_violations == 0, "Изменилась точность сумм: требуется пересмотреть DECIMAL scale"
    # Частичные платежи не должны повторно увеличивать сумму предписанного взноса.
    installment_conflicts = c.execute("""SELECT count(*) FROM (
      SELECT SK_ID_CURR,SK_ID_PREV,NUM_INSTALMENT_VERSION,NUM_INSTALMENT_NUMBER
      FROM eligible_installments GROUP BY ALL
      HAVING count(DISTINCT DAYS_INSTALMENT)>1 OR count(DISTINCT AMT_INSTALMENT)>1)""").fetchone()[0]
    # Если внутри одной версии противоречат сроки/суммы, сохраняем их в отчёт
    # и исключаем из obligation-метрик: MAX не должен молча изобретать график.
    export("""SELECT SK_ID_CURR,SK_ID_PREV,NUM_INSTALMENT_VERSION,NUM_INSTALMENT_NUMBER,
        count(*) AS PAYMENT_ROWS, max(DAYS_INSTALMENT) AS DUE_DAY,
        max(CAST(AMT_INSTALMENT AS DECIMAL(24,3))) AS AMOUNT_DUE,
        sum(CAST(AMT_PAYMENT AS DECIMAL(24,3))) AS AMOUNT_PAID,
        max(DAYS_ENTRY_PAYMENT) AS LAST_PAYMENT_DAY
        FROM eligible_installments GROUP BY SK_ID_CURR,SK_ID_PREV,NUM_INSTALMENT_VERSION,NUM_INSTALMENT_NUMBER
        HAVING count(DISTINCT DAYS_INSTALMENT)=1 AND count(DISTINCT AMT_INSTALMENT)=1""", agg_dir / "installment_obligations.parquet")
    c.execute(f"CREATE VIEW obligations AS SELECT *, greatest(LAST_PAYMENT_DAY-DUE_DAY,0) AS DAYS_LATE FROM read_parquet({sql_path(agg_dir / 'installment_obligations.parquet')})")
    aggregate("installments", "obligations", ["PAYMENT_ROWS", "DUE_DAY", "AMOUNT_DUE", "AMOUNT_PAID", "DAYS_LATE"],
              {"LATE_SHARE": "avg(CAST(LAST_PAYMENT_DAY > DUE_DAY AS INTEGER))",
               "UNDERPAID_SHARE": "avg(CASE WHEN DUE_DAY < 0 AND AMOUNT_PAID IS NOT NULL THEN CAST(AMOUNT_PAID < AMOUNT_DUE AS INTEGER) END)",
               "PAID_TO_DUE": safe_ratio("sum(AMOUNT_PAID)", "sum(AMOUNT_DUE)"),
               "MOST_RECENT_PAYMENT_DAY": "max(LAST_PAYMENT_DAY)"},
              "Взносы после объединения частичных платежей по договору, версии и номеру: просрочка по последней оплате, недоплата и paid/due. Только наблюдаемые оплаченные взносы, не вся совокупность обязательств; неоплаченные отсутствующие строки не восстанавливаем.", "SK_ID_PREV")

    record_cutoff("previous", "cohort_previous", "DAYS_DECISION < 0", "Предшествующие решения. Плановые даты могут быть положительными; 365243 заменяется NULL. Спорные финальные даты исключены.")
    c.execute("CREATE VIEW eligible_previous AS SELECT * EXCLUDE(DAYS_FIRST_DUE,DAYS_LAST_DUE_1ST_VERSION), NULLIF(DAYS_FIRST_DUE,365243) AS DAYS_FIRST_DUE, NULLIF(DAYS_LAST_DUE_1ST_VERSION,365243) AS DAYS_LAST_DUE_1ST_VERSION FROM cohort_previous WHERE DAYS_DECISION < 0")
    aggregate("previous", "eligible_previous", ["AMT_ANNUITY", "AMT_APPLICATION", "AMT_CREDIT", "AMT_DOWN_PAYMENT", "AMT_GOODS_PRICE", "DAYS_DECISION", "CNT_PAYMENT", "RATE_DOWN_PAYMENT", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION"],
              {"APPROVED_SHARE": "avg(CASE WHEN NAME_CONTRACT_STATUS IS NOT NULL THEN CAST(NAME_CONTRACT_STATUS='Approved' AS INTEGER) END)",
               "REFUSED_SHARE": "avg(CASE WHEN NAME_CONTRACT_STATUS IS NOT NULL THEN CAST(NAME_CONTRACT_STATUS='Refused' AS INTEGER) END)",
               "CREDIT_TO_REQUEST_AVG": "avg(" + safe_ratio("AMT_CREDIT", "AMT_APPLICATION") + ")"},
              "Предыдущие заявки: суммы, первоначальный срок, даты решения, доля одобрений/отказов. DAYS_FIRST_DRAWING, DAYS_LAST_DUE, DAYS_TERMINATION исключены из-за недостаточной информации о времени появления значения.", "SK_ID_PREV")

    # Реальный объём: все связанные строки входа и реально включённые записи.
    for name in ["bureau", "bb", "pos", "cc", "installments", "previous"]:
        n, clients = c.execute(f"SELECT count(*),count(DISTINCT SK_ID_CURR) FROM eligible_{name}").fetchone()
        raw_n = c.execute(f"SELECT count(*) FROM cohort_{name}").fetchone()[0]
        coverage.append({"table": name, "cohort_raw_rows": raw_n, "eligible_rows": n,
                         "eligible_clients": clients, "eligible_client_share": clients / len(ids)})

    # Базовая популяция определяется application, ни один клиент не отбрасывается.
    blocks_by_level = {1: ["application"], 2: ["application","bureau","bb"],
                       3: ["application","bureau","bb","pos","cc"],
                       4: ["application","bureau","bb","pos","cc","installments","previous"]}
    # Физический порядок Parquet — часть контракта с row_ids.parquet.
    # Финальные записи и проверки выполняются последовательно в явном ORDER BY.
    c.execute("SET preserve_insertion_order=true")
    c.execute("SET threads=1")
    level_features, shapes = {}, []
    for level, blocks in blocks_by_level.items():
        joined = "agg_application a"
        expressions = [f"a.{qi(col)}" for col in feature_blocks["application"]]
        names = feature_blocks["application"].copy()
        for block in blocks[1:]:
            joined += f" LEFT JOIN agg_{block} {block} USING(SK_ID_CURR)"
            expressions.extend(f"{block}.{qi(col)}" for col in feature_blocks[block])
            names.extend(feature_blocks[block])
            # Только структурный индикатор получает 0, сами агрегаты остаются NULL.
            flag = f"HAS_{block.upper()}"
            expressions.append(f"CAST({block}.SK_ID_CURR IS NOT NULL AS INTEGER) AS {flag}")
            names.append(flag)
            if level == min(l for l,b in blocks_by_level.items() if block in b):
                metadata.append({"feature": flag, "block": block,
                                 "description_ru": "Наличие пригодной истории после временной/структурной проверки; 0 не означает нулевые суммы.",
                                 "sql_expression": expressions[-1]})
        assert len(names) == len(set(names)) and not (set(names) & FORBIDDEN)
        assert not any(name.startswith("SK_ID_") for name in names)
        n, unique_n = c.execute(f"SELECT count(*),count(DISTINCT a.SK_ID_CURR) FROM {joined}").fetchone()
        assert n == unique_n == len(ids)
        current_ids = c.execute(f"SELECT a.SK_ID_CURR FROM {joined} ORDER BY a.SK_ID_CURR").fetchnumpy()["SK_ID_CURR"]
        assert np.array_equal(current_ids, ids.SK_ID_CURR.to_numpy())
        path = out_dir / f"X_level_{level}.parquet"
        export("SELECT " + ", ".join(expressions) + f" FROM {joined} ORDER BY a.SK_ID_CURR", path)
        actual_names = c.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_path(path)})").df().column_name.tolist()
        assert actual_names == names
        level_features[str(level)] = names
        dump_json(out_dir / f"features_level_{level}.json", names)
        shapes.append({"level": level, "rows": n, "features": len(names), "bytes": path.stat().st_size})
        if level > 1:
            assert names[:len(level_features[str(level-1)])] == level_features[str(level-1)]
        print(f"Level {level}: {n:,} x {len(names)}; {path.stat().st_size / 1024**2:.1f} MiB", flush=True)

    ids[["SK_ID_CURR"]].to_parquet(out_dir / "row_ids.parquet", index=False)
    ids.to_parquet(out_dir / "target.parquet", index=False)
    fairness = c.execute("SELECT SK_ID_CURR,CODE_GENDER,DAYS_BIRTH FROM cohort_app ORDER BY SK_ID_CURR").df()
    fairness["AGE_YEARS"] = (-fairness.DAYS_BIRTH / 365.25).where(fairness.DAYS_BIRTH < 0)
    fairness["AGE_GROUP"] = np.where(fairness.AGE_YEARS.isna(), "unknown", np.where(fairness.AGE_YEARS < 25, "<25", ">=25"))
    fairness["GENDER_GROUP"] = fairness.CODE_GENDER.where(fairness.CODE_GENDER.isin(["M","F"]), "unknown")
    fairness.to_parquet(out_dir / "fairness.parquet", index=False)
    assert np.array_equal(fairness.SK_ID_CURR, ids.SK_ID_CURR)

    # 80/20 фиксируется один раз; выборки не перегенерируются для моделей/уровней.
    dev, test = train_test_split(ids, test_size=0.20, stratify=ids.TARGET, random_state=42)
    dev, test = dev.sort_values("SK_ID_CURR"), test.sort_values("SK_ID_CURR")
    split_dir = out_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    split_summary = []

    def save_split(frame, name):
        assert frame.SK_ID_CURR.is_unique
        frame[["SK_ID_CURR"]].to_parquet(split_dir / f"{name}_ids.parquet", index=False)
        split_summary.append({"split": name, "n": len(frame), "target_0": int((frame.TARGET == 0).sum()),
                              "target_1": int((frame.TARGET == 1).sum()), "positive_rate": float(frame.TARGET.mean())})

    save_split(dev, "development")
    save_split(test, "test")
    assert set(dev.SK_ID_CURR).isdisjoint(test.SK_ID_CURR)
    assert set(dev.SK_ID_CURR) | set(test.SK_ID_CURR) == set(ids.SK_ID_CURR)
    repeat_signatures = []
    for seed in SEEDS:
        repeat, _ = train_test_split(dev, train_size=0.90, stratify=dev.TARGET, random_state=seed)
        repeat = repeat.sort_values("SK_ID_CURR")
        save_split(repeat, f"train_seed_{seed}")
        assert set(repeat.SK_ID_CURR) <= set(dev.SK_ID_CURR)
        assert set(repeat.SK_ID_CURR).isdisjoint(test.SK_ID_CURR)
        repeat_signatures.append(hashlib.sha256(repeat.SK_ID_CURR.to_numpy().tobytes()).hexdigest())
    assert len(set(repeat_signatures)) == 5
    # Для всех 28 поисков один и тот же заранее фиксированный 3-fold на development.
    folds = dev[["SK_ID_CURR"]].copy()
    folds["validation_fold"] = -1
    for fold, (_, val) in enumerate(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(dev, dev.TARGET)):
        folds.iloc[val, folds.columns.get_loc("validation_fold")] = fold
    folds.to_parquet(split_dir / "cv_folds.parquet", index=False)
    assert folds.validation_fold.isin([0,1,2]).all()

    # Проверяем записанные файлы, включая отсутствие бесконечностей и пропуски без истории.
    for level in range(1,5):
        path = out_dir / f"X_level_{level}.parquet"
        schema = c.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_path(path)})").df()
        num_cols = schema.loc[~schema.column_type.isin(["VARCHAR", "BOOLEAN"]), "column_name"].tolist()
        expr = ", ".join(f"count(*) FILTER (WHERE isinf({qi(col)}))" for col in num_cols)
        assert sum(c.execute(f"SELECT {expr} FROM read_parquet({sql_path(path)})").fetchone()) == 0
        assert c.execute(f"SELECT count(*) FROM read_parquet({sql_path(path)})").fetchone()[0] == len(ids)
    for block in blocks_by_level[4][1:]:
        expressions = " OR ".join(f"{qi(col)} IS NOT NULL" for col in feature_blocks[block])
        assert c.execute(f"SELECT count(*) FROM read_parquet({sql_path(out_dir / 'X_level_4.parquet')}) WHERE HAS_{block.upper()}=0 AND ({expressions})").fetchone()[0] == 0
    # Общие признаки должны быть не только вложены по именам, но и одинаковы по значениям.
    for level in range(1,4):
        selected = ",".join(qi(col) for col in level_features[str(level)])
        left, right = sql_path(out_dir / f"X_level_{level}.parquet"), sql_path(out_dir / f"X_level_{level+1}.parquet")
        # Row numbers attach to stored sorted order; EXCEPT ALL also preserves multiplicities.
        mismatch = c.execute(f"SELECT count(*) FROM ((SELECT row_number() OVER () AS rn,{selected} FROM read_parquet({left})) EXCEPT ALL (SELECT row_number() OVER () AS rn,{selected} FROM read_parquet({right})))").fetchone()[0]
        assert mismatch == 0

    pd.DataFrame(metadata).to_csv(report_dir / "feature_catalog.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(time_checks).to_csv(report_dir / "temporal_filters.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(report_dir / "history_coverage.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stage_times).to_csv(report_dir / "stage_timings.csv", index=False)
    pd.DataFrame(split_summary).to_csv(report_dir / "split_balance.csv", index=False)
    dump_json(out_dir / "feature_levels.json", level_features)
    summary = {"scope": "full" if sample_size is None else "technical_sample", "n_applicants": len(ids),
               "target_0": int((ids.TARGET == 0).sum()), "target_1": int((ids.TARGET == 1).sum()),
               "positive_rate": float(ids.TARGET.mean()), "shapes": shapes, "splits": split_summary,
               "history_coverage": coverage, "installment_conflicting_obligations_excluded": installment_conflicts,
               "payment_amount_decimal_scale": 3, "payment_precision_violations": precision_violations,
               "checks": {"unique_applicants": True, "same_clients_all_levels": True,
                          "nested_features_and_equal_values": True, "no_target_ids_gender_age_in_X": True,
                          "no_train_test_overlap": True, "repeats_subset_of_development": True,
                          "no_infinite_values": True, "missing_history_remains_null": True},
               "elapsed_seconds": round(time.perf_counter()-started,3),
               "versions": {name: importlib.metadata.version(name) for name in ["duckdb","pandas","numpy","pyarrow","scikit-learn"]},
               "python": platform.python_version(), "available_ram_end_gib": psutil.virtual_memory().available / 1024**3}
    dump_json(report_dir / "validation_summary.json", summary)
    c.close()
    gc.collect()
    return summary


def compare_pilot_to_full(pilot_dir, full_dir):
    """Одинаковый клиент должен получить одинаковые признаки в пилоте и полном наборе."""
    pilot_ids = pd.read_parquet(Path(pilot_dir) / "row_ids.parquet").SK_ID_CURR.to_numpy()
    full_ids = pd.read_parquet(Path(full_dir) / "row_ids.parquet").SK_ID_CURR.to_numpy()
    positions = np.searchsorted(full_ids, pilot_ids)
    assert np.array_equal(full_ids[positions], pilot_ids)
    for level in range(1,5):
        # Столбцы читаются группами, чтобы эта проверка также была ограничена по RAM.
        cols = json.loads((Path(pilot_dir)/f"features_level_{level}.json").read_text(encoding="utf-8"))
        for start in range(0, len(cols), 30):
            selected = cols[start:start+30]
            small = pd.read_parquet(Path(pilot_dir)/f"X_level_{level}.parquet",columns=selected)
            full = pd.read_parquet(Path(full_dir)/f"X_level_{level}.parquet",columns=selected).iloc[positions].reset_index(drop=True)
            pd.testing.assert_frame_equal(small, full, check_exact=False, rtol=1e-10, atol=1e-10)
            del small, full
    return {"pilot_clients": len(pilot_ids), "levels_compared": 4, "same_values": True, "tolerance": 1e-10}
