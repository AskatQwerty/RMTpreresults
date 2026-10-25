"""Проверка неполных ключей и сохранности частичных платежей.

Не изменяет CSV, X, признаки, TARGET и сохранённые разбиения. В отчёты попадают
только агрегированные количества; идентификаторы отдельных клиентов не выводятся.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import duckdb
import pandas as pd


CANDIDATE_KEY = ["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT"]
OBLIGATION_KEY = ["SK_ID_CURR", "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]
PAYMENT_RULE = (
    "DAYS_ENTRY_PAYMENT < 0 AND DAYS_ENTRY_PAYMENT <> 365243 "
    "AND DAYS_INSTALMENT IS NOT NULL AND DAYS_INSTALMENT <> 365243"
)


def _path(path):
    return "'" + str(Path(path).resolve()).replace("\\", "/").replace("'", "''") + "'"


def audit_payment_keys(data_dir, prepared_dir, report_dir):
    """Аудировать весь installments CSV и существующие полные артефакты.

    Development/test используются только для разбивки числа дефектных строк,
    без повторной генерации split и без анализа целевого исхода.
    """
    started = time.perf_counter()
    data_dir, prepared_dir, report_dir = map(Path, (data_dir, prepared_dir, report_dir))
    report_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}

    def save(frame, name):
        path = report_dir / name
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        artifacts[name] = str(path.resolve())

    with tempfile.TemporaryDirectory(prefix="homecredit_payment_audit_") as temp_dir:
        c = duckdb.connect()
        try:
            c.execute("SET memory_limit='1GB'")
            c.execute("SET threads=1")
            c.execute("SET preserve_insertion_order=false")
            c.execute(f"SET temp_directory={_path(temp_dir)}")
            c.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv_auto({_path(data_dir/'installments_payments.csv')}, sample_size=100000, ignore_errors=false)")
            c.execute(f"CREATE VIEW prepared_ids AS SELECT * FROM read_parquet({_path(prepared_dir/'row_ids.parquet')})")
            c.execute(f"CREATE VIEW dev AS SELECT * FROM read_parquet({_path(prepared_dir/'splits/development_ids.parquet')})")
            c.execute(f"CREATE VIEW test AS SELECT * FROM read_parquet({_path(prepared_dir/'splits/test_ids.parquet')})")
            c.execute(f"CREATE VIEW saved_obligations AS SELECT * FROM read_parquet({_path(prepared_dir/'aggregates/installment_obligations.parquet')})")
            c.execute(f"CREATE VIEW saved_aggregates AS SELECT * FROM read_parquet({_path(prepared_dir/'aggregates/installments.parquet')})")
            assert c.execute("SELECT count(*) FROM dev INNER JOIN test USING(SK_ID_CURR)").fetchone()[0] == 0
            c.execute("""CREATE VIEW labeled AS SELECT i.SK_ID_CURR,
                CASE WHEN d.SK_ID_CURR IS NOT NULL THEN 'development'
                     WHEN t.SK_ID_CURR IS NOT NULL THEN 'test' ELSE 'unassigned' END AS split
                FROM prepared_ids i LEFT JOIN dev d USING(SK_ID_CURR)
                LEFT JOIN test t USING(SK_ID_CURR)""")
            assert c.execute("SELECT count(*) FROM labeled WHERE split='unassigned'").fetchone()[0] == 0
            null_predicate = " OR ".join(f"{col} IS NULL" for col in CANDIDATE_KEY)
            c.execute(f"CREATE TEMP TABLE incomplete AS SELECT * FROM raw WHERE {null_predicate}")
            c.execute("""CREATE VIEW incomplete_scoped AS SELECT h.*,
                coalesce(l.split,'outside_labeled_applications') AS split
                FROM incomplete h LEFT JOIN labeled l USING(SK_ID_CURR)""")
            raw_rows = c.execute("SELECT count(*) FROM raw").fetchone()[0]
            incomplete_rows = c.execute("SELECT count(*) FROM incomplete").fetchone()[0]
            cols = c.execute("DESCRIBE raw").df().column_name.tolist()
            null_counts = c.execute("SELECT " + ",".join(f"count(*) FILTER(WHERE {col} IS NULL)" for col in cols) + " FROM incomplete").fetchone()
            save(pd.DataFrame([
                {"column":col,"part_of_candidate_key":col in CANDIDATE_KEY,
                 "part_of_obligation_key":col in OBLIGATION_KEY,"null_rows":int(n),
                 "non_null_rows":incomplete_rows-int(n)}
                for col,n in zip(cols,null_counts)
            ]), "installments_incomplete_key_fields.csv")
            pattern = " || ';' || ".join(f"CASE WHEN {col} IS NULL THEN '{col}' ELSE '' END" for col in cols)
            patterns = c.execute(f"SELECT {pattern} AS null_pattern, count(*) AS rows FROM incomplete GROUP BY ALL ORDER BY rows DESC").df()
            patterns["null_pattern"] = patterns.null_pattern.str.split(';').map(lambda x: '; '.join(y for y in x if y))
            save(patterns, "installments_incomplete_key_patterns.csv")
            # Неизвестная дата оплаты => предикат UNKNOWN; WHERE не включает строку.
            scoped = c.execute(f"""SELECT split, count(*) AS incomplete_rows,
                count(DISTINCT SK_ID_CURR) AS clients,
                count(*) FILTER(WHERE {PAYMENT_RULE}) AS rows_eligible,
                count(*) FILTER(WHERE NOT coalesce(({PAYMENT_RULE}),false)) AS rows_excluded
                FROM incomplete_scoped GROUP BY split ORDER BY split""").df()
            save(scoped, "installments_incomplete_key_by_split.csv")

            keys = ",".join(OBLIGATION_KEY)
            c.execute(f"CREATE TEMP TABLE affected_obligations AS SELECT DISTINCT {keys} FROM incomplete")
            c.execute(f"""CREATE TEMP TABLE other_valid AS SELECT {keys},count(*) AS valid_payment_rows
                FROM raw INNER JOIN affected_obligations USING({keys})
                WHERE {PAYMENT_RULE} GROUP BY {keys}""")
            obligation_recovery = c.execute(f"""SELECT coalesce(l.split,'outside_labeled_applications') AS split,
                count(*) AS affected_obligations,
                count(*) FILTER(WHERE o.valid_payment_rows>0) AS obligations_with_other_valid_payments,
                count(*) FILTER(WHERE o.valid_payment_rows IS NULL) AS obligations_without_valid_payments,
                coalesce(sum(o.valid_payment_rows),0) AS other_valid_payment_rows,
                count(*) FILTER(WHERE s.SK_ID_CURR IS NOT NULL) AS obligations_in_saved_aggregation
                FROM affected_obligations a LEFT JOIN other_valid o USING({keys})
                LEFT JOIN saved_obligations s USING({keys}) LEFT JOIN labeled l USING(SK_ID_CURR)
                GROUP BY ALL ORDER BY split""").df()
            save(obligation_recovery, "installments_incomplete_obligation_recovery.csv")
            client_retention = c.execute("""WITH clients AS (SELECT DISTINCT SK_ID_CURR FROM incomplete)
                SELECT coalesce(l.split,'outside_labeled_applications') AS split, count(*) AS affected_clients,
                count(*) FILTER(WHERE l.SK_ID_CURR IS NOT NULL) AS retained_current_applicants,
                count(*) FILTER(WHERE a.SK_ID_CURR IS NOT NULL) AS clients_with_other_saved_installment_history
                FROM clients i LEFT JOIN labeled l USING(SK_ID_CURR)
                LEFT JOIN saved_aggregates a USING(SK_ID_CURR) GROUP BY ALL ORDER BY split""").df()
            save(client_retention, "installments_incomplete_client_retention.csv")
            n_applicants = c.execute("SELECT count(*) FROM prepared_ids").fetchone()[0]
            for level in range(1,5):
                assert c.execute(f"SELECT count(*) FROM read_parquet({_path(prepared_dir/f'X_level_{level}.parquet')})").fetchone()[0] == n_applicants

            # У платежных событий нет уникального ID: проверяем два разных понятия дубля.
            exact_unique = c.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM raw)").fetchone()[0]
            candidate = ",".join(CANDIDATE_KEY)
            duplicate_stats = c.execute(f"""WITH g AS (SELECT {candidate},count(*) AS n,
                    count(DISTINCT AMT_PAYMENT) AS payment_amount_variants FROM raw GROUP BY {candidate})
                SELECT count(*) FILTER(WHERE n>1), coalesce(sum(n-1),0),
                    coalesce(sum(n) FILTER(WHERE n>1),0),max(n),
                    count(*) FILTER(WHERE n>1 AND payment_amount_variants>1) FROM g""").fetchone()
            c.execute(f"CREATE VIEW eligible_labeled AS SELECT r.* FROM raw r INNER JOIN prepared_ids USING(SK_ID_CURR) WHERE {PAYMENT_RULE}")
            eligible = c.execute("""SELECT count(*),count(AMT_PAYMENT),
                sum(CAST(AMT_PAYMENT AS DECIMAL(24,3))) FROM eligible_labeled""").fetchone()
            saved = c.execute("""SELECT count(*),sum(PAYMENT_ROWS),sum(AMOUNT_PAID),
                count(*) FILTER(WHERE PAYMENT_ROWS>1),sum(PAYMENT_ROWS) FILTER(WHERE PAYMENT_ROWS>1),
                max(PAYMENT_ROWS) FROM saved_obligations""").fetchone()
            assert int(eligible[0]) == int(saved[1]), "Потеря/повтор строк при объединении частичных платежей"
            assert eligible[2] == saved[2], "Сумма фактических оплат изменилась при агрегации"
            duplicate_report = {
                "raw_rows":raw_rows, "exact_duplicate_row_surplus":raw_rows-exact_unique,
                "repeated_candidate_key_groups":int(duplicate_stats[0]),
                "candidate_key_duplicate_surplus":int(duplicate_stats[1]),
                "rows_in_repeated_candidate_keys":int(duplicate_stats[2]),
                "max_rows_per_candidate_key":int(duplicate_stats[3]),
                "repeated_keys_with_distinct_payment_amounts":int(duplicate_stats[4]),
                "eligible_labeled_payment_rows":int(eligible[0]),
                "eligible_labeled_nonmissing_payment_amounts":int(eligible[1]),
                "saved_obligations":int(saved[0]), "saved_payment_rows_sum":int(saved[1]),
                "eligible_labeled_amount_paid_decimal":str(eligible[2]),
                "saved_amount_paid_decimal":str(saved[2]),
                "multiple_payment_obligations":int(saved[3]),
                "rows_in_multiple_payment_obligations":int(saved[4]),
                "max_payments_per_obligation":int(saved[5]),
                "rows_preserved":True, "exact_decimal_amount_preserved":True,
            }
            save(pd.DataFrame([duplicate_report]), "installments_partial_payments_preservation.csv")
            report = {
                "scope":"Все строки installments_payments; существующие development/test только для разбивки диагностик",
                "candidate_key":CANDIDATE_KEY, "obligation_key":OBLIGATION_KEY,
                "missing_key_rows":incomplete_rows,
                "null_fields":[col for col,n in zip(cols,null_counts) if n],
                "eligibility_rule":PAYMENT_RULE,
                "by_split":scoped.to_dict('records'),
                "obligation_recovery":obligation_recovery.to_dict('records'),
                "client_retention":client_retention.to_dict('records'),
                "partial_payment_preservation":duplicate_report,
                "interpretation_ru":[
                    "Кандидатный ключ включает дату фактической оплаты, но не является гарантированно уникальным ID события.",
                    "Ключ обязательства состоит из текущей заявки, предыдущего договора, версии графика и номера взноса; дата оплаты в него не входит.",
                    "Строки без даты фактической оплаты не проходят временной фильтр. Их сумма оплаты не заполняется; факт неоплаты не выводится из отсутствия даты.",
                    "Текущие заявки сохраняются независимо от наличия пригодных платежей; другие платежи того же взноса/клиента остаются доступными.",
                    "Полные дубли только диагностируются. Повторения кандидатного ключа и частичные оплаты не удаляются; AMT_PAYMENT суммируется в DECIMAL(24,3), AMT_INSTALMENT учитывается один раз на взнос.",
                    "Повтор суммы AMT_INSTALMENT по нескольким платежным строкам не означает несколько обязательств.",
                ],
                "artifacts":artifacts,
                "elapsed_seconds":round(time.perf_counter()-started,3),
            }
            (report_dir/'installments_key_quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
            return report
        finally:
            c.close()
