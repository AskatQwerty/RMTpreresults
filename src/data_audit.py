"""Последовательный, точный аудит исходных таблиц Home Credit.

Запросы выполняются DuckDB с лимитом памяти и spill-to-disk, заданными вызывающим
кодом. CSV не меняются. Повторения ключей истории диагностируются, а не удаляются:
несколько платежей по одному взносу могут быть корректными частичными платежами.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd


SOURCES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bb": "bureau_balance.csv",
    "pos": "POS_CASH_balance.csv",
    "cc": "credit_card_balance.csv",
    "installments": "installments_payments.csv",
    "previous": "previous_application.csv",
}
DESCRIPTION_FILE = "HomeCredit_columns_description.csv"
GRAINS = {
    "application": ["SK_ID_CURR"],
    "bureau": ["SK_ID_BUREAU"],
    "bb": ["SK_ID_BUREAU", "MONTHS_BALANCE"],
    "pos": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "cc": ["SK_ID_PREV", "MONTHS_BALANCE"],
    "installments": [
        "SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER", "DAYS_ENTRY_PAYMENT"
    ],
    "previous": ["SK_ID_PREV"],
}


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _s(value: str | Path) -> str:
    return "'" + str(value).replace("\\", "/").replace("'", "''") + "'"


def register_sources(con, data_dir: Path) -> dict:
    """Зарегистрировать raw_* views. Проверить наличие всех восьми файлов."""
    data_dir = Path(data_dir)
    missing = [name for name in [*SOURCES.values(), DESCRIPTION_FILE] if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError("Отсутствуют обязательные файлы: " + ", ".join(missing))
    for name, filename in SOURCES.items():
        con.execute(
            f"CREATE OR REPLACE VIEW raw_{name} AS "
            f"SELECT * FROM read_csv_auto({_s(data_dir / filename)}, "
            "header=true, sample_size=100000, ignore_errors=false)"
        )
    return {name: f"raw_{name}" for name in SOURCES}


def _number_type(dtype: str) -> bool:
    return any(dtype.startswith(t) for t in (
        "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
        "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE", "DECIMAL"
    ))


def _save(df: pd.DataFrame, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path.resolve())


def audit_sources(con, data_dir: Path, output_dir: Path) -> dict:
    """Аудит полного набора CSV; возвращает компактный отчёт и пути файлов.

    Отсутствие SK_ID_CURR в application_train само по себе не является ошибкой:
    история содержит также клиентов без опубликованного TARGET. Такие записи
    отличаются от настоящих сирот SK_ID_BUREAU/SK_ID_PREV и неверного владельца
    предыдущего договора. Точная проверка дублей учитывает все исходные столбцы.
    """
    started = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    register_sources(con, data_dir)
    source_stats = {
        filename: (data_dir / filename).stat()
        for filename in [*SOURCES.values(), DESCRIPTION_FILE]
    }
    dictionary = pd.read_csv(data_dir / DESCRIPTION_FILE, encoding="cp1252")
    for name in ("Table", "Row", "Description"):
        dictionary[name] = dictionary[name].astype("string").str.strip()
    dictionary["Table"] = dictionary["Table"].replace({
        "application_{train|test}.csv": "application_train.csv",
        "POS_CASH_balance": "POS_CASH_balance.csv",
    })
    dictionary["Row_original"] = dictionary["Row"]
    dictionary.loc[
        dictionary["Table"].isin(["bureau.csv", "bureau_balance.csv"])
        & dictionary["Row"].eq("SK_BUREAU_ID"), "Row"
    ] = "SK_ID_BUREAU"
    dictionary_lookup = dictionary.groupby(["Table", "Row"], dropna=False)["Description"].agg(
        lambda values: " | ".join(values.dropna().astype(str).unique())
    ).to_dict()
    table_rows, column_rows, coverage_rows = [], [], []
    for name, filename in SOURCES.items():
        step_started = time.perf_counter()
        view = f"raw_{name}"
        schema = con.execute(f"DESCRIBE SELECT * FROM {view}").fetchdf()
        expressions = ["COUNT(*) AS total_rows"]
        column_specs = []
        for i, row in schema.iterrows():
            col, dtype = str(row["column_name"]), str(row["column_type"])
            quoted = _q(col)
            expressions.append(f"COUNT(*) - COUNT({quoted}) AS missing_{i}")
            numeric = _number_type(dtype)
            temporal = col.startswith("DAYS_") or col == "MONTHS_BALANCE"
            if numeric:
                expressions += [f"MIN({quoted}) AS min_{i}", f"MAX({quoted}) AS max_{i}"]
            if temporal and numeric:
                expressions += [
                    f"COUNT(*) FILTER (WHERE {quoted} > 0) AS positive_{i}",
                    f"COUNT(*) FILTER (WHERE {quoted} = 0) AS zero_{i}",
                    f"COUNT(*) FILTER (WHERE {quoted} = 365243) AS placeholder_{i}",
                ]
            column_specs.append((i, col, dtype, numeric, temporal))
        stats = con.execute(f"SELECT {', '.join(expressions)} FROM {view}").fetchdf().iloc[0]
        n_rows = int(stats["total_rows"])
        for i, col, dtype, numeric, temporal in column_specs:
            description = dictionary_lookup.get((filename, col), "")
            record = {
                "table": filename, "column": col, "dtype": dtype,
                "rows": n_rows, "missing_count": int(stats[f"missing_{i}"]),
                "missing_fraction": float(stats[f"missing_{i}"]) / n_rows if n_rows else None,
                "min": stats[f"min_{i}"] if numeric else None,
                "max": stats[f"max_{i}"] if numeric else None,
                "positive_count": int(stats[f"positive_{i}"]) if temporal and numeric else None,
                "zero_count": int(stats[f"zero_{i}"]) if temporal and numeric else None,
                "placeholder_365243_count": int(stats[f"placeholder_{i}"]) if temporal and numeric else None,
                "description": description,
            }
            column_rows.append(record)
            coverage_rows.append({
                "table": filename, "column": col,
                "description_found": bool(description), "description": description,
            })
        # Exact duplicates are an audit result, not an instruction to discard rows.
        distinct_rows = int(con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {view})").fetchone()[0])
        keys = GRAINS[name]
        key_list = ", ".join(map(_q, keys))
        key_stats = con.execute(
            f"SELECT COUNT(*) AS groups, COALESCE(SUM(n - 1),0) AS surplus, "
            f"COALESCE(MAX(n),0) AS max_rows FROM "
            f"(SELECT {key_list}, COUNT(*) AS n FROM {view} GROUP BY {key_list})"
        ).fetchone()
        key_nulls = int(con.execute(
            f"SELECT COUNT(*) FROM {view} WHERE " + " OR ".join(f"{_q(k)} IS NULL" for k in keys)
        ).fetchone()[0])
        table_rows.append({
            "table": filename, "rows": n_rows, "columns": len(schema),
            "size_bytes": source_stats[filename].st_size,
            "exact_duplicate_rows": n_rows - distinct_rows,
            "candidate_key": ", ".join(keys), "key_groups": int(key_stats[0]),
            "key_duplicate_surplus": int(key_stats[1]), "max_rows_per_key": int(key_stats[2]),
            "rows_with_null_key": key_nulls,
            "key_expected_unique": name != "installments",
            "key_note": "Частичные платежи могут повторять этот кандидатный ключ" if name == "installments" else "",
            "audit_seconds": round(time.perf_counter() - step_started, 3),
        })
        print(f"Аудит {filename}: {n_rows:,} строк, {len(schema)} столбцов; "
              f"полных дублей {n_rows - distinct_rows:,}", flush=True)
        # Save incremental results as well, so diagnostics survive a later interruption.
        _save(pd.DataFrame(table_rows), output_dir, "table_summary.csv")
        _save(pd.DataFrame(column_rows), output_dir, "column_audit.csv")
    relation_rows = []

    def relation(table, metric, value, interpretation):
        relation_rows.append({"table": table, "check": metric, "rows_or_ids": int(value), "interpretation": interpretation})

    for name in ("bureau", "pos", "cc", "installments", "previous"):
        values = con.execute(
            f"SELECT COUNT(*) FILTER (WHERE a.SK_ID_CURR IS NULL), "
            f"COUNT(DISTINCT h.SK_ID_CURR) FILTER (WHERE a.SK_ID_CURR IS NULL), "
            f"COUNT(*) FILTER (WHERE a.SK_ID_CURR IS NOT NULL) "
            f"FROM raw_{name} h LEFT JOIN (SELECT DISTINCT SK_ID_CURR FROM raw_application) a "
            f"ON h.SK_ID_CURR=a.SK_ID_CURR"
        ).fetchone()
        relation(SOURCES[name], "rows_outside_labeled_applications", values[0],
                 "Ожидаемо возможны клиенты без опубликованного TARGET; не использовать в X")
        relation(SOURCES[name], "distinct_clients_outside_labeled_applications", values[1],
                 "Не означает сиротский ключ в полном конкурсе; application_test не используется")
        relation(SOURCES[name], "rows_linked_to_labeled_applications", values[2],
                 "История с владельцем из application_train")

    # Group dimension keys before joining: audit itself must not amplify row counts.
    bvalues = con.execute(
        "WITH b AS (SELECT SK_ID_BUREAU, MIN(SK_ID_CURR) AS SK_ID_CURR FROM raw_bureau GROUP BY SK_ID_BUREAU), "
        "a AS (SELECT DISTINCT SK_ID_CURR FROM raw_application) "
        "SELECT COUNT(*) FILTER (WHERE b.SK_ID_BUREAU IS NULL), "
        "COUNT(DISTINCT h.SK_ID_BUREAU) FILTER (WHERE b.SK_ID_BUREAU IS NULL), "
        "COUNT(*) FILTER (WHERE b.SK_ID_BUREAU IS NOT NULL AND a.SK_ID_CURR IS NULL), "
        "COUNT(*) FILTER (WHERE a.SK_ID_CURR IS NOT NULL) "
        "FROM raw_bb h LEFT JOIN b USING (SK_ID_BUREAU) LEFT JOIN a ON b.SK_ID_CURR=a.SK_ID_CURR"
    ).fetchone()
    for metric, value, meaning in zip(
        ("rows_without_bureau_parent", "bureau_ids_without_parent", "rows_with_unlabeled_bureau_owner", "rows_linked_via_bureau_to_labeled_applications"),
        bvalues,
        ("Сиротские строки bureau_balance: родитель bureau отсутствует", "Число отсутствующих родительских SK_ID_BUREAU", "Родитель существует, но владелец вне application_train", "Полная связь bureau_balance → bureau → текущая заявка"),
    ):
        relation(SOURCES["bb"], metric, value, meaning)

    parent_ambiguous = con.execute(
        "SELECT COUNT(*) FROM (SELECT SK_ID_PREV FROM raw_previous GROUP BY SK_ID_PREV "
        "HAVING COUNT(DISTINCT SK_ID_CURR)>1)"
    ).fetchone()[0]
    relation(SOURCES["previous"], "previous_ids_with_multiple_owners", parent_ambiguous,
             "Один предыдущий договор не должен принадлежать нескольким текущим заявкам")
    bureau_ambiguous = con.execute(
        "SELECT COUNT(*) FROM (SELECT SK_ID_BUREAU FROM raw_bureau GROUP BY SK_ID_BUREAU "
        "HAVING COUNT(DISTINCT SK_ID_CURR)>1)"
    ).fetchone()[0]
    relation(SOURCES["bureau"], "bureau_ids_with_multiple_owners", bureau_ambiguous,
             "Один бюро-договор не должен принадлежать нескольким текущим заявкам")
    for name in ("pos", "cc", "installments"):
        values = con.execute(
            "WITH p AS (SELECT SK_ID_PREV, MIN(SK_ID_CURR) AS SK_ID_CURR FROM raw_previous GROUP BY SK_ID_PREV) "
            "SELECT COUNT(*) FILTER (WHERE p.SK_ID_PREV IS NULL), "
            "COUNT(DISTINCT h.SK_ID_PREV) FILTER (WHERE p.SK_ID_PREV IS NULL), "
            "COUNT(*) FILTER (WHERE p.SK_ID_PREV IS NOT NULL AND h.SK_ID_CURR <> p.SK_ID_CURR) "
            f"FROM raw_{name} h LEFT JOIN p ON h.SK_ID_PREV=p.SK_ID_PREV"
        ).fetchone()
        relation(SOURCES[name], "rows_without_previous_parent", values[0], "Сиротские строки: отсутствует предыдущая заявка")
        relation(SOURCES[name], "previous_ids_without_parent", values[1], "Количество предыдущих договоров без родителя")
        relation(SOURCES[name], "rows_with_previous_owner_mismatch", values[2], "SK_ID_CURR истории не совпадает с владельцем SK_ID_PREV")
    target = con.execute(
        "SELECT TARGET, COUNT(*) AS rows, COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () AS fraction "
        "FROM raw_application GROUP BY TARGET ORDER BY TARGET"
    ).fetchdf()

    table_rows.append({
        "table": DESCRIPTION_FILE, "rows": len(dictionary), "columns": len(dictionary.columns) - 1,
        "size_bytes": source_stats[DESCRIPTION_FILE].st_size,
        "exact_duplicate_rows": int(dictionary.duplicated().sum()),
        "candidate_key": "Table, Row", "key_groups": len(dictionary.drop_duplicates(["Table", "Row"])),
        "key_duplicate_surplus": int(dictionary.duplicated(["Table", "Row"]).sum()),
        "max_rows_per_key": int(dictionary.groupby(["Table", "Row"], dropna=False).size().max()),
        "rows_with_null_key": int(dictionary[["Table", "Row"]].isna().any(axis=1).sum()),
        "key_expected_unique": True, "key_note": "Словарь прочитан в кодировке cp1252", "audit_seconds": None,
    })
    for col in dictionary.columns.drop("Row_original"):
        column_rows.append({
            "table": DESCRIPTION_FILE, "column": col, "dtype": str(dictionary[col].dtype),
            "rows": len(dictionary), "missing_count": int(dictionary[col].isna().sum()),
            "missing_fraction": float(dictionary[col].isna().mean()),
        })
    manifest = []
    for filename, before in source_stats.items():
        after = (data_dir / filename).stat()
        unchanged = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
        if not unchanged:
            raise RuntimeError(f"Исходный файл изменился во время аудита: {filename}")
        manifest.append({
            "file": filename, "path": str((data_dir / filename).resolve()),
            "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "size_and_mtime_unchanged": unchanged,
        })
    actual_columns = {(row["table"], row["column"]) for row in coverage_rows}
    dictionary_extras = dictionary[
        ~pd.MultiIndex.from_frame(dictionary[["Table", "Row"]]).isin(actual_columns)
    ]
    paths = {
        "table_summary": _save(pd.DataFrame(table_rows), output_dir, "table_summary.csv"),
        "column_audit": _save(pd.DataFrame(column_rows), output_dir, "column_audit.csv"),
        "relationships": _save(pd.DataFrame(relation_rows), output_dir, "relationships.csv"),
        "dictionary": _save(dictionary, output_dir, "column_dictionary_utf8.csv"),
        "description_coverage": _save(pd.DataFrame(coverage_rows), output_dir, "description_coverage.csv"),
        "dictionary_extra_rows": _save(dictionary_extras, output_dir, "dictionary_extra_rows.csv"),
        "target_balance": _save(target, output_dir, "target_balance.csv"),
        "source_manifest": _save(pd.DataFrame(manifest), output_dir, "source_manifest.csv"),
    }
    report = {
        "scope": "Полный аудит всех восьми обязательных исходных CSV",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "tables": [{k: v for k, v in row.items() if k in ("table", "rows", "columns", "exact_duplicate_rows", "key_duplicate_surplus")} for row in table_rows],
        "dictionary_unmatched_columns": [
            {"table": row["table"], "column": row["column"]}
            for row in coverage_rows if not row["description_found"]
        ],
        "files": paths,
        "notes": [
            "Нет автоматического удаления дублей или повторяющихся частичных платежей.",
            "Положительные даты и 365243 лишь посчитаны; решения об их доступности принимаются отдельно по смыслу поля.",
            "Отсутствие клиента в application_train отличается от отсутствующего родительского договора.",
            "Типы определены DuckDB по sample_size=100000; остальные строки читаются строго без ignore_errors.",
        ],
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
