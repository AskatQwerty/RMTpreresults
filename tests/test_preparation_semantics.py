"""Регрессионные проверки временной доступности и сохранения популяции.

Все значения сценария синтетические; реальные CSV читаются только для получения
схем. Эти результаты не подменяют аудит или объём обработанного исходного набора.
Запуск: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_preparation import FORBIDDEN, SEEDS, run_preparation


SOURCE_DIR = Path(os.environ.get(
    "HOME_CREDIT_DATA_DIR", r"C:\Users\askat\Study\RMT\home-credit-default-risk"
))
CLIENT = 100000
NO_HISTORY_CLIENT = 100001
MISSING_PREVIOUS_ID = 399999
DECIMAL_CLIENT = CLIENT + 4


def build_synthetic_sources(source_dir: Path, fixture_dir: Path) -> None:
    """Создать 60 заявок и адресные сценарии в реальных схемах Home Credit."""
    fixture_dir.mkdir(parents=True)
    names = (
        "application_train.csv", "bureau.csv", "bureau_balance.csv",
        "POS_CASH_balance.csv", "credit_card_balance.csv",
        "installments_payments.csv", "previous_application.csv",
    )
    schemas = {name: pd.read_csv(source_dir / name, nrows=64) for name in names}

    def row(name, **changes):
        sample = schemas[name]
        values = {
            col: "Unknown" if pd.api.types.is_object_dtype(dtype) else 1.0
            for col, dtype in sample.dtypes.items()
        }
        for col in values:
            if col.startswith("DAYS_") or col == "MONTHS_BALANCE":
                values[col] = -10.0
        values.update(changes)
        unknown = set(values) - set(sample.columns)
        assert not unknown, f"Тест содержит несуществующие поля: {unknown}"
        return values

    def write(name, rows):
        pd.DataFrame(rows, columns=schemas[name].columns).to_csv(fixture_dir / name, index=False)

    apps = [row(
        "application_train.csv", SK_ID_CURR=CLIENT + i, TARGET=i % 2,
        CODE_GENDER="M" if i % 2 == 0 else "F", DAYS_BIRTH=-15000,
        DAYS_EMPLOYED=-100, AMT_INCOME_TOTAL=100, AMT_CREDIT=200,
        AMT_ANNUITY=20, AMT_GOODS_PRICE=180, CNT_FAM_MEMBERS=2,
    ) for i in range(60)]
    apps[0].update(
        CODE_GENDER="XNA", DAYS_BIRTH=-20 * 365.25, DAYS_EMPLOYED=365243,
        AMT_INCOME_TOTAL=0, AMT_CREDIT=100, AMT_ANNUITY=10,
        AMT_GOODS_PRICE=0, CNT_FAM_MEMBERS=0,
    )
    write("application_train.csv", apps)
    write("bureau.csv", [
        row("bureau.csv", SK_ID_CURR=CLIENT, SK_ID_BUREAU=200001,
            DAYS_CREDIT=-500, DAYS_CREDIT_UPDATE=-10, DAYS_CREDIT_ENDDATE=120,
            DAYS_ENDDATE_FACT=365243, CREDIT_ACTIVE="Active", CREDIT_TYPE="Consumer credit"),
        row("bureau.csv", SK_ID_CURR=CLIENT + 2, SK_ID_BUREAU=200002,
            DAYS_CREDIT=-100, DAYS_CREDIT_UPDATE=-10, DAYS_CREDIT_ENDDATE=60,
            CREDIT_ACTIVE="Closed", CREDIT_TYPE="Credit card"),
        row("bureau.csv", SK_ID_CURR=CLIENT, SK_ID_BUREAU=200003,
            DAYS_CREDIT=-100, DAYS_CREDIT_UPDATE=3, CREDIT_ACTIVE="Active", CREDIT_TYPE="Credit card"),
    ])
    write("bureau_balance.csv", [
        row("bureau_balance.csv", SK_ID_BUREAU=200001, MONTHS_BALANCE=-2, STATUS="1"),
        row("bureau_balance.csv", SK_ID_BUREAU=200001, MONTHS_BALANCE=-1, STATUS="C"),
        row("bureau_balance.csv", SK_ID_BUREAU=200001, MONTHS_BALANCE=0, STATUS="5"),
        row("bureau_balance.csv", SK_ID_BUREAU=200002, MONTHS_BALANCE=-1, STATUS="0"),
        row("bureau_balance.csv", SK_ID_BUREAU=200003, MONTHS_BALANCE=-1, STATUS="5"),
    ])
    write("previous_application.csv", [
        row("previous_application.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300001,
            DAYS_DECISION=-100, DAYS_FIRST_DUE=30, DAYS_LAST_DUE_1ST_VERSION=100,
            DAYS_FIRST_DRAWING=20, NAME_CONTRACT_STATUS="Approved"),
        row("previous_application.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300002,
            DAYS_DECISION=-50, DAYS_FIRST_DUE=365243, DAYS_LAST_DUE_1ST_VERSION=365243,
            NAME_CONTRACT_STATUS="Refused"),
        row("previous_application.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300003,
            DAYS_DECISION=10, DAYS_FIRST_DUE=40, DAYS_LAST_DUE_1ST_VERSION=200,
            NAME_CONTRACT_STATUS="Approved"),
    ])
    write("POS_CASH_balance.csv", [
        row("POS_CASH_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300001,
            MONTHS_BALANCE=-1, SK_DPD=2, CNT_INSTALMENT_FUTURE=4),
        row("POS_CASH_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300001,
            MONTHS_BALANCE=0, SK_DPD=99),
        # Direct SK_ID_CURR and its own past snapshot remain usable without previous_application.
        row("POS_CASH_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=MISSING_PREVIOUS_ID,
            MONTHS_BALANCE=-2, SK_DPD=6, CNT_INSTALMENT_FUTURE=2),
        row("POS_CASH_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=MISSING_PREVIOUS_ID,
            MONTHS_BALANCE=1, SK_DPD=999),
    ])
    write("credit_card_balance.csv", [
        row("credit_card_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300001,
            MONTHS_BALANCE=-1, AMT_BALANCE=100, AMT_CREDIT_LIMIT_ACTUAL=0),
        row("credit_card_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=300001,
            MONTHS_BALANCE=0, AMT_BALANCE=9999, AMT_CREDIT_LIMIT_ACTUAL=100),
        row("credit_card_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=MISSING_PREVIOUS_ID,
            MONTHS_BALANCE=-2, AMT_BALANCE=60, AMT_CREDIT_LIMIT_ACTUAL=200),
        row("credit_card_balance.csv", SK_ID_CURR=CLIENT, SK_ID_PREV=MISSING_PREVIOUS_ID,
            MONTHS_BALANCE=2, AMT_BALANCE=9999, AMT_CREDIT_LIMIT_ACTUAL=100),
    ])

    def payment(number, due, paid_at, amount_due, amount_paid, previous_id=300001, client_id=CLIENT):
        return row(
            "installments_payments.csv", SK_ID_CURR=client_id, SK_ID_PREV=previous_id,
            NUM_INSTALMENT_VERSION=1, NUM_INSTALMENT_NUMBER=number,
            DAYS_INSTALMENT=due, DAYS_ENTRY_PAYMENT=paid_at,
            AMT_INSTALMENT=amount_due, AMT_PAYMENT=amount_paid,
        )

    write("installments_payments.csv", [
        payment(1, -10, -12, 100, 40),
        payment(1, -10, -11, 100, 60),  # two partial payments, one obligation
        payment(1, -10, 1, 100, 999),  # actual future payment must not inflate paid amount
        payment(2, 20, -1, 50, 50),    # positive scheduled due + already made prepayment
        payment(3, 100, 2, 900, 900),
        payment(4, -10, np.nan, 123, 123),
        payment(5, -10, 365243, 123, 123),
        payment(6, 365243, -1, 123, 123),
        payment(8, -20, -22, 30, 30),
        payment(8, -21, -22, 30, 30),  # conflicting schedule cannot silently use MAX
        payment(9, -10, -2, 800, 800, previous_id=MISSING_PREVIOUS_ID),
        payment(10, -10, 1, 1000, 1000, previous_id=MISSING_PREVIOUS_ID),
        # Equal decimal totals in reversed summation orders. Binary floating point
        # can turn .5+.2+.1 into .7999999999999999 and falsely label it underpaid.
        *[
            payment(number, -10, -20 + position, due, amount,
                    previous_id=399998, client_id=DECIMAL_CLIENT)
            for number, due, parts in (
                (101, 0.8, (0.5, 0.2, 0.1)),
                (102, 0.8, (0.1, 0.2, 0.5)),
                (103, 0.6, (0.1, 0.2, 0.3)),
                (104, 0.6, (0.3, 0.2, 0.1)),
            )
            for position, amount in enumerate(parts)
        ],
    ])
    dictionary = source_dir / "HomeCredit_columns_description.csv"
    (fixture_dir / dictionary.name).write_bytes(dictionary.read_bytes())


class PreparationSemanticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SOURCE_DIR / "application_train.csv").is_file():
            raise unittest.SkipTest("Реальные CSV необходимы только для схем; задайте HOME_CREDIT_DATA_DIR")
        cls.temp = tempfile.TemporaryDirectory(prefix="homecredit_semantics_")
        cls.base = Path(cls.temp.name)
        cls.addClassCleanup(cls.temp.cleanup)
        cls.input_dir = cls.base / "input"
        cls.output_dir = cls.base / "prepared"
        build_synthetic_sources(SOURCE_DIR, cls.input_dir)
        cls.summary = run_preparation(cls.input_dir, cls.output_dir, cls.base / "reports")
        cls.ids = pd.read_parquet(cls.output_dir / "row_ids.parquet").SK_ID_CURR
        cls.levels = {
            level: pd.read_parquet(cls.output_dir / f"X_level_{level}.parquet").set_axis(cls.ids)
            for level in range(1, 5)
        }

    def test_application_ratios_and_placeholders(self):
        subject = self.levels[1].loc[CLIENT]
        for feature in ("APP_CREDIT_TO_INCOME", "APP_ANNUITY_TO_INCOME",
                        "APP_CREDIT_TO_GOODS", "APP_INCOME_PER_MEMBER", "DAYS_EMPLOYED"):
            self.assertTrue(pd.isna(subject[feature]), feature)
        self.assertEqual(subject.APP_EMPLOYMENT_DATE_UNKNOWN, 1)
        self.assertAlmostEqual(subject.APP_ANNUITY_TO_CREDIT, 0.1)

    def test_bureau_mapping_and_asof_months(self):
        subject = self.levels[2].loc[CLIENT]
        other_owner = self.levels[2].loc[CLIENT + 2]
        self.assertEqual(subject.BUREAU_RECORD_COUNT, 1)
        self.assertEqual(subject.BUREAU_DAYS_CREDIT_ENDDATE_AVG, 120)
        self.assertTrue(pd.isna(subject.BUREAU_DAYS_ENDDATE_FACT_AVG))
        self.assertEqual(subject.BB_RECORD_COUNT, 2)
        self.assertEqual(subject.BB_CONTRACT_COUNT, 1)
        self.assertEqual(subject.BB_DELINQUENT_SHARE, 0.5)
        self.assertEqual(subject.BB_MAX_DPD_STATUS, 1)
        self.assertEqual(other_owner.BB_RECORD_COUNT, 1)
        self.assertEqual(other_owner.BB_DELINQUENT_SHARE, 0)

    def test_history_uses_own_cutoffs_and_retains_missing_previous(self):
        subject = self.levels[3].loc[CLIENT]
        previous_ids = pd.read_csv(self.input_dir / "previous_application.csv").SK_ID_PREV
        self.assertNotIn(MISSING_PREVIOUS_ID, set(previous_ids))
        # Two past snapshots: DPD 2 and 6, future installment counts 4 and 2.
        self.assertEqual(subject.POS_RECORD_COUNT, 2)
        self.assertEqual(subject.POS_CONTRACT_COUNT, 2)
        self.assertEqual(subject.POS_SK_DPD_MAX, 6)
        self.assertEqual(subject.POS_CNT_INSTALMENT_FUTURE_AVG, 3)
        # Balances 100 and 60; zero limit is undefined, leaving utilization 60/200.
        self.assertEqual(subject.CC_RECORD_COUNT, 2)
        self.assertEqual(subject.CC_CONTRACT_COUNT, 2)
        self.assertEqual(subject.CC_AMT_BALANCE_AVG, 80)
        self.assertEqual(subject.CC_AMT_CREDIT_LIMIT_ACTUAL_MIN, 0)
        self.assertAlmostEqual(subject.CC_UTILIZATION_AVG, 0.3)
        self.assertAlmostEqual(subject.CC_UTILIZATION_MAX, 0.3)

    def test_partial_payments_and_future_scheduled_prepayment(self):
        subject = self.levels[4].loc[CLIENT]
        # Three obligations: 40+60 / 100, prepaid 50 / 50, and 800 / 800
        # without a previous_application parent. Their own dates govern inclusion.
        self.assertEqual(subject.INSTALLMENTS_RECORD_COUNT, 3)
        self.assertEqual(subject.INSTALLMENTS_CONTRACT_COUNT, 2)
        self.assertAlmostEqual(subject.INSTALLMENTS_AMOUNT_DUE_AVG, (100 + 50 + 800) / 3)
        self.assertAlmostEqual(subject.INSTALLMENTS_AMOUNT_PAID_AVG, (40 + 60 + 50 + 800) / 3)
        self.assertAlmostEqual(subject.INSTALLMENTS_PAYMENT_ROWS_AVG, (2 + 1 + 1) / 3)
        self.assertEqual(subject.INSTALLMENTS_PAID_TO_DUE, 1)
        self.assertEqual(subject.INSTALLMENTS_DUE_DAY_MAX, 20)
        self.assertEqual(subject.INSTALLMENTS_MOST_RECENT_PAYMENT_DAY, -1)
        self.assertAlmostEqual(subject.INSTALLMENTS_LATE_SHARE, 1 / 3)
        self.assertEqual(subject.INSTALLMENTS_DAYS_LATE_MAX, 8)
        self.assertEqual(self.summary["installment_conflicting_obligations_excluded"], 1)

    def test_previous_planned_dates_and_placeholder(self):
        subject = self.levels[4].loc[CLIENT]
        self.assertEqual(subject.PREVIOUS_RECORD_COUNT, 2)
        self.assertEqual(subject.PREVIOUS_DAYS_FIRST_DUE_AVG, 30)
        self.assertEqual(subject.PREVIOUS_DAYS_LAST_DUE_1ST_VERSION_AVG, 100)
        self.assertFalse(any("DAYS_FIRST_DRAWING" in name for name in subject.index))

    def test_decimal_partial_payments_are_exact_and_order_independent(self):
        subject = self.levels[4].loc[DECIMAL_CLIENT]
        self.assertEqual(subject.INSTALLMENTS_RECORD_COUNT, 4)
        self.assertEqual(subject.INSTALLMENTS_UNDERPAID_SHARE, 0)
        self.assertEqual(subject.INSTALLMENTS_PAID_TO_DUE, 1)
        obligations = pd.read_parquet(
            self.output_dir / "aggregates" / "installment_obligations.parquet"
        )
        obligations = obligations.loc[
            obligations.SK_ID_CURR == DECIMAL_CLIENT
        ].set_index("NUM_INSTALMENT_NUMBER")
        self.assertEqual(set(obligations.index), {101, 102, 103, 104})
        for number, expected in ((101, "0.8"), (102, "0.8"), (103, "0.6"), (104, "0.6")):
            obligation = obligations.loc[number]
            self.assertEqual(obligation.PAYMENT_ROWS, 3)
            self.assertEqual(Decimal(str(obligation.AMOUNT_DUE)), Decimal(expected))
            self.assertEqual(Decimal(str(obligation.AMOUNT_PAID)), Decimal(expected))
            self.assertEqual(obligation.AMOUNT_PAID, obligation.AMOUNT_DUE)
        self.assertEqual(obligations.loc[101, "AMOUNT_PAID"], obligations.loc[102, "AMOUNT_PAID"])
        self.assertEqual(obligations.loc[103, "AMOUNT_PAID"], obligations.loc[104, "AMOUNT_PAID"])

    def test_missing_history_preserves_nulls_and_people(self):
        subject = self.levels[4].loc[NO_HISTORY_CLIENT]
        for block in ("BUREAU", "BB", "POS", "CC", "INSTALLMENTS", "PREVIOUS"):
            self.assertEqual(subject[f"HAS_{block}"], 0)
            block_features = [name for name in subject.index if name.startswith(block + "_")]
            self.assertTrue(subject[block_features].isna().all(), block)
        for level, frame in self.levels.items():
            self.assertEqual(len(frame), 60)
            self.assertTrue(frame.index.is_unique)
            self.assertEqual(frame.index.tolist(), self.ids.tolist())
            if level > 1:
                prior = self.levels[level - 1]
                pd.testing.assert_frame_equal(prior, frame.loc[:, prior.columns])

    def test_sensitive_fields_only_in_sidecar(self):
        for frame in self.levels.values():
            self.assertFalse(set(frame.columns) & FORBIDDEN)
            self.assertFalse(any(name.startswith("SK_ID_") for name in frame.columns))
        fairness = pd.read_parquet(self.output_dir / "fairness.parquet").set_index("SK_ID_CURR")
        self.assertEqual(fairness.loc[CLIENT, "AGE_YEARS"], 20)
        self.assertEqual(fairness.loc[CLIENT, "AGE_GROUP"], "<25")
        self.assertEqual(fairness.loc[CLIENT, "GENDER_GROUP"], "unknown")
        self.assertEqual(fairness.loc[NO_HISTORY_CLIENT, "GENDER_GROUP"], "F")

    def test_shared_stratified_splits_and_seed_repeats(self):
        target = pd.read_parquet(self.output_dir / "target.parquet")
        splits = self.output_dir / "splits"
        dev = pd.read_parquet(splits / "development_ids.parquet").SK_ID_CURR
        test = pd.read_parquet(splits / "test_ids.parquet").SK_ID_CURR
        self.assertEqual((len(dev), len(test)), (48, 12))
        self.assertFalse(set(dev) & set(test))
        self.assertEqual(set(dev) | set(test), set(self.ids))
        expected_dev, expected_test = train_test_split(
            target, test_size=0.2, stratify=target.TARGET, random_state=42
        )
        self.assertEqual(set(dev), set(expected_dev.SK_ID_CURR))
        self.assertEqual(set(test), set(expected_test.SK_ID_CURR))
        dev_frame = target[target.SK_ID_CURR.isin(dev)].sort_values("SK_ID_CURR")
        signatures = []
        for seed in SEEDS:
            actual = pd.read_parquet(splits / f"train_seed_{seed}_ids.parquet").SK_ID_CURR
            expected, _ = train_test_split(
                dev_frame, train_size=0.9, stratify=dev_frame.TARGET, random_state=seed
            )
            self.assertEqual(set(actual), set(expected.SK_ID_CURR))
            self.assertFalse(set(actual) & set(test))
            self.assertEqual(len(actual), int(0.9 * len(dev)))
            self.assertEqual(set(target.loc[target.SK_ID_CURR.isin(actual), "TARGET"]), {0, 1})
            signatures.append(tuple(actual))
        self.assertEqual(len(set(signatures)), 5)
        folds = pd.read_parquet(splits / "cv_folds.parquet")
        self.assertEqual(set(folds.SK_ID_CURR), set(dev))
        self.assertEqual(set(folds.validation_fold), {0, 1, 2})
        for fold in range(3):
            fold_ids = folds.loc[folds.validation_fold == fold, "SK_ID_CURR"]
            labels = target.loc[target.SK_ID_CURR.isin(fold_ids), "TARGET"]
            self.assertEqual(labels.value_counts().to_dict(), {0: 8, 1: 8})
        levels = json.loads((self.output_dir / "feature_levels.json").read_text(encoding="utf-8"))
        self.assertEqual(set(levels), {"1", "2", "3", "4"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
