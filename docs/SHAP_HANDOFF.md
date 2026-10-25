# Передача RQ1/RQ2 → SHAP/RQ3

## Граница готовой работы

Завершены подготовка данных, пилот `pilot_20k_5config` и предварительная оценка RQ1/RQ2. Семь моделей × четыре уровня дают 28 сохранённых полных Pipeline. Каждый выбран через пять конфигураций и общие три CV-folds, затем автоматически обучен один раз на всех 20 000 пилотных train-клиентов. Общий test содержит 61 503 клиента. Готовые метрики, paired bootstrap CI, DeLong, прибыль, fairness, калибровка и анализ ошибок находятся в `reports/pilot_20k_5config/evaluation/`.

> Сохранён один итоговый fit каждой модели на каждом уровне. Текущие артефакты позволяют сравнивать объяснения между моделями. Устойчивость после переобучения требует дополнительных независимо обученных повторов и в этой части работы не измерялась. Повторный запуск объяснителя не является повторным обучением.

Модели старого ресурсного пилота на 221 407 клиентах не входят в текущий эксперимент и не являются его дополнительными повторами. Пакет содержит только `pilot_20k_5config`. Новые обучения, подбор параметров, bootstrap и SHAP при его сборке не выполняются.

## Получение и распаковка

Получите от участника команды Git-версию проекта и локальный архив `handoff/pilot_20k_5config_shap.zip`. Архив не входит в Git и автоматически никуда не отправляется. Распакуйте его в отдельный каталог: внутри уже находится структура корня проекта. Не накладывайте архив на рабочий каталог с чужими изменениями. Откройте терминал в корне распакованного пакета, где находятся `src/`, `docs/`, `requirements.txt` и `package_manifest.json`.

`package_manifest.json` фиксирует относительные пути, размеры, SHA-256 файлов и связь с Git commit исходного кода. Сверьте его с переданной версией репозитория; сохраните manifest рядом с будущими SHAP-результатами. Полные Pipeline загрузите только из согласованного командного пакета: `joblib` восстанавливает Python-объекты.

Архив собирается сопровождающим через `python scripts/build_shap_handoff_package.py --prepare`, затем после обычного локального commit — через `python scripts/build_shap_handoff_package.py --finalize`. Получателю достаточно распаковки; эти команды не нужны для загрузки моделей.

## Окружение

Нужно окружение исходного обучения: **Python 3.13.9**, версии из `requirements.txt` и `docs/software_versions.json`. Для чтения графиков/отчётов добавлен `requirements-reporting.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-reporting.txt
```

Версия SHAP пока не выбрана и SHAP не установлен этой задачей. В своей части зафиксируйте версию объяснителя, параметры, способ маскирования и выход модели; не меняйте незаметно версии sklearn/XGBoost/LightGBM ради установки SHAP. Сохраните отдельный файл зависимостей RQ3 и проверьте совместимость загрузки готовых Pipeline. Не вызывайте `fit` или `fit_transform` для подготовки объяснений.

## Точные пути внутри пакета

Ниже `P = artifacts/training/pilot_20k_5config`, `D = data/handoff/pilot_20k_5config`. Все пути относительны корню распаковки.

| Файл | Содержимое |
|---|---|
| `P/<model>/level_<N>/fit_checkpoint/pipeline.joblib` | Полный обученный Pipeline; включены все 28 сочетаний |
| `P/<model>/level_<N>/fit_checkpoint/best_params.json` | Выбранные параметры и CV log-loss |
| `P/<model>/level_<N>/fit_checkpoint/fit_diagnostics.json` | Атрибуты обученных компонентов и их размеры; для MLP — фактический `n_iter_`, `loss_curve_`, настройки |
| `P/<model>/level_<N>/fit_checkpoint/warnings.json` | Предупреждения поиска и refit, включая CV-кандидатов |
| `P/<model>/level_<N>/fit_checkpoint/cv_results.parquet` | Сохранённые результаты пяти конфигураций |
| `P/<model>/level_<N>/probabilities.parquet` | `SK_ID_CURR`, `TARGET`, `probability_default`, scope/model/level/split |
| `P/cohort_ids.parquet`, `P/cohort_targets.parquet` | Оригинальные 20 000 ID пилота и TARGET |
| `P/test_ids.parquet`, `P/cv_folds.parquet` | Оригинальные test-ID и три пилотных CV-folds |
| `P/experiment_manifest.json`, `P/cohort_contract.json` | Протокол и хэши исходных артефактов |
| `P/source_snapshot/` | Снимок пяти основных модулей, использованных при обучении |
| `D/X_train_level4.parquet`, `D/X_test_level4.parquet` | Готовые исходные признаки Level 4 для train/test; **SK_ID_CURR — отдельный служебный столбец**, его нельзя подавать в модель |
| `D/train_ids.parquet`, `D/test_ids.parquet` | Порядок строк train/test, совпадающий с сохранёнными ID пилота |
| `D/target_train.parquet`, `D/target_test.parquet` | TARGET отдельно, с ключом `SK_ID_CURR` |
| `D/fairness_train.parquet`, `D/fairness_test.parquet` | Отдельные `CODE_GENDER`, `DAYS_BIRTH`, `AGE_YEARS`, `AGE_GROUP`, `GENDER_GROUP` с ID |
| `D/features_level_1.json` … `D/features_level_4.json` | **Точные имена и порядок** входных признаков каждой модели соответствующего уровня |
| `D/feature_levels.json`, `D/feature_schema.json` | Вложенность уровней, схема данных и порядок столбцов |
| `D/model_schema.json` | Проверенные списки преобразованных признаков каждого сохранённого Pipeline; у ансамблей — отдельно по каждой базе |
| `D/cv_folds.parquet` | Копия сохранённых пилотных folds для воспроизводимости |

Заменяйте `<model>` одним из `logistic_regression`, `random_forest`, `xgboost`, `lightgbm`, `mlp`, `soft_voting`, `stacking`; `<N>` — 1, 2, 3 или 4. Level 4 содержит все признаки предыдущих уровней, поэтому отдельные большие X-файлы уровней 1–3 в пакете не нужны: выбирайте соответствующий сохранённый список колонок. Полный development не включён; доступных 20 000 train-клиентов достаточно для выбора train-only background.

## Загрузка Pipeline, ID, TARGET и входных признаков без обучения

Следующий код только читает файлы, согласует ключи и обращается к сохранённым атрибутам. Он не выполняет `fit`, `predict` или SHAP. Запускайте из корня пакета, чтобы функции `src.model_pipelines` были доступны при `joblib.load`.

```python
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path.cwd()
P = ROOT / "artifacts/training/pilot_20k_5config"
D = ROOT / "data/handoff/pilot_20k_5config"
model, level = "lightgbm", 4
pair = P / model / f"level_{level}"
pipeline = joblib.load(pair / "fit_checkpoint/pipeline.joblib")
features = json.loads((D / f"features_level_{level}.json").read_text(encoding="utf-8"))

test_ids = pd.read_parquet(D / "test_ids.parquet")
test_raw = pd.read_parquet(D / "X_test_level4.parquet")
target = pd.read_parquet(D / "target_test.parquet")
saved = pd.read_parquet(pair / "probabilities.parquet")
assert test_ids.SK_ID_CURR.is_unique and len(test_ids) == 61503
assert test_raw.SK_ID_CURR.is_unique and target.SK_ID_CURR.is_unique
assert set(test_raw.SK_ID_CURR) == set(target.SK_ID_CURR) == set(test_ids.SK_ID_CURR)
assert saved.SK_ID_CURR.is_unique and set(saved.SK_ID_CURR) == set(test_ids.SK_ID_CURR)
aligned = test_ids.merge(test_raw, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
aligned_target = test_ids.merge(target, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
aligned_prob = test_ids.merge(saved, on="SK_ID_CURR", how="left", validate="one_to_one", sort=False)
assert np.array_equal(aligned.SK_ID_CURR, test_ids.SK_ID_CURR)
assert aligned_target.TARGET.notna().all()
assert np.array_equal(aligned_target.TARGET, aligned_prob.TARGET)
assert aligned_prob.probability_default.between(0, 1).all()
X_test = aligned.loc[:, features]  # строго этот порядок; без ID/TARGET/fairness
y_test = aligned_target.TARGET
assert list(X_test.columns) == features
assert not {"SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH", "AGE_YEARS"} & set(features)
classes = np.asarray(pipeline.classes_)
positive_column = int(np.flatnonzero(classes == 1)[0])
print(model, level, X_test.shape, "Столбец TARGET=1:", positive_column)
```

В сохранённых прогнозах **`probability_default` = P(TARGET=1)**. При последующей работе объяснителя, которому нужна функция прогноза, выбирайте столбец `TARGET=1` через `classes_`, а не полагайтесь на его номер без проверки. Для бинарного решения в RQ1/RQ2 дефолт предсказывался при `p_default >= t`, одобрение — при `p_default < t`.

## Названия преобразованных признаков и ансамбли

Для пяти самостоятельных моделей обученная предобработка находится в `pipeline.named_steps["preprocess"]`:

```python
preprocess = pipeline.named_steps["preprocess"]
transformed_names = preprocess.get_feature_names_out().tolist()
assert list(preprocess.feature_names_in_) == features
# При необходимости только transform готовых строк; не fit_transform:
# transformed = preprocess.transform(X_test.loc[:, features])
```

Порядок `transformed_names` соответствует сохранённому `ColumnTransformer`: числовые значения, индикаторы исходных пропусков и закодированные категории. Не создавайте новый encoder и не восстанавливайте словарь категорий по test. Полностью пустой train-столбец сохранён с фиксированным fallback 0 и индикатором; неизвестные категории обрабатываются готовым encoder. LR/MLP используют сохранённое масштабирование; XGBoost получает dense float32 с явными нулями. Преобразования и списки всегда берите у конкретной модели.

При сборке пакета загрузкой объектов без `fit`, `transform` и `predict` проверены все 28 Pipeline: классы `[0, 1]`, точный порядок исходных признаков и доступность имён преобразований. Для L1/L2/L3/L4 получены соответственно 359/467/639/755 преобразованных признаков (124/178/264/322 исходных); полные имена записаны в `D/model_schema.json`.

У soft voting и stacking **нет общей внешней предобработки**. Каждый обученный base Pipeline имеет свою:

```python
ensemble = pipeline.named_steps["model"]
base_feature_names = {
    name: base.named_steps["preprocess"].get_feature_names_out().tolist()
    for name, base in ensemble.named_estimators_.items()
}
# Ключи: lr, xgb, lgbm. Это fitted named_estimators_, не шаблоны estimators.
```

Не выдавайте объяснение отдельной базы за объяснение итогового ансамбля. У stacking метамодель получает три OOF-вероятности базовых моделей при обучении (`passthrough=False`), а не 322 исходных признака. Объяснение метамодели в пространстве трёх прогнозов и объяснение всей функции stacking в исходных признаках — разные объекты. Аналогично, агрегирование OHE/индикаторов обратно к исходным полям нужно описать и делать по реальным именам, а не по позициям из другой модели.

## Признаки наличия истории и fairness

Индикаторы в Level 4: `HAS_BUREAU`, `HAS_BB`, `HAS_POS`, `HAS_CC`, `HAS_INSTALLMENTS`, `HAS_PREVIOUS`. Им соответствуют счётчики `BUREAU_RECORD_COUNT`, `BB_RECORD_COUNT`, `POS_RECORD_COUNT`, `CC_RECORD_COUNT`, `INSTALLMENTS_RECORD_COUNT`, `PREVIOUS_RECORD_COUNT` и агрегаты каждого источника. `APP_EMPLOYMENT_DATE_UNKNOWN` отмечает обработанную заглушку занятости. Отсутствие истории нельзя переинтерпретировать как фактически нулевой долг/платёж: пропуски и индикаторы сохранены отдельно.

Fairness-колонки не входили в X. Используйте их для группового анализа объяснений через ID, не добавляйте обратно в модель. В общем test неизвестен пол у двух клиентов; неизвестного возраста нет. Неизвестные группы остаются отдельными и участвуют в общих метриках, но не в бинарном разрыве F/M. Возраст ровно 25 лет относится к `>=25`.

## План согласованного SHAP-анализа

1. Один раз выберите и сохраните одинаковые test-ID для всех сравниваемых моделей/уровней. Сохраняйте порядок и присоединяйте результаты по ID. Фиксируйте seed и правило выбора, а не только число клиентов.
2. Background выбирайте из `D/train_ids.parquet`/`D/X_train_level4.parquet` (это development-подвыборка пилота), без test. Один сохранённый список background-ID и seed используйте в сопоставимых объяснениях; различия метода/ограничений объяснителя документируйте.
3. До расчёта согласуйте объясняемый выход: **вероятность дефолта** либо **log-odds дефолта**, выбор link/output, baseline и способ маскирования. Нельзя молча сопоставлять SHAP вероятности одной модели с SHAP log-odds другой. Различия между объяснителями и пространствами признаков обозначайте явно.
4. Сохраняйте тип/версию объяснителя, параметры, seed, ID объясняемых клиентов и background, имена признаков, ожидаемое значение, выход модели и проверку аддитивности в выбранной шкале. Предобработку берите из готового Pipeline; повторный fit запрещён в текущем этапе.
5. Интерпретируйте сравнения как межмодельные объяснения фиксированных fits. Оценка устойчивости после переобучения требует отдельного согласованного расширения плана; повторный запуск SHAP этой вариативности не измеряет.

## MLP, статья и ограничения

Настройки выбранного MLP и его фактический `n_iter_`, `loss_curve_` находятся в `P/mlp/level_<N>/fit_checkpoint/fit_diagnostics.json`; search+refit warnings — рядом в `warnings.json`. Публичная сводка — `reports/pilot_20k_5config/evaluation/mlp_actual_diagnostics.csv`. Фактические эпохи L1–L4: **300, 300, 300, 243**. Предупреждения CV-кандидатов не доказывают достижение max_iter выбранным refit; train loss не тождественна внешней log-loss. Не исключайте MLP из сравнения и не объявляйте причиной слабого качества переобучение или ошибку предобработки без новых подтверждений.

Основные ограничения RQ1/RQ2 остаются: 20k train вместо полного development, пять конфигураций вместо двадцати, один refit вместо пяти повторов; CI exploratory, не учитывают обучение; fairness-коррекции и калибратор отложены. Полное правило A2 выбора модели не подтверждено. CPU/RAM обучения не записаны. При сборке статьи нужно добавить раздел RQ3, согласовать обозначения и шкалы SHAP, проверить LaTeX в шаблоне и сохранить эти ограничения. Целевую выборку и результаты RQ1/RQ2 ради SHAP не подменять.
