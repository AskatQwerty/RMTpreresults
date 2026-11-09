# RMT: кредитный риск Home Credit

**Актуальные результаты: `pilot_20k_20config_mlp_updated`.** Набор содержит 28 готовых Pipeline: семь моделей × четыре уровня. Четыре MLP заменены результатами дополнительного исследования; остальные 24 модели совпадают с итоговыми моделями поиска по 20 конфигурациям.

- [Отчёт RQ1/RQ2, profit и fairness](reports/pilot_20k_20config_mlp_updated/evaluation/results_report.html)
- [Таблицы CSV / LaTeX](reports/pilot_20k_20config_mlp_updated/evaluation/tables) · [Графики](reports/pilot_20k_20config_mlp_updated/evaluation/figures)
- [Архив всех 28 моделей и инструкция загрузки](model_bundles/README.md)
- [Указатель актуальной версии](docs/active_results.json)

Источник: [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data). Единица анализа — текущая заявка; `TARGET=1` — дефолт. Основная PR-метрика — **PR-AUC (average precision, AP)**. Исходные CSV, готовые X, ID клиентов, индивидуальные прогнозы и bootstrap-выборки в Git не публикуются. Отдельное исключение — проверенный архив обученных Pipeline без клиентских записей.

## Данные и обучение

Сохранены development **246 008** и test **61 503** клиента. Все модели используют те же **20 000** train-клиентов: 18 385 класса 0 и 1 615 класса 1, без изменения баланса. Test содержит 56 538 / 4 965 клиентов классов 0/1. Общие сохранённые три стратифицированных CV-фолда, seed 42, не менялись.

| Уровень | Добавленные данные | Признаков до кодирования |
|---|---|---:|
| 1 | application и признаки анкеты | 124 |
| 2 | bureau, bureau_balance | 178 |
| 3 | POS_CASH_balance, credit_card_balance | 264 |
| 4 | installments_payments, previous_application | 322 |

Модели: Logistic Regression L2, Random Forest, XGBoost, LightGBM, MLP, soft voting и stacking. Базовые модели ансамблей — LR, XGBoost и LightGBM. Предобработка обучается внутри каждого train, включая OOF-модели stacking. TARGET, ID, пол и прямое представление возраста исключены из признаков.

Завершены **1 680 внешних CV-обучений**: 28 пар × 20 конфигураций × 3 фолда. [Исходные оценки и выбранные параметры](reports/training/pilot_20k_20config_combined/) сохранены. Затем были обучены 28 итоговых моделей на всех 20 000 клиентах.

Для MLP дополнительно проверены четыре заранее заданных варианта × четыре уровня × три прежних фолда = **48 CV-оценок**, затем обучены четыре итоговых MLP. Внутренние 15% текущего train служат для выбора числа эпох по validation log-loss; после этого новая предобработка и сеть обучаются на всём train. Активные MLP ограничивают числовые z-значения до ±5; alpha=0,00001 на Level 1 и 0,1 на Levels 2–4. Итоговые эпохи: **27, 24, 16, 21**.

Это изменение протокола MLP после просмотра прежнего test. Варианты выбирались по CV до повторной оценки test, но анализ остаётся **дополнительным исследовательским**. Активные MLP нельзя описывать как неизменённых победителей исходного поиска по 20 конфигурациям. Пять повторных обучений A2 выполняются отдельно и не подменяются bootstrap-интервалами.

## Обновлённая оценка

Оценка использует сохранённые вероятности; при замене моделей новые fit/predict не выполнялись. Пересчитаны **1 000 общих парных bootstrap-повторов**, seed 42. 95% percentile CI — pointwise, unadjusted, exploratory, при фиксированных моделях.

Обновлены качество, прирост AP между уровнями, калибровка, шесть DeLong-сравнений ROC-AUC с LR на Level 4 с Bonferroni, условная прибыль и fairness. Пороги прежние: **0,5; 1/6; 1/11**. Одобрение — `p(default) < threshold`; прибыль в условных единицах `m=1`, `L=1,5,10` вычисляется по TARGET. Fairness использует прежние группы F/M и возраста <25/≥25; неизвестные группы остаются в общих метриках, но исключаются из бинарных разрывов.

MLP Level 4: test log-loss **0,249034**, AP **0,235569**, ROC-AUC **0,754453**. Точные интервалы, profit и fairness — в актуальном отчёте. [Отчёт об изменении MLP](reports/mlp_repair_20k_v1/results_report.html) сохраняет сравнение с прежней версией.

## Архивы и RQ3

- [Пререзультаты пяти конфигураций](reports/pilot_20k_5config/evaluation/results_report.html) и [их прежний README](docs/archive/README_pilot_20k_5config.md) сохранены.
- [Результаты 20 конфигураций до замены MLP](reports/pilot_20k_20config_final/evaluation/results_report.html) сохранены как предыдущая версия.
- [Ноябрьский RQ3 / SHAP](rq3_november/README.md): все семь финальных моделей уровня 4 и пять переобучений каждой модели объяснены на одних и тех же 1 000 тестовых клиентах. [Результаты устойчивости](rq3_november/rq3/out/tables/rq3_stability.csv) · [Октябрьский RQ3](rq3/README_RQ3.md).

## Воспроизведение

Python **3.13.9**, версии в [requirements.txt](requirements.txt) и [docs/software_versions.json](docs/software_versions.json). Для графиков дополнительно установите `requirements-reporting.txt`. Готовые модели загружаются без нового обучения; клиентские данные передаются отдельно.

- `scripts/import_pilot_cv_results.py` объединяет проверенные 1 680 CV-fit.
- `scripts/run_final_pilot_background.py` выполняет первоначальные 28 итоговых обучений.
- `scripts/run_mlp_repair.py` по умолчанию проверяет конфигурацию; `--mode run` запускает или продолжает дополнительное исследование MLP.
- `scripts/promote_repaired_mlp.py` собирает активную версию из сохранённых моделей; существующую версию не перезаписывает.
- `python -m src.updated_pilot_evaluation` пересчитывает актуальную оценку из локальных сохранённых вероятностей, без fit.
- `scripts/validate_updated_pilot.py` проверяет метрики, происхождение моделей, profit/fairness и целостность.
- `scripts/build_active_model_bundle.py` экспортирует модели без клиентских записей.

Notebook 01–04 сохраняют первоначальную подготовку и пилот пяти конфигураций. Их outputs исторические; для актуальной статьи используйте `reports/pilot_20k_20config_mlp_updated/evaluation/`. Не запускайте обучение заново ради просмотра отчётов.

## Лицензия и зафиксированная версия

Код проекта опубликован под [MIT License](LICENSE). Версия для ссылки в статье: [v1.0](https://github.com/AskatQwerty/RMTpreresults/releases/tag/v1.0).
