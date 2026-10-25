"""Создать отдельный notebook исследовательского пилота 20k / 5 конфигураций.

Builder только записывает notebook; обучение выполняется отдельным явным запуском.
Существующий notebooks/02_model_training.ipynb не изменяется.
"""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source, **metadata):
    cells.append(nbf.v4.new_code_cell(source, metadata=metadata))


md("""# 03. Исследовательский пилот: 20 000 заявок, 5 конфигураций

Этот notebook описывает отдельный эксперимент со scope **`pilot_20k_5config`**. Он использует готовые признаки четырёх уровней и одну сохранённую стратифицированную подвыборку из development. Исходные X, development/test, пять основных train-подвыборок и CV-folds полного исследования остаются прежними.

Для каждой из 28 пар «модель + уровень» выполняется один RandomizedSearchCV: **5 конфигураций × 3 CV-фолда**, выбор по `neg_log_loss`, затем **один встроенный refit** лучшего Pipeline на всей пилотной train-выборке. Результат — 420 внешних CV-fit и 28 итоговых refit при успешном выполнении без повторов прерванных этапов. Внутренние обучения voting/stacking учитываются отдельно: всего 1 344 примитивных fit.

Полный план Assignment 2 — 28 поисков по 20 конфигураций и 140 основных обучений — сохраняется отдельно в notebook 02. Здесь нет пяти повторных обучений каждой пары. Пилот позволяет получить предварительные сравнения, но не измеряет устойчивость результатов по пяти train-подвыборкам.

`RUN_PILOT=False` по умолчанию. Создание notebook, инициализация ID/CV и чтение отчётов не запускают обучение. Для чтения существующих результатов достаточно конфигурации и последней отчётной ячейки.""")

code("""from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display

ROOT=Path.cwd()
if ROOT.name=='notebooks':
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))

from src.research_pilot import initialize_pilot, run_research_pilot
from src.pilot_reporting import export_pilot_reports
from src.model_pipelines import experiment_spec

PREPARED=ROOT/'data/processed/full'
PILOT_ROOT=ROOT/'artifacts/training/pilot_20k_5config'
REPORT_DIR=ROOT/'reports/pilot_20k_5config'
SCOPE='pilot_20k_5config'
SAMPLE_N=20000
THREADS=2
RUN_PILOT=False  # True только для явного запуска или продолжения этого пилота.

print('Scope:',SCOPE)
print('Артефакты:',PILOT_ROOT)
print('Отчёты:',REPORT_DIR)
print('Запуск обучения включён:',RUN_PILOT)""", tags=['pilot_configuration'])

md("""## 1. Одна общая пилотная выборка и собственные CV-folds

`initialize_pilot` создаёт и сохраняет стратифицированную выборку из 20 000 development-ID, общую для всех моделей и уровней. Для неё один раз создаётся собственная `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`; назначения сохраняются в `cv_folds.parquet` внутри каталога пилота. При продолжении используются уже сохранённые ID и фолды.

Это новые пилотные CV-folds; назначения полного development не переписываются. Test не участвует в выборе пилотной выборки или формировании фолдов. На этом этапе допускаются чтение схем, ID и проверка принадлежности разбиениям; модели не обучаются.

Следующая ячейка показывает фактический состав выборки из `cohort_summary.json`. Значения не выводятся из запланированного размера и не подменяются ожидаемыми числами.""")

code("""pilot_context=initialize_pilot(PREPARED,PILOT_ROOT,threads=THREADS,sample_n=SAMPLE_N)
cohort_path=PILOT_ROOT/'cohort_summary.json'
with cohort_path.open(encoding='utf-8') as stream:
    cohort_summary=json.load(stream)
print('Размер пилотной выборки:',cohort_summary['sample_n'])
print('Стратифицированный выбор, seed:',cohort_summary['random_state'])
print('Баланс TARGET в пилотной выборке:')
display(pd.DataFrame(cohort_summary['cohort_balance']))
print('Общие пилотные CV-folds:')
display(pd.DataFrame([{key:fold[key] for key in ['fold','train_rows','validation_rows']}
                      for fold in cohort_summary['cv']]))
print('Manifest:',PILOT_ROOT/'experiment_manifest.json')
print('Сохранённые пилотные CV-folds:',PILOT_ROOT/'cv_folds.parquet')""", tags=['pilot_initialize_without_fit'])

md("""## 2. Порядок выполнения и бюджет

Сначала выполняются четыре уровня Logistic Regression, затем четыре уровня LightGBM, XGBoost, soft voting, stacking, MLP и Random Forest. Пары запускаются последовательно в отдельных worker-процессах. Сначала обрабатываются более быстрые семейства; порядок не зависит от результатов test.

Полные Pipeline и заранее заданные пространства гиперпараметров сохраняются в manifest. Каждый поиск получает необученный Pipeline целиком. `refit=True` обучает выбранный Pipeline один раз после CV; отдельный ручной `fit` после поиска не требуется.

Числа ниже — план успешного исполнения. Если прерванный поиск приходится повторить, фактическое число попыток больше; это должно быть видно в сохранённых событиях и отчёте.""")

code("""MODEL_ORDER=['logistic_regression','lightgbm','xgboost','soft_voting','stacking','mlp','random_forest']
plan_rows=[]
for model in MODEL_ORDER:
    primitive_per_fit=experiment_spec(model)['primitive_fits_per_outer_fit']
    plan_rows.append({'model':model,'levels':4,'searches':4,'configurations_per_search':5,
        'cv_folds':3,'candidate_cv_fits':4*5*3,'final_refits':4,
        'primitive_fits_including_ensembles':4*(5*3+1)*primitive_per_fit})
pilot_plan=pd.DataFrame(plan_rows)
display(pilot_plan)
assert int(pilot_plan['searches'].sum())==28
assert int(pilot_plan['candidate_cv_fits'].sum())==420
assert int(pilot_plan['final_refits'].sum())==28
assert int(pilot_plan['primitive_fits_including_ensembles'].sum())==1344
print('Всего: 28 поисков, 420 CV-fit, 28 встроенных refit; 1 344 примитивных fit.')""")

md("""## 3. Предобработка и итоговая оценка

Числовые медианы, обработка полностью пустого train-столбца, индикаторы пропусков, заполнение категорий, OneHotEncoder и необходимое масштабирование входят в Pipeline. На каждом CV-fit они обучаются только на обучающей части фолда. Validation и test преобразуются сохранёнными статистиками; неизвестные категории обрабатываются без повторного fit.

У stacking каждая базовая модель содержит собственную предобработку также во внутренних OOF-fit. Параметры выбираются только по средней CV log-loss. TARGET и ID не являются входными признаками; пол и возраст остаются вне X согласно подготовке A2.

После поиска fitted `best_estimator_` сохраняется **до обращения к test для прогнозирования**. Итоговые вероятности и метрики рассчитываются на существующем test; параметры, правила обработки и порог классификации по этим результатам не подбираются. Сравнения пилота следует интерпретировать с учётом единственной train-выборки и ограниченного поиска пяти конфигураций.

**RQ1 в A2 — выбор модели при совместной оценке точности, прибыли и fairness.** Пилот предоставляет описательные сравнения моделей, Brier score и калибровочные частоты. Для Level 4 из уже сохранённых вероятностей дополнительно вычисляются заранее заданные сценарии прибыли и групповых различий. Они не требуют нового обучения или вызова `predict`.

Нормировка: доход от погашенного кредита `m=1`, убыток при дефолте `L/m` равен **1, 5 или 10**. Для каждого отношения сравниваются фиксированный порог **0.5** и порог безубыточности **`1/(1+L/m)`**. Заявка одобряется строго при `p(default)<t`; при равенстве порогу отклоняется. Средняя реализованная прибыль по test считается по всем заявителям: `mean(approve * ((1-TARGET) - (L/m)*TARGET))`. Это условные единицы при заданных допущениях, а не реальные денежные доходы кредитора. При `L/m=1` оба правила намеренно совпадают.

Fairness-группы берутся из сохранённого файла: пол F/M и возраст **`<25` / `>=25`**, поэтому ровно 25 лет относится к старшей группе. В A2 равенство границе отдельно не определено; здесь сохраняется уже принятое правило подготовки. Неизвестные группы показываются отдельно и не входят в бинарные разрывы. Approval-rate gap и TPR gap — абсолютные разности долей. Положительный класс — дефолт (`TARGET=1`), поэтому TPR означает долю обнаруженных дефолтов среди фактических дефолтов группы. При нулевом знаменателе показатель остаётся неопределённым. Калибровочная таблица использует 10 фиксированных интервалов вероятности без обучения калибратора.

Показываются все семь моделей Level 4 без фильтра допустимости по точности. Полное правило A2 сначала проверяет точность, затем выбирает прибыль при ограничении fairness; этот выбор **не выполняется** в пилоте. Итоговый победитель, значимость различий, bootstrap CI, DeLong, Bonferroni и fairness-коррекции остаются для последующего полного анализа.

**RQ2 — польза кредитной и платёжной истории и способность моделей её использовать.** Ему соответствуют приросты метрик каждого типа модели при добавлении истории, прежде всего Levels 2–4 относительно application-only Level 1. Соседние уровни дают дополнительное описательное сравнение. Единичный пилотный прирост не доказывает значимость или устойчивость после повторных обучений. SHAP и RQ3 в этот пилот не входят.""")

md("""## 4. Явный запуск / продолжение пилота

Для запуска из чистого ядра установите `RUN_PILOT=True` в конфигурационной ячейке, выполните конфигурацию и ячейку ниже. Инициализация данных также выполняется контроллером, если ещё необходима. Массовый эксперимент notebook 02 этим вызовом не запускается.

Контрольные точки поиска и оценки разделены. Сначала сохраняются fitted Pipeline, выбранные параметры, CV-результаты и диагностика обучения. Затем выполняются прогнозирование и экспорт. Если прогнозирование или экспорт прерваны, уже сохранённая модель загружается без нового fit. Если прерван сам поиск до его контрольной точки, повторяется тот же поиск; завершённые пары пропускаются после проверки сохранённых артефактов.

Используйте тот же `PILOT_ROOT` для продолжения. Несовместимые данные, параметры или код нельзя смешивать в одном каталоге. Существующие результаты сохраняются, а состояние запуска определяется по артефактам на диске.""")

code("""if RUN_PILOT:
    pilot_run_summary=run_research_pilot(PREPARED,PILOT_ROOT,REPORT_DIR,threads=THREADS)
    display(pilot_run_summary)
else:
    print('Обучение не запускалось этой ячейкой: RUN_PILOT=False.')
    print('Для запуска/продолжения: изменить флаг в конфигурации и выполнить эту ячейку.')
print('Артефакты пилота:',PILOT_ROOT)""", tags=['research_pilot_explicit_opt_in'])

md("""## 5. Независимая сводка сохранённых результатов

Ячейка ниже использует только пути из конфигурации и файлы на диске. Ей не нужны `pilot_context`, `cohort_summary`, `pilot_plan` или `pilot_run_summary` в памяти; её можно выполнить после конфигурации из чистого ядра. Она не загружает Pipeline для обучения и не запускает fit.

Таблица `metrics.csv` содержит реальные завершённые оценки; `metrics.tex` предназначена для включения в отчёт. `report_metadata.json` фиксирует scope и полноту результата. Ниже показываются компактная сводка, предварительное сравнение RQ1 на Level 4 и RQ2 по четырём уровням. Значения и готовые разности берутся из CSV; метрики и прогнозы здесь не пересчитываются. Отсутствующие пары остаются незавершёнными и отображаются пропусками.

Для RQ1 AP, ROC-AUC, log-loss и Brier показывают только predictive accuracy и качество вероятностей. Таблицы сценариев прибыли и fairness сохранены отдельно в `decision_scenarios.csv` и `decision_groups.csv`; core-модель по этим результатам не выбирается. В RQ2 меньший log-loss и больший AP означают лучшее значение на данном test. Один refit на пару не даёт оснований для вывода о статистической значимости. При наличии завершённых MLP показываются фактические `n_iter_`, длина `loss_curve_` и сохранённые предупреждения.

Большие артефакты моделей, вероятностей и контрольных точек хранятся в `artifacts/`, уже исключённом из Git. Небольшие отчётные таблицы находятся в `reports/pilot_20k_5config/`.""")

code("""if (PILOT_ROOT/'experiment_manifest.json').exists():
    pilot_disk_report=export_pilot_reports(PILOT_ROOT,REPORT_DIR)
    metadata_path=REPORT_DIR/'report_metadata.json'
    if metadata_path.exists():
        with metadata_path.open(encoding='utf-8') as stream:
            report_metadata=json.load(stream)
    else:
        report_metadata=pilot_disk_report
    status_fields={'scope':'Scope','status':'Состояние','completed_pairs':'Завершено пар',
                   'pending_pairs':'Ожидают','invalid_pairs':'Повреждено'}
    display(pd.DataFrame([{title:report_metadata.get(key) for key,title in status_fields.items()}]))

    def read_report_csv(name):
        path=REPORT_DIR/name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    comparison=read_report_csv('rq1_model_comparison.csv')
    saved_metrics=read_report_csv('metrics.csv')
    print('RQ1 — Level 4: предварительное сравнение качества, без выбора core-модели.')
    if not comparison.empty:
        level4=comparison.loc[comparison['level']==4].copy()
        if not saved_metrics.empty and 'search_and_refit_seconds' in saved_metrics:
            runtime=saved_metrics.loc[saved_metrics['level']==4,['model','search_and_refit_seconds']]
            level4=level4.merge(runtime,on='model',how='left',validate='one_to_one')
        rq1_labels={'model_label':'Модель','status':'Статус','test_pr_auc_average_precision':'AP',
            'test_roc_auc':'ROC-AUC','test_log_loss':'Log-loss','test_brier_score':'Brier',
            'search_and_refit_seconds':'Поиск + refit, с'}
        rq1_table=level4[[name for name in rq1_labels if name in level4]].rename(columns=rq1_labels)
        formats={name:'{:.4f}' for name in ['AP','ROC-AUC','Log-loss','Brier'] if name in rq1_table}
        if 'Поиск + refit, с' in rq1_table:
            formats['Поиск + refit, с']='{:.1f}'
        display(rq1_table.style.format(formats,na_rep='—'))
    else:
        print('Сохранённой таблицы RQ1 пока нет.')

    history=read_report_csv('rq2_history_gains.csv')
    print('RQ2 — сравнение четырёх уровней; разности Δ L4−L1 уже сохранены в CSV.')
    if not history.empty:
        model_order=history['model_label'].drop_duplicates().tolist()
        for metric,delta,title in [
            ('test_log_loss','delta_log_loss_vs_level1','Log-loss: меньше — лучше'),
            ('test_pr_auc_average_precision','delta_pr_auc_average_precision_vs_level1','AP: больше — лучше')]:
            table=history.pivot(index='model_label',columns='level',values=metric)
            table=table.reindex(index=model_order,columns=[1,2,3,4])
            table.columns=['Level 1','Level 2','Level 3','Level 4']
            table['Δ L4−L1']=history.loc[history['level']==4].set_index('model_label')[delta]
            table.index.name='Модель'
            print(title)
            display(table.style.format('{:.4f}',na_rep='—'))
    else:
        print('Сохранённой таблицы RQ2 пока нет.')

    mlp_diagnostics=read_report_csv('mlp_diagnostics.csv')
    if not mlp_diagnostics.empty:
        print('MLP — фактические эпохи и сохранённая кривая обучения:')
        diagnostic_labels={'model':'Модель','level':'Level','n_iter_':'Фактически эпох',
            'max_iter':'Лимит эпох','loss_curve_points':'Точек loss_curve_',
            'loss_curve_length_matches_n_iter':'Длина кривой = n_iter_'}
        display(mlp_diagnostics[[name for name in diagnostic_labels if name in mlp_diagnostics]]
                .rename(columns=diagnostic_labels))
    else:
        print('Сохранённой диагностики завершённых MLP пока нет.')
    saved_warnings=read_report_csv('warnings.csv')
    mlp_warnings=saved_warnings.loc[saved_warnings['model']=='mlp'] if not saved_warnings.empty else pd.DataFrame()
    if not mlp_warnings.empty:
        print('Сохранённые предупреждения MLP (полный текст — warnings.csv):')
        with pd.option_context('display.max_colwidth',140):
            display(mlp_warnings[['level','stage','category','count','message']])
    elif not mlp_diagnostics.empty:
        print('Предупреждений MLP в сохранённом warnings.csv нет.')
    print('Один refit на пару: победитель и статистическая значимость здесь не определяются.')
    print('Сохранённые отчётные файлы:')
    for report_path in sorted(REPORT_DIR.glob('*')):
        if report_path.is_file():
            print(report_path)
else:
    print('Пилот ещё не инициализирован; отчётных результатов нет.')""", tags=['pilot_disk_reports_independent'])

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python'},
    'research_scope': 'pilot_20k_5config',
})
destination = ROOT/'notebooks/03_pilot_20k_5config.ipynb'
nbf.write(notebook, destination)
print(f'Created {destination}; {len(cells)} cells; RUN_PILOT=False')
