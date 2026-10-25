"""Build the independent saved-prediction evaluation notebook; never run training."""
from pathlib import Path

import nbformat as nbf


ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text,**metadata): cells.append(nbf.v4.new_code_cell(text,metadata=metadata))


md("""# 04. Предварительная оценка RQ1/RQ2: сохранённый пилот 20k

Здесь анализируются **только сохранённые test-вероятности** 28 Pipeline пилота `pilot_20k_5config`. Новых `fit`, `predict`, поиска гиперпараметров, калибратора или fairness-коррекций нет. X, разбиения, параметры, модели и прогнозы остаются неизменными. Ресурсный пилот в эти сравнения не входит.

Выборка обучения пилота: 20 000 клиентов development, пять конфигураций и один refit на пару. Это предварительные результаты Assignment 3, а не замена полного A2 с 20 конфигурациями и пятью повторными обучениями. Остальные development-заявки не использованы при обучении этих моделей.

`RUN_ANALYSIS=False` по умолчанию. Явный запуск анализа безопасен для завершённых bootstrap-checkpoint: уже рассчитанные повторы не пересчитываются. Для просмотра существующих материалов достаточно выполнить конфигурацию и последнюю отчётную ячейку.""")

code("""from pathlib import Path
import json,sys
import pandas as pd
from IPython.display import display,SVG

ROOT=Path.cwd()
if ROOT.name=='notebooks':
    ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))

from src.pilot_evaluation import run_evaluation
from src.pilot_evaluation_reporting import export_evaluation_report

PREPARED=ROOT/'data/processed/full'
PILOT_ROOT=ROOT/'artifacts/training/pilot_20k_5config'
PREVIOUS_REPORTS=ROOT/'reports/pilot_20k_5config'
EVALUATION_DIR=PREVIOUS_REPORTS/'evaluation'
N_BOOTSTRAP=1000
SEED=42
RUN_ANALYSIS=False

print('Оценка сохранённых вероятностей; новых fit/predict нет.')
print('Пилот:',PILOT_ROOT)
print('Новые результаты:',EVALUATION_DIR)
print('Запуск анализа включён:',RUN_ANALYSIS)""",tags=['evaluation_configuration'])

md("""## 1. Согласование test-ID и проверка исходных оценок

Вероятности всех 28 сочетаний явно согласуются с сохранёнными test-ID и TARGET. Проверяются уникальность и одинаковый состав 61 503 клиентов, положительный класс `TARGET=1`, отсутствие NaN/inf и диапазон [0,1]. Основная PR-метрика во всём анализе — **PR-AUC (average precision, AP)**, рассчитанная через `average_precision_score`.

AP, ROC-AUC, log-loss и Brier пересчитываются и сравниваются с прежним `rq1_model_comparison.csv`. Несогласие не скрывается: подробности находятся в `existing_metrics_comparison.csv`, а проверки ID — в `alignment_validation.json`. Входные хэши сохраняются до и после анализа. Повторный аудит исходных CSV не выполняется.

Precision с нулевым знаменателем остаётся неопределённой. Например, если модель не предсказывает дефолтов, это не precision=0. Recall и F1 могут быть равны нулю при наличии фактических дефолтов; такие нули сохраняются.""")

md("""## 2. Общий парный bootstrap и сравнения

Выполняются 1 000 повторов с `default_rng(42)`. Единица ресемплирования — клиент test. Одинаковые клиентские веса на каждом повторе применяются ко всем моделям и уровням. Веса и состояние RNG сохраняются по повтору, поэтому анализ продолжается с checkpoint после прерывания.

95% percentile-интервалы берутся по квантилям 0.025/0.975. Они обозначаются **pointwise, unadjusted, exploratory** и отражают только test-неопределённость фиксированных обученных моделей. Они не учитывают повторное обучение и не являются средним ± SD пяти train-запусков.

Сравнения AP: шесть моделей против LR на Level 4; внутри каждой модели L2/L3/L4 относительно L1, а также L2−L1, L3−L2, L4−L3. Все разности используют уже общие bootstrap-результаты. Bootstrap p-values не вычисляются.

Для ROC-AUC отдельно выполняется парный двухсторонний DeLong: шесть сравнений с LR на Level 4 и Bonferroni в этом семействе. DeLong не является тестом AP. Отсутствие значимой разницы не означает эквивалентность, а правило A2 «не хуже лучшего по AP» не считается формально выполненным. Эти детали анализа зафиксированы сейчас, но не объявляются заранее зарегистрированными.""")

md("""## 3. Прибыль, fairness, калибровка и ошибки

На Level 4 рассматриваются m=1 и L=1,5,10. Порог фиксирован: 0.5 либо m/(m+L). При L=1 оба совпадают, поэтому результат сохраняется один раз: всего пять сценариев на модель. Одобрение означает `p_default < t`, предсказанный дефолт — `p_default >= t`.

Прибыль на всех заявителей: `mean(approve * ((1-TARGET)*m - TARGET*L))`. Для отклонённых прибыль равна нулю. Используется наблюдённый TARGET, а не ожидаемый исход из вероятности. Approve-all/reject-all — ориентиры. Это условные единицы, не фактическая банковская прибыль; пороги по test не оптимизируются.

Fairness-группы присоединяются по ID: пол и возраст `<25` / `>=25`. Неизвестные группы показаны отдельно; они исключены из бинарных разрывов, но остаются в общих метриках и прибыли. TPR относится к обнаружению фактического дефолта (`TARGET=1`). Оба наблюдаемых **approval-rate gap ≤5 п.п.** проверяются отдельно по полу и возрасту; TPR-gap не подменяет это ограничение. После просмотра результатов предел не ослабляется.

Калибровочные кривые используют десять фиксированных равных интервалов вероятности; последняя правая граница 1 включена. Число наблюдений каждого bin сохранено. Калибратор не обучается. Ошибки сохраняются для всех 28 сочетаний при t=0.5 и для Level 4 при порогах прибыли.

MLP остаётся в таблицах. Фактические `n_iter_`, настройки и предупреждения читаются из сохранённых журналов. Предупреждения всего search+refit нельзя приписывать только выбранному refit. Train loss_ и внешняя log-loss не тождественны; слабое качество само по себе не доказывает переобучение или ошибку предобработки.""")

md("""## 4. Явный запуск или продолжение анализа — без обучения

Чтобы выполнить анализ, установите `RUN_ANALYSIS=True` в конфигурации и запустите следующую ячейку. Она читает готовые вероятности и продолжает незавершённые bootstrap-повторы; завершённые bootstrap-повторы не вычисляются заново. Проверки входов, точечные сводки и интервалы могут быть обновлены из сохранённых данных и checkpoint. Новые файлы создаются только в `reports/pilot_20k_5config/evaluation/`.""")

code("""if RUN_ANALYSIS:
    evaluation_result=run_evaluation(PREPARED,PILOT_ROOT,PREVIOUS_REPORTS,EVALUATION_DIR,
                                     n_bootstrap=N_BOOTSTRAP,seed=SEED)
    print('Статус анализа:',evaluation_result.get('status'))
    print('Готово bootstrap-повторов:',evaluation_result.get('completed_bootstrap'))
    print('Новых fit/predict:',evaluation_result.get('new_fit_calls'),evaluation_result.get('new_predict_calls'))
else:
    print('Эта ячейка не запускала расчёты: RUN_ANALYSIS=False.')
    print('Существующие отчёты доступны в следующей ячейке.')""",tags=['evaluation_explicit_opt_in'])

md("""## 5. Независимый отчёт и материалы для статьи

Эта ячейка зависит только от конфигурации и сохранённых файлов: ей не нужен `evaluation_result`. Она обновляет CSV/LaTeX и читаемый HTML/TXT из рассчитанных результатов. Bootstrap, fit, predict и построение новых PDF здесь не запускаются.

Основные таблицы — RQ1 Level 4 с качеством и CI; AP четырёх уровней и парные приросты для RQ2; прибыль/fairness по фиксированным сценариям. Подробности остаются в полном CSV и приложении. Служебные Status/reason исключены из таблиц для статьи. Подписи таблиц и графиков английские, пояснения notebook и читаемого отчёта русские.

PDF/SVG рисунки сохраняются отдельным контролируемым экспортом и проходят визуальную проверку. Здесь только отображаются уже готовые SVG. `results_readiness.json` различает готовые материалы и конкретные незавершённые проверки. Отсутствующее описание оборудования обучения отмечается как ограничение документации и не подменяется текущей машиной.""")

code("""readiness=export_evaluation_report(PILOT_ROOT,EVALUATION_DIR,previous_reports=PREVIOUS_REPORTS,
                                   create_figures=False)
print('Статус материалов:',readiness.get('status'))
if readiness.get('missing_tables'):
    print('Ещё нет таблиц анализа:',', '.join(readiness['missing_tables']))
else:
    print('Bootstrap-повторов:',readiness.get('completed_bootstrap'))
    print('Готово для предварительного раздела статьи:',readiness.get('ready_for_article'))
    for message in readiness.get('blockers',[]):
        print('Осталось:',message)
    for name,title in [('rq1_level4','RQ1: качество Level 4'),
                       ('rq2_ap_levels','RQ2: AP четырёх уровней'),
                       ('rq2_baseline_gains','RQ2: парные приросты AP относительно L1'),
                       ('profit_fairness_L5','Фиксированный сценарий L/m=5; остальные сценарии в полном отчёте'),
                       ('errors_level4','Ошибки Level 4 при t=0.5')]:
        path=EVALUATION_DIR/'tables'/f'{name}.csv'
        if path.exists():
            print(title)
            display(pd.read_csv(path).fillna('—'))
    mlp_path=EVALUATION_DIR/'mlp_actual_diagnostics.csv'
    if mlp_path.exists():
        print('MLP: фактические эпохи и предупреждения всего search+refit')
        mlp=pd.read_csv(mlp_path)
        display(mlp[['level','n_iter_','max_iter','loss_curve_points','warning_count_search_and_refit']])
    for name in ['ap_by_level','calibration_level4']:
        path=EVALUATION_DIR/'figures'/f'{name}.svg'
        if path.exists():
            display(SVG(filename=str(path)))
    print('Читаемый отчёт:',EVALUATION_DIR/'results_report.html')
    print('Текстовый отчёт:',EVALUATION_DIR/'results_report.txt')
    print('Готовность и ограничения:',EVALUATION_DIR/'results_readiness.json')
    print('Таблицы CSV/LaTeX:',EVALUATION_DIR/'tables')
    print('Рисунки PDF/SVG:',EVALUATION_DIR/'figures')
print('Новые обучения и новые predict в этом notebook не выполняются.')""",tags=['evaluation_disk_reports_independent'])

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
    'language_info':{'name':'python'},'research_scope':'pilot_20k_5config_evaluation'})
destination=ROOT/'notebooks/04_pilot_results_analysis.ipynb'
nbf.write(notebook,destination)
print(f'Created {destination}; RUN_ANALYSIS=False; no calculations executed.')
