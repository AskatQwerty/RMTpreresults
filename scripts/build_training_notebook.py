"""Создать 02_model_training.ipynb: первоначальный запуск строго технический."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(source): cells.append(nbf.v4.new_markdown_cell(source))
def code(source,**metadata): cells.append(nbf.v4.new_code_cell(source,metadata=metadata))

md("""# 02. Обучение моделей: Pipeline и протокол A2

Используются готовые `data/processed/full/`, четыре списка признаков, development/test, пять 90%-ных train-подвыборок и сохранённые CV-folds. **X и разбиения не создаются заново.** Пол и возраст остаются во внешнем fairness-файле; ID служит только индексом. TARGET передаётся отдельно.

Технический прогон, ресурсный пилот и полный эксперимент разделены. Массовый эксперимент отключён флагом `RUN_FULL_EXPERIMENT=False`; ресурсный пилот включается отдельным `RUN_RESOURCE_PILOT`. Технические результаты не являются оценкой качества полного эксперимента и не определяют его гиперпараметры. Итоговые сводки читают артефакты с диска и работают после запуска конфигурации из чистого ядра, даже если разделы технической проверки пропущены.

Реализация: `src/fold_preprocessing.py` — числовые пропуски; `src/model_pipelines.py` — полные Pipeline семи моделей; `src/model_training.py` — загрузка по ID, поиск, повторы, оценки и сохранение. Версии зависимостей закреплены в `requirements.txt`.""")

code("""from pathlib import Path
import sys, json, time, importlib.metadata
import pandas as pd
import psutil
from IPython.display import display

ROOT=Path.cwd()
if ROOT.name=='notebooks': ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from src.model_training import (PreparedData, full_experiment_plan, run_technical,
    run_full_experiment, atomic_json, sha256, PACKAGES)
from src.model_pipelines import MODEL_NAMES, make_estimator, parameter_space, experiment_spec
from src.data_quality import artifact_manifest
from src.training_reporting import write_full_report, write_technical_reports, write_resource_reports
from src.resource_pilot import run_resource_pilot

PREPARED=ROOT/'data/processed/full'
REPORTS=ROOT/'reports/training'
TECHNICAL_PARENT=ROOT/'artifacts/training/technical'
FULL_OUTPUT=ROOT/'artifacts/training/a2_full_v2'
RESOURCE_PARENT=ROOT/'artifacts/training/resource_pilot'
THREADS=2
TECHNICAL_TRAIN_N=1200
TECHNICAL_VALIDATION_N=400
RUN_FULL_EXPERIMENT=False  # Менять на True только для сознательного полного длительного запуска.
RUN_RESOURCE_PILOT=False  # Отдельные 7 fit с полными настройками; не основной эксперимент.
REPORTS.mkdir(parents=True,exist_ok=True)
NOTEBOOK_STARTED=time.perf_counter()
print('Python:',sys.version.split()[0])
print('Версии:',{name:importlib.metadata.version(name) for name in PACKAGES})
print('Доступная RAM, GiB:',round(psutil.virtual_memory().available/1024**3,2))
print('Полный эксперимент включён:',RUN_FULL_EXPERIMENT)""")

md("""## 1. Контракт данных и существующие разбиения

При загрузке проверяются ID, порядок TARGET, уникальность, непересечение development/test, принадлежность повторов development, покрытие CV-folds и вложенность списков признаков. Числовые/категориальные столбцы определяются **по сохранённой схеме**, без обучения категорий на всей популяции.

`PreparedData.load()` сканирует Parquet порциями и возвращает только запрошенные ID. Один физический блок файла может содержать разные split; до DataFrame/fit/score выбираются только нужные строки. В техническом режиме запрос test-ID запрещён assertion. SHA-256 входных файлов фиксируется перед работой и сравнивается после.""")
code("""data=PreparedData(PREPARED)
contract=data.contract_report()
frozen_before=artifact_manifest(PREPARED)
atomic_json(REPORTS/'input_contract.json',contract)
atomic_json(REPORTS/'prepared_sha256_before.json',frozen_before)
display(pd.DataFrame(contract['levels']))
display(pd.DataFrame(contract['cv']))
display(pd.DataFrame(contract['repeats']))
print('Development:',contract['development'],'test:',contract['test'])
print('Непересечение, ID/TARGET, вложенность и сохранённые split: проверены.')""")

md("""## 2. Полная предобработка внутри CV

**Числа.** Используется существующий `make_numeric_pipeline`: медианы fit только на train текущего фолда/повтора; полностью пустой train-столбец сохраняется с фиксированным fallback 0. Отдельный индикатор исходного пропуска создаётся для каждого числового столбца, включая новые пропуски в validation. Реальные значения validation/test не заменяются повторно обученной статистикой.

**Категории.** Детерминированное приведение непустых категорий к строкам и None/pd.NA к NaN → `SimpleImputer(strategy='constant', fill_value='__RMT_MISSING__', keep_empty_features=True)` → `OneHotEncoder(handle_unknown='ignore')`. Это также обрабатывает реальный Boolean-столбец `EMERGENCYSTATE_MODE` в сохранённом Parquet: булевы значения и строковый маркер пропуска не смешиваются в одном OHE-столбце. Категории encoder обучаются только на текущем train. Неизвестная категория получает нулевой вектор соответствующего OHE-блока; полностью пустой train-столбец не ломает схему.

**Масштабирование.** StandardScaler включён только в числовую ветку LR и MLP; параметры fit на текущем train, индикаторы не масштабируются. У ансамблей масштабируется собственная LR-база.

**Формат и память.** Используется float32; LR, RF, LightGBM и MLP получают CSR, XGBoost — плотный float32, поскольку XGBoost трактует отсутствующие элементы CSR как missing, а числовой 0 нужно сохранить. См. [официальное пояснение XGBoost](https://xgboost.readthedocs.io/en/stable/faq.html#why-do-i-see-different-results-with-sparse-and-dense-data). Плотная матрица XGB занимает `n_rows × n_encoded_features × 4` байт, плюс рабочие копии. Поиск и ансамбли работают последовательно (`n_jobs=1`), отдельные библиотеки/BLAS ограничены двумя потоками. Измерения RAM для заранее выбранных полных конфигураций находятся в ресурсном пилоте раздела 7 и сводке раздела 8; они не определяют потребление памяти всех кандидатов поиска. Для RF пространство заранее ограничено глубиной до 24 и минимумом 5 наблюдений в листе.

**Stacking.** Состав voting и stacking заранее фиксирован: LR + XGBoost + LightGBM. У каждой базы собственный полный Pipeline. В stacking нет общего imputer/encoder перед ансамблем: каждый внутренний OOF-fit заново обучает свою предобработку на внутреннем train. Метамодель — LR L2 на трёх OOF-вероятностях; `passthrough=False`, внутренний stratified 3-fold. После OOF базовые Pipeline отдельно fit на полном train соответствующего внешнего обучения. Сохранённый внешний validation/test ни в один из этих fit не входит.

Ни один `fit_transform` не вызывается заранее на полном development. Весь estimator Pipeline передаётся в RandomizedSearchCV.""")
code("""specs=pd.DataFrame([experiment_spec(name) for name in MODEL_NAMES])
display(specs[['model','base_models','full_parameter_configurations',
               'base_full_fits_per_outer_fit','base_oof_fits_per_outer_fit',
               'meta_fits_per_outer_fit','primitive_fits_per_outer_fit']])
numeric,categorical=data.columns(4)
unfitted_example=make_estimator('stacking',numeric,categorical,seed=42,threads=THREADS)
print('Stacking: внешний Pipeline содержит только',list(unfitted_example.named_steps))
print('Каждая база:',[(name,list(est.named_steps)) for name,est in unfitted_example.named_steps['model'].estimators])
print('Все гиперпараметрические пространства сохранены заранее; код не использует test для их изменения.')
atomic_json(REPORTS/'parameter_spaces.json',{name:parameter_space(name) for name in MODEL_NAMES})""")

md("""## 3. Полный согласованный эксперимент — спецификация

Семь моделей: Logistic Regression L2, Random Forest, XGBoost, LightGBM, MLP, soft voting и stacking. На каждой паре «модель + уровень» выполняется **один** RandomizedSearchCV: `n_iter=20`, сохранённые три CV-folds, `scoring='neg_log_loss'`, `random_state=42`, `refit=False`, `error_score='raise'`. Выбирается максимальный negative log-loss, то есть минимальный log-loss. Пространства конечные и содержат не менее 20 конфигураций; выбор без повторов.

После поиска его параметры фиксируются. Для seeds 42–46 создаются пять новых Pipeline и fit только на соответствующих сохранённых 90%-ных train-подвыборках. Меняется seed и train, но выбранные гиперпараметры остаются одинаковыми. Сохранённые медианы/категории не переносятся между повторами. Первичный протокол не использует веса классов, ресемплинг, калибровку, feature selection или раннюю остановку по test. MLP имеет `early_stopping=False`; количество итераций заранее задано, предупреждения сходимости сохраняются.

**Бюджет:** 28 поисков × 20 × 3 = 1 680 внешних CV-fit; 7 × 4 × 5 = 140 основных fit. `refit=False` не добавляет 28 промежуточных обучений на полном development. Один внешний voting-fit содержит 3 базовых fit; stacking-fit — 3 полных + 9 внутренних OOF-fit баз и 1 fit метамодели. Итого по плану **5 200 базовых + 260 метамоделей = 5 460 примитивных fit**. Это другой способ подсчёта той же работы, его нельзя складывать с 1 820 внешними вызовами. Повторы прерванных этапов могут увеличить фактическое число вызовов.""")
code("""plan=full_experiment_plan()
atomic_json(REPORTS/'full_experiment_plan.json',plan)
display(pd.DataFrame(plan['model_specs'])[['model','searches','candidate_cv_outer_fits','main_outer_fits','primitive_fits']])
print('Поисков:',plan['searches'],'CV-fit:',plan['candidate_cv_outer_fits'],
      'основных fit:',plan['main_outer_fits'],'примитивных fit:',plan['primitive_fits_including_ensembles'])""")

md("""### 3.1. Настройки и фактическая диагностика MLP

Полные рабочие настройки не изменены по техническим метрикам или test: `max_iter=300`, `tol=1e-4`, `n_iter_no_change=20`, `early_stopping=False`, `solver='adam'`. Фиксированная конфигурация ресурсного пилота использует `(64,32)` скрытых узлов, `alpha=0.001`, `batch_size=256`, `learning_rate_init=0.001`, seed 42. В основном поиске меняются только заранее заданные параметры пространства, а эти ограничения остановки остаются прежними.

`max_iter` — верхний предел, а фактическое число завершённых эпох читается из `n_iter_`. После каждого сохраняемого повторного обучения и fit ресурсного пилота в `fit_diagnostics.json` записываются `n_iter_`, полная `loss_curve_`, итоговая `loss_`; предупреждения, включая ConvergenceWarning, сохраняются в JSON этапа. Наличие кривой и число её элементов проверяются; сериализованная Pipeline также содержит эти атрибуты. RandomizedSearchCV с `refit=False` не сохраняет сами модели кандидатов: предупреждения поиска записываются агрегированно, отдельные кривые кандидатов CV не сохраняются. Ресурсный пилот не подбирает более удобные значения после измерений.""")
code("""mlp_numeric,mlp_categorical=PreparedData(PREPARED).columns(4)
mlp_full=make_estimator('mlp',mlp_numeric,mlp_categorical,seed=42,technical=False,threads=THREADS)
mlp_settings=mlp_full.named_steps['model'].get_params(deep=False)
mlp_display_keys=['hidden_layer_sizes','activation','solver','alpha','batch_size',
                  'learning_rate_init','max_iter','tol','n_iter_no_change','early_stopping','random_state']
display(pd.DataFrame([{'parameter':key,'full_mode_value':mlp_settings[key]} for key in mlp_display_keys]))
atomic_json(REPORTS/'mlp_full_settings.json',{key:mlp_settings[key] for key in mlp_display_keys})""")

md("""## 4. Технический прогон — только development

Для проверки выбираются 1 200 клиентов из **сохранённого** `train_seed_42`, а 400 validation-клиентов — из оставшейся части development, не входящей в этот train. Выбор клиентов стратифицирован по TARGET с seed 2026. Это малые технические подмножества; официальные split остаются неизменными. В техническом поиске используются сохранённые CV-назначения этих 1 200 ID, без генерации новых fold.

На Level 4 (все источники) проверяются все семь типов. Для каждого запускается уменьшенный поиск **2 конфигурации × 3 сохранённых fold** и один технический fit с выбранными параметрами; число деревьев, ширина/итерации MLP уменьшены. Это 42 технических внешних CV-fit и 7 технических итоговых fit, всего 147 примитивных fit с учётом ансамблей. Дополнительно механически проверены схемы всех четырёх уровней.

В fit, подбор и метрики этой проверки не попадает ни один test-клиент. Технические гиперпараметры и метрики не переносятся в полный эксперимент. Файлы пишутся в отдельный каталог с fingerprint контекста. Сохраняются реальные Pipeline, результаты поиска и вероятности с ID; каждая модель повторно загружается из joblib и проверяется на совпадение вероятностей.""")
code("""technical=run_technical(PREPARED,TECHNICAL_PARENT,
    train_n=TECHNICAL_TRAIN_N,validation_n=TECHNICAL_VALIDATION_N,threads=THREADS)
atomic_json(REPORTS/'technical_first_invocation.json',technical)
technical_root=Path(technical['root'])
technical_metrics=pd.read_parquet(technical_root/'technical_metrics.parquet')
technical_metrics.to_csv(REPORTS/'technical_metrics.csv',index=False)
display(technical_metrics[['model','train_rows','rows','pr_auc_average_precision','roc_auc',
                           'log_loss','brier_score','fit_seconds','warning_count']])
display(pd.DataFrame(technical['saved_cv_subset']))
print('Прочитанные для обучения/оценки подмножества:')
display(pd.DataFrame(technical['data_access']))
print('Test не использовался:',not any(item['contains_test'] for item in technical['data_access']))
print('Результаты технического запуска:',technical_root)
print('Время, секунд:',round(technical['elapsed_seconds_this_invocation'],2))""")

md("""## 5. Проверка resume и правила восстановления

Сохраняются атомарно `cv_results.parquet`, `best_params.json`, `pipeline.joblib`, `probabilities.parquet` (ID, TARGET, вероятность, модель, уровень, seed), `metrics.json`. Файл `complete.json` записывается **последним**, содержит SHA-256 всех артефактов этапа. Метрики: log-loss, PR-AUC как average precision, ROC-AUC, Brier, precision/recall/F1, accuracy и confusion matrix при фиксированном пороге 0,5. Profit-пороги, fairness и SHAP — последующий этап; по test здесь не выбираются пороги и правила.

Resume пропускает только завершённые поиски и отдельные повторы после проверки файлов. Незавершённый поиск RandomizedSearchCV начинает те же 20 конфигураций заново — восстановления внутри кандидата/фолда нет. Незавершённый повтор обучается заново. Ошибка сохраняется в `failure.json`, события — в `events.jsonl`; предупреждения, включая ConvergenceWarning, сохраняются отдельно в JSON этапа и не маскируются.

Manifest фиксирует SHA-256 подготовленных данных, split и Python-исходников, пространства поиска, версии пакетов и настройки. При изменении контекста существующий полный каталог отклоняется: нужно сохранить прежний код/окружение или выбрать новый `FULL_OUTPUT`. Исполняемый notebook не хэшируется, потому что его outputs меняются при запуске. Параллельная запись в один каталог блокируется `run.lock`. При аварийном закрытии ядра проверьте PID из lock и удалите только устаревший lock, когда процесс уже не работает.

После добавления диагностики обучения новый полный каталог — `a2_full_v2`. Предыдущий прерванный запуск `a2_full_v1` и его исходники сохранены отдельно; их manifest не переписывается. Состав моделей, X, разбиения и бюджет A2 остаются прежними.

Следующая ячейка реально повторяет техническую команду с тем же контекстом. Все семь поисков и семь fit должны быть возобновлены **без нового обучения**, а хэши сохранённых моделей должны совпасть.""")
code("""models_before={str(p.relative_to(technical_root)):sha256(p) for p in technical_root.rglob('pipeline.joblib')}
resume=run_technical(PREPARED,TECHNICAL_PARENT,
    train_n=TECHNICAL_TRAIN_N,validation_n=TECHNICAL_VALIDATION_N,threads=THREADS)
assert resume['run_counters']=={'new_searches':0,'resumed_searches':7,'new_fits':0,'resumed_fits':7}
models_after={str(p.relative_to(technical_root)):sha256(p) for p in technical_root.rglob('pipeline.joblib')}
assert models_before==models_after and len(models_after)==7
atomic_json(REPORTS/'technical_resume_check.json',{'counters':resume['run_counters'],
    'model_files_sha256_unchanged':True,'elapsed_seconds':resume['elapsed_seconds_this_invocation']})
print('Resume проверен:',resume['run_counters'])
warning_rows=[]
for stage_name,records in [('search',technical['searches']),('fit',technical['fits'])]:
    for record in records:
        for warning in record['warnings']:
            warning_rows.append({'stage':stage_name,'model':record['model'],**warning})
warnings_frame=pd.DataFrame(warning_rows,columns=['stage','model','category','message','count'])
warnings_frame.to_csv(REPORTS/'technical_warnings.csv',index=False)
display(warnings_frame)
print('Предупреждения сходимости означают необходимость контроля итераций в полном протоколе; технический прогон не доказывает сходимость полного обучения.')""")

md("""## 6. ПОЛНЫЙ ДЛИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ — ОТКЛЮЧЁН ПРИ ПЕРВОЙ ПРОВЕРКЕ

Для запуска установите **`RUN_FULL_EXPERIMENT=True`** в конфигурационной ячейке и выполните конфигурацию, затем ячейку ниже. Перезапуск всех предыдущих этапов не обязателен. Используйте прежний `FULL_OUTPUT` для resume; текущая версия кода и окружения должны совпадать с manifest.

Запуск выполняет весь утверждённый бюджет последовательно. Модели и вероятности могут занять значительное место. Измерения времени, RAM и размера моделей для фиксированных полных конфигураций сохранены ресурсным пилотом (разделы 7–8); по ним нельзя надёжно оценить все 20 кандидатов каждой пары и полную длительность 28 поисков / 140 повторов. Держите приложение/ядро активным. Успешные этапы сохраняются сразу. Все входные Parquet остаются неизменными.

Во время полного запуска test используется только после fit каждого повторного Pipeline для конечных вероятностей/метрик. Эти результаты не меняют фиксированный поиск, параметры, preprocessing или порог 0,5.""")
code("""full_result=None
if RUN_FULL_EXPERIMENT:
    full_result=run_full_experiment(PREPARED,FULL_OUTPUT,threads=THREADS)
    display(full_result)
else:
    print('Эта ячейка не запускала массовый эксперимент: RUN_FULL_EXPERIMENT=False.')
    print('Для запуска: RUN_FULL_EXPERIMENT=True; затем выполнить эту ячейку.')
    print('Каталог полного эксперимента:',FULL_OUTPUT)""",tags=['full_experiment_explicit_opt_in'])

md("""## 7. Отдельный ресурсный пилот — Level 4 / train_seed_42

Включается только флагом `RUN_RESOURCE_PILOT=True`. Используются все **221 407** ID сохранённой `train_seed_42`, Level 4 и seed 42. Для каждого из семи типов заранее фиксируется одна конфигурация фабрики `technical=False`, затем выполняется ровно один внешний fit. RandomizedSearchCV не вызывается; validation/test для оценки не читаются. У voting/stacking остаются внутренние базовые/OOF-fit их штатной архитектуры — это не дополнительные варианты конфигурации.

Выбранные параметры сохраняются до fit. Для каждого типа фиксируются число строк, исходных и преобразованных признаков (у ансамбля — по базам и входу метамодели), длительность, пик RAM процесса, размер сохранённой модели, фактическая диагностика обучения и предупреждения. Пилот выполняется последовательно; его результаты находятся отдельно от основного эксперимента в `artifacts/training/resource_pilot/`. Точный способ измерения пика RAM и интервал выборки записываются в отчёт. Измерение конфигурации не является гарантией памяти для всех кандидатов основного поиска.

Для запуска из чистого ядра достаточно конфигурационной ячейки и ячейки ниже. Выбор параметров не зависит от технических метрик/test. Основной бюджет 28 поисков и 140 повторов этим пилотом не расходуется. Если доступная системная RAM остаётся ниже 512 MiB три секунды, монитор завершает только свой worker и сохраняет частичный результат. Размер train и настройки автоматически не уменьшаются; незавершённую попытку нужно учитывать отдельно от успешного fit.

Сохранённый пилот виден в разделе 8 и при `RUN_RESOURCE_PILOT=False`. Повторный вызов с тем же manifest проверяет checkpoint и пропускает завершённые типы; незавершённый fit начинается заново. Эквивалентный запуск из терминала: `python -m src.resource_pilot --prepared data/processed/full --output-parent artifacts/training/resource_pilot`. Для обновления только отчётных ячеек без fit: `python scripts/execute_training_notebook.py --reports-only`.

**Фактическое выполнение:** сохранены все семь типов моделей. Было девять внешних попыток: первый RF остановлен защитой RAM, а первый stacking завершил fit, но не сохранил Pipeline из-за временной блокировки служебного JSON в Windows. После освобождения памяти RF выполнен повторно; для JSON добавлены ограниченные повторные попытки записи. Шесть готовых Pipeline переиспользованы без изменения байтов, повторён только stacking. Исходные manifest, исходники и checkpoint сохранены; перенос проверяет неизменность данных, параметров и всего кода за исключением функции записи JSON. Завершённый каталог — `resource_de05e94a7cb7`; исходный частичный запуск показывается в отчёте отдельно. Полная история и время попыток — `reports/training/resource_pilot_attempts.csv`, проверки — `resource_pilot_validation.json`. Основной бюджет A2 не изменён.""")
code("""if RUN_RESOURCE_PILOT:
    resource_result=run_resource_pilot(PREPARED,RESOURCE_PARENT,threads=THREADS)
    atomic_json(REPORTS/'resource_pilot_summary.json',resource_result)
    resource_metrics_path=Path(resource_result['root'])/'resource_metrics.parquet'
    if resource_metrics_path.exists():
        pd.read_parquet(resource_metrics_path).to_csv(REPORTS/'resource_pilot_metrics.csv',index=False)
    display(resource_result)
else:
    print('Ресурсный пилот не запускался этой ячейкой: RUN_RESOURCE_PILOT=False.')
    print('Для запуска выполните конфигурацию с RUN_RESOURCE_PILOT=True, затем эту ячейку.')
print('Отдельный каталог ресурсного пилота:',RESOURCE_PARENT)""",tags=['resource_pilot_explicit_opt_in'])

md("""## 8. Раздельные сводки из сохранённых артефактов

Эта ячейка зависит **только от путей из конфигурации**. Ей не нужны `technical`, `technical_root`, `warnings_frame`, `frozen_before` или результат предыдущего вызова fit. Она работает после последовательности «чистое ядро → конфигурация → раздел 6 → этот раздел» и формирует полный отчёт даже без технического прогона в этом ядре.

Полный отчёт читает experiment manifest, проверяет хэши этапов с `complete.json` и считает реальные завершённые поиски/повторы. Статусы различают отсутствие запуска, активный/прерванный частичный запуск, повреждённые артефакты и полный план 28/140. Прогресс существовавшего до этого ядра запуска также виден; старые артефакты не удаляются. Сохранность входов сверяется с `experiment_manifest.inputs`, а не с переменной `frozen_before`.

Технические сводки записываются отдельно в `reports/training/technical/`, полные — в `reports/training/full/`. Ни отсутствие технических файлов, ни пропущенные технические ячейки не мешают полному отчёту. Метрики технической выборки не смешиваются с итоговыми test-метриками. Ресурсные результаты остаются в своём отдельном каталоге и не используются для выбора победителя по качеству.

Все крупные артефакты находятся в `artifacts/` и исключены из Git; небольшие агрегированные сводки — в `reports/training/`. Загружать joblib следует только из собственных доверенных файлов этого эксперимента.""")
code("""technical_disk_report=write_technical_reports(TECHNICAL_PARENT,REPORTS/'technical',prepared_dir=PREPARED)
full_disk_report=write_full_report(FULL_OUTPUT,REPORTS/'full',prepared_dir=PREPARED)
resource_disk_report=write_resource_reports(RESOURCE_PARENT,REPORTS/'resource')
atomic_json(REPORTS/'training_notebook_summary.json',{
    'schema_version':2,'reports_built_from_disk':True,
    'technical_report':str(REPORTS/'technical/technical_runs.json'),
    'full_report':str(REPORTS/'full/full_summary.json'),
    'resource_report':str(REPORTS/'resource/resource_runs.json'),
    'technical_runs_found':technical_disk_report['runs_found'],
    'full_status':full_disk_report['status'],
    'resource_runs_found':resource_disk_report['runs_found']})
print('Технические запуски на диске:',technical_disk_report['runs_found'])
display(pd.DataFrame(technical_disk_report['runs']))
print('Отдельная полная сводка:')
display(pd.DataFrame([full_disk_report]))
print('Отдельные ресурсные пилоты на диске:',resource_disk_report['runs_found'])
resource_display=pd.DataFrame(resource_disk_report['model_metrics'])
if not resource_display.empty:
    resource_display['processed_features']=resource_display['processed_features_json'].map(
        lambda raw:'; '.join((item['path'].split('.bases.')[-1].replace('pipeline.model.meta','meta')+
            ': '+str(item['features'])) for item in json.loads(raw)) or 'нет данных в сводке')
    resource_display['model_MiB']=pd.to_numeric(resource_display['model_size_bytes'])/(1024**2)
    for resource_name,resource_rows in resource_display.groupby('run',sort=False):
        print(resource_name)
        display(resource_rows[['model','status','rows','raw_features','processed_features',
            'fit_seconds','peak_ram_gib','model_MiB','warning_count','mlp_n_iter_json']])
print('CSV ресурсных метрик по моделям:',resource_disk_report['model_metrics_csv'])
print('CSV предупреждений ресурсного пилота:',resource_disk_report['warnings_csv'])
print('Сводки:',REPORTS/'technical/technical_runs.json',REPORTS/'full/full_summary.json')""",tags=['disk_reports_independent_of_technical'])

notebook=nbf.v4.new_notebook(cells=cells,metadata={
    'kernelspec':{'display_name':'Python 3 (ipykernel)','language':'python','name':'python3'},
    'language_info':{'name':'python'}})
nbf.write(notebook,ROOT/'notebooks/02_model_training.ipynb')
print('Created',len(cells),'cells; RUN_FULL_EXPERIMENT=False')
