Validation and consolidation of saved RMT CV results

Completed: 1680 / 1680 CV fits (7 models × 4 levels × 20 configurations × 3 folds).
Each batch 01–05, 06–10, 11–15, 16–20 contains 420 results; there are no missing or duplicate keys.
Training: the same 20 000 clients, class counts 18 385 / 1 615. Saved IDs and CV folds match.
This is an expanded pilot on 20 000 clients, not the full-development A2 experiment.

Best mean CV log-loss: 0.247222941, soft_voting, Level 4, configuration 9.
This is a CV selection result, not a final test metric.

Files:
all_1680_cv_folds.csv — exact scores for each saved fold and their sources.
all_560_candidates.csv — mean and standard deviation across three folds for each candidate.
best_28_by_cv.csv / selected_parameters.json — best parameters for each model and level.
best_cv_log_loss_by_level.csv — a 7 × 4 table.
warnings.csv — saved warnings with the available attribution detail.
import_audit.json — checksums, checks, sources and limitations.

Batch 11–15 exception: for 60 RF fits after WinError 5, joblib threading was replaced with ThreadPoolExecutor while retaining n_jobs=2.
Model code, parameters and splits were unchanged. The adaptation was outside the original package fingerprint; its text and SHA are saved separately.
Bitwise equivalence of the execution mechanisms was not checked. Timings from different computers cannot be directly compared as model speeds.
Stopped incomplete attempts are not counted as additional completed CV fits.

Only existing results were read and combined here: no new fit/predict/bootstrap calls.
Final evaluation of winners selected from 20 configurations requires separate training of the selected Pipelines on train and evaluation on the unchanged test.
The old 28 Pipelines and their test reports come from the 5-configuration search; they must not be presented as new results.
The five A2 repeats were not performed here. Original results, X, test and splits were preserved unchanged.
