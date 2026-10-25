"""
RQ3 — consistency and stability of SHAP explanations (Assignment 3, Results).

Works directly on the team package `pilot_20k_5config` (28 saved Pipelines + handoff data).
Run from the package root (the folder that contains src/, data/, artifacts/):

    python rq3/rq3_shap.py select                       # fix explained applicants + background (once)
    python rq3/rq3_shap.py pilot --n 20                 # timing + exact-vs-approximate agreement
    python rq3/rq3_shap.py explain --run fixed          # SHAP of the saved package fits (Level 4)
    python rq3/rq3_shap.py explain --run r1 --refit-seed 1   # retrain with saved params on a 90% subsample, explain
    python rq3/rq3_shap.py analyze                      # stability / agreement tables + figures

Design (A2 Methods, Sec. C.4 and D; docs/SHAP_HANDOFF.md):
  * one fixed list of explained TEST applicants (saved once; same for every model and run)
  * one fixed BACKGROUND sample drawn from the 20k pilot TRAIN set (never test)
  * explained output: P(TARGET=1) ("probability", default) or its logit ("logodds"); ONE scale for all models
  * exact explainers where they exist on that scale (Tree SHAP, interventional, for RF / XGBoost / LightGBM;
    Linear SHAP for LR on log-odds); otherwise a permutation Shapley estimator (forward + reverse permutation,
    averaged over the background) that masks whole RAW features.  Masking is done in the transformed space of
    each fitted preprocessor: every raw feature owns its value column, its missing indicator and its one-hot
    columns, and the preprocessing is column-wise, so replacing those columns by a background row is exactly
    "replace the raw feature by the background value".  Saved preprocessors are only used with transform().
  * SHAP values of the transformed columns are summed back to the 322 raw Level-4 features.
  * `--refit-seed` builds a NEW estimator with src.model_pipelines.make_estimator and the saved best params
    (the retraining extension agreed with the RQ1/RQ2 author); the saved package Pipelines are never refitted.
"""
from __future__ import annotations

import argparse, json, os, sys, time, platform, itertools, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCOPE = "pilot_20k_5config"
P = ROOT / "artifacts/training" / SCOPE
D = ROOT / "data/handoff" / SCOPE
OUT = ROOT / "rq3" / "out"
ID = "SK_ID_CURR"
MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "mlp", "soft_voting", "stacking"]
SHORT = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB", "lightgbm": "LGBM",
         "mlp": "MLP", "soft_voting": "VOTE", "stacking": "STACK"}
EPS = 1e-6          # clip used only on the log-odds scale
warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------- data
def load_data(level=4):
    feats = json.loads((D / f"features_level_{level}.json").read_text(encoding="utf-8"))
    schema = pq.read_schema(D / "X_train_level4.parquet")
    numeric = [f.name for f in schema if f.name in feats and (pa.types.is_integer(f.type) or pa.types.is_floating(f.type))]
    categorical = [f for f in feats if f not in numeric]
    def part(split):
        ids = pd.read_parquet(D / f"{split}_ids.parquet")
        X = ids.merge(pd.read_parquet(D / f"X_{split}_level4.parquet"), on=ID, how="left", validate="one_to_one", sort=False)
        y = ids.merge(pd.read_parquet(D / f"target_{split}.parquet"), on=ID, how="left", validate="one_to_one", sort=False)
        assert np.array_equal(X[ID], ids[ID]) and y.TARGET.isin([0, 1]).all()
        return X, y.TARGET.to_numpy()
    Xtr, ytr = part("train"); Xte, yte = part("test")
    assert len(Xtr) == 20000 and len(Xte) == 61503
    fair = pd.read_parquet(D / "fairness_test.parquet")
    return dict(level=level, feats=feats, numeric=numeric, categorical=categorical,
                X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte, fairness_test=fair)


def load_pipeline(model, level=4):
    return joblib.load(P / model / f"level_{level}/fit_checkpoint/pipeline.joblib")


def best_params(model, level=4):
    return json.loads((P / model / f"level_{level}/fit_checkpoint/best_params.json").read_text(encoding="utf-8"))["params"]


def pos_column(clf):
    return int(np.flatnonzero(np.asarray(clf.classes_) == 1)[0])


# ----------------------------------------------------------------------------- fixed samples
def select(data, n_per_group=500, n_bg=100, seed=42):
    """Rule: from the fixed test set draw n_per_group applicants with HAS_BUREAU=1 and n_per_group with
    HAS_BUREAU=0 at random (seed), keeping each group's natural default rate; background = n_bg random
    rows of the 20k pilot train set (same seed stream). Both lists are saved once and reused by every run."""
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    X = data["X_test"]
    parts = []
    for hb in (1, 0):
        pool = np.flatnonzero(X["HAS_BUREAU"].to_numpy() == hb)
        parts.append(np.sort(rng.choice(pool, n_per_group, replace=False)))
    idx = np.concatenate(parts)
    exp = pd.DataFrame({ID: X[ID].to_numpy()[idx], "HAS_BUREAU": X["HAS_BUREAU"].to_numpy()[idx],
                        "TARGET": data["y_test"][idx]})
    exp = exp.merge(data["fairness_test"][[ID, "GENDER_GROUP", "AGE_GROUP"]], on=ID, how="left")
    bg_idx = np.sort(rng.choice(len(data["X_train"]), n_bg, replace=False))
    bg = pd.DataFrame({ID: data["X_train"][ID].to_numpy()[bg_idx]})
    exp.to_parquet(OUT / "explained_ids.parquet", index=False); bg.to_parquet(OUT / "background_ids.parquet", index=False)
    json.dump({"rule": "stratified by HAS_BUREAU, n_per_group from each group, natural default rate kept",
               "n_per_group": n_per_group, "n_background": n_bg, "seed": seed,
               "source_explained": "test_ids.parquet (fixed test set)", "source_background": "train_ids.parquet (20k pilot train)",
               "default_rate_explained": float(exp.TARGET.mean())}, open(OUT / "selection.json", "w"), indent=2)
    return exp, bg


def fixed_frames(data, n=None):
    exp = pd.read_parquet(OUT / "explained_ids.parquet"); bg = pd.read_parquet(OUT / "background_ids.parquet")
    if n:   # pilot: first n/2 of each group
        exp = pd.concat([exp[exp.HAS_BUREAU == 1].head(n // 2), exp[exp.HAS_BUREAU == 0].head(n - n // 2)])
    Xe = exp[[ID]].merge(data["X_test"], on=ID, how="left", validate="one_to_one", sort=False)
    Xb = bg[[ID]].merge(data["X_train"], on=ID, how="left", validate="one_to_one", sort=False)
    assert np.array_equal(Xe[ID], exp[ID]) and np.array_equal(Xb[ID], bg[ID])
    return exp.reset_index(drop=True), Xe, Xb


# ----------------------------------------------------------------------------- transformed space
class TransformedMap:
    """raw_index[j] = index (in feats) of the raw feature behind transformed column j of a fitted `preprocess`."""
    def __init__(self, preprocess, feats):
        ct = preprocess.named_steps["columns"]
        blocks = []
        for name, tr, cols in ct.transformers_:
            if name == "numeric":
                blocks += [feats.index(c) for c in cols]          # values
                blocks += [feats.index(c) for c in cols]          # missing indicators
            elif name == "categorical":
                for c, cats in zip(cols, tr.named_steps["onehot"].categories_):
                    blocks += [feats.index(c)] * len(cats)
        self.raw_index = np.asarray(blocks)
        names = preprocess.get_feature_names_out()
        assert len(names) == len(self.raw_index), (len(names), len(self.raw_index))
        for n, j in zip(names, self.raw_index):
            assert feats[j] in n, (n, feats[j])
        self.M = len(feats)
        self.members = [np.flatnonzero(self.raw_index == j) for j in range(self.M)]

    def aggregate(self, V):
        out = np.zeros((V.shape[0], self.M)); np.add.at(out.T, self.raw_index, V.T); return out


def dense(T):
    return T.toarray() if hasattr(T, "toarray") else np.asarray(T)


class ModelFn:
    """Callable on masked transformed inputs; reproduces pipe.predict_proba exactly (checked at construction).
    For standalone models: one transformed matrix. For ensembles: one per base (same raw layout)."""
    def __init__(self, model, pipe, feats, Xe, Xb, scale):
        self.model, self.scale = model, scale
        if model in ("soft_voting", "stacking"):
            ens = pipe.named_steps["model"]
            self.bases = [(name, base.named_steps["preprocess"], base.named_steps["model"]) for name, base in ens.named_estimators_.items()]
            self.ens = ens
        else:
            self.bases = [("self", pipe.named_steps["preprocess"], pipe.named_steps["model"])]
            self.ens = None
        self.maps = [TransformedMap(pre, feats) for _, pre, _ in self.bases]
        for m in self.maps[1:]:
            assert np.array_equal(m.raw_index, self.maps[0].raw_index)
        self.Te = [dense(pre.transform(Xe[feats])) for _, pre, _ in self.bases]
        self.Tb = [dense(pre.transform(Xb[feats])) for _, pre, _ in self.bases]
        pos = pos_column(pipe)
        ref = pipe.predict_proba(Xe[feats])[:, pos].astype(np.float64)
        err = float(np.max(np.abs(self.prob(self.Te) - ref)))
        assert err < 1e-5, f"ModelFn does not reproduce the pipeline for {model}: max err {err}"
        self.check_err = err

    def prob(self, Ts):
        """Ts: list of transformed matrices (one per base, aligned rows) -> P(default)."""
        if self.ens is None:
            _, _, clf = self.bases[0]
            return clf.predict_proba(Ts[0])[:, pos_column(clf)].astype(np.float64)
        ps = [clf.predict_proba(T)[:, pos_column(clf)] for (_, _, clf), T in zip(self.bases, Ts)]
        if self.model == "soft_voting":
            w = self.ens.weights if self.ens.weights is not None else [1] * len(ps)
            return np.average(np.column_stack(ps), axis=1, weights=w).astype(np.float64)
        meta = self.ens.final_estimator_
        return meta.predict_proba(np.column_stack(ps))[:, pos_column(meta)].astype(np.float64)

    def f(self, Ts):
        p = self.prob(Ts)
        if self.scale == "logodds":
            p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
        return p


# ----------------------------------------------------------------------------- permutation Shapley (raw-feature groups)
def perm_shap(fn: ModelFn, n_perm=1, seed=0, verbose=False):
    """Returns (values (n, M), base (n,)). For each applicant and each permutation of the M raw features:
    start from every background row, switch the raw features on one by one in the permuted order (forward),
    then in the reversed order (antithetic); the marginal change of f, averaged over background rows and
    both passes, is the contribution. base = mean f over background rows. values.sum(1) + base = f(x)."""
    rng = np.random.default_rng(seed)
    M = fn.maps[0].M; members = fn.maps[0].members
    n = fn.Te[0].shape[0]; B = fn.Tb[0].shape[0]
    V = np.zeros((n, M)); base = np.zeros(n)
    # column masks for "first k groups of the order are switched on" are built per permutation
    for i in range(n):
        t0 = time.time()
        acc = np.zeros(M)
        for p in range(n_perm):
            order = rng.permutation(M)
            for order_ in (order, order[::-1]):
                # state k (k=0..M): groups order_[:k] take the applicant's columns, the rest keep the background
                Ts = []
                for Te, Tb in zip(fn.Te, fn.Tb):
                    ncol = Tb.shape[1]
                    col_state = np.zeros((M + 1, ncol), dtype=bool)
                    step = np.zeros(ncol, dtype=bool)
                    for k, g in enumerate(order_, start=1):
                        step[members[g]] = True; col_state[k] = step
                    S = np.where(col_state[:, None, :], Te[i][None, None, :], Tb[None, :, :])   # (M+1, B, ncol)
                    Ts.append(S.reshape((M + 1) * B, ncol).astype(Tb.dtype, copy=False))
                g = fn.f(Ts).reshape(M + 1, B).mean(1)                                             # (M+1,)
                acc[order_] += np.diff(g)
                if p == 0 and order_ is order: base[i] = g[0]
        V[i] = acc / (2 * n_perm)
        if verbose and (i % 50 == 0): print(f"    applicant {i+1}/{n} ({time.time()-t0:.2f}s each)", flush=True)
    return V, base


def exact_shap(model, fn: ModelFn, scale):
    """Exact explainers on the same transformed space / background; aggregated to raw. None if unavailable."""
    import shap
    if model in ("soft_voting", "stacking", "mlp"): return None
    _, _, clf = fn.bases[0]; Te, Tb = fn.Te[0], fn.Tb[0]
    if model == "logistic_regression":
        if scale != "logodds": return None
        ex = shap.LinearExplainer(clf, Tb); V = np.asarray(ex.shap_values(Te)); base = float(np.ravel(ex.expected_value)[-1])
        f = clf.decision_function(Te)
    elif model == "random_forest":
        if scale != "probability": return None
        ex = shap.TreeExplainer(clf, data=Tb, feature_perturbation="interventional")
        V = np.asarray(ex.shap_values(Te, check_additivity=False)); V = V[:, :, 1] if V.ndim == 3 else V
        base = float(np.ravel(ex.expected_value)[-1]); f = clf.predict_proba(Te)[:, 1]
    else:
        ex = shap.TreeExplainer(clf, data=Tb, feature_perturbation="interventional",
                                model_output="probability" if scale == "probability" else "raw")
        V = np.asarray(ex.shap_values(Te, check_additivity=False)); V = V[:, :, 1] if V.ndim == 3 else V
        base = float(np.ravel(ex.expected_value)[-1])
        f = clf.predict_proba(Te)[:, 1] if scale == "probability" else (
            clf.predict(Te, output_margin=True) if model == "xgboost" else clf.predict(Te, raw_score=True))
    add_err = float(np.max(np.abs(V.sum(1) + base - f)))
    return fn.maps[0].aggregate(V), np.full(len(Te), base), {"explainer": type(ex).__name__, "additivity_max_abs_err": add_err}


def explain_model(model, pipe, data, Xe, Xb, scale, seed, n_perm, prefer_exact=True, verbose=False):
    fn = ModelFn(model, pipe, data["feats"], Xe, Xb, scale)
    info = {"model": model, "scale": scale, "n_explained": len(Xe), "n_background": len(Xb), "pipeline_check_max_err": fn.check_err}
    r = exact_shap(model, fn, scale) if prefer_exact else None
    if r is not None:
        V, base, i = r; info.update(kind="exact", **i)
    else:
        t = time.time(); V, base = perm_shap(fn, n_perm, seed, verbose); info.update(kind="approximate", explainer="permutation", n_permutations=n_perm, seconds=time.time() - t)
        fx = fn.f(fn.Te); info["additivity_max_abs_err"] = float(np.max(np.abs(V.sum(1) + base - fx)))
    return V, base, info


# ----------------------------------------------------------------------------- refit (retraining extension)
def refit(model, data, seed, frac=0.9, threads=2):
    from sklearn.model_selection import StratifiedShuffleSplit
    from src.model_pipelines import make_estimator
    est = make_estimator(model, data["numeric"], data["categorical"], seed=seed, technical=False, threads=threads)
    est.set_params(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in best_params(model).items()})
    sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
    tr, _ = next(sss.split(data["X_train"], data["y_train"]))
    t = time.time(); est.fit(data["X_train"].iloc[tr][data["feats"]], data["y_train"][tr]); sec = time.time() - t
    return est, {"refit_seed": seed, "train_rows": int(len(tr)), "fit_seconds": sec, "params": best_params(model)}


# ----------------------------------------------------------------------------- measures
def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b); ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def topk_mask(V, k=5, mode="abs"):
    v = np.abs(V) if mode == "abs" else V
    idx = np.argpartition(-v, k - 1, axis=1)[:, :k]
    M = np.zeros_like(V, dtype=bool); np.put_along_axis(M, idx, True, axis=1); return M


def topk_overlap(Va, Vb, k=5, mode="abs"):
    return (topk_mask(Va, k, mode) & topk_mask(Vb, k, mode)).sum(1) / k


def per_applicant_spearman(Va, Vb):
    return np.array([spearman(np.abs(a), np.abs(b)) for a, b in zip(Va, Vb)])


# ----------------------------------------------------------------------------- commands
def cmd_select(a):
    data = load_data(); exp, bg = select(data, a.n_per_group, a.bg, a.seed)
    print(exp.groupby("HAS_BUREAU").TARGET.agg(["size", "mean"])); print("background", len(bg))


def cmd_pilot(a):
    data = load_data(); exp, Xe, Xb = fixed_frames(data, n=a.n); Xb = Xb.head(a.bg)
    rows = []
    for model in a.models.split(","):
        pipe = load_pipeline(model)
        fn = ModelFn(model, pipe, data["feats"], Xe, Xb, a.scale)
        t = time.time(); r = exact_shap(model, fn, a.scale); te = time.time() - t
        t = time.time(); V, b = perm_shap(fn, a.n_perm, seed=1); tp = time.time() - t
        t = time.time(); V2, _ = perm_shap(fn, a.n_perm, seed=2); tp2 = time.time() - t
        fx = fn.f(fn.Te)
        row = dict(model=SHORT[model], n=len(Xe), bg=len(Xb), scale=a.scale, n_perm=a.n_perm, pipeline_check_err=fn.check_err,
                   exact_sec=round(te, 2) if r else None, exact_add_err=round(r[2]["additivity_max_abs_err"], 7) if r else None,
                   perm_sec_per_applicant=round(tp / len(Xe), 2), perm_add_err=round(float(np.max(np.abs(V.sum(1) + b - fx))), 7),
                   noise_top5_two_seeds=float(topk_overlap(V, V2).mean()), noise_spearman_applicant=float(per_applicant_spearman(V, V2).mean()),
                   noise_spearman_global=spearman(np.abs(V).mean(0), np.abs(V2).mean(0)))
        if r:
            row.update(top5_exact_vs_perm=float(topk_overlap(r[0], V).mean()), spearman_applicant_exact_vs_perm=float(per_applicant_spearman(r[0], V).mean()),
                       spearman_global_exact_vs_perm=spearman(np.abs(r[0]).mean(0), np.abs(V).mean(0)))
        if model == "logistic_regression" and a.scale == "probability":
            fn2 = ModelFn(model, pipe, data["feats"], Xe, Xb, "logodds"); r2 = exact_shap(model, fn2, "logodds")
            row.update(LR_top5_exactlogodds_vs_permprob=float(topk_overlap(r2[0], V).mean()),
                       LR_spearman_global_exactlogodds_vs_permprob=spearman(np.abs(r2[0]).mean(0), np.abs(V).mean(0)))
        rows.append(row); print(row, flush=True)
    OUT.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(OUT / f"pilot_{a.scale}_n{a.n}_bg{a.bg}.csv", index=False)


def cmd_explain(a):
    data = load_data(); exp, Xe, Xb = fixed_frames(data, n=a.n)
    run_dir = OUT / "runs" / a.run; run_dir.mkdir(parents=True, exist_ok=True)
    for model in a.models.split(","):
        if (run_dir / f"{model}.npy").exists() and not a.force:
            print("skip (exists)", model); continue
        meta = {}
        if a.refit_seed is None:
            pipe = load_pipeline(model); meta["source"] = "package fit (pilot_20k_5config, one refit on 20k)"
        else:
            print(f"refitting {model} seed={a.refit_seed} ...", flush=True)
            pipe, meta = refit(model, data, a.refit_seed); meta["source"] = "refit with saved best params on 90% stratified subsample"
            if a.save_models:
                joblib.dump(pipe, run_dir / f"{model}_pipeline.joblib")
        # test-set metrics of this fit (full locked test set, 61,503 applicants): gives mean +- SD over retrainings
        from sklearn.metrics import average_precision_score, roc_auc_score, log_loss, brier_score_loss
        pt = pipe.predict_proba(data["X_test"][data["feats"]])[:, pos_column(pipe)].astype(np.float64)
        np.save(run_dir / f"{model}_prob_test.npy", pt.astype(np.float32))
        meta["test_metrics"] = {"ap": float(average_precision_score(data["y_test"], pt)), "roc_auc": float(roc_auc_score(data["y_test"], pt)),
                                "log_loss": float(log_loss(data["y_test"], pt, labels=[0, 1])), "brier": float(brier_score_loss(data["y_test"], pt)), "n_test": int(len(pt))}
        print(f"explaining {model} ({len(Xe)} applicants, bg {len(Xb)}) ...", flush=True)
        n_perm = a.n_perm_ensemble if model in ("soft_voting", "stacking") else a.n_perm
        V, base, info = explain_model(model, pipe, data, Xe, Xb, a.scale, seed=a.seed, n_perm=n_perm, verbose=True)
        np.save(run_dir / f"{model}.npy", V); np.save(run_dir / f"{model}_base.npy", base)
        p = pipe.predict_proba(Xe[data["feats"]])[:, pos_column(pipe)]
        np.save(run_dir / f"{model}_prob.npy", p)
        info.update(meta); info.update(run=a.run, seed_explainer=a.seed, python=platform.python_version(), machine=platform.machine(),
                                       cpu_count=os.cpu_count(), time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        json.dump(info, open(run_dir / f"{model}.json", "w"), indent=2, default=float)
        print({k: info[k] for k in info if k in ("model", "kind", "explainer", "seconds", "fit_seconds", "additivity_max_abs_err")}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--n_per_group", type=int, default=500); s.add_argument("--bg", type=int, default=100); s.add_argument("--seed", type=int, default=42)
    s = sub.add_parser("pilot"); s.add_argument("--n", type=int, default=20); s.add_argument("--bg", type=int, default=50); s.add_argument("--n_perm", type=int, default=1)
    s.add_argument("--scale", default="probability", choices=["probability", "logodds"]); s.add_argument("--models", default=",".join(MODELS))
    s = sub.add_parser("explain"); s.add_argument("--run", required=True); s.add_argument("--refit-seed", type=int, default=None)
    s.add_argument("--n", type=int, default=None); s.add_argument("--n_perm", type=int, default=2); s.add_argument("--n_perm_ensemble", type=int, default=1)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--scale", default="probability", choices=["probability", "logodds"]); s.add_argument("--models", default=",".join(MODELS))
    s.add_argument("--force", action="store_true"); s.add_argument("--save-models", action="store_true")
    a = ap.parse_args()
    {"select": cmd_select, "pilot": cmd_pilot, "explain": cmd_explain}[a.cmd](a)
