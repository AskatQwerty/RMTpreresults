"""
RQ3 analysis: tables (CSV + LaTeX) and figures (PDF) from the SHAP runs written by rq3_shap.py.

    python rq3/rq3_analyze.py      # uses rq3/out/runs/fixed (consistency), fixed_noise (noise floor), r* (stability)

Measures (A2, Sec. D): Spearman correlation of global feature rankings (mean |SHAP| over applicants) and the share
of an applicant's top-5 reasons (largest |SHAP|) that coincide. Uncertainty: 95% percentile bootstrap over
applicants (1,000 resamples, seed 42); differences vs LR use a paired bootstrap at a Bonferroni-adjusted level
for the six planned comparisons. Pre-set criteria (A2 / poster): stable if mean Spearman >= 0.8 and
>= 4 of 5 top reasons kept (overlap >= 0.8).

A run may explain fewer applicants than the fixed list (the first n/2 of each HAS_BUREAU group, as
rq3_shap.fixed_frames does); every comparison is made on the applicants common to both explanations.
"""
from __future__ import annotations
import json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT / "rq3" / "out"; RUNS = OUT / "runs"; TAB = OUT / "tables"; FIG = OUT / "figures"
MODELS = ["logistic_regression", "random_forest", "xgboost", "lightgbm", "mlp", "soft_voting", "stacking"]
SHORT = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGB", "lightgbm": "LGBM", "mlp": "MLP", "soft_voting": "VOTE", "stacking": "STACK"}
LONG = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
        "mlp": "MLP", "soft_voting": "Soft voting", "stacking": "Stacking"}
K, B, SEED, N_COMP = 5, 1000, 42, 6
CRIT_SP, CRIT_TOP = 0.8, 0.8


# ----------------------------------------------------------------------------- measures
def spearman_rows(A, Bm):
    ra = np.apply_along_axis(rankdata, -1, A); rb = np.apply_along_axis(rankdata, -1, Bm)
    ra = ra - ra.mean(-1, keepdims=True); rb = rb - rb.mean(-1, keepdims=True)
    return (ra * rb).sum(-1) / np.sqrt((ra ** 2).sum(-1) * (rb ** 2).sum(-1))


def sp1(a, b): return float(spearman_rows(np.asarray(a)[None], np.asarray(b)[None])[0])


def topk_mask(V, k=K):
    idx = np.argpartition(-np.abs(V), k - 1, axis=1)[:, :k]
    M = np.zeros(V.shape, dtype=bool); np.put_along_axis(M, idx, True, axis=1); return M


def overlap(Va, Vb, k=K):
    return (topk_mask(Va, k) & topk_mask(Vb, k)).sum(1) / k


def boot_mean_ci(x, level=95.0, seed=SEED):
    rng = np.random.default_rng(seed); x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (B, len(x)))].mean(1); a = (100 - level) / 2
    return float(x.mean()), float(np.percentile(m, a)), float(np.percentile(m, 100 - a))


def boot_global_spearman(absA, absB, seed=SEED):
    rng = np.random.default_rng(seed); n = absA.shape[0]; idx = rng.integers(0, n, (B, n))
    ga = np.stack([absA[i].mean(0) for i in idx]); gb = np.stack([absB[i].mean(0) for i in idx])
    s = spearman_rows(ga, gb)
    return sp1(absA.mean(0), absB.mean(0)), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


# ----------------------------------------------------------------------------- loading / alignment
def subset_index(exp, n):
    g1 = np.flatnonzero(exp.HAS_BUREAU.to_numpy() == 1); g0 = np.flatnonzero(exp.HAS_BUREAU.to_numpy() == 0)
    return np.arange(len(exp)) if n >= len(exp) else np.concatenate([g1[: n // 2], g0[: n - n // 2]])


def load_run(run, exp):
    d = RUNS / run; res = {}
    for m in MODELS:
        if (d / f"{m}.npy").exists() and (d / f"{m}.json").exists():
            V = np.load(d / f"{m}.npy")
            res[m] = dict(V=V, idx=subset_index(exp, V.shape[0]), base=np.load(d / f"{m}_base.npy"), prob=np.load(d / f"{m}_prob.npy"),
                          info=json.load(open(d / f"{m}.json")))
    return res


def common(a, b):
    """positions pa, pb into a['idx'], b['idx'] of the applicants explained in both."""
    _, pa, pb = np.intersect1d(a["idx"], b["idx"], return_indices=True); return pa, pb


def fmt(m, lo, hi, d=2): return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def latex_table(df, caption, label, cols, path, env="table"):
    lines = [f"\\begin{{{env}}}[t]", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\centering", "\\small",
             "\\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", "\\hline", " & ".join(cols) + " \\\\", "\\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join(str(r[c]) for c in cols) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", f"\\end{{{env}}}"]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- consistency (saved fits)
def consistency(exp, feats, fixed, noise, fixed_for_noise=None):
    fixed_for_noise = fixed if fixed_for_noise is None else fixed_for_noise
    rows = []
    LR = fixed["logistic_regression"]
    for m in fixed:
        if m == "logistic_regression": continue
        pa, pb = common(LR, fixed[m]); VL, Vm = LR["V"][pa], fixed[m]["V"][pb]; ex = exp.iloc[LR["idx"][pa]]
        for g, mask in {"all": np.ones(len(ex), bool), "with bureau": ex.HAS_BUREAU.to_numpy() == 1, "no bureau": ex.HAS_BUREAU.to_numpy() == 0}.items():
            ov = overlap(VL[mask], Vm[mask]); spa = spearman_rows(np.abs(VL[mask]), np.abs(Vm[mask]))
            o, olo, ohi = boot_mean_ci(ov); s, slo, shi = boot_mean_ci(spa); gs, glo, ghi = boot_global_spearman(np.abs(VL[mask]), np.abs(Vm[mask]))
            rows.append(dict(group=g, model=m, n=int(mask.sum()), explainer=fixed[m]["info"]["kind"], top5_shared_with_LR=o, top5_lo=olo, top5_hi=ohi,
                             spearman_applicant=s, sp_lo=slo, sp_hi=shi, spearman_global=gs, gsp_lo=glo, gsp_hi=ghi))
    t = pd.DataFrame(rows); t.to_csv(TAB / "rq3_consistency_with_LR.csv", index=False)
    names = [SHORT[m] for m in fixed]
    mat_sp = pd.DataFrame(index=names, columns=names, dtype=float); mat_ov = mat_sp.copy()
    for a, b in itertools.product(fixed, fixed):
        pa, pb = common(fixed[a], fixed[b])
        mat_sp.loc[SHORT[a], SHORT[b]] = sp1(np.abs(fixed[a]["V"][pa]).mean(0), np.abs(fixed[b]["V"][pb]).mean(0))
        mat_ov.loc[SHORT[a], SHORT[b]] = float(overlap(fixed[a]["V"][pa], fixed[b]["V"][pb]).mean())
    mat_sp.to_csv(TAB / "rq3_global_spearman_matrix.csv"); mat_ov.to_csv(TAB / "rq3_top5_overlap_matrix.csv")
    # explainer noise floor: same fit, second explainer seed
    nz = []
    for m in fixed:
        if m in noise and m in fixed_for_noise:
            A_, B_ = fixed_for_noise[m], noise[m]
            pa, pb = common(A_, B_); Vf, Vn = A_["V"][pa], B_["V"][pb]
            o, olo, ohi = boot_mean_ci(overlap(Vf, Vn)); gs, glo, ghi = boot_global_spearman(np.abs(Vf), np.abs(Vn))
            nz.append(dict(model=m, n=int(len(pa)), explainer=B_["info"].get("explainer"), n_permutations=B_["info"].get("n_permutations"),
                           top5_same_fit_two_seeds=o, top5_lo=olo, top5_hi=ohi, spearman_applicant=float(spearman_rows(np.abs(Vf), np.abs(Vn)).mean()),
                           spearman_global=gs, gsp_lo=glo, gsp_hi=ghi))
        elif fixed[m]["info"]["kind"] == "exact":
            nz.append(dict(model=m, n=int(len(fixed[m]["idx"])), explainer=fixed[m]["info"].get("explainer"), n_permutations=None, top5_same_fit_two_seeds=1.0,
                           top5_lo=1.0, top5_hi=1.0, spearman_applicant=1.0, spearman_global=1.0, gsp_lo=1.0, gsp_hi=1.0))
    nz = pd.DataFrame(nz); nz.to_csv(TAB / "rq3_explainer_noise.csv", index=False)
    a = t[t.group == "all"].copy(); a["Model"] = a.model.map(LONG); a["Explainer"] = a.explainer; a["n"] = a.n
    a["Top-5 shared with LR"] = [fmt(*x) for x in zip(a.top5_shared_with_LR, a.top5_lo, a.top5_hi)]
    a["Spearman (applicant)"] = [fmt(*x) for x in zip(a.spearman_applicant, a.sp_lo, a.sp_hi)]
    a["Spearman (global)"] = [fmt(*x) for x in zip(a.spearman_global, a.gsp_lo, a.gsp_hi)]
    latex_table(a, "Agreement of SHAP explanations with logistic regression on the same test applicants (saved fits, Level 4, probability of default). "
                   "Mean and 95\\% bootstrap CI over applicants; $n$ = applicants explained by both models.", "tab:rq3_agree",
                ["Model", "Explainer", "n", "Top-5 shared with LR", "Spearman (applicant)", "Spearman (global)"], TAB / "rq3_consistency_with_LR.tex", env="table*")
    imp = pd.DataFrame({SHORT[m]: np.abs(fixed[m]["V"]).mean(0) for m in fixed}, index=feats); imp.to_csv(TAB / "rq3_global_importance_fixed.csv")
    top = pd.DataFrame({SHORT[m]: imp[SHORT[m]].sort_values(ascending=False).head(10).index.tolist() for m in fixed}); top.index = np.arange(1, 11)
    top.to_csv(TAB / "rq3_top10_features_fixed.csv")
    return t, mat_sp, mat_ov, nz, imp


# ----------------------------------------------------------------------------- stability (retrainings)
def stability(exp, feats, runs):
    names = sorted(runs); R = len(names)
    if R < 2: return None, None, None
    pairs = list(itertools.combinations(range(R), 2))
    models = [m for m in MODELS if all(m in runs[r] for r in names)]
    rows, per_app, metrics = [], {}, []
    for m in models:
        idx = runs[names[0]][m]["idx"]
        assert all(np.array_equal(runs[r][m]["idx"], idx) for r in names), f"{m}: runs explain different applicants"
        ex = exp.iloc[idx]; Vs = [runs[r][m]["V"] for r in names]; A = [np.abs(v) for v in Vs]
        for g, mask in {"all": np.ones(len(ex), bool), "with bureau": ex.HAS_BUREAU.to_numpy() == 1, "no bureau": ex.HAS_BUREAU.to_numpy() == 0}.items():
            G = np.stack([a[mask].mean(0) for a in A])
            sp_pairs = np.array([sp1(G[i], G[j]) for i, j in pairs])
            ov = np.mean([overlap(Vs[i][mask], Vs[j][mask]) for i, j in pairs], axis=0)
            spa = np.mean([spearman_rows(A[i][mask], A[j][mask]) for i, j in pairs], axis=0)
            o, olo, ohi = boot_mean_ci(ov); s, slo, shi = boot_mean_ci(spa)
            rng = np.random.default_rng(SEED); n = int(mask.sum()); As = np.stack([a[mask] for a in A]); boots = np.empty(B)
            for b in range(B):
                ii = rng.integers(0, n, n); Gb = As[:, ii, :].mean(1); boots[b] = np.mean([sp1(Gb[i], Gb[j]) for i, j in pairs])
            prob = np.stack([runs[r][m]["prob"][mask] for r in names])
            rows.append(dict(group=g, model=m, n=n, runs=R, explainer=runs[names[0]][m]["info"]["kind"],
                             spearman_global_mean=float(sp_pairs.mean()), spearman_global_sd=float(sp_pairs.std(ddof=1)) if len(pairs) > 1 else np.nan,
                             gsp_lo=float(np.percentile(boots, 2.5)), gsp_hi=float(np.percentile(boots, 97.5)),
                             top5_kept=o, top5_lo=olo, top5_hi=ohi, spearman_applicant=s, sp_lo=slo, sp_hi=shi,
                             criterion_spearman=bool(sp_pairs.mean() >= CRIT_SP), criterion_top5=bool(o >= CRIT_TOP),
                             mean_sd_prob_across_runs=float(prob.std(0, ddof=1).mean())))
            if g == "all": per_app[m] = dict(idx=idx, ov=ov, spa=spa)
        # test-set metrics of the retrained models (full test set)
        tm = [runs[r][m]["info"].get("test_metrics") for r in names]
        if all(tm):
            for k in ("ap", "roc_auc", "log_loss", "brier"):
                v = np.array([x[k] for x in tm]); metrics.append(dict(model=m, metric=k, runs=R, mean=float(v.mean()), sd=float(v.std(ddof=1)), min=float(v.min()), max=float(v.max())))
        fs = [runs[r][m]["info"].get("fit_seconds") for r in names]
        if all(fs): metrics.append(dict(model=m, metric="fit_seconds", runs=R, mean=float(np.mean(fs)), sd=float(np.std(fs, ddof=1)), min=float(min(fs)), max=float(max(fs))))
    t = pd.DataFrame(rows); t.to_csv(TAB / "rq3_stability.csv", index=False)
    mt = pd.DataFrame(metrics); mt.to_csv(TAB / "rq3_refit_test_metrics.csv", index=False)
    if len(mt):
        w = mt[mt.metric != "fit_seconds"].copy(); w["Model"] = w.model.map(LONG)
        w["val"] = [f"{m:.4f} $\\pm$ {s:.4f}" for m, s in zip(w["mean"], w["sd"])]
        piv = w.pivot(index="Model", columns="metric", values="val").reindex([LONG[m] for m in models])[["ap", "roc_auc", "log_loss", "brier"]].reset_index()
        piv.columns = ["Model", "PR-AUC", "ROC-AUC", "Log-loss", "Brier"]
        latex_table(piv, f"Test-set metrics of the {R} retrained models per type (Level 4, same locked test set, 61,503 applicants): mean $\\pm$ s.d.\\ over retrainings.",
                    "tab:rq3_refit_metrics", ["Model", "PR-AUC", "ROC-AUC", "Log-loss", "Brier"], TAB / "rq3_refit_test_metrics.tex", env="table*")
    diff = []
    if "logistic_regression" in per_app:
        lr = per_app["logistic_regression"]
        for m in models:
            if m == "logistic_regression": continue
            _, pa, pb = np.intersect1d(lr["idx"], per_app[m]["idx"], return_indices=True)
            d = per_app[m]["ov"][pb] - lr["ov"][pa]; mean, lo, hi = boot_mean_ci(d, level=100 - 5.0 / N_COMP)
            diff.append(dict(model=m, n=int(len(d)), stability_top5_minus_LR=mean, ci_lo_bonf=lo, ci_hi_bonf=hi, excludes_zero=bool(lo > 0 or hi < 0),
                             n_comparisons=N_COMP, level=f"{100-5.0/N_COMP:.2f}%"))
    diff = pd.DataFrame(diff); diff.to_csv(TAB / "rq3_stability_diff_vs_LR.csv", index=False)
    a = t[t.group == "all"].copy(); a["Model"] = a.model.map(LONG); a["Explainer"] = a.explainer
    a["Spearman between runs"] = [(f"{m:.2f} $\\pm$ {s:.2f} " if not np.isnan(s) else f"{m:.2f} ") + f"[{lo:.2f}, {hi:.2f}]"
                                  for m, s, lo, hi in zip(a.spearman_global_mean, a.spearman_global_sd, a.gsp_lo, a.gsp_hi)]
    a["Top-5 kept"] = [fmt(*x) for x in zip(a.top5_kept, a.top5_lo, a.top5_hi)]
    dmap = {r.model: r for r in diff.itertuples()} if len(diff) else {}
    a["$\\Delta$ top-5 vs LR"] = ["--" if m == "logistic_regression" else (f"{dmap[m].stability_top5_minus_LR:+.2f} [{dmap[m].ci_lo_bonf:+.2f}, {dmap[m].ci_hi_bonf:+.2f}]" + ("*" if dmap[m].excludes_zero else "")) for m in a.model]
    a["Criteria met"] = [("yes" if (x and y) else "no") for x, y in zip(a.criterion_spearman, a.criterion_top5)]
    latex_table(a, f"Stability of SHAP explanations across {R} retrainings of each model (Level 4, $n$ = {int(a.n.iloc[0])} applicants). Spearman: mean $\\pm$ s.d.\\ over the {len(pairs)} run pairs, "
                   "95\\% bootstrap CI over applicants in brackets; top-5 kept: mean share of an applicant's five largest reasons present in both retrainings; "
                   "$\\Delta$: paired difference vs logistic regression with a Bonferroni-adjusted CI (six comparisons; * excludes zero). Criteria: Spearman $\\ge$ 0.8 and top-5 kept $\\ge$ 0.8.",
                "tab:rq3_stability", ["Model", "Explainer", "Spearman between runs", "Top-5 kept", "$\\Delta$ top-5 vs LR", "Criteria met"], TAB / "rq3_stability.tex", env="table*")
    return t, per_app, mt


# ----------------------------------------------------------------------------- figures
def figures(cons, mat_sp, stab, per_app, imp):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    c, c2, c3 = "#1b6ca8", "#e07b39", "#4caf50"
    a = cons[cons.group != "all"]; models = list(dict.fromkeys(a.model)); fig, ax = plt.subplots(figsize=(3.5, 2.4)); w = 0.38
    for i, (g, col) in enumerate([("with bureau", c), ("no bureau", c2)]):
        s = a[a.group == g].set_index("model").reindex(models); x = np.arange(len(models)) + (i - 0.5) * w
        ax.bar(x, s.top5_shared_with_LR, w, yerr=[s.top5_shared_with_LR - s.top5_lo, s.top5_hi - s.top5_shared_with_LR], capsize=2, color=col, label=g)
    ax.set_xticks(np.arange(len(models))); ax.set_xticklabels([SHORT[m] for m in models]); ax.set_ylim(0, 1); ax.set_ylabel("Top-5 reasons shared with LR"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig_rq3_agreement_with_LR.pdf"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(3.3, 2.9)); im = ax.imshow(mat_sp.values.astype(float), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(mat_sp))); ax.set_xticklabels(mat_sp.columns); ax.set_yticks(range(len(mat_sp))); ax.set_yticklabels(mat_sp.index)
    for i in range(len(mat_sp)):
        for j in range(len(mat_sp)):
            v = float(mat_sp.values[i, j]); ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Spearman ρ of global rankings"); fig.tight_layout(); fig.savefig(FIG / "fig_rq3_spearman_matrix.pdf"); plt.close(fig)
    sel = [m for m in ["LR", "LGBM", "STACK"] if m in imp.columns]
    feats_top = list(dict.fromkeys(sum([imp[m].sort_values(ascending=False).head(8).index.tolist() for m in sel], [])))[:15]
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    for i, m in enumerate(sel):
        r = imp[m].rank(ascending=False); ax.plot([r[f] for f in feats_top], np.arange(len(feats_top)), "o", ms=4, label=m, color=[c, c2, c3][i])
    ax.set_yticks(np.arange(len(feats_top))); ax.set_yticklabels([f[:26] for f in feats_top], fontsize=7); ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xlabel("Rank of mean |SHAP|, log scale"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / "fig_rq3_top_features.pdf"); plt.close(fig)
    if stab is None: return
    a = stab[stab.group == "all"]; fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6)); x = np.arange(len(a))
    ax[0].errorbar(x, a.spearman_global_mean, yerr=[a.spearman_global_mean - a.gsp_lo, a.gsp_hi - a.spearman_global_mean], fmt="o", capsize=3, color=c)
    ax[0].axhline(CRIT_SP, ls="--", color="gray", lw=0.8); ax[0].set_xticks(x); ax[0].set_xticklabels([SHORT[m] for m in a.model]); ax[0].set_ylabel("Spearman ρ between retrainings"); ax[0].set_title("(a) Global ranking stability"); ax[0].set_ylim(0, 1.02)
    ax[1].boxplot([per_app[m]["ov"] for m in a.model], showfliers=False); ax[1].set_xticks(np.arange(1, len(a) + 1)); ax[1].set_xticklabels([SHORT[m] for m in a.model])
    ax[1].axhline(CRIT_TOP, ls="--", color="gray", lw=0.8); ax[1].set_ylabel("Share of top-5 reasons kept"); ax[1].set_title("(b) Per-applicant stability"); ax[1].set_ylim(0, 1.02)
    for z in ax: z.set_xlabel("Model")
    fig.tight_layout(); fig.savefig(FIG / "fig_rq3_stability.pdf"); plt.close(fig)


def error_analysis(exp, feats, fixed, per_app, runs):
    rows = []
    for m in fixed:
        V = fixed[m]["V"]; absV = np.abs(V); p = fixed[m]["prob"]
        rows.append(dict(model=m, n=len(V), share_applicants_max_abs_shap_below_1e4=float((absV.max(1) < 1e-4).mean()), median_sum_abs_shap=float(np.median(absV.sum(1))),
                         n_features_with_mean_abs_shap_above_1e6=int((absV.mean(0) > 1e-6).sum()), share_prob_below_1e6=float((p < 1e-6).mean()), share_prob_above_1minus1e6=float((p > 1 - 1e-6).mean())))
    pd.DataFrame(rows).to_csv(TAB / "rq3_error_analysis_fixed.csv", index=False)
    if per_app:
        rows, sw = [], []
        for m, d in per_app.items():
            ov = d["ov"]; ex = exp.iloc[d["idx"]]; p = np.mean([runs[r][m]["prob"] for r in runs if m in runs[r]], axis=0); lo = ov < 0.6
            rows.append(dict(model=m, n=len(ov), share_applicants_top5_kept_below_06=float(lo.mean()), mean_prob_unstable=float(p[lo].mean()) if lo.any() else np.nan,
                             mean_prob_stable=float(p[~lo].mean()) if (~lo).any() else np.nan, default_rate_unstable=float(ex.TARGET.to_numpy()[lo].mean()) if lo.any() else np.nan,
                             share_no_bureau_unstable=float((ex.HAS_BUREAU.to_numpy()[lo] == 0).mean()) if lo.any() else np.nan))
            names = sorted(r for r in runs if m in runs[r]); Rk = np.stack([rankdata(-np.abs(runs[r][m]["V"]).mean(0)) for r in names])
            for j in np.argsort(Rk.mean(0))[:10]:
                sw.append(dict(model=m, feature=feats[j], mean_rank=float(Rk[:, j].mean()), min_rank=float(Rk[:, j].min()), max_rank=float(Rk[:, j].max())))
        pd.DataFrame(rows).to_csv(TAB / "rq3_error_analysis_stability.csv", index=False)
        pd.DataFrame(sw).to_csv(TAB / "rq3_top10_rank_range_across_retrainings.csv", index=False)


if __name__ == "__main__":
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    exp = pd.read_parquet(OUT / "explained_ids.parquet")
    feats = json.loads((ROOT / "data/handoff/pilot_20k_5config/features_level_4.json").read_text(encoding="utf-8"))
    fixed = load_run("fixed", exp)
    # explainer noise floor: two independent explanations of the SAME fits at the retraining-run settings (noise_a vs noise_b);
    # fallback: fixed vs fixed_noise
    noise = {}
    if (RUNS / "noise_a").exists() and (RUNS / "noise_b").exists():
        na, nb = load_run("noise_a", exp), load_run("noise_b", exp)
        noise = {m: nb[m] for m in nb if m in na}; fixed_for_noise = {m: na[m] for m in na}
    elif (RUNS / "fixed_noise").exists():
        noise = load_run("fixed_noise", exp); fixed_for_noise = fixed
    else:
        fixed_for_noise = fixed
    runs = {d.name: load_run(d.name, exp) for d in sorted(RUNS.glob("r[0-9]*"))}
    runs = {k: v for k, v in runs.items() if v}
    cons, mat_sp, mat_ov, nz, imp = consistency(exp, feats, fixed, noise, fixed_for_noise)
    pd.set_option("display.width", 250)
    print(cons[cons.group == "all"].round(3).to_string(index=False)); print(); print(mat_sp.round(2)); print(); print(nz.round(3).to_string(index=False))
    stab, per_app, mt = stability(exp, feats, runs) if len(runs) >= 2 else (None, None, None)
    if stab is not None:
        print(); print(stab[stab.group == "all"].round(3).to_string(index=False)); print(pd.read_csv(TAB / "rq3_stability_diff_vs_LR.csv").round(3).to_string(index=False))
        if mt is not None and len(mt): print(); print(mt.round(4).to_string(index=False))
    figures(cons, mat_sp, stab, per_app, imp)
    error_analysis(exp, feats, fixed, per_app, runs)
    print("\nwritten:", sorted(p.name for p in TAB.iterdir()), sorted(p.name for p in FIG.iterdir()))
