"""Парный DeLong ROC-AUC по сохранённым вероятностям; без обучения моделей.

Ковариация DeLong (1988), рассчитанная через mid-ranks Sun & Xu (2014).
Равные scores дают вклад 1/2. Строки всех моделей должны соответствовать одним
и тем же клиентам в одинаковом порядке; ID проверяет вызывающий слой ввода.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pandas as pd


DELONG_REFERENCES=[
    {'authors':'DeLong ER, DeLong DM, Clarke-Pearson DL',
     'year':1988,
     'title':'Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach',
     'journal':'Biometrics','volume':'44','issue':'3','pages':'837-845',
     'doi':'10.2307/2531595','url':'https://doi.org/10.2307/2531595',
     'primary_record_url':'https://pubmed.ncbi.nlm.nih.gov/3203132/'},
    {'authors':'Sun X, Xu W','year':2014,
     'title':"Fast Implementation of DeLong's Algorithm for Comparing the Areas Under Correlated Receiver Operating Characteristic Curves",
     'journal':'IEEE Signal Processing Letters','volume':'21','issue':'11','pages':'1389-1393',
     'doi':'10.1109/LSP.2014.2337313','url':'https://doi.org/10.1109/LSP.2014.2337313',
     'author_manuscript_url':'https://pamixsun.github.io/papers/sun2014fast.pdf',
     'author_algorithm_url':'https://pamixsun.github.io/update/2015/08/31/roc-analysis-tool-with-delongs-method/'},
]

DELONG_METADATA={
    'method':'paired DeLong ROC-AUC; Sun-Xu midrank implementation',
    'alternative':'two-sided; delta = model ROC-AUC minus reference ROC-AUC',
    'null_hypothesis':'equal population ROC-AUC for model and reference',
    'covariance':'sample covariance of positive and negative placements, ddof=1, divided by class sizes',
    'ties':'exact equal scores receive one-half concordance via average ranks',
    'normal_approximation':True,
    'positive_class':1,
    'bonferroni_family_size':6,
    'familywise_alpha':0.05,
    'probability_rows':'same ordered, complete client rows for every model; no missing-row deletion',
    'independent_sampling_unit':'one current application/client; no cluster correction',
    'uncertainty_scope':'test-client sampling conditional on already fitted pipelines; no training/search variability',
    'zero_variance_equal_auc':'z=0 and p=1 convention; no observed AUC difference',
    'zero_variance_unequal_auc':'z and p undefined (NaN); degenerate Wald test, no significance declaration',
    'not_equivalence_test':True,
    'ap_p_values_computed':False,
    'limitations':[
        'Asymptotic DeLong inference; no exact finite-sample claim.',
        'Non-rejection does not establish equivalence.',
        'Assumes independent clients within each TARGET class and paired predictions across models.',
        'No sample weighting, missing-score imputation, bootstrap p-value or AP hypothesis test.',
    ],
    'references':DELONG_REFERENCES,
}


def _midranks(values):
    """Средние 1-based ранги: сортировка и линейное заполнение блоков ties."""
    order=np.argsort(values,kind='stable')
    ordered=values[order]
    starts=np.r_[0,np.flatnonzero(ordered[1:]!=ordered[:-1])+1]
    stops=np.r_[starts[1:],len(values)]
    ranks=np.empty(len(values),dtype=np.float64)
    # В блоке [start,stop) средний 1-based ранг равен (start+stop+1)/2.
    ranks[order]=np.repeat((starts+stops+1)/2.0,stops-starts)
    return ranks


def delong_auc_covariance(y,scores):
    """Возвращает (K ROC-AUC, K×K paired covariance) для scores формы K×N.

    В отличие от публичной таблицы, scores могут быть любыми конечными
    вещественными оценками. Нужно минимум два positive и два negative клиента.
    Никаких матриц positive×negative: память O(KN), ранжирование O(KN log N).
    """
    labels=np.asarray(y)
    if labels.ndim!=1 or not np.isin(labels,[0,1]).all():
        raise ValueError('y must be one-dimensional, complete and binary (0/1)')
    predictions=np.asarray(scores,dtype=np.float64)
    if predictions.ndim==1:
        predictions=predictions[np.newaxis,:]
    if predictions.ndim!=2 or predictions.shape[0]<1 or predictions.shape[1]!=len(labels):
        raise ValueError('scores must have shape (models, len(y))')
    if not np.isfinite(predictions).all():
        raise ValueError('scores must be finite; no rows may be silently dropped')
    positive=labels==1
    m,n=int(positive.sum()),int((~positive).sum())
    if min(m,n)<2:
        raise ValueError('DeLong covariance needs at least two observations in each class')
    k=len(predictions)
    positive_placements=np.empty((k,m),dtype=np.float64)
    negative_placements=np.empty((k,n),dtype=np.float64)
    for index,values in enumerate(predictions):
        pooled_ranks=_midranks(values)
        positive_placements[index]=(pooled_ranks[positive]-_midranks(values[positive]))/n
        negative_placements[index]=1.0-(pooled_ranks[~positive]-_midranks(values[~positive]))/m
    auc=positive_placements.mean(axis=1)
    centered_positive=positive_placements-positive_placements.mean(axis=1,keepdims=True)
    centered_negative=negative_placements-negative_placements.mean(axis=1,keepdims=True)
    covariance=(centered_positive@centered_positive.T/(m*(m-1))
                +centered_negative@centered_negative.T/(n*(n-1)))
    return auc,covariance


def paired_delong_table(y,probabilities_by_model:dict[str,np.ndarray],reference='logistic_regression'):
    """Шесть парных сравнений Level 4 с общей reference и Bonferroni family=6.

    Возвращает DataFrame; описание метода и библиография находятся в attrs.
    Уровень и ID не выводятся из scores: вызывающий код обязан передать именно
    семь заранее заданных Level-4 моделей на одних сохранённых test ID.
    """
    if reference not in probabilities_by_model:
        raise ValueError(f'Reference model is missing: {reference}')
    if len(probabilities_by_model)!=7:
        raise ValueError('The prespecified Bonferroni family requires exactly seven models (six comparisons)')
    names=list(probabilities_by_model)
    if not all(isinstance(name,str) and name for name in names):
        raise ValueError('Model names must be nonempty strings')
    arrays=[]
    for name in names:
        values=np.asarray(probabilities_by_model[name],dtype=np.float64)
        if values.ndim!=1 or len(values)!=len(y):
            raise ValueError(f'Probability shape differs from y: {name}')
        if not np.isfinite(values).all() or ((values<0)|(values>1)).any():
            raise ValueError(f'Probabilities must be complete, finite and within [0,1]: {name}')
        arrays.append(values)
    auc,covariance=delong_auc_covariance(y,np.vstack(arrays))
    reference_index=names.index(reference)
    n_positive=int(np.sum(np.asarray(y)==1))
    n_negative=len(y)-n_positive
    rows=[]
    for index,name in enumerate(names):
        if name==reference:
            continue
        delta=float(auc[index]-auc[reference_index])
        variance=float(covariance[index,index]+covariance[reference_index,reference_index]
                       -2*covariance[index,reference_index])
        tolerance=32*np.finfo(np.float64).eps*max(
            abs(covariance[index,index]),abs(covariance[reference_index,reference_index]),
            abs(covariance[index,reference_index]),np.finfo(np.float64).tiny)
        if variance<0:
            if variance < -tolerance:
                raise ArithmeticError(f'Negative paired DeLong variance: {name}')
            variance=0.0  # Только отрицательная погрешность округления, не положительная variance.
        se=math.sqrt(variance)
        if se==0 and delta==0:
            z,p_raw,status=0.0,1.0,'zero_difference_zero_variance'
        elif se==0:
            z,p_raw,status=math.nan,math.nan,'undefined_zero_variance_nonzero_difference'
        else:
            z=delta/se
            p_raw=math.erfc(abs(z)/math.sqrt(2.0))
            status='ok'
        p_adjusted=min(1.0,6*p_raw) if math.isfinite(p_raw) else math.nan
        rows.append({'model':name,'reference':reference,'auc_model':float(auc[index]),
            'auc_reference':float(auc[reference_index]),'delta_roc_auc':delta,'se':se,'z':z,
            'p_raw':p_raw,'p_bonferroni':p_adjusted,
            'reject_bonferroni_0_05':bool(math.isfinite(p_adjusted) and p_adjusted<=0.05),
            'n_positive':n_positive,'n_negative':n_negative,'inference_status':status})
    table=pd.DataFrame(rows)
    metadata=deepcopy(DELONG_METADATA)
    metadata.update({'reference':reference,'comparison_models':[name for name in names if name!=reference],
                     'n_rows':len(y),'n_positive':n_positive,'n_negative':n_negative})
    table.attrs['methodology']=metadata
    return table
