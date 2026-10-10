#!/usr/bin/env python3
"""Gaussian diagnostics for FROZEN unnormalized document scores (CPU only).

Inputs: white-box factorial scores.npz; black-box T_theta scores.npz;
fresh *_ltheta scores.npz; or verified exact document_metrics.csv.
No LLM loading, token replay, detector fitting, score flipping or calibration tuning.
Default: repeated two-fold diagnostic fitting of scalar Gaussians, NOT training.
See README.md for statistical definitions and limitations.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

VERSION = '2026.09.22-gaussian-diagnostics-v1'
MODELS = ['falcon-7b', 'llama-2-7b', 'opt-2.7b', 'opt-6.7b']
DATASETS = ['c4', 'pubmed', 'squad', 'writing', 'xsum']
COLORS = {'human': '#2c6584', 'machine': '#bb6529'}


def csv_write(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def clean_json(x):
    if isinstance(x, dict):
        return {k: clean_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean_json(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return clean_json(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def json_write(path, x):
    Path(path).write_text(json.dumps(clean_json(x), indent=2, ensure_ascii=False,
                                    allow_nan=False) + '\n', encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def seeded(seed, *parts):
    value = int.from_bytes(hashlib.sha256('|'.join(map(str, parts)).encode()).digest()[:8], 'little')
    return np.random.default_rng((seed + value) % (2 ** 64))


def vector(x, name, minimum=20):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) < minimum or not np.isfinite(x).all():
        raise ValueError(f'{name}: need a finite 1D array with >= {minimum} scores; got {x.shape}')
    if np.std(x, ddof=1) <= 0:
        raise ValueError(f'{name}: constant scores; Gaussian scale is undefined')
    return x


def resolve(root, model, dataset, variant):
    cell = root / model / dataset
    # Explicitly prefer per-variant result over combined result, never recurse into other runs.
    for p in (cell / variant / 'scores.npz', cell / 'scores.npz', cell / 'document_metrics.csv'):
        if p.is_file():
            return p
    raise FileNotFoundError(f'No score file for {model}/{dataset}. Checked {cell}/[{variant}/]scores.npz and document_metrics.csv')


def load_scores(path, variant, declaration):
    meta_path = path.with_name('results.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    protocol = meta.get('protocol', {})
    if not isinstance(protocol, dict):
        protocol = {}
    if path.suffix == '.csv':
        if protocol.get('moments') != 'exact' or protocol.get('score_verification_skipped', True):
            raise ValueError(f'{path}: document CSV must come from verified exact replay. Use original frozen scores.npz instead.')
        rows = list(csv.DictReader(path.open(encoding='utf-8')))
        arrays = {}
        ids = {}
        for split in ('human', 'machine', 'calibration'):
            rr = sorted((r for r in rows if r['variant'] == variant and r['split'] == split),
                        key=lambda r: int(r['document_index']))
            ix = [int(r['document_index']) for r in rr]
            if len(ix) != len(set(ix)):
                raise ValueError(f'{path}: duplicate document indices in {split}')
            arrays[split] = vector([float(r['score']) for r in rr], split)
            ids[split] = ix
        layout = 'verified_exact_document_csv'
    else:
        with np.load(path, allow_pickle=False) as z:
            layouts = [
                ('factorial', {s: f'{variant}__{s}' for s in ('human', 'machine', 'calibration')}),
                ('explicit_ltheta', {s: f'{s}_ltheta' for s in ('human', 'machine', 'calibration')}),
                ('plain', {s: s for s in ('human', 'machine', 'calibration')}),
            ]
            hit = next(((name, keys) for name, keys in layouts if all(k in z for k in keys.values())), None)
            if hit is None:
                raise ValueError(f'{path}: no supported score keys. Available: {z.files}')
            layout, keys = hit
            if layout == 'explicit_ltheta' and variant != 'full':
                raise ValueError('The *_ltheta layout contains only the full detector; use --variant full.')
            if layout == 'plain':
                recorded_variant = meta.get('variant', 'full')
                if recorded_variant != variant:
                    raise ValueError(f'{path}: plain scores belong to {recorded_variant}, not requested {variant}')
            arrays = {s: vector(z[k], f'{path}:{k}') for s, k in keys.items()}
        score_text = ' '.join(str(v) for v in [meta.get('primary_score', ''), meta.get('score_definition', ''),
                    protocol.get('primary_score', ''), protocol.get('deployment_score', '')])
        recognized = ('T_theta' in score_text or 'L_theta' in score_text or 'theta^T U' in score_text)
        if layout != 'explicit_ltheta' and not recognized and declaration != 'linear':
            raise ValueError(f'{path}: cannot verify unnormalized score semantics. Supply results.json or explicitly --score-declaration linear after checking it is T_theta, not S_theta.')
        ids = {}
    if any(np.array_equal(arrays['calibration'], arrays[s]) for s in ('human', 'machine')):
        raise ValueError(f'{path}: calibration scores equal a test array; verify independent splits')
    # Never determine score direction from test labels.
    return arrays, {'path': str(path.resolve()), 'sha256': digest(path), 'layout': layout,
                    'metadata_path': str(meta_path.resolve()) if meta_path.exists() else None,
                    'metadata_sha256': digest(meta_path) if meta_path.exists() else None,
                    'builder_version': meta.get('builder_version'), 'run_signature': meta.get('run_signature'),
                    'n': {s: len(v) for s, v in arrays.items()}, 'document_indices': ids,
                    'score_declaration': declaration}


def gaussian_fit(h, m):
    vh, vm = np.var(h, ddof=1), np.var(m, ddof=1)
    pooled = ((len(h) - 1) * vh + (len(m) - 1) * vm) / (len(h) + len(m) - 2)
    if min(vh, vm, pooled) <= 0:
        raise ValueError('Zero variance in a Gaussian fitting partition')
    return {'mu_h': float(np.mean(h)), 'mu_m': float(np.mean(m)),
            'sd_h': float(np.sqrt(vh)), 'sd_m': float(np.sqrt(vm)), 'sd_shared': float(np.sqrt(pooled))}


def pred_metrics(fit, alpha, shared):
    sh = fit['sd_shared'] if shared else fit['sd_h']
    sm = fit['sd_shared'] if shared else fit['sd_m']
    threshold = fit['mu_h'] + sh * stats.norm.isf(alpha)
    power = stats.norm.sf((threshold - fit['mu_m']) / sm)
    auc = stats.norm.cdf((fit['mu_m'] - fit['mu_h']) / np.hypot(sh, sm))
    return float(threshold), float(power), float(auc)


def auc(h, m):
    return float(roc_auc_score(np.r_[np.zeros(len(h)), np.ones(len(m))], np.r_[h, m]))


def empirical_roc(h, m, alphas):
    f, t, _ = roc_curve(np.r_[np.zeros(len(h)), np.ones(len(m))], np.r_[h, m], drop_intermediate=False)
    # Last point at each repeated FPR is the upper endpoint of that vertical segment.
    ux = np.unique(f)
    ut = np.array([np.max(t[f == v]) for v in ux])
    return np.interp(alphas, ux, ut)


def conformal_threshold(cal, alpha):
    # Exact decimal floor avoids e.g. 0.05 * 100 rounding down incorrectly.
    k = int((Decimal(str(alpha)) * (len(cal) + 1)).to_integral_value(rounding=ROUND_FLOOR))
    return float('inf') if k < 1 else float(np.sort(cal)[len(cal) - k])


def wilson(k, n):
    z = stats.norm.ppf(0.975)
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return float(center-half), float(center+half)


def fitted_ks(x):
    x = np.asarray(x)
    z = np.sort((x - x.mean(axis=-1, keepdims=True)) / x.std(axis=-1, ddof=1, keepdims=True), axis=-1)
    f = stats.norm.cdf(z)
    n = x.shape[-1]
    return np.maximum(np.max(np.arange(1, n+1)/n - f, axis=-1),
                      np.max(f - np.arange(n)/n, axis=-1))


def shape_diagnostic(x, rng, reps):
    d = float(fitted_ks(x))
    exceed = 0
    for start in range(0, reps, 100):
        b = min(100, reps-start)
        # Re-estimate location AND scale for each simulated sample (Lilliefors-type test).
        exceed += np.count_nonzero(fitted_ks(rng.normal(size=(b, len(x)))) >= d)
    p = (exceed+1)/(reps+1) if reps else None
    prob = (np.arange(len(x)) + 0.5)/len(x)
    q = stats.norm.ppf(prob)
    z = np.sort((x-x.mean())/x.std(ddof=1))
    return {'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std(ddof=1)),
            'skewness': float(stats.skew(x, bias=False)),
            'excess_kurtosis': float(stats.kurtosis(x, fisher=True, bias=False)),
            'qq_rmse': float(np.sqrt(np.mean((z-q)**2))),
            'fitted_normal_ks': d, 'normality_mc_p': p}


def diagnostics_partitions(h, m, repeats, rng, paired):
    if paired and len(h) != len(m):
        raise ValueError('--paired requires equal human/machine counts in matching order')
    for repeat in range(repeats):
        ih = rng.permutation(len(h))
        im = ih.copy() if paired else rng.permutation(len(m))
        hh, mm = np.array_split(ih, 2), np.array_split(im, 2)
        for fold in (0, 1):
            yield repeat, fold, hh[1-fold], mm[1-fold], hh[fold], mm[fold]


def bh(rows):
    valid = [(i, r['normality_mc_p']) for i, r in enumerate(rows) if r.get('normality_mc_p') is not None]
    order = sorted(valid, key=lambda t: t[1])
    prev = 1.0
    for rank in range(len(order), 0, -1):
        i, p = order[rank-1]
        prev = min(prev, p*len(order)/rank)
        rows[i]['normality_mc_bh_q'] = prev


def savefig(fig, path):
    fig.savefig(str(path)+'.pdf', bbox_inches='tight')
    fig.savefig(str(path)+'.png', bbox_inches='tight', dpi=180)
    plt.close(fig)


def cell_plot(h, m, out, title):
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    for cls, x in [('human', h), ('machine', m)]:
        z = np.sort((x-x.mean())/x.std(ddof=1))
        q = stats.norm.ppf((np.arange(len(x))+0.5)/len(x))
        axes[0].plot(q, z, '.', ms=3, alpha=.65, label=cls.capitalize(), color=COLORS[cls])
        axes[1].hist(x, bins='fd', density=True, histtype='step', lw=1.4, color=COLORS[cls], label=cls.capitalize())
        grid = np.linspace(x.min(), x.max(), 300)
        axes[1].plot(grid, stats.norm.pdf(grid, x.mean(), x.std(ddof=1)), color=COLORS[cls], ls='--', lw=1.2)
    lim = axes[0].get_xlim()
    axes[0].plot(lim, lim, color='0.4', lw=.8)
    axes[0].set(xlabel='Standard normal quantile', ylabel='Standardized score quantile', title='Classwise Q–Q (descriptive)')
    axes[1].set(xlabel='Frozen document score', ylabel='Density', title='Observed scores and fitted normals')
    for ax in axes:
        ax.legend(frameon=False, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(title, fontsize=11)
    savefig(fig, out / 'score_distribution')


def external_fit(root, model, dataset, variant, score_arrays):
    # Intentionally distinct filename/keys to prevent silent reuse of test scores as training scores.
    p = root / model / dataset / 'fit_scores.npz'
    with np.load(p, allow_pickle=False) as z:
        hk, mk = f'{variant}__fit_human', f'{variant}__fit_machine'
        if hk not in z or mk not in z:
            hk, mk = 'fit_human', 'fit_machine'
        h, m = vector(z[hk], f'{p}:{hk}'), vector(z[mk], f'{p}:{mk}')
    if np.array_equal(h, score_arrays['human']) or np.array_equal(m, score_arrays['machine']):
        raise ValueError(f'{p}: fitting scores equal test scores; independent fit is required')
    return h, m, {'path': str(p.resolve()), 'sha256': digest(p)}


def analyze(args, model, dataset):
    source = resolve(args.scores_root, model, dataset, args.variant)
    data, provenance = load_scores(source, args.variant, args.score_declaration)
    h, m, cal = (data[s] for s in ('human', 'machine', 'calibration'))
    if min(len(h), len(m), len(cal)) < args.min_documents:
        raise ValueError(f'{model}/{dataset}: fewer than --min-documents={args.min_documents}; use a complete run')
    out = args.output_root / model / dataset
    out.mkdir(parents=True, exist_ok=True)
    base = {'model': model, 'dataset': dataset, 'variant': args.variant}
    shapes = []
    for cls, x in [('human', h), ('machine', m)]:
        shapes.append({**base, 'class': cls, **shape_diagnostic(x, seeded(args.seed, model, dataset, cls, 'shape'), args.normality_reps)})
    fit_all = gaussian_fit(h, m)
    cell = {**base, 'n_human': len(h), 'n_machine': len(m), 'n_calibration': len(cal),
            'auroc': auc(h, m), 'variance_ratio_machine_over_human': float(np.var(m,ddof=1)/np.var(h,ddof=1)),
            'mean_gap': fit_all['mu_m']-fit_all['mu_h'],
            'diagnostic_fit_source': 'external' if args.fit_scores_root else 'two_fold_test_score_diagnostics',
            'direction_warning': bool(fit_all['mu_m'] < fit_all['mu_h'])}
    if cell['direction_warning']:
        print(f'NOTE {model}/{dataset}: machine mean below human mean; direction is preserved.', flush=True)
    # Ratio bootstrap: within-class documents; paired resampling only on explicit --paired.
    rng = seeded(args.seed, model, dataset, 'variance_bootstrap')
    ratios = []
    if args.paired and len(h) != len(m):
        raise ValueError('--paired requires equal counts')
    for _ in range(args.bootstrap_reps):
        ih = rng.integers(len(h), size=len(h))
        im = ih if args.paired else rng.integers(len(m), size=len(m))
        den = np.var(h[ih], ddof=1)
        if den > 0:
            ratios.append(np.var(m[im], ddof=1)/den)
    if ratios:
        cell['variance_ratio_bootstrap_lo'], cell['variance_ratio_bootstrap_hi'] = map(float, np.quantile(ratios,[.025,.975]))
    empirical = empirical_roc(h, m, args.alphas)
    operation = []
    for alpha, ep in zip(args.alphas, empirical):
        threshold = conformal_threshold(cal, alpha)
        kh, km = int(np.sum(h > threshold)), int(np.sum(m > threshold))
        fl, fu = wilson(kh,len(h)); pl, pu = wilson(km,len(m))
        operation.append({**base, 'alpha': alpha, 'conformal_threshold': threshold,
            'empirical_roc_power': float(ep), 'conformal_fpr': kh/len(h), 'conformal_power': km/len(m),
            'fpr_wilson_lo': fl, 'fpr_wilson_hi': fu, 'power_wilson_lo': pl, 'power_wilson_hi': pu})
    records, splits = [], []
    if args.fit_scores_root:
        fh, fm, ext = external_fit(args.fit_scores_root, model, dataset, args.variant, data)
        provenance['external_fit'] = ext
        jobs = [(0, 0, fh, fm, h, m)]
    else:
        jobs = []
        for rep, fold, ih, im, eh, em in diagnostics_partitions(h,m,args.repeats,seeded(args.seed,model,dataset,'split'),args.paired):
            splits.append({'repeat':rep,'fold':fold,'fit_human_indices':ih.tolist(),'fit_machine_indices':im.tolist(),
                           'eval_human_indices':eh.tolist(),'eval_machine_indices':em.tolist()})
            jobs.append((rep,fold,h[ih],m[im],h[eh],m[em]))
    for rep, fold, fh, fm, eh, em in jobs:
        fit = gaussian_fit(fh,fm)
        ea = auc(eh,em)
        eroc = empirical_roc(eh,em,args.alphas)
        for family in ('shared', 'separate'):
            for alpha, ep in zip(args.alphas,eroc):
                threshold, pp, pa = pred_metrics(fit,alpha,family=='shared')
                # Conformal threshold uses ONLY calibration humans; Gaussian M is fit separately.
                ccal = conformal_threshold(cal,alpha)
                sm = fit['sd_shared'] if family=='shared' else fit['sd_m']
                sh = fit['sd_shared'] if family=='shared' else fit['sd_h']
                records.append({**base,'repeat':rep,'fold':fold,'family':family,'alpha':alpha,
                    'n_fit_human':len(fh),'n_fit_machine':len(fm),'n_eval_human':len(eh),'n_eval_machine':len(em),
                    **fit,'gaussian_threshold':threshold,'predicted_power':pp,'empirical_roc_power':float(ep),
                    'power_prediction_error':pp-float(ep),'predicted_auc':pa,'empirical_auc':ea,
                    'gaussian_threshold_actual_fpr':float(np.mean(eh>threshold)),
                    'gaussian_threshold_actual_power':float(np.mean(em>threshold)),
                    'predicted_conformal_fpr':float(stats.norm.sf((ccal-fit['mu_h'])/sh)),
                    'predicted_conformal_power':float(stats.norm.sf((ccal-fit['mu_m'])/sm)),
                    'actual_conformal_fpr':float(np.mean(eh>ccal)),
                    'actual_conformal_power':float(np.mean(em>ccal)),
                })
    aggregates = []
    for family in ('shared','separate'):
        for alpha in args.alphas:
            rr = [r for r in records if r['family']==family and r['alpha']==alpha]
            keys = ['predicted_power','empirical_roc_power','power_prediction_error','predicted_auc','empirical_auc',
                    'gaussian_threshold_actual_fpr','gaussian_threshold_actual_power','predicted_conformal_fpr',
                    'predicted_conformal_power','actual_conformal_fpr','actual_conformal_power']
            row = {**base,'family':family,'alpha':alpha,'n_diagnostic_partitions':len(rr)}
            for key in keys:
                row[key] = float(np.mean([r[key] for r in rr]))
            row['partition_mean_abs_power_error'] = float(np.mean([abs(r['power_prediction_error']) for r in rr]))
            row['conformal_power_prediction_error'] = row['predicted_conformal_power']-row['actual_conformal_power']
            aggregates.append(row)
    provenance.update({'version': VERSION, 'fixed_direction':True, 'detector_refitted':False,
                       'gaussian_parameters_fitted_on_test_partitions':not bool(args.fit_scores_root),
                       'test_scores_not_changed':True, 'paired_partitions':args.paired})
    json_write(out/'provenance.json',provenance)
    json_write(out/'diagnostic_partitions.json',splits)
    csv_write(out/'shape_diagnostics.csv',shapes)
    csv_write(out/'operating_points.csv',operation)
    csv_write(out/'gaussian_prediction_partitions.csv',records)
    csv_write(out/'gaussian_prediction.csv',aggregates)
    csv_write(out/'cell_summary.csv',[cell])
    if not args.no_plots:
        cell_plot(h,m,out,f'{model} / {dataset}')
    return cell, shapes, operation, aggregates


def summary_plot(args, cells, predictions, operations):
    alpha = 0.05 if 0.05 in args.alphas else args.alphas[-1]
    fig, axes = plt.subplots(1,3,figsize=(11.4,3.4),constrained_layout=True)
    for fam,color in [('shared','#2c6584'),('separate','#bb6529')]:
        rows = [r for r in predictions if r['family']==fam and r['alpha']==alpha]
        axes[0].scatter([r['predicted_power'] for r in rows],[r['empirical_roc_power'] for r in rows],
                        label=f'{fam.capitalize()} variance',color=color,s=26,alpha=.8)
    axes[0].plot([0,1],[0,1],color='0.5',ls='--',lw=.8)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Gaussian predicted power',ylabel='Empirical ROC power',title=f'Power at FPR = {alpha:.0%}')
    axes[0].legend(fontsize=7,frameon=False)
    ratios=np.array([c['variance_ratio_machine_over_human'] for c in cells])
    xx=np.arange(len(cells))
    axes[1].scatter(xx,ratios,s=18,color='#2c6584')
    if all('variance_ratio_bootstrap_lo' in c for c in cells):
        lo=np.array([c['variance_ratio_bootstrap_lo'] for c in cells])
        hi=np.array([c['variance_ratio_bootstrap_hi'] for c in cells])
        axes[1].vlines(xx,lo,hi,color='#2c6584',alpha=.6,lw=1)
    axes[1].axhline(1,color='0.5',ls='--',lw=.8)
    axes[1].set(yscale='log',xlabel='Setting index (see cell_index.csv)',ylabel='Machine / human score variance',title='Projected variance ratio')
    for a in args.alphas:
        rr=[r for r in operations if r['alpha']==a]
        axes[2].scatter([a]*len(rr),[r['conformal_fpr'] for r in rr],s=18,alpha=.6,color='#2c6584')
    hi=max(max(args.alphas)*1.25,max(r['conformal_fpr'] for r in operations)*1.1)
    axes[2].plot([0,hi],[0,hi],ls='--',color='0.5',lw=.8)
    axes[2].set(xlim=(0,hi),ylim=(0,hi),xlabel='Nominal FPR',ylabel='Observed conformal FPR',title='Independent human calibration')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
    savefig(fig,args.output_root/'diagnostic_overview')


def self_test():
    f={'mu_h':0.,'mu_m':1.,'sd_h':1.,'sd_m':1.,'sd_shared':1.}
    for a in [.01,.05]:
        t,p,au=pred_metrics(f,a,True)
        assert np.isclose(p,stats.norm.cdf(1-stats.norm.isf(a)))
        assert np.isclose(au,stats.norm.cdf(1/np.sqrt(2)))
        assert pred_metrics(f,a,False)==(t,p,au)
    # Inclusive conformal >=, strict score threshold >, finite resolution, and ties.
    cal=np.array([0.,0.,1.,1.,2.,2.,3.,3.,4.])
    x=np.array([-1.,0.,.5,1.,2.,3.,4.,4.1,9.])
    for a in [.01,.1,.2,.5,.9]:
        pp=(1+(cal[:,None]>=x).sum(axis=0))/(len(cal)+1)
        assert np.array_equal(pp<=a,x>conformal_threshold(cal,a))
    rng=np.random.default_rng(71)
    h=rng.normal(size=20000);m=rng.normal(1,1,size=20000)
    ff=gaussian_fit(h,m)
    assert abs(pred_metrics(ff,.05,True)[1]-empirical_roc(h,m,[.05])[0])<.025
    hh=np.arange(500.);mm=hh+1
    jobs=list(diagnostics_partitions(hh,mm,2,np.random.default_rng(1),True))
    for rep,fold,ih,im,eh,em in jobs:
        assert not set(ih)&set(eh)
        assert set(ih)|set(eh)==set(range(500))
        assert np.array_equal(ih,im) and np.array_equal(eh,em)
    d=shape_diagnostic(rng.normal(size=500),rng,99)
    assert 0<d['normality_mc_p']<=1
    # Location/scale invariance of the fitted-normal discrepancy.
    assert np.isclose(fitted_ks(h[:500]),fitted_ks(4*h[:500]+3))
    print('PASS: analytic Gaussian power/AUC; tie-aware conformal decisions; empirical ROC; disjoint paired partitions; fitted-normal test.')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scores-root',type=Path,help='Exact root containing model/dataset scores; no dated defaults.')
    p.add_argument('--output-root',type=Path)
    p.add_argument('--models',default=','.join(MODELS))
    p.add_argument('--datasets',default=','.join(DATASETS))
    p.add_argument('--variant',default='full')
    p.add_argument('--alphas',default='0.01,0.05')
    p.add_argument('--repeats',type=int,default=10,help='Repeated two-fold scalar Gaussian diagnostics; repetitions are NOT independent observations.')
    p.add_argument('--normality-reps',type=int,default=999)
    p.add_argument('--bootstrap-reps',type=int,default=1000)
    p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--min-documents',type=int,default=100)
    p.add_argument('--paired',action='store_true',help='Use only when human/machine arrays have matching source documents in the same order.')
    p.add_argument('--fit-scores-root',type=Path,help='Optional independent Gaussian fitting scores: model/dataset/fit_scores.npz.')
    p.add_argument('--score-declaration',choices=['auto','linear'],default='auto')
    p.add_argument('--no-plots',action='store_true')
    p.add_argument('--overwrite',action='store_true')
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test();return 0
    if args.scores_root is None or args.output_root is None:
        p.error('--scores-root and --output-root are required')
    args.scores_root=args.scores_root.resolve();args.output_root=args.output_root.resolve()
    if not args.scores_root.is_dir():
        p.error(f'Input root does not exist: {args.scores_root}')
    if args.output_root==args.scores_root or args.output_root in args.scores_root.parents or args.scores_root in args.output_root.parents:
        p.error('Input and output directories must be separate, not ancestors/descendants')
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        p.error('Output is nonempty; choose a new directory or explicitly --overwrite')
    args.alphas=[float(x) for x in args.alphas.split(',')]
    if not args.alphas or any(not 0<a<.5 for a in args.alphas):
        p.error('Use 0 < alpha < 0.5')
    if args.repeats<1 or args.normality_reps<0 or args.bootstrap_reps<0 or args.min_documents<40:
        p.error('Require repeats>=1, reps>=0, min-documents>=40')
    models=[x.strip() for x in args.models.split(',') if x.strip()]
    datasets=[x.strip() for x in args.datasets.split(',') if x.strip()]
    if not models or not datasets or len(set(models))!=len(models) or len(set(datasets))!=len(datasets):
        p.error('Nonempty unique model/dataset lists required')
    # Preflight all requested cells before creating output; no silent skipped cells.
    for model in models:
        for dataset in datasets:
            path=resolve(args.scores_root,model,dataset,args.variant)
            arrays,_=load_scores(path,args.variant,args.score_declaration)
            if min(map(len,arrays.values()))<args.min_documents:
                raise ValueError(f'{path}: too few documents; do not use a 20-document smoke test')
            if args.fit_scores_root:
                external_fit(args.fit_scores_root,model,dataset,args.variant,arrays)
    args.output_root.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.titlesize':10,
                         'pdf.fonttype':42,'ps.fonttype':42})
    config=vars(args).copy()
    json_write(args.output_root/'run_manifest.json',{'status':'running','version':VERSION,'args':config,
        'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__})
    cells,shapes,operations,predictions=[],[],[],[]
    for model in models:
        for dataset in datasets:
            print(f'Analyzing {model}/{dataset} ({args.variant})',flush=True)
            c,s,o,g=analyze(args,model,dataset)
            cells.append(c);shapes.extend(s);operations.extend(o);predictions.extend(g)
    bh(shapes)
    for model in models:
        for dataset in datasets:
            csv_write(args.output_root/model/dataset/'shape_diagnostics.csv',
                      [r for r in shapes if r['model']==model and r['dataset']==dataset])
    csv_write(args.output_root/'all_cells_summary.csv',cells)
    csv_write(args.output_root/'all_shape_diagnostics.csv',shapes)
    csv_write(args.output_root/'all_operating_points.csv',operations)
    csv_write(args.output_root/'all_gaussian_predictions.csv',predictions)
    csv_write(args.output_root/'cell_index.csv',[{'index':i,'model':c['model'],'dataset':c['dataset']} for i,c in enumerate(cells)])
    macro=[]
    for fam in ('shared','separate'):
        for alpha in args.alphas:
            rr=[r for r in predictions if r['family']==fam and r['alpha']==alpha]
            macro.append({'family':fam,'alpha':alpha,'n_cells':len(rr),
                          'mean_predicted_power':float(np.mean([r['predicted_power'] for r in rr])),
                          'mean_empirical_roc_power':float(np.mean([r['empirical_roc_power'] for r in rr])),
                          'cell_mean_absolute_power_error':float(np.mean([abs(r['power_prediction_error']) for r in rr])),
                          'mean_abs_conformal_power_error':float(np.mean([abs(r['conformal_power_prediction_error']) for r in rr]))})
    csv_write(args.output_root/'macro_prediction_errors.csv',macro)
    if not args.no_plots:
        summary_plot(args,cells,predictions,operations)
    json_write(args.output_root/'run_manifest.json',{'status':'complete','version':VERSION,'args':config,
        'n_cells':len(cells),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
        'interpretation':'Scalar frozen-score diagnostics, not a proof of multivariate Gaussianity, shared covariance, MCLT or an optimal projection. Default Gaussian fits use held-out score partitions, not original detector training scores.',
        'uncertainty':'Variance-ratio CI is document bootstrap; operating-point Wilson intervals condition on the fixed calibration set. Repeated diagnostic partitions are dependent; no across-repeat CI is claimed.'})
    print(f'COMPLETE: {len(cells)} cells -> {args.output_root}',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (ValueError,FileNotFoundError,KeyError) as e:
        print(f'ERROR: {e}',file=sys.stderr)
        raise SystemExit(2)
