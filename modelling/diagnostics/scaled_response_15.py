"""Resume paired Model 1/3 SF/MF LOO checks at response_15 (2.6825 um)."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/model3-scaling-mpl')
import numpy as np
import pandas as pd
from src.data_load import load_full_data, create_data_indexes
from src import model_1, model_3

DATA = None
PARAMS = None


def initialize(lf_count):
    global DATA, PARAMS
    full = load_full_data()
    columns = create_data_indexes(full)['x_columns']
    lf = full[full.is_hf.eq(0)]
    if lf_count < len(lf):
        lf = lf.sample(n=lf_count, random_state=42).sort_index()
    DATA = pd.concat([lf, full[full.is_hf.eq(1)]])[[*columns, 'is_hf', 'response_15']]
    PARAMS = pd.read_csv(ROOT / 'results/cv/loo/lanes/loo_model3_w015_parameters.csv')


def fit_task(task):
    name, variant, source = task
    start = perf_counter()
    train, valid = DATA.drop(index=source), DATA.loc[[source]]
    hf_only = variant == 'sf'
    if name == 'model1':
        gp = model_1.fit_model1(train, HF_only=hf_only)
        pred = model_1.predict_model1(valid, HF_only=hf_only)
    else:
        row = PARAMS[(PARAMS.held_out_source_index == source) & (PARAMS.variant == variant)].iloc[0]
        params = {key.removeprefix('tree_'): value for key, value in row.items()
                  if key.startswith('tree_') and key != 'tree_num_boost_round' and pd.notna(value)}
        for key in ['min_data_in_leaf', 'max_depth', 'num_leaves', 'max_bin']:
            if key in params:
                params[key] = int(params[key])
        model = model_3.fit_model3(train, HF_only=hf_only, verbose_eval=False,
            tuning_result={'best_params': params, 'best_iter': int(row.tree_num_boost_round)})
        gp = model.gp_model
        pred = model_3.predict_model3(valid, HF_only=hf_only, model=model)
    cov = gp.get_cov_pars(std_err=False).iloc[0].to_dict()
    gp_var = cov['GP_var'] if hf_only else cov['rho']**2 * cov['low_GP_var'] + cov['discrepancy_GP_var']
    result = dict(model=name, variant=variant, source_index=int(source), response_index=15,
        wavelength=2.6825, y_true=float(valid.iloc[0,-1]), y_pred=float(pred['mu'][0]),
        variance=float(pred['var'][0]), hf_prior_gp_variance=gp_var,
        gp_fraction=gp_var/(gp_var+cov['Error_var']), seconds=perf_counter()-start, **cov)
    assert np.isfinite([result['y_pred'], result['variance'], gp_var]).all()
    assert result['variance'] >= 0
    return result


def summarize(out):
    records = [json.loads(path.read_text()) for path in sorted((out/'fits').glob('*.json'))]
    if not records:
        return
    new = pd.DataFrame(records)
    assert not new.duplicated(['model', 'variant', 'source_index']).any()
    new.to_csv(out/'predictions.csv', index=False)
    old = pd.concat([pd.read_csv(ROOT/f'results/cv/loo/lanes/loo_{name}_w015_predictions.csv')
                     for name in ['model1','model3']]).rename(columns={'held_out_source_index':'source_index'})
    old = old.merge(new[['model','variant','source_index']], on=['model','variant','source_index'], validate='one_to_one')
    assert len(old) == len(new)
    old_params = pd.concat([pd.read_csv(ROOT/f'results/cv/loo/lanes/loo_{name}_w015_parameters.csv')
                            for name in ['model1','model3']]).rename(columns={'held_out_source_index':'source_index'})
    old_params['hf_prior_gp_variance'] = np.where(old_params.variant.eq('sf'), old_params.GP_var,
        old_params.rho**2 * old_params.low_GP_var + old_params.discrepancy_GP_var)
    old_params['gp_fraction'] = old_params.hf_prior_gp_variance/(old_params.hf_prior_gp_variance+old_params.Error_var)
    old = old.merge(old_params[['model','variant','source_index','hf_prior_gp_variance','gp_fraction']],
                    on=['model','variant','source_index'], validate='one_to_one')
    old['setup'], new['setup'] = 'saved_raw', 'scaled'
    combined = pd.concat([old,new],ignore_index=True)
    combined['se'] = (combined.y_pred-combined.y_true)**2
    combined['covered95'] = (combined.y_pred-combined.y_true).abs()<=1.96*np.sqrt(combined.variance)
    summary = combined.groupby(['model','variant','setup']).agg(n=('y_pred','size'),
        mse=('se','mean'),coverage95=('covered95','mean'),
        median_gp_variance=('hf_prior_gp_variance','median'),median_gp_fraction=('gp_fraction','median'))
    summary['rmse']=np.sqrt(summary.pop('mse'))
    summary.to_csv(out/'summary.csv')
    quick = combined[combined.source_index.isin([10000,10021,10068])]
    quick.to_csv(out/'quick_predictions.csv',index=False)
    quick_summary = quick.groupby(['model','variant','setup']).agg(n=('y_pred','size'),
        mse=('se','mean'),median_gp_variance=('hf_prior_gp_variance','median'),
        median_gp_fraction=('gp_fraction','median'))
    quick_summary['rmse'] = np.sqrt(quick_summary.pop('mse'))
    quick_summary.to_csv(out/'quick_summary.csv')
    print(summary.to_string(),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants',nargs='+',choices=['sf','mf'],default=['sf','mf'])
    parser.add_argument('--sources',nargs='+',type=int,default=list(range(10000,10097)))
    parser.add_argument('--lf-count',type=int,default=10000)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    out=ROOT/f'results/scaled_response_15/lf{args.lf_count}'
    (out/'fits').mkdir(parents=True,exist_ok=True)
    tasks=[(name,variant,source) for variant in args.variants for source in args.sources for name in ['model1','model3']]
    def path(task):
        return out/'fits'/f'{task[0]}_{task[1]}_{task[2]}.json'
    pending=[task for task in tasks if not path(task).exists()]
    print(f'{len(pending)} pending fits; LF={args.lf_count}; workers={args.workers}',flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.lf_count,)) as pool:
        futures={pool.submit(fit_task,task):task for task in pending}
        for future in as_completed(futures):
            task=futures[future]
            result=future.result()
            path(task).write_text(json.dumps(result,indent=2))
            print(f'{task}: {result["seconds"]:.1f}s; GP fraction={result["gp_fraction"]:.4g}',flush=True)
    summarize(out)


if __name__=='__main__':
    main()
