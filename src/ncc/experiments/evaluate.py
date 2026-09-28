"""Evaluation: results pkl -> metrics -> CSV."""
import logging

import numpy as np

from ncc.experiments import paths
from ncc.experiments.config import dataset_name
from ncc.experiments.files import load_pickle, write_csv
from ncc.experiments.metrics import METRIC_NAMES, eval_metrics

logger = logging.getLogger(__name__)


def evaluate(exp, seed=None, all_seeds=False, clip=False, sort=False):
    """Writes the metric CSV of one seed (default: exp.seed), or of all exp.seeds, and returns its path.

    One seed: results/csvs/<exp_id>_<seed>.csv, per region, with the median and mean rows on top.
    All seeds: results/csvs/<exp_id>_multi_seeds.csv, per seed, with the mean row on top.
    clip: only evaluate the steps in data/<dataset>_clip_idx.pkl (all steps if there is no such file).
    sort: sort the interval widths at every step so they shrink as alpha grows.
    """
    clip_idx_list = load_clip_idx(exp) if clip else None
    if all_seeds:
        output_path = aggregate_seeds(exp, clip_idx_list, sort_res=sort)
    else:
        seed = exp.seed if seed is None else seed
        metrics = eval_one_exp(exp, seed, clip_idx_list=clip_idx_list, sort_res=sort)
        output_path = paths.csv_dir() / f'{exp.exp_id}_{seed}.csv'
        save_metrics2csv(metrics, output_path)
    logger.info('Metrics saved to %s', output_path)
    return output_path


def prepare_data(saved_pred_file, alphas):
    """Converts saved NCC results to preds[region][week_ahead] = (y_trues, y_preds, lowers, uppers).
    Negative q-hats are set to zero."""
    predictions = load_pickle(saved_pred_file)
    preds_in_right_format = {}
    for region, region_predictions in predictions.items():
        preds_in_right_format[region] = {}
        for week_ahead, cur_predictions in region_predictions.items():
            y_trues = []
            y_preds = []
            lowers = {alpha: [] for alpha in alphas}
            uppers = {alpha: [] for alpha in alphas}
            for q_hats, y_hat, y in cur_predictions:
                q_hats = np.array(q_hats)
                q_hats[q_hats < 0] = 0
                y_val = y[0]
                y_hat_val = y_hat[0]
                y_trues.append(y_val)
                y_preds.append(y_hat_val)
                for i, alpha in enumerate(alphas):
                    lowers[alpha].append(y_hat_val - q_hats[i])
                    uppers[alpha].append(y_hat_val + q_hats[i])
            preds_in_right_format[region][week_ahead] = (y_trues, y_preds, lowers, uppers)
    return preds_in_right_format


def list_idx(mylist, myidx):
    return [mylist[i] for i in myidx]


def clip_ts_with_index_list(preds, idxes):
    """Keep only the time steps in idxes, e.g. the respiratory season for covid."""
    clipped_preds = {}
    for region in preds:
        clipped_preds[region] = {}
        for week_ahead, (y_trues, y_preds, lowers, uppers) in preds[region].items():
            clipped_preds[region][week_ahead] = (
                list_idx(y_trues, idxes),
                list_idx(y_preds, idxes),
                {alpha: list_idx(lowers[alpha], idxes) for alpha in lowers},
                {alpha: list_idx(uppers[alpha], idxes) for alpha in uppers},
            )
    return clipped_preds


def sort_preds(preds):
    """At every time step, reassign the half interval widths so they shrink as alpha grows."""
    sorted_preds = {}
    for region in preds:
        sorted_preds[region] = {}
        for week_ahead, (y_trues, y_preds, lowers, uppers) in preds[region].items():
            alphas = list(lowers.keys())
            sorted_lowers = {alpha: [] for alpha in alphas}
            sorted_uppers = {alpha: [] for alpha in alphas}
            for i in range(len(y_preds)):
                halfPIs = {alpha: y_preds[i] - lowers[alpha][i] for alpha in alphas}  # half PIs in one time step
                sorted_keys = sorted(list(halfPIs.keys()))
                sorted_vals = sorted(list(halfPIs.values()), reverse=True)
                for j in range(len(sorted_keys)):
                    sorted_lowers[sorted_keys[j]].append(y_preds[i] - sorted_vals[j])
                    sorted_uppers[sorted_keys[j]].append(y_preds[i] + sorted_vals[j])
            sorted_preds[region][week_ahead] = (y_trues, y_preds, sorted_lowers, sorted_uppers)
    return sorted_preds


def load_clip_idx(exp):
    """Evaluation indexes of the experiment's dataset (data/<dataset>_clip_idx.pkl), or None if there are none."""
    dataset = dataset_name(exp)
    clip_idx_file_path = paths.clip_idx(dataset)
    if not clip_idx_file_path.exists():
        logger.info('clip index file not found for dataset %s, evaluating all test steps', dataset)
        return None
    return load_pickle(clip_idx_file_path)


def eval_one_exp(exp, seed, clip_idx_list=None, sort_res=False):
    """Returns metrics[region][week_ahead] = {metric name: value}."""
    preds = prepare_data(paths.results(exp.exp_id, seed), exp.ncc.alphas)
    if sort_res:
        preds = sort_preds(preds)
    if clip_idx_list is not None:
        preds = clip_ts_with_index_list(preds, clip_idx_list)
    metrics = {}
    for region in preds:
        metrics[region] = {}
        for week_ahead, (y_trues, y_preds, lowers, uppers) in preds[region].items():
            metrics[region][week_ahead] = eval_metrics(y_preds, y_trues, lowers, uppers)
    return metrics


def summary_rows(rows, label, summarize):
    """One row per week ahead that summarizes the metric columns of rows = [[name, week_ahead, *metrics], ...]."""
    week_aheads = list(dict.fromkeys(row[1] for row in rows))
    summaries = []
    for week_ahead in week_aheads:
        vals = [row[2:] for row in rows if row[1] == week_ahead]
        summaries.append([label, week_ahead] + [summarize(col) for col in zip(*vals)])
    return summaries


def region_rows(metrics):
    return [[region, week_ahead] + [metric[name] for name in METRIC_NAMES]
            for region in metrics for week_ahead, metric in metrics[region].items()]


def save_metrics2csv(metrics, path):
    """Per-region metrics of one run, with the median and mean over regions on top."""
    rows = region_rows(metrics)
    header = ['region', 'week ahead'] + METRIC_NAMES
    write_csv(path, [header] + summary_rows(rows, 'median', np.median) + summary_rows(rows, 'mean', np.mean) + rows)


def aggregate_seeds(exp, clip_idx_list=None, sort_res=False):
    """Region-averaged metrics of every seed in exp.seeds, with the mean over seeds on top."""
    seed_rows = []
    for seed in exp.seeds:
        if not paths.results(exp.exp_id, seed).exists():
            logger.warning('Seed %s does not exist!', seed)
            continue
        metrics = eval_one_exp(exp, seed, clip_idx_list=clip_idx_list, sort_res=sort_res)
        seed_rows += summary_rows(region_rows(metrics), f'{seed}', np.mean)
    header = ['seed', 'week ahead'] + METRIC_NAMES
    output_path = paths.csv_dir() / f'{exp.exp_id}_multi_seeds.csv'
    write_csv(output_path, [header] + summary_rows(seed_rows, 'mean', np.mean) + seed_rows)
    return output_path
