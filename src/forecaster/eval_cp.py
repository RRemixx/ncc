import argparse
from pathlib import Path

import numpy as np

from forecaster.utils import pickle_load, WriteRows2CSV
from forecaster.metrics import eval_metrics
from forecaster.e2ecp import load_all_params, BASE_PRED_PATH, RESULTS_PATH

METRIC_NAMES = ['rmse', 'nrmse', 'mape', 'cs_score', 'norm_crps_val', 'mean_wis', 'mono', 'cov_50', 'cov_95', 'cov_90', 'pi_length_90', 'dvs', 'qice']
CSV_PATH = RESULTS_PATH / 'csvs'
DATA_PATH = Path('../../data')


def prepare_data(saved_pred_file, alphas):
    """Converts saved NCC results to preds[region][week_ahead] = (y_trues, y_preds, lowers, uppers).
    Negative q-hats are set to zero."""
    predictions = pickle_load(saved_pred_file)
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


def load_clip_idx(e2ecp_params):
    """Evaluation indexes of the experiment's dataset (data/<dataset>_clip_idx.pkl), or None if there are none."""
    base_pred_file = BASE_PRED_PATH / f'saved_pred_{e2ecp_params["data_file_id"]}.pickle'
    dataset = pickle_load(base_pred_file)['params']['dataset']
    clip_idx_file_path = DATA_PATH / f'{dataset}_clip_idx.pkl'
    if not clip_idx_file_path.exists():
        print(f'clip index file not found for dataset {dataset}, evaluating all test steps')
        return None
    return pickle_load(clip_idx_file_path)


def eval_one_exp(e2ecp_params, exp_name, clip_idx_list=None, sort_res=False):
    """Returns metrics[region][week_ahead], in the order of METRIC_NAMES."""
    preds = prepare_data(RESULTS_PATH / f'{exp_name}.pkl', e2ecp_params['alphas'])
    if sort_res:
        preds = sort_preds(preds)
    if clip_idx_list is not None:
        preds = clip_ts_with_index_list(preds, clip_idx_list)
    metrics = {}
    for region in preds:
        metrics[region] = {}
        for week_ahead, (y_trues, y_preds, lowers, uppers) in preds[region].items():
            metrics[region][week_ahead] = eval_metrics(y_preds, y_trues, lowers, uppers, verbose=False)
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
    return [[region, week_ahead] + list(metric) for region in metrics for week_ahead, metric in metrics[region].items()]


def save_metrics2csv(metrics, path):
    """Per-region metrics of one run, with the median and mean over regions on top."""
    rows = region_rows(metrics)
    header = ['region', 'week ahead'] + METRIC_NAMES
    WriteRows2CSV([header] + summary_rows(rows, 'median', np.median) + summary_rows(rows, 'mean', np.mean) + rows, path)


def aggregrate_results_across_exps(exp_id, e2ecp_params, clip_idx_list=None, sort_res=False):
    """Region-averaged metrics of every seed in e2ecp_params['seeds'], with the mean over seeds on top."""
    seed_rows = []
    for seed in e2ecp_params['seeds']:
        exp_name = f'{exp_id}_{seed}'
        if not (RESULTS_PATH / f'{exp_name}.pkl').exists():
            print(f'Seed {seed} does not exist!')
            continue
        metrics = eval_one_exp(e2ecp_params, exp_name, clip_idx_list=clip_idx_list, sort_res=sort_res)
        seed_rows += summary_rows(region_rows(metrics), f'{seed}', np.mean)
    header = ['seed', 'week ahead'] + METRIC_NAMES
    output_path = CSV_PATH / f'{exp_id}_multi_seeds.csv'
    WriteRows2CSV([header] + summary_rows(seed_rows, 'mean', np.mean) + seed_rows, output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Experiment id, i.e. setup/cp_exp_params/<id>.yaml')
    parser.add_argument('--seed', '-s', type=int, help='Seed to evaluate. Defaults to the seed in the experiment file.')
    parser.add_argument('--multi_seeds', '-u', action='store_true', help='Evaluate all seeds in the experiment file')
    parser.add_argument('--clip', '-c', action='store_true', help='Only evaluate the steps in data/<dataset>_clip_idx.pkl')
    parser.add_argument('--sort', '-o', action='store_true', help='Sort interval widths so they shrink as alpha grows')
    args = parser.parse_args()

    e2ecp_params = load_all_params(args.input)
    clip_idx_list = load_clip_idx(e2ecp_params) if args.clip else None
    if args.multi_seeds:
        output_path = aggregrate_results_across_exps(args.input, e2ecp_params, clip_idx_list, sort_res=args.sort)
    else:
        seed = args.seed if args.seed is not None else e2ecp_params['seed']
        exp_name = f'{args.input}_{seed}'
        metrics = eval_one_exp(e2ecp_params, exp_name, clip_idx_list=clip_idx_list, sort_res=args.sort)
        output_path = CSV_PATH / f'{exp_name}.csv'
        save_metrics2csv(metrics, output_path)
    print(f'Metrics saved to {output_path}')


if __name__ == '__main__':
    main()
