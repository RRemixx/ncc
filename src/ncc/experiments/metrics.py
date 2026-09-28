"""Evaluation metrics of the paper. The metric bodies are kept as published, to protect the reported numbers."""
import numpy as np
import properscoring as ps
from sklearn.preprocessing import MinMaxScaler

EPS = 1e-6

# the metrics returned by eval_metrics, in the order of the CSV columns
METRIC_NAMES = ['rmse', 'nrmse', 'mape', 'cs_score', 'norm_crps_val', 'mean_wis', 'mono', 'cov_50', 'cov_95',
                'cov_90', 'pi_length_90', 'dvs', 'qice']


def rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    return np.sqrt(((predictions - targets) ** 2).mean())


def norm_rmse(predictions, targets):
    """
    Root Mean Squared Error
    Args:
        predictions (np.ndarray): Point Predictions of the model
        targets (np.ndarray): Point Targets of the model
    Returns:
        float: RMSE
    """
    try:
        scale = MinMaxScaler()
        targets = scale.fit_transform(targets[:,None])
        predictions = scale.transform(predictions[:,None])
    finally:
       return np.sqrt(((predictions - targets) ** 2).mean())


def mape(predictions, targets):
    """
    Mean Absolute Percentage Error
    Args:
        predictions (np.ndarray): Predictions of the model
        targets (np.ndarray): Targets of the model
    Returns:
        float: MAPE
    """
    # TODO(paper): writes NaN into targets in place, which affects the metrics computed after it
    targets[targets==0] = np.nan
    return np.nanmean(np.abs((predictions - targets) / targets)) * 100


def crps(mean, std, targets):
    """
    Quantile-based CRPS
    Args:
        mean (np.ndarray): Mean of the distribution (N)
        std (np.ndarray): Standard Deviation of the distribution (N)
        targets (np.ndarray): Targets of the model (N)
    Returns:
        float: CRPS
    """
    std_clip = np.clip(std, EPS, None)
    ans1 = ps.crps_gaussian(targets, mean, std_clip)
    ans2 = np.abs(targets - mean)
    ans = (~np.isclose(std, 0, atol=1e-6)) * ans1 + (np.isclose(std, 0, atol=1e-6)) * ans2
    return ans.mean()


def calculate_stds_helper(y_pred, lower):
    val_prob = {}
    alphas = sorted(list(lower.keys()))
    for i in range(len(alphas)):
        if i == len(alphas) - 1:
            alpha1 = alphas[i]
            itv1 = y_pred - lower[alpha1]
            val_prob[y_pred] = 1 - alpha1
        else:
            alpha1 = alphas[i]
            alpha2 = alphas[i + 1]
            itv1 = y_pred - lower[alpha1]
            itv2 = y_pred - lower[alpha2]
            val_prob[y_pred + 1/2 * (itv1 + itv2)] = (alpha2 - alpha1) / 2
            val_prob[y_pred - 1/2 * (itv1 + itv2)] = (alpha2 - alpha1) / 2
    variance = 0
    total_prob = 0
    for val, prob in val_prob.items():
        total_prob += prob
        variance += (val - y_pred) ** 2 * prob
    return variance ** 0.5


def calculate_stds(y_preds, lowers):
    stds = []
    # take one week from data
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        lower = {}
        for alpha in lowers:
            lower[alpha] = float(lowers[alpha][i])
        current_std = calculate_stds_helper(y_pred, lower)
        stds.append(current_std)
    return stds


def norm_crps(y_preds, y_trues, lowers):
    scale = MinMaxScaler()
    y_trues = scale.fit_transform(y_trues[:, None]).reshape(-1)
    y_preds = scale.transform(y_preds[:, None]).reshape(-1)
    for alpha in lowers:
        lowers[alpha] = scale.transform(np.array(lowers[alpha])[:, None]).reshape(-1)
    stds = calculate_stds(y_preds, lowers)
    crps_val = crps(y_preds, stds, y_trues)
    return crps_val


def interval_score(y_true, upper, lower, alpha):
    return (upper - lower) + 2 / alpha * (lower - y_true) * (y_true < lower) + 2 / alpha * (y_true - upper) * (upper < y_true)


def weighted_IS_helper(y_pred, y_true, lower):
    # alphas = sorted(list(lower.keys()), reverse=True)[:-1]
    alphas = sorted(list(lower.keys()), reverse=True)
    K = len(alphas)
    sum_term = 0
    for i in range(len(alphas)):
        alpha = alphas[i]
        upper = y_pred - lower[alpha] + y_pred
        sum_term += alphas[i] / 2 * interval_score(y_true, upper, lower[alpha], alpha)
    return 1 / (K + 1/2) * (alphas[0] / 2 * abs(y_true - y_pred) + sum_term)


def weighted_IS(y_preds, y_trues, lowers):
    weighted_IS_vals = []
    scale = MinMaxScaler()
    y_trues = scale.fit_transform(y_trues[:, None]).reshape(-1)
    y_preds = scale.transform(y_preds[:, None]).reshape(-1)
    for alpha in lowers:
        lowers[alpha] = scale.transform(np.array(lowers[alpha])[:, None]).reshape(-1)
    for i in range(len(y_preds)):
        y_pred = y_preds[i]
        y_true = y_trues[i]
        lower = {}
        for alpha in lowers:
            lower[alpha] = float(lowers[alpha][i])
        weighted_IS_vals.append(weighted_IS_helper(y_pred, y_true, lower))
    return weighted_IS_vals


def monotonicity(y_preds, lowers):
    """
    Measure monotinicity: for a list x of length l. counter=0. For i from 0 to l, if x[i] > any other element behind, counter++. counter/l
    """
    num_weeks = len(y_preds)
    alphas = list(lowers.keys())
    mono_metrics = []
    for i in range(num_weeks):
        y_pred = y_preds[i]
        q_hats = []
        counter = 0
        for alpha in alphas:
            q_hats.append(y_pred - lowers[alpha][i])
        # validate
        # prev_qhat = q_hats[0]
        # for qhat in q_hats:
        #     assert qhat >= 0
        #     assert qhat <= prev_qhat
        #     prev_qhat = qhat
        for k, q_hat in enumerate(q_hats):
            greater_than_right = True
            for j_ in range(len(q_hats)-k-1):
                j = j_ + k + 1
                if q_hat < q_hats[j]:
                    greater_than_right = False
                    break
            if greater_than_right:
                counter += 1
        mono_metrics.append(counter / len(alphas))
    # for item in mono_metrics:
    #     assert item == 1
    return np.mean(mono_metrics)


def validity_score(y_preds, lowers):
    """
    Measure the percentage of valid distributions: for each time step, if its prediction intervals are monotonic w.r.t the confidence levels, then this is a valid distribution.
    """
    num_weeks = len(y_preds)
    alphas = list(lowers.keys())
    num_valid_points = 0
    for i in range(num_weeks):
        y_pred = y_preds[i]
        q_hats = []
        for alpha in alphas:
            q_hats.append(y_pred - lowers[alpha][i])
        valid_point = 1
        for k in range(len(q_hats)-1):
            if q_hats[k] < q_hats[k+1]:
                valid_point = 0
                break
        num_valid_points = num_valid_points + valid_point
    return num_valid_points / num_weeks


def interval_length(y_preds, lowers, target_error_rate=0.1):
    target_lower = lowers[target_error_rate]
    half_PI_lengths = np.abs(y_preds - target_lower)
    return np.mean(half_PI_lengths)


def x_percentage_coverage(x, y_preds, y_trues, lowers, uppers):
    alpha = round(1-x, 4)
    in_range_count = 0
    for i in range(len(y_preds)):
        if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
            in_range_count += 1
    cov_rate = in_range_count / len(y_preds)
    return cov_rate


def QICE(y_preds, lowers, n_bins=5):
    def intvl_counts(data, n_intvls, intvl_lb, intvl_ub):
        # get intervals
        intvls = np.linspace(intvl_lb, intvl_ub, num=n_intvls+1)
        # count number
        counts = np.zeros(n_intvls)
        for i in range(n_intvls):
            lb = intvls[i]
            ub = intvls[i+1]
            for k in range(len(data)):
                if data[k] <= ub and data[k] >= lb:
                    counts[i] += 1
        return counts
    alphas = list(lowers.keys())
    exp_counts = intvl_counts(alphas, n_intvls=n_bins, intvl_lb=0, intvl_ub=1)
    exp_counts = np.flip(exp_counts)
    qice_scores = []
    for i in range(len(y_preds)):
        cur_qs = [np.abs(y_preds[i] - lowers[alpha][i]) for alpha in lowers]
        total_counts = len(cur_qs)
        cur_counts = intvl_counts(data=cur_qs, n_intvls=n_bins, intvl_lb=np.min(cur_qs), intvl_ub=np.max(cur_qs))
        qice_score = np.mean(np.abs(cur_counts/total_counts - exp_counts/total_counts)) 
        qice_scores.append(qice_score)
    return np.mean(qice_scores)


def eval_metrics(y_preds, y_trues, lowers, uppers):
    """Returns {metric name: value}, keyed by METRIC_NAMES in that order.

    Args:
        y_preds (list): []
        y_trues (list): []
        lowers (dict): alpha -> []
        uppers (dict): alpha -> []
    """
    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)

    # weighted IS
    weighted_IS_vals = weighted_IS(y_preds, y_trues, lowers.copy())
    mean_wis = np.mean(weighted_IS_vals)
    
    # crps
    norm_crps_val = norm_crps(y_preds, y_trues, lowers.copy())
    
    rmse_val = rmse(y_preds, y_trues)
    nrmse_val = norm_rmse(y_preds, y_trues)
    mape_val = mape(y_preds, y_trues)
    # cs score
    cs_score = 0
    for alpha in lowers:
        in_range_count = 0
        for i in range(len(y_preds)):
            if y_trues[i] > lowers[alpha][i] and y_trues[i] < uppers[alpha][i]:
                in_range_count += 1
        cov_rate = in_range_count / len(y_preds)
        cs_score += 1 / len(lowers) * abs(cov_rate - 1 + alpha)
    # monotonicity
    mono_score = monotonicity(y_preds, lowers)
    
    # validity
    dist_valid_score = validity_score(y_preds, lowers)
    
    # 90% PI half length
    pi_length_90 = interval_length(y_preds, lowers, target_error_rate=0.1)
    
    # 90% coverage
    cov_90 = x_percentage_coverage(0.90, y_preds, y_trues, lowers, uppers)
    
    # 50% coverage
    cov_50 = x_percentage_coverage(0.5, y_preds, y_trues, lowers, uppers)
    
    # 95% coverage
    cov_95 = x_percentage_coverage(0.95, y_preds, y_trues, lowers, uppers)
    
    # QICE
    qice_score = QICE(y_preds, lowers, n_bins=5)
    
    values = (rmse_val, nrmse_val, mape_val, cs_score, norm_crps_val, mean_wis, mono_score, cov_50, cov_95, cov_90,
              pi_length_90, dist_valid_score, qice_score)
    return dict(zip(METRIC_NAMES, values))
