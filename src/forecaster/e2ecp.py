import argparse
import random
from pathlib import Path

import numpy as np
import torch

from forecaster.utils import pickle_load, pickle_save, load_yaml_params
from forecaster.e2ecp_utils import End2endCPTrainerForecaster, End2endCPDataset, init_model

BASE_PRED_PATH = Path('../../results/base_pred')
RESULTS_PATH = Path('../../results')


def e2ecp_train(e2ecp_params, exp_name, train):
    """Train NCC on the training weeks (train=True), or calibrate the test weeks with a trained model."""
    training_params = e2ecp_params['training_params']
    week_ahead_list = e2ecp_params['week_ahead_list']
    alphas = e2ecp_params['alphas']

    seed = e2ecp_params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load the base forecaster's predictions
    base_pred = pickle_load(BASE_PRED_PATH / f'saved_pred_{e2ecp_params["data_file_id"]}.pickle')['base_pred']
    e2ecp_dataset = End2endCPDataset(base_pred, week_ahead_list)
    data, scalar = e2ecp_dataset.e2ecp_seq_data(training_params)
    regions = e2ecp_dataset.get_regions()

    # prepare model
    model = init_model(e2ecp_params['model_params'], alphas, training_params['cov_window_size'])
    trainer = End2endCPTrainerForecaster(
        model, regions, week_ahead_list, alphas, scalar=scalar,
        cf_lr=training_params['cf_lr'],
        tta_reg_factor=training_params['tta_reg_factor'],
        tta_lr=training_params['tta_lr'],
    )
    model_path = Path(e2ecp_params['model_path'])
    if train:
        losses = trainer.stage_train(training_params, data)
        pickle_save(model_path / f'losses_{exp_name}.pkl', losses)
        torch.save(model.state_dict(), model_path / f'{exp_name}.pt')
    else:
        model.load_state_dict(torch.load(model_path / f'{exp_name}.pt', weights_only=True))
        all_predictions = trainer.forecast4eval(training_params, data)
        pickle_save(RESULTS_PATH / f'{exp_name}.pkl', all_predictions)


def load_all_params(input_file):
    """Default params from setup/e2ecp.yaml, overridden by setup/cp_exp_params/<input_file>.yaml."""
    e2ecp_params = load_yaml_params('../../setup/e2ecp.yaml')
    e2ecp_exp_params = load_yaml_params(f'../../setup/cp_exp_params/{input_file}.yaml')
    for key, value in e2ecp_exp_params.items():
        if key == 'training_params' or key == 'model_params':
            e2ecp_params[key].update(value)
        else:
            e2ecp_params[key] = value
    return e2ecp_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='Experiment id, i.e. setup/cp_exp_params/<id>.yaml')
    parser.add_argument('--train', '-t', action='store_true', help='Train the model. Without it, calibrate the test weeks.')
    parser.add_argument('--seed', '-s', type=int, help='Overrides the seed in the experiment file')
    args = parser.parse_args()
    e2ecp_params = load_all_params(args.input)
    if args.seed is not None:
        e2ecp_params['seed'] = args.seed
    e2ecp_train(e2ecp_params, f'{args.input}_{e2ecp_params["seed"]}', args.train)


if __name__ == '__main__':
    main()
