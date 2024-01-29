import argparse
import json
import os
from itertools import product
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from train_utils import DepthModelHandler, run_test
from data_utils import DepthDataHandler


def main(config_file):
    """
    Starts hyperparameter search and training-evaluation
    :param config_file: str, path to config file
    """
    with open(config_file) as json_file:
        config = json.load(json_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['model']['cuda_visible_devices']
    hyper_parameters = config['hyper_parameters']
    param_values = [v for v in hyper_parameters.values()]
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = f'{config["model"]["results_dir"]}/{timestamp}'
    max_ssim = 0
    best_exp = 0
    for i, params in enumerate(product(*param_values)):
        config['hyper_parameters'] = {param: value for param, value in zip(hyper_parameters.keys(), params)}
        config['model']['results_dir'] = f'{results_dir}/exp_{i}'
        os.makedirs(config['model']['results_dir'])
        with open(f'{config["model"]["results_dir"]}/config.json', 'w') as fp:
            json.dump(config, fp)
        # Tensorboard object
        summary_writer = SummaryWriter(f'{results_dir}/exp_{i}/{str(config["hyper_parameters"])}')
        # Get data loaders
        data_handler = DepthDataHandler(config['data'], config['hyper_parameters']['batch_size'])
        train_loader, val_loader, test_loader = data_handler.get_data()
        # Run training-testing
        model_handler = DepthModelHandler(config, train_loader, val_loader, summary_writer)
        val_metrics, ssim = model_handler.run()
        # Results
        val_metrics = {f'hparam/{metric}': list(value.values())[-1] for metric, value in val_metrics.items()}
        summary_writer.add_hparams(config["hyper_parameters"], val_metrics)
        summary_writer.close()
        if ssim > max_ssim:
            max_ssim = ssim
            best_model = model_handler.model

    # Load the best model of the best experiment
    best_sd = torch.load(f'{results_dir}/exp_{best_exp}/model/best_model.pth.tar')
    best_model.load_state_dict(best_sd['model_weights'])
    # Plot best experiment results on test set
    run_test(best_model, test_loader, f'{results_dir}/exp_{best_exp}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='Json file with parameters for experiments')

    args = parser.parse_args()
    main(args.config)
