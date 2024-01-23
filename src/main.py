import argparse
import json
import os
import pandas as pd
from itertools import product
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from train_utils import DepthModelHandler
from data_utils import DepthDataHandler


def main(config_file: str):
    with open(config_file) as json_file:
        config = json.load(json_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = config['model']['cuda_visible_devices']
    hyper_parameters = config['hyper_parameters']
    param_values = [v for v in hyper_parameters.values()]
    results = []
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    results_dir = f'{config["model"]["results_dir"]}/{timestamp}'
    for i, params in enumerate(product(*param_values)):
        config['hyper_parameters'] = {param: value for param, value in zip(hyper_parameters.keys(), params)}
        config['model']['results_dir'] = f'{results_dir}/exp_{i}'
        os.makedirs(config['model']['results_dir'])
        with open(f'{config["model"]["results_dir"]}/config.json', 'w') as fp:
            json.dump(config, fp)
        # Tensorboard object
        summary_writer = SummaryWriter(f'{results_dir}/exp_{i}/{str(config["hyper_parameters"])}')
        # Get data loaders

        data_handler = DepthDataHandler(config['data_params'], config['hyper_parameters']['batch_size'])
        train_loader, test_loader = data_handler.get_data()
        # Run training-testing
        model_handler = get_model_handler(config, train_loader, test_loader,
                                          summary_writer, config['data']['window_size'])
        test_metrics = model_handler.run()
        # Results
        test_metrics = {f'hparam/{metric}': value[-1] for metric, value in test_metrics.items()}
        summary_writer.add_hparams(config["hyper_parameters"], test_metrics)
        config['hyper_parameters'].update(test_metrics)
        df = pd.DataFrame(config['hyper_parameters'], index=[i])
        df['model'] = [model_handler.model]
        df['test_loader'] = [test_loader]
        df['class_names'] = [model_handler.classes]
        results.append(df)
        summary_writer.close()

    results_df = pd.concat(results)
    best_exp = results_df[results_df['hparam/loss'] == results_df['hparam/loss'].min()]
    if config['model']['log_predictions']:
        model_handler.tensorboard_predictions(best_exp, results_dir)
    best_exp = best_exp.drop(['model', 'test_loader', 'class_names'], axis=1)
    results_df = results_df.drop(['model', 'test_loader', 'class_names'], axis=1)
    print('Best model:')
    print(best_exp.T)
    results_csv = os.path.dirname(config['model']['results_dir']) + '/results.csv'
    results_df.to_csv(results_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json',
                        help='Json file with parameters for experiments')

    args = parser.parse_args()
    main(args.config)
