import torch
from typing import Sequence, Dict


def load_checkpoint(checkpoint_path, model):
    print('Loading checkpoint from {}'.format(checkpoint_path))
    state_dicts = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dicts['model_state_dict'])
    print('\tEpoch {}'.format(state_dicts['epoch']))
    return


def determine_best_epoch_from_log(
        log_file: str, metrics: Sequence[str],
        metric_weights: Sequence[float] = None, highest: bool = True, delimiter: str = "\t",
    ) -> Dict[str, int or float]:
    """Reads the log file and determines which epoch is best based on the specified metrics and their weights.

    Args:
        log_file (str): The path to the log file.
        metrics (Sequence[str]): The names of the metrics to consider when ranking.
        metric_weights (Sequence[float], optional): The weights used when combining multiple metrics. If not provided, an equal weighting
            is used.
        highest (bool, optional): Whether or not the best epoch should be considered the highest scoring or the lowest scoring. Defaults to True.
        delimiter (str, optional): The delimiter used in the log file. Default is '\t'.

    Raises:
        ValueError: Metrics must be a subset of the header columns found in the log file.
        ValueError: Number of metrics must match number of metric weights (if provided).

    Returns:
        Dict[str, int | float]: A dictionary mapping the metrics of the best epoch to their corresponding values.
    """
    
    # Parse the log file
    with open(log_file, 'r') as f:
        log_lines = f.readlines()
    log_lines = [line for line in log_lines if line[0] != '#']
        
    # Make sure that the desired metrics are found in the log file via the header
    header_cols = [header for header in log_lines[0].lower().strip().split(delimiter) if header]
    epoch_col = header_cols.index('epoch')
    metric_cols = []
    for metric in metrics:
        if metric.lower() in header_cols:
            metric_cols.append(header_cols.index(metric.lower()))
        else:
            raise ValueError(f'Desired metric {metric} not found in the log file. Please choose one of {header_cols}.')
        
    # Provide equal weighting if no weights provided.
    if metric_weights == None:
        metric_weights = [1.] * len(metrics)
    else:
        if len(metrics) != len(metric_weights):
            raise ValueError(f'The number of metrics must equal the number of weights. {len(metrics)} != {len(metric_weights)}.')
        
    # Loop over epochs and determine best based on desired metrics and corresponding weights
    epoch_scores = []
    for epoch_line in log_lines[1:]:
        if epoch_line:
            cols = epoch_line.strip().split(delimiter)
            epoch = int(cols[epoch_col])
            if epoch == 0:
                continue
            metric_vals = [float(cols[metric_col]) for metric_col in metric_cols]
            epoch_score = sum([weight * val for val, weight in zip(metric_vals, metric_weights)])
            epoch_scores.append( (epoch, epoch_score) )
            
    # Determine best epoch based on the score
    sorted_epochs = sorted(epoch_scores, key=lambda x: x[1], reverse=highest)
    
    # Create results dictionary
    results = {'epoch': sorted_epochs[0][0]}
    for epoch_line in log_lines[1:]:
        if epoch_line:
            cols = epoch_line.strip().split(delimiter)
            epoch = int(cols[epoch_col])
            if results['epoch'] == epoch:
                for i, col in enumerate(header_cols):
                    if col != 'epoch':
                        results.update({'_'.join(col.split()): float(cols[i])})
    
    return results
