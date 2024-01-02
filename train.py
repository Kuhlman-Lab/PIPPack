import time, os
import shutil, random
import torch
import hydra
import logging
from omegaconf import DictConfig
import lightning

# Library code
from utils.utils import count_parameters, create_subdirs
from utils.train_utils import determine_best_epoch_from_log, load_checkpoint
from model.optim import get_std_opt
from model.loss import MetricLogger


logger = logging.getLogger(__name__)


def train_loop(exp_cfg, model, optimizer, train_dataloader, valid_dataloader, metric_logger, device) -> None:
    # Sanity validation run
    model.eval()
    with torch.no_grad():
        for batch in valid_dataloader:
            # Determine how many recycles to do.
            if exp_cfg.n_recycle > 0:
                n_cyc = exp_cfg.n_recycle
            else:
                n_cyc = 0
            
            # Move batch to device.
            batch = batch.to(device)
            
            # Run through model and compute loss.
            output = model(batch, n_recycle=n_cyc)
            _ = model.compute_loss(output, batch, use_sc_bf_mask=True, _logger=metric_logger, _log_prefix="val")
        
    # Perform logging.
    metric_logger.log(epoch=0, precision=exp_cfg.logging_precision)
    
    for e in range(1, exp_cfg.epochs + 1):
        # Training epoch
        model.train()
        for batch in train_dataloader:
            # Determine how many recycles to do.
            if exp_cfg.n_recycle > 0:
                n_cyc = random.randint(0, exp_cfg.n_recycle)
            else:
                n_cyc = 0
                
            # Move batch to device.
            batch = batch.to(device)
            
            # Run through model and compute loss.
            output = model(batch, n_recycle=n_cyc)
            loss = model.compute_loss(output, batch, use_sc_bf_mask=exp_cfg.use_b_factor_mask, _logger=metric_logger)
            
            # Perform backprop and update.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        # Validation epoch
        if e % exp_cfg.validate_every_n_epochs == 0:
            model.eval()
            with torch.no_grad():
                for batch in valid_dataloader:
                    # Determine how many recycles to do.
                    if exp_cfg.n_recycle > 0:
                        n_cyc = exp_cfg.n_recycle
                    else:
                        n_cyc = 0
                    
                    # Move batch to device.
                    batch = batch.to(device)
                    
                    # Run through model and compute loss.
                    output = model(batch, n_recycle=n_cyc)
                    _ = model.compute_loss(output, batch, use_sc_bf_mask=True, _logger=metric_logger, _log_prefix="val")
        
        # Perform logging.
        metric_logger.log(epoch=e, precision=exp_cfg.logging_precision)
                         
        # Save the model checkpoint.
        checkpoint_filename = os.path.join(os.getcwd(), 'checkpoints', f'epoch_{e}.pt')
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_filename)

@hydra.main(version_base=None, config_path="./config", config_name="train")
def main(cfg: DictConfig) -> None:
    
    # Set up RNG and device
    seed = lightning.seed_everything(cfg.experiment.seed)
    logger.info(f"Using seed={seed} for RNG.")
    device = torch.device("cuda:0" if (torch.cuda.is_available() and not cfg.experiment.force_cpu) else "cpu")

    # Load dataset and get dataloaders
    dm: lightning.LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    dm.prepare_data()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    valid_loader = dm.val_dataloader()
    logger.info(f'Training: {len(dm.data_train)}, Validation: {len(dm.data_val)}')

    # Instantiate the model from config.
    model: torch.nn.Module = hydra.utils.instantiate(cfg.model).to(device)
    logger.info(f'Number of parameters: {count_parameters(model):,}')
    if "weights_path" in cfg.experiment:
        logger.info(f"Loading weights from {cfg.experiment.weights_path}")
        # Find the checkpoint to load into model
        checkpoint = os.path.join(cfg.experiment.weights_path, f'{cfg.experiment.model_name}_ckpt.pt')
        load_checkpoint(checkpoint, model)

    # Build optimizer
    if "finetune" in cfg.experiment.name:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    else:
        optimizer = get_std_opt(model.parameters(), cfg.model.hidden_dim)

    # Build directories for experiment
    create_subdirs(os.getcwd(), ['checkpoints', 'results'])

    # Create metric logger
    log_file = os.path.join(os.getcwd(), "results", "train_log.csv")
    metrics = [mode + " " + metric for metric in model.metric_names for mode in ["train", "val"]]
    metric_logger = MetricLogger(log_file, metrics).to(device)

    # Train
    start_train = time.time()
    train_loop(cfg.experiment, model, optimizer, train_loader, valid_loader, metric_logger, device)
    train_elapsed = time.time() - start_train
    logger.info(f"Total training time: {train_elapsed:.2f} sec")
        
    # Determine best model via early stopping on validation
    best_results = determine_best_epoch_from_log(log_file, [model.monitor_metric], highest=False, delimiter=",")
    best_ckpt = os.path.join(os.getcwd(), 'checkpoints', f"epoch_{best_results['epoch']}.pt")
    best_ckpt_copy = os.path.join(os.getcwd(), 'results', f"best_ckpt_epoch_{best_results['epoch']}.pt")
    shutil.copy(best_ckpt, best_ckpt_copy)
    
    # Write results
    with open(os.path.join(os.getcwd(), 'results', 'train_results.txt'), 'w') as f:
        f.write(f"Best Epoch: {best_results['epoch']}")
        prev_metric = ""
        for metric in best_results:
            if metric != "epoch":
                cur_metric = " ".join(metric.split("_")[1:-1])
                mode = metric.split("_")[0]
                if prev_metric != cur_metric:
                    f.write(f"\n{cur_metric}:")
                    prev_metric = cur_metric
                f.write(f"\n\t{mode}: {best_results[metric]}")


if __name__ == "__main__":
    main()
