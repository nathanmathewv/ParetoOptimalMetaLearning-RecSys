import argparse
import json
import os
import torch
import numpy as np
import math
import sys
import logging
from collections import defaultdict
from tqdm import tqdm

LOG_EVERY_N_BATCHES = 50  # log running averages to file every N batches

from utils.data_loader import get_dataloader
from models.hete_ml import FairMetaHIN
from utils.losses import get_path_fairness_loss

# ─── Weight path helpers ──────────────────────────────────────────────────────

PARETO_WEIGHTS_DIR   = "models"
ORIGINAL_WEIGHTS_DIR = os.path.join("MetaHIN-master", "original_model_weights")

def get_weight_path(model_type, dataset, epoch):
    base_dir = PARETO_WEIGHTS_DIR if model_type == 'pareto' else ORIGINAL_WEIGHTS_DIR
    if str(epoch).lower() == 'latest':
        fname = f"{dataset}_{model_type}_latest.pth"
    else:
        fname = f"{dataset}_{model_type}_epoch_{epoch}.pth"
    return os.path.join(base_dir, fname)

# ─── Metrics ──────────────────────────────────────────────────────────────────

def dcg_at_k(scores):
    if len(scores) == 0:
        return 0.0
    return scores[0] + sum(sc / math.log(ind + 1, 2)
                           for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

def ndcg_at_k(real_scores, predicted_scores, k=5):
    if len(real_scores) < 2:
        return 0.0
    top_k_idx = np.argsort(predicted_scores)[::-1][:k]
    dcg  = dcg_at_k(list(real_scores[top_k_idx]))
    ideal = sorted(real_scores, reverse=True)[:k]
    idcg = dcg_at_k(ideal)
    return (dcg / idcg) if idcg > 0 else 0.0

# ─── Core evaluation function ────────────────────────────────────────────────

def evaluate_model(model_type, config_path, checkpoint_path, logger):
    with open(config_path, 'r') as f:
        config = json.load(f)

    device = torch.device("cuda" if config.get('use_cuda', False) and torch.cuda.is_available() else "cpu")
    data_dir = config.get('data_dir', 'data/movielens')

    # ── Build model ──
    if model_type == 'pareto':
        model = FairMetaHIN(config).to(device)
    elif model_type == 'original':
        sys.path.append('MetaHIN-master/code')
        from HeteML_new import HML
        model = HML(config, 'hml').to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # ── Load weights ──
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        logger.info(f"Loaded {model_type} weights from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint NOT FOUND: {checkpoint_path} — evaluating with random weights!")

    model.eval()

    states = ['warm_up', 'user_cold_testing', 'item_cold_testing', 'user_and_item_cold_testing']
    all_results = {}

    for state in states:
        logger.info(f"Evaluating: {state}")
        try:
            dataloader = get_dataloader(data_dir, state=state,
                                        batch_size=config.get('batch_size', 16), shuffle=False)
        except FileNotFoundError:
            logger.warning(f"  data for '{state}' not found, skipping.")
            continue

        mae_list, rmse_list, ndcg_list = [], [], []
        gender_mse = defaultdict(list)
        path_fairness_list = []

        total_batches = len(dataloader)
        pbar = tqdm(dataloader, desc=f"  {state}", unit="batch",
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for batch_idx, batch in enumerate(pbar, 1):
            for task in batch:
                supp_x  = task['supp_x'].to(device)
                supp_y  = task['supp_y'].to(device)
                query_x = task['query_x'].to(device)
                query_y = task['query_y'].to(device)

                if supp_x.shape[0] == 0 or query_x.shape[0] == 0:
                    continue

                supp_mp  = {k: [v_t.to(device) for v_t in v] for k, v in task['supp_mp'].items()}
                query_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['query_mp'].items()}

                try:
                    if model_type == 'pareto':
                        query_y_pred, mp_att = model(supp_x, supp_y, supp_mp,
                                                     query_x, query_y, query_mp)
                        y_pred = query_y_pred.detach().cpu().numpy()
                        y_true = query_y.cpu().numpy()
                        path_fairness_list.append(mp_att.detach())

                        mae  = float(np.mean(np.abs(y_pred - y_true)))
                        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                        ndcg = float(ndcg_at_k(y_true, y_pred, k=min(5, len(y_true))))
                        mae_list.append(mae)
                        rmse_list.append(rmse)
                        ndcg_list.append(ndcg)
                        gender_mse[task['gender']].append(rmse ** 2)

                    else:
                        _loss, _mae, _rmse, _ndcg = model.mp_update(
                            supp_x, supp_y, supp_mp, query_x, query_y, query_mp
                        )
                        mae_list.append(float(_mae))
                        rmse_list.append(float(_rmse))
                        ndcg_list.append(float(_ndcg))

                except Exception as e:
                    logger.error(f"  task error: {e}")
                    continue

            # Update tqdm postfix with running averages
            if mae_list:
                avg_mae  = np.mean(mae_list)
                avg_rmse = np.mean(rmse_list)
                avg_ndcg = np.mean(ndcg_list)
                pbar.set_postfix(
                    MAE=f"{avg_mae:.4f}",
                    RMSE=f"{avg_rmse:.4f}",
                    NDCG5=f"{avg_ndcg:.4f}",
                    n=len(mae_list)
                )
                # Periodic log to file
                if batch_idx % LOG_EVERY_N_BATCHES == 0 or batch_idx == total_batches:
                    logger.info(
                        f"  [{state}] batch {batch_idx}/{total_batches}  "
                        f"n={len(mae_list)}  MAE={avg_mae:.4f}  "
                        f"RMSE={avg_rmse:.4f}  NDCG@5={avg_ndcg:.4f}"
                    )

        pbar.close()

        # ── Aggregate ──
        result = {
            'MAE':     float(np.mean(mae_list))  if mae_list  else None,
            'RMSE':    float(np.mean(rmse_list)) if rmse_list else None,
            'NDCG@5':  float(np.mean(ndcg_list)) if ndcg_list else None,
            'n_tasks': len(mae_list),
        }

        if model_type == 'pareto':
            male_mse   = gender_mse.get(0, [])
            female_mse = gender_mse.get(1, [])
            result['User_Fairness_Gap'] = (
                float(abs(np.mean(male_mse) - np.mean(female_mse)))
                if male_mse and female_mse else None
            )
            result['Path_Exposure_Var'] = (
                float(get_path_fairness_loss(path_fairness_list).item())
                if path_fairness_list else None
            )

        all_results[state] = result

        # Log state summary (goes to both terminal and log file)
        summary_parts = [f"{k}={v:.4f}" for k, v in result.items()
                         if k != 'n_tasks' and v is not None]
        logger.info(f"  ✓ {state}  ({result['n_tasks']} tasks)  {' | '.join(summary_parts)}")

    return all_results

# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MetaHIN models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['pareto', 'original'])
    parser.add_argument('--epoch', type=str, default='latest',
                        help='1,2,3,4,5 or latest')
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset = config.get('dataset', 'movielens')
    ckpt = get_weight_path(args.model, dataset, args.epoch)

    # ── Logging: summary-only to file, tqdm handles terminal progress ──
    log_file = f"evaluation_{args.model}_{args.epoch}.log"
    logger = logging.getLogger(f"eval_{args.model}_{args.epoch}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(sh)

    logger.info(f"Model: {args.model} | Epoch: {args.epoch} | Checkpoint: {ckpt}")

    results = evaluate_model(args.model, args.config, ckpt, logger)

    # ── Final summary ──
    logger.info(f"{'═'*60}")
    logger.info(f"  FINAL — {args.model.upper()} epoch={args.epoch}")
    logger.info(f"{'═'*60}")
    for state, m in results.items():
        parts = [f"{k}={v:.4f}" for k, v in m.items() if k != 'n_tasks' and v is not None]
        logger.info(f"  {state:>30s}  ({m['n_tasks']:>4d} tasks)  {' | '.join(parts)}")
    logger.info(f"{'═'*60}")

    out_file = f"results_{args.model}_{args.epoch}.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Saved → {out_file}  |  Log → {log_file}")


if __name__ == "__main__":
    main()
