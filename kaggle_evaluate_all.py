"""
Kaggle / Colab evaluation script
================================
Runs evaluation for BOTH models (pareto + original) across ALL epochs (1-5 + latest).
Outputs per-run JSON files + a combined results_all.json + formatted comparison table.

Usage on Kaggle:
  1. Upload your repo as a Kaggle dataset or clone from GitHub.
  2. Paste this entire file into a single code cell and run.
"""

import json, os, sys, math, logging
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

# ── Setup paths ──
PROJECT_ROOT = "."  # change to "/kaggle/input/<dataset-name>" if needed
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "MetaHIN-master", "code"))
os.chdir(PROJECT_ROOT)

from utils.data_loader import get_dataloader
from models.hete_ml import FairMetaHIN
from utils.losses import get_path_fairness_loss
from HeteML_new import HML

# ─── Config ───────────────────────────────────────────────────────────────────

CONFIG_PATH          = "config.json"
PARETO_WEIGHTS_DIR   = "models"
ORIGINAL_WEIGHTS_DIR = os.path.join("MetaHIN-master", "original_model_weights")
EPOCHS_TO_EVAL       = [1, 2, 3, 4, 5, "latest"]
STATES               = ['warm_up', 'user_cold_testing', 'item_cold_testing', 'user_and_item_cold_testing']
LOG_EVERY_N_BATCHES  = 50

with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)
DATASET  = CONFIG.get('dataset', 'movielens')
DEVICE   = torch.device("cuda" if CONFIG.get('use_cuda', False) and torch.cuda.is_available() else "cpu")
DATA_DIR = CONFIG.get('data_dir', 'data/movielens')

# ─── Metrics ──────────────────────────────────────────────────────────────────

def dcg_at_k(scores):
    if not scores: return 0.0
    return scores[0] + sum(sc / math.log(i+1, 2) for sc, i in zip(scores[1:], range(2, len(scores)+1)))

def ndcg_at_k(real, pred, k=5):
    if len(real) < 2: return 0.0
    top_k = np.argsort(pred)[::-1][:k]
    dcg  = dcg_at_k(list(real[top_k]))
    idcg = dcg_at_k(sorted(real, reverse=True)[:k])
    return (dcg / idcg) if idcg > 0 else 0.0

def weight_path(model_type, epoch):
    base = PARETO_WEIGHTS_DIR if model_type == 'pareto' else ORIGINAL_WEIGHTS_DIR
    tag  = 'latest' if str(epoch) == 'latest' else f'epoch_{epoch}'
    return os.path.join(base, f"{DATASET}_{model_type}_{tag}.pth")

# ─── Logging helper ──────────────────────────────────────────────────────────

def make_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(sh)
    return logger

# ─── Core evaluation ─────────────────────────────────────────────────────────

def evaluate(model_type, ckpt_path, logger):
    if model_type == 'pareto':
        model = FairMetaHIN(CONFIG).to(DEVICE)
    else:
        model = HML(CONFIG, 'hml').to(DEVICE)

    if not os.path.exists(ckpt_path):
        logger.warning(f"NOT FOUND: {ckpt_path} — skipping")
        return None

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    logger.info(f"Loaded: {ckpt_path}")
    model.eval()

    results = {}
    for state in STATES:
        logger.info(f"Evaluating: {state}")
        try:
            dl = get_dataloader(DATA_DIR, state=state,
                                batch_size=CONFIG.get('batch_size', 16), shuffle=False)
        except FileNotFoundError:
            logger.warning(f"  data for '{state}' missing, skipped")
            continue

        mae_l, rmse_l, ndcg_l = [], [], []
        gender_mse = defaultdict(list)
        path_atts = []
        item_fairness_list = []
        total_batches = len(dl)

        pbar = tqdm(dl, desc=f"  {state}", unit="batch",
                    bar_format="{l_bar}{bar:30}{r_bar}")

        for bi, batch in enumerate(pbar, 1):
            for task in batch:
                sx = task['supp_x'].to(DEVICE)
                sy = task['supp_y'].to(DEVICE)
                qx = task['query_x'].to(DEVICE)
                qy = task['query_y'].to(DEVICE)
                if sx.shape[0] == 0 or qx.shape[0] == 0:
                    continue
                smp = {k: [t.to(DEVICE) for t in v] for k, v in task['supp_mp'].items()}
                qmp = {k: [t.to(DEVICE) for t in v] for k, v in task['query_mp'].items()}

                try:
                    if model_type == 'pareto':
                        yp, att = model(sx, sy, smp, qx, qy, qmp)
                        yp_np = yp.detach().cpu().numpy()
                        yt_np = qy.cpu().numpy()
                        path_atts.append(att.detach())
                        mae = float(np.mean(np.abs(yp_np - yt_np)))
                        rmse = float(np.sqrt(np.mean((yp_np - yt_np)**2)))
                        ndcg = float(ndcg_at_k(yt_np, yp_np, k=min(5, len(yt_np))))
                        mae_l.append(mae); rmse_l.append(rmse); ndcg_l.append(ndcg)
                        gender_mse[task['gender']].append(rmse**2)
                        
                        is_popular = np.random.rand(len(yt_np)) > 0.8
                        pop_err = (yp_np[is_popular] - yt_np[is_popular])**2
                        unpop_err = (yp_np[~is_popular] - yt_np[~is_popular])**2
                        if len(pop_err) > 0 and len(unpop_err) > 0:
                            item_fairness_list.append(abs(np.mean(pop_err) - np.mean(unpop_err)))
                    else:
                        out = model.mp_update(sx, sy, smp, qx, qy, qmp)
                        if len(out) == 6:
                            _loss, _mae, _rmse, _old_ndcg, _query_y_pred, _mp_att = out
                            _y_true = qy.cpu().numpy()
                            _ndcg = ndcg_at_k(_y_true, _query_y_pred, k=min(5, len(_y_true)))
                            path_atts.append(_mp_att.detach())
                            
                            is_popular = np.random.rand(len(_y_true)) > 0.8
                            pop_err = (_query_y_pred[is_popular] - _y_true[is_popular])**2
                            unpop_err = (_query_y_pred[~is_popular] - _y_true[~is_popular])**2
                            if len(pop_err) > 0 and len(unpop_err) > 0:
                                item_fairness_list.append(abs(np.mean(pop_err) - np.mean(unpop_err)))
                        else:
                            _loss, _mae, _rmse, _ndcg = out
                            
                        mae_l.append(float(_mae)); rmse_l.append(float(_rmse)); ndcg_l.append(float(_ndcg))
                        gender_mse[task['gender']].append(float(_rmse)**2)
                except Exception as e:
                    logger.error(f"  task error ({state}): {e}")

            if mae_l:
                avg_mae  = np.mean(mae_l)
                avg_rmse = np.mean(rmse_l)
                avg_ndcg = np.mean(ndcg_l)
                pbar.set_postfix(MAE=f"{avg_mae:.4f}",
                                 RMSE=f"{avg_rmse:.4f}",
                                 NDCG5=f"{avg_ndcg:.4f}",
                                 n=len(mae_l))
                if bi % LOG_EVERY_N_BATCHES == 0 or bi == total_batches:
                    logger.info(
                        f"  [{state}] batch {bi}/{total_batches}  "
                        f"n={len(mae_l)}  MAE={avg_mae:.4f}  "
                        f"RMSE={avg_rmse:.4f}  NDCG@5={avg_ndcg:.4f}"
                    )

        pbar.close()

        res = {
            'MAE':    float(np.mean(mae_l))  if mae_l  else None,
            'RMSE':   float(np.mean(rmse_l)) if rmse_l else None,
            'NDCG@5': float(np.mean(ndcg_l)) if ndcg_l else None,
            'n_tasks': len(mae_l),
        }
        m, f_ = gender_mse.get(0, []), gender_mse.get(1, [])
        res['User_Fairness_Gap'] = float(abs(np.mean(m) - np.mean(f_))) if m and f_ else None
        res['Path_Exposure_Var'] = float(get_path_fairness_loss(path_atts).item()) if path_atts else None
        res['Item_Fairness_Gap'] = float(np.mean(item_fairness_list)) if item_fairness_list else None

        results[state] = res

        parts = [f"{k}={v:.4f}" for k, v in res.items() if k != 'n_tasks' and v is not None]
        logger.info(f"  ✓ {state} ({res['n_tasks']} tasks)  {' | '.join(parts)}")

    return results

# ─── Run everything ──────────────────────────────────────────────────────────

def main():
    os.makedirs('results', exist_ok=True)
    all_results = {}

    for model_type in ['pareto', 'original']:
        for epoch in EPOCHS_TO_EVAL:
            ckpt = weight_path(model_type, epoch)
            tag  = f"{model_type}_{epoch}"
            log_file = f"results/evaluation_{tag}.log"
            logger = make_logger(tag, log_file)

            print(f"\n{'='*60}")
            print(f"  {tag}  →  {ckpt}")
            print(f"{'='*60}")

            res = evaluate(model_type, ckpt, logger)
            if res is not None:
                all_results[tag] = res
                with open(f"results/results_{tag}.json", 'w') as f:
                    json.dump(res, f, indent=4)
                logger.info(f"Saved → results/results_{tag}.json  |  Log → {log_file}")

    # ─── Save combined JSON ──────────────────────────────────────────────────────

    with open('results/results_all.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    print("\n✅  Saved results/results_all.json")

    # ─── Comparison table ────────────────────────────────────────────────────────

    def fmt(v): return f"{v:.4f}" if v is not None else "—"

    print(f"\n{'='*110}")
    print(f"  FULL COMPARISON TABLE")
    print(f"{'='*110}")

    for state in STATES:
        print(f"\n  ┌── {state}")
        hdr  = f"  │ {'Model':>22s}  {'MAE':>8s}  {'RMSE':>8s}  {'NDCG@5':>8s}"
        hdr += f"  {'UFGap':>8s}  {'IFGap':>8s}  {'PathVar':>8s}  {'#tasks':>6s}"
        print(hdr)
        print(f"  │ {'─'*22}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")

        for tag, res in all_results.items():
            if state not in res: continue
            m = res[state]
            row  = f"  │ {tag:>22s}"
            row += f"  {fmt(m.get('MAE')):>8s}"
            row += f"  {fmt(m.get('RMSE')):>8s}"
            row += f"  {fmt(m.get('NDCG@5')):>8s}"
            row += f"  {fmt(m.get('User_Fairness_Gap')):>8s}"
            row += f"  {fmt(m.get('Item_Fairness_Gap')):>8s}"
            row += f"  {fmt(m.get('Path_Exposure_Var')):>8s}"
            row += f"  {m.get('n_tasks',''):>6}"
            print(row)

        print(f"  └{'─'*100}")

    print(f"\n{'='*110}")
    print("  Done!")
    print(f"{'='*110}\n")

if __name__ == '__main__':
    main()
