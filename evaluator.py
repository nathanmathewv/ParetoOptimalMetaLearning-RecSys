import argparse
import json
import os
import torch
import numpy as np
import math
import sys

from utils.data_loader import get_dataloader
from models.hete_ml import FairMetaHIN
from utils.losses import get_path_fairness_loss

def dcg_at_k(scores):
    if len(scores) == 0:
        return 0.0
    return scores[0] + sum(sc / math.log(ind+1, 2) for sc, ind in zip(scores[1:], range(2, len(scores) + 1)))

def ndcg_at_k(real_scores, predicted_scores, k=5):
    if len(real_scores) == 0:
        return 0.0
    sorted_idx = np.argsort(predicted_scores)[::-1][:k]
    r_s_at_k = real_scores[sorted_idx]
    
    idcg = dcg_at_k(sorted(real_scores, reverse=True)[:k])
    if idcg == 0:
        return 0.0
    return dcg_at_k(r_s_at_k) / idcg

def evaluate_model(model_type='pareto', config_path='config.json'):
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = torch.device("cuda" if config.get('use_cuda', False) and torch.cuda.is_available() else "cpu")
    
    data_dir = config.get('data_dir', 'data/movielens')
    
    states = ['warm_up', 'user_cold_testing', 'item_cold_testing', 'user_and_item_cold_testing']
    
    if model_type == 'pareto':
        model = FairMetaHIN(config).to(device)
        model_path = os.path.join('models', f"{config['dataset']}_pareto_latest.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded Pareto model from {model_path}")
        else:
            print(f"Warning: {model_path} not found. Evaluating with random weights.")
    elif model_type == 'original':
        sys.path.insert(0, os.path.join('MetaHIN-master', 'code'))
        if config['dataset'] == 'movielens':
            from Config import config_ml as orig_config
        elif config['dataset'] == 'yelp':
            from Config import config_yelp as orig_config
        else:
            from Config import config_db as orig_config
            
        from HeteML_new import HML
        model = HML(orig_config, 'mp_update')
        model.to(device)
            
        model_path = os.path.join('MetaHIN-master', 'res', config['dataset'], 'hml.pkl')
        if os.path.exists(model_path):
            trained_state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(trained_state_dict)
            print(f"Loaded Original model from {model_path}")
        else:
            import sys
            sys.path.append('MetaHIN-master/code')
            from HeteML_new import HML
            model = HML(config, 'hml').to(device)
            
            # Load from our new trained weights
            model_path = os.path.join("models", f"{config.get('dataset', 'movielens')}_original_latest.pth")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, map_location=device))
            else:
                logging.warning(f"Could not find weights at {model_path}. You may need to train the original model first.")
            model.eval()
            print(f"Loaded Original model from {model_path}")
    else:
        raise ValueError("Invalid model type")

    if model_type == 'pareto':
        model.eval()
    
    all_results = {}
    
    for state in states:
        print(f"Evaluating state: {state}")
        try:
            dataloader = get_dataloader(data_dir, state=state, batch_size=config.get('batch_size', 16), shuffle=False)
        except FileNotFoundError:
            print(f"State {state} not found for dataset {config['dataset']}, skipping.")
            continue
            
        mae_list, rmse_list, ndcg_list = [], [], []
        user_fairness_list, path_fairness_list = [], []
        
        for batch in dataloader:
            if model_type == 'pareto':
                batch_mp_atts = []
                for task in batch:
                    u_id = task['u_id']
                    gender = task['gender']
                    
                    supp_x = task['supp_x'].to(device)
                    supp_y = task['supp_y'].to(device)
                    query_x = task['query_x'].to(device)
                    query_y = task['query_y'].to(device)
                    
                    if supp_x.shape[0] == 0 or query_x.shape[0] == 0:
                        continue
                        
                    supp_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['supp_mp'].items()}
                    query_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['query_mp'].items()}
                    
                    with torch.no_grad():
                        query_y_pred, mp_att = model(supp_x, supp_y, supp_mp, query_x, query_y, query_mp)
                        
                        y_pred_np = query_y_pred.cpu().numpy()
                        y_true_np = query_y.cpu().numpy()
                        
                        mae = np.mean(np.abs(y_pred_np - y_true_np))
                        rmse = np.sqrt(np.mean((y_pred_np - y_true_np)**2))
                        ndcg = ndcg_at_k(y_true_np, y_pred_np, k=min(5, len(y_true_np)))
                            
                        mae_list.append(mae)
                        rmse_list.append(rmse)
                        ndcg_list.append(ndcg)
                        
                        batch_mp_atts.append(mp_att)
                        user_fairness_list.append({'gender': gender, 'mse': rmse**2})
                        
                if batch_mp_atts:
                    path_variance = get_path_fairness_loss(batch_mp_atts).item()
                    path_fairness_list.append(path_variance)
            else:
                # Evaluate Original
                for task in batch:
                    supp_x = task['supp_x'].to(device)
                    supp_y = task['supp_y'].to(device)
                    query_x = task['query_x'].to(device)
                    query_y = task['query_y'].to(device)
                    
                    if supp_x.shape[0] == 0 or query_x.shape[0] == 0:
                        continue
                        
                    supp_mp_list = [task['supp_mp'][mp_name] for mp_name in orig_config['mp']]
                    query_mp_list = [task['query_mp'][mp_name] for mp_name in orig_config['mp']]
                    
                    try:
                        # model.evaluation returns mae, rmse, ndcg_5 for a single task
                        _mae, _rmse, _ndcg_5 = model.evaluation(
                            supp_x, supp_y, supp_mp_list,
                            query_x, query_y, query_mp_list, device
                        )
                        mae_list.append(_mae)
                        rmse_list.append(_rmse)
                        ndcg_list.append(_ndcg_5)
                    except Exception as e:
                        pass

        if model_type == 'pareto' and user_fairness_list:
            male_mses = [item['mse'] for item in user_fairness_list if item['gender'] == 0]
            female_mses = [item['mse'] for item in user_fairness_list if item['gender'] == 1]
            avg_male = np.mean(male_mses) if male_mses else 0.0
            avg_female = np.mean(female_mses) if female_mses else 0.0
            user_fairness = abs(avg_male - avg_female)
        else:
            user_fairness = 0.0
            
        avg_path_fairness = np.mean(path_fairness_list) if path_fairness_list else 0.0
        
        all_results[state] = {
            'MAE': float(np.mean(mae_list)) if mae_list else 0.0,
            'RMSE': float(np.mean(rmse_list)) if rmse_list else 0.0,
            'NDCG@5': float(np.mean(ndcg_list)) if ndcg_list else 0.0,
            'User Fairness (MSE Diff)': float(user_fairness),
            'Path Exposure Variance': float(avg_path_fairness)
        }
        
    print(f"\n=== Results for {model_type} model ===")
    for state, res in all_results.items():
        print(f"State: {state}")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}")
            
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MetaHIN Models")
    parser.add_argument('--model', type=str, default='pareto', choices=['pareto', 'original'], help='Which model to evaluate')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    args = parser.parse_args()
    
    results = evaluate_model(args.model, args.config)
    
    out_file = f'results_{args.model}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved results to {out_file}")
