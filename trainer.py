import torch
import torch.nn.functional as F
from tqdm import tqdm
import logging
import json
import os
import sys
import argparse
from utils.data_loader import get_dataloader
from models.hete_ml import FairMetaHIN
from utils.losses import get_accuracy_loss, get_user_fairness_loss, get_item_fairness_loss, get_path_fairness_loss
from utils.mgda import MGDAOptimizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

def train(config, data_dir, epochs=5, model_type='pareto'):
    device = torch.device("cuda" if config.get('use_cuda', False) and torch.cuda.is_available() else "cpu")
    
    # Initialize Model
    if model_type == 'pareto':
        model = FairMetaHIN(config).to(device)
    else:
        sys.path.append('MetaHIN-master/code')
        from HeteML_new import HML
        model = HML(config, 'hml').to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    # Dataloader
    dataloader = get_dataloader(data_dir, state='meta_training', batch_size=config.get('batch_size', 16))
    
    logging.info(f"Starting training for {model_type} model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            if model_type == 'pareto':
                losses_acc, losses_user, losses_item = [], [], []
                mp_atts = []
                
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
                    
                    query_y_pred, mp_att = model(supp_x, supp_y, supp_mp, query_x, query_y, query_mp)
                    
                    l_acc = get_accuracy_loss(query_y_pred, query_y)
                    losses_acc.append(l_acc)
                    losses_user.append((gender, l_acc.detach()))
                    
                    is_popular = (torch.rand(query_y.shape[0]) > 0.8).to(device) 
                    l_item = get_item_fairness_loss(query_y_pred, query_y, is_popular)
                    losses_item.append(l_item)
                    mp_atts.append(mp_att)
                
                if not losses_acc:
                    continue
                    
                L_acc = torch.stack(losses_acc).mean()
                L_item = torch.stack(losses_item).mean()
                L_path = get_path_fairness_loss(mp_atts)
                
                male_accs = [losses_acc[i] for i, m in enumerate(losses_user) if m[0] == 0]
                female_accs = [losses_acc[i] for i, m in enumerate(losses_user) if m[0] == 1]
                avg_male_g = torch.stack(male_accs).mean() if male_accs else torch.tensor(0.0).to(device)
                avg_female_g = torch.stack(female_accs).mean() if female_accs else torch.tensor(0.0).to(device)
                L_user = torch.abs(avg_male_g - avg_female_g)
    
                losses = [L_acc, L_user, L_item, L_path]
                active_losses = [l for l in losses if l.requires_grad]
                if len(active_losses) == 0:
                    continue
                    
                grads = []
                valid_losses = []
                for l in active_losses:
                    optimizer.zero_grad()
                    l.backward(retain_graph=True)
                    
                    g = []
                    for p in model.parameters():
                        if p.grad is not None:
                            g.append(p.grad.view(-1))
                    if g:
                        grads.append(torch.cat(g))
                        valid_losses.append(l)
                        
                if len(grads) > 1:
                    weights = MGDAOptimizer.get_optimal_weights(grads)
                    final_loss = sum(w * l for w, l in zip(weights, valid_losses))
                else:
                    final_loss = valid_losses[0]
    
                optimizer.zero_grad()
                final_loss.backward()
                optimizer.step()
                total_loss += final_loss.item()
                num_batches += 1
                
            else:
                task_losses = []
                for task in batch:
                    supp_x = task['supp_x'].to(device)
                    supp_y = task['supp_y'].to(device)
                    query_x = task['query_x'].to(device)
                    query_y = task['query_y'].to(device)
                    
                    if supp_x.shape[0] == 0 or query_x.shape[0] == 0:
                        continue
                        
                    supp_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['supp_mp'].items()}
                    query_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['query_mp'].items()}
                    
                    loss, _, _, _ = model.mp_update(supp_x, supp_y, supp_mp, query_x, query_y, query_mp)
                    task_losses.append(loss)
                
                if not task_losses:
                    continue
                    
                final_loss = torch.stack(task_losses).mean()
                final_loss.backward()
                optimizer.step()
                total_loss += final_loss.item()
            
        avg_loss = total_loss / max(num_batches, 1)
        logging.info(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.6f} ({num_batches}/{len(dataloader)} batches processed)")
        
        # Save model weights
        os.makedirs("models", exist_ok=True)
        model_save_path = os.path.join("models", f"{config.get('dataset', 'movielens')}_{model_type}_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), model_save_path)
        logging.info(f"Model weights saved to {model_save_path}")
        
        latest_path = os.path.join("models", f"{config.get('dataset', 'movielens')}_{model_type}_latest.pth")
        torch.save(model.state_dict(), latest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="pareto", choices=["pareto", "original"])
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        config = json.load(f)
        
    train(config, data_dir=config.get('data_dir', 'data/movielens'), epochs=config.get('epochs', 5), model_type=args.model)
