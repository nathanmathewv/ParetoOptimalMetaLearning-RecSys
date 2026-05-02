import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.data_loader import get_dataloader
from models.hete_ml import FairMetaHIN
from utils.losses import get_accuracy_loss, get_user_fairness_loss, get_item_fairness_loss, get_path_fairness_loss
from utils.mgda import MGDAOptimizer

def train(config, data_dir, epochs=5):
    device = torch.device("cuda" if config.get('use_cuda', False) else "cpu")
    
    # Initialize Model
    model = FairMetaHIN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
    
    # Dataloader
    dataloader = get_dataloader(data_dir, state='meta_training', batch_size=config.get('batch_size', 16))
    
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            
            losses_acc, losses_user, losses_item = [], [], []
            mp_atts = []
            
            # For item popularity (dummy thresholding for now based on item feature index)
            # A real implementation would use item frequencies
            
            # We process task by task since the meta-learning update is per-user
            for task in batch:
                u_id = task['u_id']
                gender = task['gender']
                
                # Move to device
                supp_x = task['supp_x'].to(device)
                supp_y = task['supp_y'].to(device)
                query_x = task['query_x'].to(device)
                query_y = task['query_y'].to(device)
                
                if supp_x.shape[0] == 0 or query_x.shape[0] == 0:
                    continue
                    
                supp_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['supp_mp'].items()}
                query_mp = {k: [v_t.to(device) for v_t in v] for k, v in task['query_mp'].items()}
                
                # Forward pass
                query_y_pred, mp_att = model(supp_x, supp_y, supp_mp, query_x, query_y, query_mp)
                
                # Compute individual task losses
                l_acc = get_accuracy_loss(query_y_pred, query_y)
                
                # For User Fairness, we use the gender of the current user. Since this is a per-user task,
                # the variance is computed over the batch later, or we can compute MSE per gender group 
                # over the batch.
                # Let's collect them first
                losses_acc.append(l_acc)
                losses_user.append((gender, l_acc.detach()))
                
                # Item popularity: random mask for now, you should replace this with real frequency checking
                is_popular = (torch.rand(query_y.shape[0]) > 0.8).to(device) 
                l_item = get_item_fairness_loss(query_y_pred, query_y, is_popular)
                losses_item.append(l_item)
                
                mp_atts.append(mp_att)
            
            if not losses_acc:
                continue
                
            # Aggregate batch-level losses
            L_acc = torch.stack(losses_acc).mean()
            L_item = torch.stack(losses_item).mean()
            L_path = get_path_fairness_loss(mp_atts)
            
            # User Fairness: difference in average MSE between Male (0) and Female (1) in this batch
            male_mses = [m[1] for m in losses_user if m[0] == 0]
            female_mses = [m[1] for m in losses_user if m[0] == 1]
            avg_male = torch.stack(male_mses).mean() if male_mses else torch.tensor(0.0).to(device)
            avg_female = torch.stack(female_mses).mean() if female_mses else torch.tensor(0.0).to(device)
            L_user = torch.abs(avg_male - avg_female)
            L_user.requires_grad_(True) # It's constructed from detaches, but we want it in the graph. 
            
            # Actually, to compute gradients properly, L_user must be built from the attached tensors:
            male_accs = [losses_acc[i] for i, m in enumerate(losses_user) if m[0] == 0]
            female_accs = [losses_acc[i] for i, m in enumerate(losses_user) if m[0] == 1]
            avg_male_g = torch.stack(male_accs).mean() if male_accs else torch.tensor(0.0).to(device)
            avg_female_g = torch.stack(female_accs).mean() if female_accs else torch.tensor(0.0).to(device)
            L_user = torch.abs(avg_male_g - avg_female_g)

            # Gather the 4 losses
            losses = [L_acc, L_user, L_item, L_path]
            
            # Filter out 0 losses to avoid gradient issues
            active_losses = [l for l in losses if l.requires_grad and l.item() > 0]
            if len(active_losses) == 0:
                continue
                
            # Get gradients for each loss
            grads = []
            valid_losses = []
            for l in active_losses:
                optimizer.zero_grad()
                l.backward(retain_graph=True)
                
                # Flatten gradients
                g = []
                for p in model.parameters():
                    if p.grad is not None:
                        g.append(p.grad.view(-1))
                if g:
                    grads.append(torch.cat(g))
                    valid_losses.append(l)
                    
            if len(grads) > 1:
                # Use MGDA to find Pareto optimal weights
                weights = MGDAOptimizer.get_optimal_weights(grads)
                
                # Compute final weighted loss
                final_loss = sum(w * l for w, l in zip(weights, valid_losses))
            else:
                final_loss = valid_losses[0]

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            total_loss += final_loss.item()
            
        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(dataloader):.4f}")

if __name__ == "__main__":
    config = {
        'dataset': 'movielens',
        'embedding_dim': 32,
        'item_embedding_dim': 64,
        'user_embedding_dim': 128,
        'first_fc_hidden_dim': 64,
        'second_fc_hidden_dim': 64,
        'item_fea_len': 26,
        'num_rate': 6,
        'num_genre': 25,
        'num_gender': 2,
        'num_age': 7,
        'num_occupation': 21,
        'num_zipcode': 3402,
        'use_cuda': torch.cuda.is_available(),
        'mp': ['UM', 'UMUM', 'UMAM', 'UMDM'],
        'mp_update': 1,
        'local_update': 1,
        'lr': 0.001,
        'mp_lr': 0.001,
        'local_lr': 0.001,
        'batch_size': 16
    }
    train(config, data_dir='data/movielens')
