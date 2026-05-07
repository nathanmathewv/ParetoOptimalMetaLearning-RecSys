import torch
import torch.nn.functional as F
import numpy as np
from models.meta_learner import MetaLearner, MetapathLearner
from models.EmbeddingInitializer import UserEmbeddingML, ItemEmbeddingML

class FairMetaHIN(torch.nn.Module):
    def __init__(self, config):
        super(FairMetaHIN, self).__init__()
        self.config = config
        self.device = torch.device("cuda" if config.get('use_cuda', False) else "cpu")
        
        self.item_emb = ItemEmbeddingML(config)
        self.user_emb = UserEmbeddingML(config)
        
        self.mp_learner = MetapathLearner(config)
        self.meta_learner = MetaLearner(config)
        
        self.mp_lr = config.get('mp_lr', 0.001)
        self.local_lr = config.get('local_lr', 0.001)
        
        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.mp_weight_len = len(self.mp_learner.update_parameters())
        self.mp_weight_name = list(self.mp_learner.update_parameters().keys())
        
        self.transformer_liners = self.transform_mp2task()

    def transform_mp2task(self):
        liners = {}
        ml_parameters = self.meta_learner.update_parameters()
        output_dim_of_mp = 32
        for w in self.ml_weight_name:
            liners[w.replace('.', '-')] = torch.nn.Linear(output_dim_of_mp, np.prod(ml_parameters[w].shape))
        return torch.nn.ModuleDict(liners)

    def inner_update(self, support_user_emb, support_item_emb, support_set_y, support_mp_user_emb, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.meta_learner.update_parameters()

        support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict)
        loss = F.mse_loss(support_set_y_pred.view(-1), support_set_y.view(-1))
        grad = torch.autograd.grad(loss, vars_dict.values(), create_graph=False, allow_unused=True)

        fast_weights = {}
        for i, w in enumerate(vars_dict.keys()):
            if grad[i] is not None:
                fast_weights[w] = vars_dict[w] - self.local_lr * grad[i]
            else:
                fast_weights[w] = vars_dict[w]

        for idx in range(1, self.config.get('local_update', 1)):
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_user_emb, vars_dict=fast_weights)
            loss = F.mse_loss(support_set_y_pred.view(-1), support_set_y.view(-1))
            grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=False, allow_unused=True)

            for i, w in enumerate(fast_weights.keys()):
                if grad[i] is not None:
                    fast_weights[w] = fast_weights[w] - self.local_lr * grad[i]

        return fast_weights

    def forward(self, support_x, support_y, support_mp, query_x, query_y, query_mp):
        # Extract features
        item_fea_len = self.config.get('item_fea_len', 26)
        
        support_user_emb = self.user_emb(support_x[:, item_fea_len:].long())
        support_item_emb = self.item_emb(support_x[:, 0:item_fea_len].long())
        
        query_user_emb = self.user_emb(query_x[:, item_fea_len:].long())
        query_item_emb = self.item_emb(query_x[:, 0:item_fea_len].long())
        
        support_mp_enhanced_user_emb_s, query_mp_enhanced_user_emb_s = [], []
        mp_task_fast_weights_s = {}
        mp_task_loss_s = {}
        
        mp_initial_weights = self.mp_learner.update_parameters()
        ml_initial_weights = self.meta_learner.update_parameters()
        
        for mp in self.config['mp']:
            support_set_mp = support_mp[mp]
            query_set_mp = query_mp[mp]
            
            support_neighs_emb = self.item_emb(torch.cat([x.long() for x in support_set_mp]))
            support_index_list = [x.shape[0] for x in support_set_mp]
            
            query_neighs_emb = self.item_emb(torch.cat([x.long() for x in query_set_mp]))
            query_index_list = [x.shape[0] for x in query_set_mp]
            
            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp, support_index_list)
            
            # Meta-path learner update
            support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
            loss = F.mse_loss(support_set_y_pred.view(-1), support_y.view(-1))
            grad = torch.autograd.grad(loss, mp_initial_weights.values(), create_graph=False, allow_unused=True)
            
            fast_weights = {}
            for i in range(self.mp_weight_len):
                weight_name = self.mp_weight_name[i]
                if grad[i] is not None:
                    fast_weights[weight_name] = mp_initial_weights[weight_name] - self.mp_lr * grad[i]
                else:
                    fast_weights[weight_name] = mp_initial_weights[weight_name]
                    
            for idx in range(1, self.config.get('mp_update', 1)):
                support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp, support_index_list, vars_dict=fast_weights)
                support_set_y_pred = self.meta_learner(support_user_emb, support_item_emb, support_mp_enhanced_user_emb)
                loss = F.mse_loss(support_set_y_pred.view(-1), support_y.view(-1))
                grad = torch.autograd.grad(loss, fast_weights.values(), create_graph=False, allow_unused=True)
                for i in range(self.mp_weight_len):
                    weight_name = self.mp_weight_name[i]
                    if grad[i] is not None:
                        fast_weights[weight_name] = fast_weights[weight_name] - self.mp_lr * grad[i]
            
            # Compute embeddings with updated meta-path learner
            support_mp_enhanced_user_emb = self.mp_learner(support_user_emb, support_item_emb, support_neighs_emb, mp, support_index_list, vars_dict=fast_weights)
            query_mp_enhanced_user_emb = self.mp_learner(query_user_emb, query_item_emb, query_neighs_emb, mp, query_index_list, vars_dict=fast_weights)
            
            support_mp_enhanced_user_emb_s.append(support_mp_enhanced_user_emb)
            query_mp_enhanced_user_emb_s.append(query_mp_enhanced_user_emb)
            
            # Transformer for meta-learner init
            f_fast_weights = {}
            for w, liner in self.transformer_liners.items():
                w = w.replace('-', '.')
                # Mean over sample dimension
                f_fast_weights[w] = ml_initial_weights[w] * torch.sigmoid(liner(support_mp_enhanced_user_emb.mean(0))).view(ml_initial_weights[w].shape)
                
            # Inner task update
            mp_task_fast_weights = self.inner_update(support_user_emb, support_item_emb, support_y, support_mp_enhanced_user_emb, vars_dict=f_fast_weights)
            mp_task_fast_weights_s[mp] = mp_task_fast_weights
            
            # Query loss for attention
            query_set_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_mp_enhanced_user_emb, vars_dict=mp_task_fast_weights)
            q_loss = F.mse_loss(query_set_y_pred.view(-1), query_y.view(-1))
            mp_task_loss_s[mp] = q_loss
            
        # Attention across meta-paths (Exposure_p)
        mp_loss_stack = torch.stack(list(mp_task_loss_s.values()))
        mp_att = F.softmax(-mp_loss_stack, dim=0)
        
        # Aggregate weights
        agg_task_fast_weights = {}
        for idx, mp in enumerate(self.config['mp']):
            if idx == 0:
                agg_task_fast_weights = {k: v * mp_att[idx] for k, v in mp_task_fast_weights_s[mp].items()}
            else:
                tmp_weights = {k: v * mp_att[idx] for k, v in mp_task_fast_weights_s[mp].items()}
                for k in agg_task_fast_weights:
                    agg_task_fast_weights[k] += tmp_weights[k]
                    
        # Final prediction
        agg_mp_emb = torch.stack(query_mp_enhanced_user_emb_s, 1)
        query_agg_enhanced_user_emb = torch.sum(agg_mp_emb * mp_att.unsqueeze(1), 1)
        
        query_y_pred = self.meta_learner(query_user_emb, query_item_emb, query_agg_enhanced_user_emb, vars_dict=agg_task_fast_weights)
        
        return query_y_pred, mp_att
