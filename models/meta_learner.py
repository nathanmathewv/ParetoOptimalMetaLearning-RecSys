import torch
from torch.nn import functional as F

class MetaLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetaLearner, self).__init__()
        self.embedding_dim = config.get('embedding_dim', 32)
        self.fc1_in_dim = 32 + config.get('item_embedding_dim', 32)
        self.fc2_in_dim = config.get('first_fc_hidden_dim', 64)
        self.fc2_out_dim = config.get('second_fc_hidden_dim', 64)
        self.config = config

        self.vars = torch.nn.ParameterDict()

        w1 = torch.nn.Parameter(torch.ones([self.fc2_in_dim, self.fc1_in_dim]))
        torch.nn.init.xavier_normal_(w1)
        self.vars['ml_fc_w1'] = w1
        self.vars['ml_fc_b1'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w2 = torch.nn.Parameter(torch.ones([self.fc2_out_dim, self.fc2_in_dim]))
        torch.nn.init.xavier_normal_(w2)
        self.vars['ml_fc_w2'] = w2
        self.vars['ml_fc_b2'] = torch.nn.Parameter(torch.zeros(self.fc2_in_dim))

        w3 = torch.nn.Parameter(torch.ones([1, self.fc2_out_dim]))
        torch.nn.init.xavier_normal_(w3)
        self.vars['ml_fc_w3'] = w3
        self.vars['ml_fc_b3'] = torch.nn.Parameter(torch.zeros(1))

    def forward(self, user_emb, item_emb, user_neigh_emb, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars

        x_i = item_emb
        x_u = user_neigh_emb
        
        x = torch.cat((x_i, x_u), 1)
        x = F.relu(F.linear(x, vars_dict['ml_fc_w1'], vars_dict['ml_fc_b1']))
        x = F.relu(F.linear(x, vars_dict['ml_fc_w2'], vars_dict['ml_fc_b2']))
        x = F.linear(x, vars_dict['ml_fc_w3'], vars_dict['ml_fc_b3'])
        return x.squeeze(-1)

    def update_parameters(self):
        return self.vars

class MetapathLearner(torch.nn.Module):
    def __init__(self, config):
        super(MetapathLearner, self).__init__()
        self.config = config
        self.vars = torch.nn.ParameterDict()
        
        neigh_w = torch.nn.Parameter(torch.ones([32, config.get('item_embedding_dim', 32)]))
        torch.nn.init.xavier_normal_(neigh_w)
        self.vars['neigh_w'] = neigh_w
        self.vars['neigh_b'] = torch.nn.Parameter(torch.zeros(32))

    def forward(self, user_emb, item_emb, neighs_emb, mp, index_list, vars_dict=None):
        if vars_dict is None:
            vars_dict = self.vars
            
        agg_neighbor_emb = F.linear(neighs_emb, vars_dict['neigh_w'], vars_dict['neigh_b'])
        
        # In original MetaHIN, they did torch.mean over all neighbors
        # We process them sequentially per instance or just take mean
        _emb = []
        start = 0
        for idx in index_list:
            end = start + idx
            _emb.append(F.leaky_relu(torch.mean(agg_neighbor_emb[start:end], 0)))
            start = end
        
        if _emb:
            output_emb = torch.stack(_emb, 0)
        else:
            output_emb = torch.zeros(len(index_list), 32).to(neighs_emb.device)
            
        return output_emb

    def update_parameters(self):
        return self.vars
