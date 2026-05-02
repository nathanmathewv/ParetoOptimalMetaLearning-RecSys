import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

class MetaHINDataLoader(Dataset):
    def __init__(self, data_dir, state='meta_training', n_support=10, n_query=10):
        self.data_dir = data_dir
        self.state = state
        
        # Load precomputed features
        self.movie_fea_hete = np.load(os.path.join(data_dir, 'movie_feature_hete.npy'), allow_pickle=True).item()
        self.user_fea = np.load(os.path.join(data_dir, 'user_feature.npy'), allow_pickle=True).item()
        
        # Load user interactions
        with open(os.path.join(data_dir, f'{state}.json'), 'r') as f:
            self.u_movies = json.load(f)
        with open(os.path.join(data_dir, f'{state}_y.json'), 'r') as f:
            self.u_movies_y = json.load(f)
            
        self.u_ids = list(self.u_movies.keys())
        
        # Load gender mapping for User Fairness
        self.user_gender = {}
        with open(os.path.join(data_dir, 'original/users.dat'), 'r') as f:
            for line in f:
                parts = line.strip().split('::')
                uid, gender = int(parts[0]), parts[1]
                self.user_gender[str(uid)] = 1 if gender == 'F' else 0
                
        # Load graph connections for meta-paths
        with open(os.path.join(data_dir, 'movie_actor.json'), 'r') as f:
            self.m_actors = json.load(f)
        with open(os.path.join(data_dir, 'movie_director.json'), 'r') as f:
            self.m_directors = json.load(f)
            
        # Reverse dictionaries
        self.a_movies = {}
        for m, actors in self.m_actors.items():
            for a in actors:
                self.a_movies.setdefault(str(a), []).append(m)
                
        self.d_movies = {}
        for m, directors in self.m_directors.items():
            for d in directors:
                self.d_movies.setdefault(str(d), []).append(m)
                
        self.m_users = {}
        for u, movies in self.u_movies.items():
            for m in movies:
                self.m_users.setdefault(str(m), []).append(u)

    def __len__(self):
        return len(self.u_ids)

    def _get_mp_neighbors(self, u_id, m_id, movies_set):
        import random
        def sample(s, max_len=20):
            l = list(s)
            if len(l) > max_len:
                return random.sample(l, max_len)
            return l

        # UMUM: users who rated m_id -> their movies
        umum = set([str(m_id)])
        for u in sample(self.m_users.get(str(m_id), []), 10):
            umum.update([str(m) for m in sample(self.u_movies.get(str(u), []), 10)])
            
        # UMAM: actors of m_id -> their movies
        umam = set([str(m_id)])
        for a in sample(self.m_actors.get(str(m_id), []), 10):
            umam.update([str(m) for m in sample(self.a_movies.get(str(a), []), 10)])
            
        # UMDM: directors of m_id -> their movies
        umdm = set([str(m_id)])
        for d in sample(self.m_directors.get(str(m_id), []), 10):
            umdm.update([str(m) for m in sample(self.d_movies.get(str(d), []), 10)])
            
        return sample(umum, 20), sample(umam, 20), sample(umdm, 20)

    def _pack_mp_features(self, u_id, m_id, movies_set):
        umum, umam, umdm = self._get_mp_neighbors(u_id, m_id, movies_set)
        
        def safe_cat(movie_list):
            valid_m = [m for m in movie_list if int(m) in self.movie_fea_hete]
            if not valid_m:
                valid_m = [m_id]
            tensors = [self.movie_fea_hete[int(m)] for m in valid_m]
            return torch.cat(tensors, dim=0)
            
        return {
            'UM': safe_cat(movies_set),
            'UMUM': safe_cat(umum),
            'UMAM': safe_cat(umam),
            'UMDM': safe_cat(umdm)
        }

    def __getitem__(self, idx):
        u_id = self.u_ids[idx]
        movies = self.u_movies[u_id]
        ratings = self.u_movies_y[u_id]
        
        # Split into support and query randomly or sequentially
        # Here we do a simple 80-20 split
        split_idx = int(len(movies) * 0.8)
        if split_idx == 0 and len(movies) > 1:
            split_idx = 1
            
        supp_movies = movies[:split_idx]
        query_movies = movies[split_idx:]
        
        supp_ratings = ratings[:split_idx]
        query_ratings = ratings[split_idx:]
        
        # User feature
        u_fea = self.user_fea[int(u_id)]
        
        # Pack support
        supp_x, supp_mp = [], {'UM':[], 'UMUM':[], 'UMAM':[], 'UMDM':[]}
        for m in supp_movies:
            supp_x.append(torch.cat((self.movie_fea_hete[int(m)], u_fea), 1))
            mps = self._pack_mp_features(u_id, m, supp_movies)
            for k, v in mps.items(): supp_mp[k].append(v)
            
        # Pack query
        query_x, query_mp = [], {'UM':[], 'UMUM':[], 'UMAM':[], 'UMDM':[]}
        for m in query_movies:
            query_x.append(torch.cat((self.movie_fea_hete[int(m)], u_fea), 1))
            mps = self._pack_mp_features(u_id, m, supp_movies + query_movies)
            for k, v in mps.items(): query_mp[k].append(v)
            
        return {
            'u_id': int(u_id),
            'gender': self.user_gender.get(str(u_id), 0),
            'supp_x': torch.cat(supp_x, 0) if supp_x else torch.zeros(0),
            'supp_y': torch.FloatTensor(supp_ratings),
            'supp_mp': supp_mp,
            'query_x': torch.cat(query_x, 0) if query_x else torch.zeros(0),
            'query_y': torch.FloatTensor(query_ratings),
            'query_mp': query_mp
        }

def collate_fn(batch):
    return batch

def get_dataloader(data_dir, state, batch_size=32, shuffle=True):
    dataset = MetaHINDataLoader(data_dir, state)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=4)
