import os
import json
import pandas as pd
import numpy as np
import torch
import re
import random
import pickle
import os
from tqdm import tqdm
random.seed(13)
input_dir = 'original/'
output_dir = './'
melu_output_dir = '../../../MeLU/movielens/'
states = [ "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing","meta_training"]

if not os.path.exists("{}/meta_training/".format(output_dir)):
    os.mkdir("{}/log/".format(output_dir))
    for state in states:
        os.mkdir("{}/{}/".format(output_dir, state))
        os.mkdir("{}/{}/".format(melu_output_dir, state))
        if not os.path.exists("{}/{}/{}".format(output_dir, "log", state)):
            os.mkdir("{}/{}/{}".format(output_dir, "log", state))
ui_data = pd.read_csv(input_dir+'ratings.dat', names=['user', 'item', 'rating', 'timestamp'],sep="::", engine='python')
len(ui_data)
user_data = pd.read_csv(input_dir+'users.dat', names=['user', 'gender', 'age', 'occupation_code', 'zip'],
                        sep="::", engine='python')
item_data = pd.read_csv(input_dir+'movies_extrainfos.dat', names=['item', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
                        sep="::", engine='python', encoding="utf-8")
user_list = list(set(ui_data.user.tolist()) | set(user_data.user))
item_list = list(set(ui_data.item.tolist()) | set(item_data.item))
user_num = len(user_list)
item_num = len(item_list)
user_num, item_num
def load_list(fname):
    list_ = []
    with open(fname, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_
rate_list = load_list("{}/m_rate.txt".format(input_dir))
genre_list = load_list("{}/m_genre.txt".format(input_dir))
actor_list = load_list("{}/m_actor.txt".format(input_dir))
director_list = load_list("{}/m_director.txt".format(input_dir))
gender_list = load_list("{}/m_gender.txt".format(input_dir))
age_list = load_list("{}/m_age.txt".format(input_dir))
occupation_list = load_list("{}/m_occupation.txt".format(input_dir))
zipcode_list = load_list("{}/m_zipcode.txt".format(input_dir))
len(rate_list), len(genre_list), len(actor_list), len(director_list), len(gender_list), len(age_list), len(occupation_list), len(zipcode_list)
    def item_converting(row, rate_list, genre_list, director_list, actor_list):
        rate_idx = torch.tensor([[rate_list.index(str(row['rate']))]]).long()
        genre_idx = torch.zeros(1, 25).long()
        for genre in str(row['genre']).split(", "):
            idx = genre_list.index(genre)
            genre_idx[0, idx] = 1  # one-hot vector
        
        director_idx = torch.zeros(1, 2186).long()
        director_id = []
        for director in str(row['director']).split(", "):
            idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
            director_idx[0, idx] = 1
            director_id.append(idx+1)  # id starts from 1, not index
        actor_idx = torch.zeros(1, 8030).long()
        actor_id = []
        for actor in str(row['actors']).split(", "):
            idx = actor_list.index(actor)
            actor_idx[0, idx] = 1
            actor_id.append(idx+1)
        return torch.cat((rate_idx, genre_idx), 1), torch.cat((rate_idx, genre_idx, director_idx, actor_idx), 1), director_id, actor_id

    def user_converting(row, gender_list, age_list, occupation_list, zipcode_list):
        gender_idx = torch.tensor([[gender_list.index(str(row['gender']))]]).long()
        age_idx = torch.tensor([[age_list.index(str(row['age']))]]).long()
        occupation_idx = torch.tensor([[occupation_list.index(str(row['occupation_code']))]]).long()
        zip_idx = torch.tensor([[zipcode_list.index(str(row['zip'])[:5])]]).long()
        return torch.cat((gender_idx, age_idx, occupation_idx, zip_idx), 1)  # (1, 4)
# hash map for item
movie_fea_hete = {}
movie_fea_homo = {}
m_directors = {}
m_actors = {}
for idx, row in item_data.iterrows():
    m_info = item_converting(row, rate_list, genre_list, director_list, actor_list)
    movie_fea_hete[row['item']] = m_info[0]
    movie_fea_homo[row['item']] = m_info[1]
    m_directors[row['item']] = m_info[2]
    m_actors[row['item']] = m_info[3]
# hash map for user
user_fea = {}
for idx, row in user_data.iterrows():
    u_info = user_converting(row, gender_list, age_list, occupation_list, zipcode_list)
    user_fea[row['user']] = u_info
states = [ "warm_up", "user_cold_testing", "item_cold_testing", "user_and_item_cold_testing","meta_training"]
    import collections
    def reverse_dict(d):
        # {1:[a,b,c], 2:[a,f,g],...}
        re_d = collections.defaultdict(list)
        for k, v_list in d.items():
            for v in v_list:
                re_d[v].append(k)
        return dict(re_d)
a_movies = reverse_dict(m_actors)
d_movies = reverse_dict(m_directors)
len(a_movies), len(d_movies)
def jsonKeys2int(x):
    if isinstance(x, dict):
            return {int(k):v for k,v in x.items()}
    return x
state = 'meta_training'

support_u_movies = json.load(open(output_dir+state+'/support_u_movies.json','r'), object_hook=jsonKeys2int)
query_u_movies= json.load(open(output_dir+state+'/query_u_movies.json','r'), object_hook=jsonKeys2int)
support_u_movies_y = json.load(open(output_dir+state+'/support_u_movies_y.json','r'), object_hook=jsonKeys2int)
query_u_movies_y = json.load(open(output_dir+state+'/query_u_movies_y.json','r'), object_hook=jsonKeys2int)
if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))

train_u_movies = {}
if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))
for idx, u_id in tqdm(enumerate(u_id_list)):
    train_u_movies[int(u_id)] = []
    train_u_movies[int(u_id)] += support_u_movies[u_id]+query_u_movies[u_id]
len(train_u_movies)
train_u_id_list = list(u_id_list).copy()
len(train_u_id_list)
# get mp data 
print(state)

u_m_u_movies = {}
u_m_a_movies = {}
u_m_d_movies = {}

support_m_users = reverse_dict(support_u_movies)

for u in tqdm(u_id_list):
    u_m_u_movies[u] = {}
    u_m_a_movies[u] = {}
    u_m_d_movies[u] = {}
    for m in support_u_movies[u]:
        u_m_a_movies[u][m] = set([m])
        for _a in m_actors[m]:
            cur_ms = a_movies[_a]
            u_m_a_movies[u][m].update(cur_ms)
            
        u_m_d_movies[u][m] = set([m])
        for _d in m_directors[m]:
            cur_ms = d_movies[_d]
            u_m_d_movies[u][m].update(cur_ms)
        
        u_m_u_movies[u][m] = set([m])   # add itself to avoid empty tensor when build the support set
        u_m_u_movies[u][m].update(support_u_movies[u].copy())
        if m in support_m_users:  # for meta_training, only support set can be seen!!!
            for _u in support_m_users[m]:  #  only include user in training set !!!!
                cur_ms = support_u_movies[_u]  # list
                u_m_u_movies[u][m].update(cur_ms)
    
    for m in query_u_movies[u]:
        if m in u_m_a_movies[u] or m in u_m_d_movies[u] or m in u_m_u_movies[u]:
            print('error!!!')
            break
            
        u_m_a_movies[u][m] = set([m])
        for _a in m_actors[m]:
            cur_ms = a_movies[_a]
            u_m_a_movies[u][m].update(cur_ms)
            
        u_m_d_movies[u][m] = set([m])
        for _d in m_directors[m]:
            cur_ms = d_movies[_d]
            u_m_d_movies[u][m].update(cur_ms)
        
        u_m_u_movies[u][m] = set([m])   # add itself to avoid empty tensor when build the support set
        u_m_u_movies[u][m].update(support_u_movies[u].copy())
        if m in support_m_users:  # for meta_training, only support set can be seen!!!
            for _u in support_m_users[m]:  #  only include user in training set !!!!
                cur_ms = support_u_movies[_u]  # list
                u_m_u_movies[u][m].update(cur_ms)
        
print(len(u_m_u_movies), len(u_m_a_movies), len(u_m_d_movies))
for idx, u_id in  tqdm(enumerate(u_id_list)):
    support_x_app = None
    support_um_app = []
    support_umum_app = []
    support_umam_app = []
    support_umdm_app = []
        
    for m_id in support_u_movies[u_id]:
        tmp_x_converted = torch.cat((movie_fea_hete[m_id], user_fea[u_id]), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
        except:
            support_x_app = tmp_x_converted

        # meta-paths
        # UM
        support_um_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], support_u_movies[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        support_umum_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_u_movies[u_id][m_id])), dim=0))
        # UMAM
        support_umam_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_a_movies[u_id][m_id])), dim=0))
        # UMDM
        support_umdm_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_d_movies[u_id][m_id])), dim=0))
        
    support_y_app = torch.FloatTensor(support_u_movies_y[u_id])
    
    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_um_app, open("{}/{}/support_um_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_umum_app, open("{}/{}/support_umum_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_umam_app, open("{}/{}/support_umam_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_umdm_app, open("{}/{}/support_umdm_{}.pkl".format(output_dir, state, idx), "wb"))
   
    query_x_app = None
    query_um_app = []
    query_umum_app = []
    query_umam_app = []
    query_umdm_app = []
        
    for m_id in query_u_movies[u_id]:
        tmp_x_converted = torch.cat((movie_fea_hete[m_id], user_fea[u_id]), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
        except:
            query_x_app = tmp_x_converted

        # meta-paths
        # UM
        query_um_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], support_u_movies[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        query_umum_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_u_movies[u_id][m_id])), dim=0))
        # UMAM
        query_umam_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_a_movies[u_id][m_id])), dim=0))
        # UMDM
        query_umdm_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_d_movies[u_id][m_id])), dim=0))
        
    query_y_app = torch.FloatTensor(query_u_movies_y[u_id])
    
    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_um_app, open("{}/{}/query_um_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_umum_app,open("{}/{}/query_umum_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_umam_app,open("{}/{}/query_umam_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_umdm_app,open("{}/{}/query_umdm_{}.pkl".format(output_dir, state, idx), "wb"))
   
print(idx)

# state = 'warm_up'
# state = 'user_cold_testing'
# state = 'item_cold_testing'
state = 'user_and_item_cold_testing'

support_u_movies = json.load(open(output_dir+state+'/support_u_movies.json','r'), object_hook=jsonKeys2int)
query_u_movies= json.load(open(output_dir+state+'/query_u_movies.json','r'), object_hook=jsonKeys2int)
support_u_movies_y = json.load(open(output_dir+state+'/support_u_movies_y.json','r'), object_hook=jsonKeys2int)
query_u_movies_y = json.load(open(output_dir+state+'/query_u_movies_y.json','r'), object_hook=jsonKeys2int)
if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))

cur_train_u_movies =  train_u_movies.copy()

if support_u_movies.keys() == query_u_movies.keys():
    u_id_list = support_u_movies.keys()
print(len(u_id_list))
for idx, u_id in tqdm(enumerate(u_id_list)):
    if u_id not in cur_train_u_movies:
        cur_train_u_movies[u_id] = []
    cur_train_u_movies[u_id] += support_u_movies[u_id]

print(len(cur_train_u_movies),  len(train_u_movies))
print(len(set(train_u_id_list) & set(u_id_list)))

(len(u_id_list) +  len(train_u_movies) - len(set(train_u_id_list) & set(u_id_list))) == len(set(cur_train_u_movies))
# get mp data 
print(state)

u_m_u_movies = {}
u_m_a_movies = {}
u_m_d_movies = {}

cur_train_m_users = reverse_dict(cur_train_u_movies)

for u in tqdm(u_id_list):
    u_m_u_movies[u] = {}
    u_m_a_movies[u] = {}
    u_m_d_movies[u] = {}
    for m in support_u_movies[u]:
        u_m_a_movies[u][m] = set([m])
        for _a in m_actors[m]:
            cur_ms = a_movies[_a]
            u_m_a_movies[u][m].update(cur_ms)
            
        u_m_d_movies[u][m] = set([m])
        for _d in m_directors[m]:
            cur_ms = d_movies[_d]
            u_m_d_movies[u][m].update(cur_ms)
        
        u_m_u_movies[u][m] = set([m])   # add itself to avoid empty tensor when build the support set
        u_m_u_movies[u][m].update(cur_train_u_movies[u].copy())
        if m in cur_train_m_users:  # for meta_training, only support set can be seen!!!
            for _u in cur_train_m_users[m]:  #  only include user in training set !!!!
                cur_ms = cur_train_u_movies[_u]  # list
                u_m_u_movies[u][m].update(cur_ms)
    
    for m in query_u_movies[u]:
        if m in u_m_a_movies[u] or m in u_m_d_movies[u] or m in u_m_u_movies[u]:
            print('error!!!')
            break
            
        u_m_a_movies[u][m] = set([m])
        for _a in m_actors[m]:
            cur_ms = a_movies[_a]
            u_m_a_movies[u][m].update(cur_ms)
            
        u_m_d_movies[u][m] = set([m])
        for _d in m_directors[m]:
            cur_ms = d_movies[_d]
            u_m_d_movies[u][m].update(cur_ms)
        
        u_m_u_movies[u][m] = set([m])   # add itself to avoid empty tensor when build the support set
        u_m_u_movies[u][m].update(cur_train_u_movies[u].copy())
        if m in cur_train_m_users:  # for meta_training, only support set can be seen!!!
            for _u in cur_train_m_users[m]:  #  only include user in training set !!!!
                cur_ms = cur_train_u_movies[_u]  # list
                u_m_u_movies[u][m].update(cur_ms)
        
print(len(u_m_u_movies), len(u_m_a_movies), len(u_m_d_movies))
for idx, u_id in  tqdm(enumerate(u_id_list)):
    support_x_app = None
    support_um_app = []
    support_umum_app = []
    support_umam_app = []
    support_umdm_app = []
        
    for m_id in support_u_movies[u_id]:
        tmp_x_converted = torch.cat((movie_fea_hete[m_id], user_fea[u_id]), 1)
        try:
            support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
        except:
            support_x_app = tmp_x_converted

        # meta-paths
        # UM
        support_um_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], cur_train_u_movies[u_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        support_umum_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_u_movies[u_id][m_id])), dim=0))
        # UMAM
        support_umam_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_a_movies[u_id][m_id])), dim=0))
        # UMDM
        support_umdm_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_d_movies[u_id][m_id])), dim=0))
        
    support_y_app = torch.FloatTensor(support_u_movies_y[u_id])
    
    pickle.dump(support_x_app, open("{}/{}/support_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_y_app, open("{}/{}/support_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_um_app, open("{}/{}/support_um_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_umum_app, open("{}/{}/support_umum_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_umam_app, open("{}/{}/support_umam_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(support_umdm_app, open("{}/{}/support_umdm_{}.pkl".format(output_dir, state, idx), "wb"))
   
    query_x_app = None
    query_um_app = []
    query_umum_app = []
    query_umam_app = []
    query_umdm_app = []
        
    for m_id in query_u_movies[u_id]:
        tmp_x_converted = torch.cat((movie_fea_hete[m_id], user_fea[u_id]), 1)
        try:
            query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
        except:
            query_x_app = tmp_x_converted

        # meta-paths
        # UM
        query_um_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], cur_train_u_movies[u_id]+[m_id])), dim=0))  # each element: (#neighbor, 26=1+25)
        # UMUM
        query_umum_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_u_movies[u_id][m_id])), dim=0))
        # UMAM
        query_umam_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_a_movies[u_id][m_id])), dim=0))
        # UMDM
        query_umdm_app.append(torch.cat(list(map(lambda x: movie_fea_hete[x], u_m_d_movies[u_id][m_id])), dim=0))
        
    query_y_app = torch.FloatTensor(query_u_movies_y[u_id])
    
    pickle.dump(query_x_app, open("{}/{}/query_x_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_y_app, open("{}/{}/query_y_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_um_app, open("{}/{}/query_um_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_umum_app,open("{}/{}/query_umum_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_umam_app,open("{}/{}/query_umam_{}.pkl".format(output_dir, state, idx), "wb"))
    pickle.dump(query_umdm_app,open("{}/{}/query_umdm_{}.pkl".format(output_dir, state, idx), "wb"))
   
print(idx)
len(support_umum_app)
query_umum_app[1].shape
u_id, m_id

