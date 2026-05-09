import sys
import copy
import random
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

with open("data/category_mapping.json", "r") as f:
    item_category_mapping = json.load(f)["product_to_category"]

cat_to_items = defaultdict(list)
for item_id, cat_id in item_category_mapping.items():
    cat_to_items[cat_id].append(int(item_id))

# Pre-convert to sets for faster subtraction logic
cat_to_items_set = {k: set(v) for k, v in cat_to_items.items()}
all_items_set = {int(i) for i in item_category_mapping.keys()}
item_to_cat = {int(item_id): cat_id for item_id, cat_id in item_category_mapping.items()}

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def computeRePos(time_seq, time_span):
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix

def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED, cat_to_items_set, item_to_cat):
    def sample(user):
        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        
        user_history = user_train[user]
        ts = set(map(lambda x: x[0], user_history))
        nxt = user_history[-1][0]
        idx = maxlen - 1
        
        for i in reversed(user_history[:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt

            if nxt != 0:
                target_cat = item_to_cat.get(nxt)
                # 1. Ensure we only pull candidates that are within the valid embedding range
                candidates = [i for i in list(cat_to_items_set.get(target_cat, set()) - ts) if i <= itemnum]
                
                if len(candidates) > 0:
                    neg[idx] = random.choice(candidates)
                else:
                    neg[idx] = random_neq(1, itemnum + 1, ts)
                
            nxt = i[0]
            idx -= 1
            if idx == -1: break
            
        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            user = np.random.randint(1, usernum + 1)
            while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)
            one_batch.append(sample(user))

        result_queue.put(zip(*one_batch))

class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10, n_workers=1, cat_to_items_set=None, item_to_cat=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []

        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                    usernum,
                    itemnum,
                    batch_size,
                    maxlen,
                    relation_matrix,
                    self.result_queue,
                    np.random.randint(2e9),
                    cat_to_items_set, # Pass it to the worker
                    item_to_cat      # Pass it to the worker
                )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map

def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u+1
    for i, item in enumerate(item_set):
        item_map[item] = i+1
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    
    print('Preparing data...')
    f = open('data/%s.txt' % fname, 'r')
    time_set = set()

    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        user_count[u]+=1
        item_count[i]+=1
    f.close()
    f = open('data/%s.txt' % fname, 'r') # try?...ugly data pre-processing code
    for line in f:
        try:
            u, i, rating, timestamp = line.rstrip().split('\t')
        except:
            u, i, timestamp = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        timestamp = float(timestamp)
        if user_count[u]<5 or item_count[i]<5: # hard-coded
            continue
        time_set.add(timestamp)
        User[u].append([i, timestamp])
    f.close()
    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    print('Preparing done...')
    return [user_train, user_valid, user_test, usernum, itemnum, timenum]

def evaluate(model, dataset, args, item_to_cat, cat_to_items_set):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    # Safety catalog for fallback sampling
    all_items_set = set(range(1, itemnum + 1))

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
        
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        # Build the input sequence (ending with the validation item)
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
            
        # Target item and its category
        target_item = test[u][0][0]
        target_cat = item_to_cat.get(target_item)
        
        # Items the user has already seen (to avoid false negatives)
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(valid[u][0][0])
        rated.add(target_item)
        rated.add(0)

        # START HARD NEGATIVE SAMPLING
        item_idx = [target_item]
        
        # Find candidates in the same category
        candidates = list(cat_to_items_set.get(target_cat, set()) - rated)
        
        if len(candidates) >= 100:
            item_idx.extend(random.sample(candidates, 100))
        else:
            # Not enough in category? Take all available and fill from global pool
            item_idx.extend(candidates)
            shortfall = 100 - len(candidates)
            remaining_pool = list(all_items_set - rated - set(candidates))
            item_idx.extend(random.sample(remaining_pool, shortfall))
        # END HARD NEGATIVE SAMPLING

        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        predictions = predictions[0]

        # Rank of the first item (target_item) among the 101 items
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, item_to_cat, cat_to_items_set):
    [train, valid, test, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    all_items_set = set(range(1, itemnum + 1))

    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        # Valid logic uses train history only
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        target_item = valid[u][0][0]
        target_cat = item_to_cat.get(target_item)
        
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(target_item)
        rated.add(0)

        # HARD NEGATIVE SAMPLING
        item_idx = [target_item]
        candidates = list(cat_to_items_set.get(target_cat, set()) - rated)
        
        if len(candidates) >= 100:
            item_idx.extend(random.sample(candidates, 100))
        else:
            item_idx.extend(candidates)
            shortfall = 100 - len(candidates)
            remaining_pool = list(all_items_set - rated - set(candidates))
            item_idx.extend(random.sample(remaining_pool, shortfall))

        time_matrix = computeRePos(time_seq, args.time_span)
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
            
        if valid_user % 100 == 0:
            print('.', end='')
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user
