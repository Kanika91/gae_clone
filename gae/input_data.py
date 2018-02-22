import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def ismember(a, b, tol=5):
    rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
    return (np.all(np.any(rows_close, axis=-1), axis=-1) and
            np.all(np.any(rows_close, axis=0), axis=0))

def load_data_pkl(dataset):
    if dataset == "gowalla":
        with open ('gowalla/traindata_small.pkl' , 'rb' ) as f:
            pkl_data = pkl.load(f)
            traindata = pkl_data['graph']

        with open ('gowalla/testdata_small.pkl' , 'rb' ) as f:
            pkl_data = pkl.load(f)
            testdata = pkl_data['graph']

        with open ('gowalla/inter_small.pkl' , 'rb' ) as f:
            pkl_data = pkl.load(f)
            user_enum = pkl_data['user_enum']
            spot_enum = pkl_data['spot_enum']
            user_graph = pkl_data['user_graph']
            spot_graph = pkl_data['spot_graph']

    elif dataset == "movielens":
        traindata = []
        spot_enum = set()
        with open("../../baselines/neural_collaborative_filtering/Data/ml-1m.train.rating",'r') as f:
            for line in f:
                user, item, rating, timestamp = line.strip().split("\t")
                traindata.append((int(user),int(item)))
                spot_enum.add(int(item))

        testdata = []
        user_enum = set()

        with open("../../baselines/neural_collaborative_filtering/Data/ml-1m.test.rating",'r') as f:
            for line in f:
                user, item, rating, timestamp = line.strip().split("\t")
                testdata.append((int(user),int(item)))
                user_enum.add(int(user))
                spot_enum.add(int(item))
        user_graph = []
        spot_graph = []

    #Create adjacency matrix
    print("User count ",len(user_enum))
    print("Spot count ",len(spot_enum), "user graph ",len(user_graph),'spot graph ',len(spot_graph))
    user_count = len(user_enum)
    item_count = len(spot_enum)
    index = len(user_enum) + len(spot_enum)


    edgelist = [] #Add user graph
    num_val = int(len(traindata)*0.2)
    edgelist = edgelist + [(x,y+user_count) for (x,y) in traindata]

    # Internally shuffle training set (before splitting off validation set)
    rand_idx = range(len(edgelist))
    np.random.seed(42)
    np.random.shuffle(rand_idx)
    edgearray = np.array(edgelist)
    edgearray = edgearray[rand_idx]
    edgelist = edgearray.tolist()

    val_edges = edgelist[:num_val]
    print("Length of val edges before ",len(val_edges))
    val_edges = val_edges + [(y,x) for (x,y) in val_edges]
    print("Length of val edges after ",len(val_edges))

    train_edges = edgelist[num_val:]
    train_edges = train_edges + [(y,x) for (x,y) in train_edges]

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, user_count)
        idx_j = np.random.randint(0, item_count)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j+user_count], np.array(train_edges)):
            continue
        if ismember([idx_i, idx_j+user_count], np.array(val_edges)):
            continue
        if val_edges_false:
            if ismember([idx_i, idx_j+user_count], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j+user_count])
        val_edges_false.append([idx_j+user_count, idx_i])

    #Load test data
    test_edges = [(x,y+user_count) for (x,y) in testdata]
    test_edges = test_edges + [(y,x) for (x,y) in test_edges]
    edges_all = np.concatenate((train_edges,val_edges,test_edges),axis=0)


    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, user_count)
        idx_j = np.random.randint(0, item_count)
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j+user_count], edges_all):
            continue
        if ismember([idx_i, idx_j+user_count], np.array(val_edges_false)):
            continue
        if test_edges_false:
            if ismember([idx_i, idx_j+user_count], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j+user_count])
        test_edges_false.append([idx_j+user_count, idx_i])

    print("Length of different edges ",len(train_edges), len(val_edges), len(val_edges_false), len(test_edges), len(test_edges_false))
    # assert ~ismember(test_edges_false, edges_all)
    # assert ~ismember(val_edges_false, edges_all)
    # assert ~ismember(val_edges, np.array(train_edges))
    # assert ~ismember(test_edges, np.array(train_edges))
    # assert ~ismember(val_edges, np.array(test_edges))

    #print('length ',len(edgelist),len(user_graph),len(traindata))
    edgelist = user_graph #Add user graph
    edgelist = edgelist + [(x+user_count,y+user_count) for (x,y) in spot_graph] #Add item graph
    edgelist = edgelist + [(y,x) for (x,y) in edgelist]
    edgelist = train_edges + edgelist
    train_edges = edgelist

    data = np.ones(len(train_edges))
    train_edges_arr = np.array(train_edges)
    adj_train = sp.csr_matrix((data, (train_edges_arr[:, 0], train_edges_arr[:, 1])), shape=(index,index))
    data = np.ones(len(edges_all))
    edges_all_arr = np.array(edges_all)
    adj = sp.csr_matrix((data, (edges_all_arr[:, 0], edges_all_arr[:, 1])), shape=(index,index))

    #adj_train = nx.adjacency_matrix(nx.from_edgelist(edgelist))
    #adj = nx.adjacency_matrix(nx.from_edgelist(edges_all))
    print(adj[70,91],adj[91,70])
    print("adjacency matrix shape ", adj.shape[0], adj.shape[1], adj_train.shape[0], adj_train.shape[1])

    #return adj,np.zeros((adj.shape[0],adj.shape[0]))
    #val_edges_false = []
    #test_edges_false = []
    return adj, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false, np.zeros((adj.shape[0],adj.shape[0]))

def load_data(dataset):

    if dataset == 'gowalla' or dataset == 'movielens':
        return load_data_pkl(dataset)

    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        objects.append(pkl.load(open("data/ind.{}.{}".format(dataset, names[i]))))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features
