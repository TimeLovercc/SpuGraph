# From "Discovering Invariant Rationales for Graph Neural Networks" and we made some modification

from .BA3_loc import *
from pathlib import Path
import random
from tqdm import tqdm
import pickle


def gen_dataset(global_b, data_path):
    n_node = 0
    n_edge = 0
    for _ in range(1000):
        # small:
        width_basis=np.random.choice(range(3,4))     # tree    #Node 32.55 #Edge 35.04
        # width_basis=np.random.choice(range(8,12))  # ladder  #Node 24.076 #Edge 34.603
        # width_basis=np.random.choice(range(15,20)) # wheel   #Node 21.954 #Edge 40.264
        # large:
        # width_basis=np.random.choice(range(3,6))   # tree    #Node 111.562 #Edge 117.77
        # width_basis=np.random.choice(range(30,50)) # ladder  #Node 83.744 #Edge 128.786
        # width_basis=np.random.choice(range(60,80)) # wheel   #Node 83.744 #Edge 128.786
        G, role_id, name = get_crane(basis_type="tree", nb_shapes=1,
                                            width_basis=width_basis,
                                            feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        ground_truth = find_gd(edge_index, role_id)

    #     pos = nx.spring_layout(G)
    #     nx.draw_networkx_nodes(G, pos=pos, nodelist=range(len(G.nodes())), node_size=150,
    #                            node_color=role_id, cmap='bwr',
    #                            linewidths=.1, edgecolors='k')

    #     nx.draw_networkx_labels(G, pos,
    #                             labels={i: str(role_id[i]) for i in range(len(G.nodes))},
    #                             font_size=10,
    #                             font_weight='bold', font_color='k'
    #                             )
    #     nx.draw_networkx_edges(G, pos=pos, edgelist=G.edges(), edge_color='black')
    #     plt.show()

        n_node += len(role_id)
        n_edge += edge_index.shape[1]
    print("#Node", n_node/1000, "#Edge", n_edge/1000)

    # Training Dataset
    edge_index_list = []
    label_list = []
    env_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    bias = float(global_b)
    e_mean = []
    n_mean = []
    for _ in tqdm(range(3000)):
        base_num = np.random.choice([1,2,3], p=[bias,(1-bias)/2,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(0)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(3000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,bias,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(1)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(3000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(2)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))
    
    np.save(data_path / 'train.npy', (edge_index_list, label_list, env_list, ground_truth_list, role_id_list, pos_list))

    def reindex_edges_and_data(FG_edge_index, ground_truth, role_id, pos):
        # Determine unique nodes in the FG_edge_index
        unique_nodes = np.unique(FG_edge_index)

        # Create a mapping from old node indices to new indices
        mapping = {node: i for i, node in enumerate(unique_nodes)}

        # Update the edge indices based on the mapping
        reindexed_edge_index = np.vectorize(mapping.get)(FG_edge_index)

        # Reindex ground_truth, role_id, and pos based on the unique nodes
        reindexed_ground_truth = ground_truth[unique_nodes]
        reindexed_role_id = role_id[unique_nodes]
        reindexed_pos = {mapping[node]: pos[node] for node in unique_nodes}

        return reindexed_edge_index, reindexed_ground_truth, reindexed_role_id, reindexed_pos

    FG_edge_index_list_updated = []
    FG_ground_truth_list = []
    FG_role_id_list = []
    FG_pos_list = []
    for i in range(len(edge_index_list)):
        FG_edge_index = edge_index_list[i][:, ground_truth_list[i] == 1]
        
        # Reindex the FG_edge_index and associated data
        reindexed_edge_index, reindexed_ground_truth, reindexed_role_id, reindexed_pos = reindex_edges_and_data(
            FG_edge_index, ground_truth_list[i], role_id_list[i], pos_list[i]
        )
        FG_edge_index_list_updated.append(reindexed_edge_index)
        FG_ground_truth_list.append(reindexed_ground_truth)
        FG_role_id_list.append(reindexed_role_id)
        FG_pos_list.append(reindexed_pos)
    print(len(FG_edge_index_list_updated))
    np.save(data_path / 'train_fg.npy', (FG_edge_index_list_updated, label_list, env_list, FG_ground_truth_list, FG_role_id_list, FG_pos_list))


    # Validation Dataset
    edge_index_list = []
    label_list = []
    env_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    bias = 1.0/3
    e_mean = []
    n_mean = []
    for _ in tqdm(range(1000)):
        base_num = np.random.choice([1,2,3], p=[bias,(1-bias)/2,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(0)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(1000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,bias,(1-bias)/2])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(1)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(1000)):
        base_num = np.random.choice([1,2,3], p=[(1-bias)/2,(1-bias)/2,bias])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,4))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(8,12))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(15,20))

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(2)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))
    np.save(data_path / 'val.npy', (edge_index_list, label_list, env_list, ground_truth_list, role_id_list, pos_list))

    FG_edge_index_list_updated = []
    FG_ground_truth_list = []
    FG_role_id_list = []
    FG_pos_list = []
    for i in range(len(edge_index_list)):
        FG_edge_index = edge_index_list[i][:, ground_truth_list[i] == 1]
        
        # Reindex the FG_edge_index and associated data
        reindexed_edge_index, reindexed_ground_truth, reindexed_role_id, reindexed_pos = reindex_edges_and_data(
            FG_edge_index, ground_truth_list[i], role_id_list[i], pos_list[i]
        )
        FG_edge_index_list_updated.append(reindexed_edge_index)
        FG_ground_truth_list.append(reindexed_ground_truth)
        FG_role_id_list.append(reindexed_role_id)
        FG_pos_list.append(reindexed_pos)
    print(len(FG_edge_index_list_updated))

    np.save(data_path / 'val_fg.npy', (FG_edge_index_list_updated, label_list, env_list, FG_ground_truth_list, FG_role_id_list, FG_pos_list))


    # Test Dataset

    edge_index_list = []
    label_list = []
    env_list = []
    ground_truth_list = []
    role_id_list = []
    pos_list = []

    e_mean = []
    n_mean = []
    for _ in tqdm(range(2000)):
        base_num = np.random.choice([1,2,3])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,6))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(30,50))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(60,80))

        G, role_id, name = get_cycle(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(0)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(2000)):
        base_num = np.random.choice([1,2,3])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,6))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(30,50))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(60,80))

        G, role_id, name = get_house(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(1)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))

    e_mean = []
    n_mean = []
    for _ in tqdm(range(2000)):
        base_num = np.random.choice([1,2,3])

        if base_num == 1:
            base = 'tree'
            width_basis=np.random.choice(range(3,6))
        if base_num == 2:
            base = 'ladder'
            width_basis=np.random.choice(range(30,50))
        if base_num == 3:
            base = 'wheel'
            width_basis=np.random.choice(range(60,80))

        G, role_id, name = get_crane(basis_type=base, nb_shapes=1,
                                        width_basis=width_basis, feature_generator=None, m=3, draw=False)
        role_id = np.array(role_id)
        edge_index = np.array(G.edges, dtype=int).T
        row, col = edge_index
        e_mean.append(len(G.edges))
        n_mean.append(len(G.nodes))
        edge_index_list.append(edge_index)
        label_list.append(2)
        env_list.append(base_num)
        ground_truth = find_gd(edge_index, role_id)
        ground_truth_list.append(ground_truth)
        role_id_list.append(role_id)
        pos = nx.spring_layout(G)
        pos_list.append(pos)
    print(np.mean(n_mean), np.mean(e_mean))
    print(len(ground_truth_list))
    np.save(data_path / 'test.npy', (edge_index_list, label_list, env_list, ground_truth_list, role_id_list, pos_list))

    FG_edge_index_list_updated = []
    FG_ground_truth_list = []
    FG_role_id_list = []
    FG_pos_list = []
    for i in range(len(edge_index_list)):
        FG_edge_index = edge_index_list[i][:, ground_truth_list[i] == 1]
        
        # Reindex the FG_edge_index and associated data
        reindexed_edge_index, reindexed_ground_truth, reindexed_role_id, reindexed_pos = reindex_edges_and_data(
            FG_edge_index, ground_truth_list[i], role_id_list[i], pos_list[i]
        )
        FG_edge_index_list_updated.append(reindexed_edge_index)
        FG_ground_truth_list.append(reindexed_ground_truth)
        FG_role_id_list.append(reindexed_role_id)
        FG_pos_list.append(reindexed_pos)
    print(len(FG_edge_index_list_updated))
    np.save(data_path / 'test_fg.npy', (FG_edge_index_list_updated, label_list, env_list, FG_ground_truth_list, FG_role_id_list, FG_pos_list))


def get_house(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["house"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_cycle(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["dircycle"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name


def get_crane(basis_type, nb_shapes=80, width_basis=8, feature_generator=None, m=3, draw=True):
    """ Synthetic Graph #5:

    Start with a tree and attach grid-shaped subgraphs.

    Args:
        nb_shapes         :  The number of shapes (here 'houses') that should be added to the base graph.
        width_basis       :  The width of the basis graph (here a random 'grid').
        feature_generator :  A `FeatureGenerator` for node features. If `None`, add constant features to nodes.
        m                 :  The tree depth.

    Returns:
        G                 :  A networkx graph
        role_id           :  Role ID for each node in synthetic graph
        name              :  A graph identifier
    """
    list_shapes = [["crane"]] * nb_shapes

    if draw:
        plt.figure(figsize=figsize)

    G, role_id, _ = synthetic_structsim.build_graph(
        width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
    )
    G = perturb([G], 0.05, id=role_id)[0]

    if feature_generator is None:
        feature_generator = featgen.ConstFeatureGen(1)
    feature_generator.gen_node_features(G)

    name = basis_type + "_" + str(width_basis) + "_" + str(nb_shapes)

    return G, role_id, name
