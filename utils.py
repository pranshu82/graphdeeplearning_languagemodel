import networkx as nx
import torch
import numpy as np
import scipy.sparse as sp
from prettytable import PrettyTable

def print_graph_detail(graph):
    dst = {"nodes"    : nx.number_of_nodes(graph),
           "edges"    : nx.number_of_edges(graph),
           "selfloops": nx.number_of_selfloops(graph),
           "isolates" : nx.number_of_isolates(graph),
           "index"      : 1 - nx.number_of_isolates(graph) / nx.number_of_nodes(graph), }
    print_table(dst)

def print_table(dst):
    table_title = list(dst.keys())
    table = PrettyTable(field_names=table_title, header_style="title", header=True, border=True,
                        hrules=1, padding_width=2, align="c")
    table.float_format = "0.4"
    table.add_row([dst[i] for i in table_title])
    print(table)

def read_file(path, mode='r', encoding=None):
    if mode not in {"r", "rb"}:
        raise ValueError("only read")
    return open(path, mode=mode, encoding=encoding)

def preprocess_adj(adj, is_sparse=False):
    """Preprocessing of adjacency matrix for simple pygGCN model and conversion to
    tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    if is_sparse:
        adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
        return adj_normalized
    else:
        return torch.from_numpy(adj_normalized.A).float()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()