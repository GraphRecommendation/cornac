import dgl.utils
from torch import nn


class HypergraphLayer(nn.Module):
	def __init__(self, in_dim, out_dim):
		super().__init__()

		self.linear = nn.Linear(in_dim, out_dim)
		self.activation = nn.LeakyReLU()

	def message(self, lhs_field, rhs_field, out):
		def func(edges):
			norm = edges.src['norm'] * edges.dst['norm'] * edges.data['norm']
			m = edges.src[lhs_field] * norm.unsqueeze(-1)
			return {out: m}

		return func

	def forward(self, g: dgl.DGLGraph, x):
		with g.local_scope():
			g.ndata['h'] = x

			g.update_all(self.message('h', 'h', 'm'), dgl.function.sum('m', 'h'))

			return self.activation(self.linear(dgl.mean_nodes(g, 'h')))


class Model(nn.Module):
	def __init__(self, n_nodes):
		super().__init__()

		self.node_embedding = nn.Embedding(n_nodes, 64)
		self.review_conv = HypergraphLayer(64, 32)
		self.review_agg = dgl.nn.GATv2Conv(32, 16, 3)
		self.loss_fn = nn.MSELoss(reduction='mean')

	def forward(self, blocks, x):
		x = self.review_conv(blocks[0], x)
		x = self.review_agg(blocks[1], x)

		return x

	def graph_predict(self, g: dgl.DGLGraph, x):
		with g.local_scope():
			g.ndata['h'] = x
			g.apply_edges(dgl.function.u_dot_v('h', 'h', 'm'))
			return g.edata['m']

	def loss(self, preds, target):
		return self.loss_fn(preds, target)