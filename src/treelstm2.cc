#include "treelstm2.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

TreeLSTMBuilder2::TreeLSTMBuilder2(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : cg(nullptr) {
  node_builder = LSTMBuilder(layers, input_dim, hidden_dim, model);
}

void TreeLSTMBuilder2::new_graph_impl(ComputationGraph& cg) {
  node_builder.new_graph(cg);
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void TreeLSTMBuilder2::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  node_builder.start_new_sequence(hinit);
}

Expression TreeLSTMBuilder2::add_input(int id, vector<int> children, const Expression& x) {
  assert (id >= 0 && h.size() == (unsigned)id);

  RNNPointer prev = (RNNPointer)(-1);
  Expression embedding = node_builder.add_input(prev, x);
  prev = node_builder.state();

  for (unsigned child : children) {
    embedding = node_builder.add_input(prev, h[child]);
    prev = node_builder.state();
  }
  h.push_back(embedding);
  return embedding;
}

BidirectionalTreeLSTMBuilder2::BidirectionalTreeLSTMBuilder2(unsigned layers,
                         unsigned input_dim,
                         unsigned hidden_dim,
                         Model* model) : cg(nullptr) {
  assert (hidden_dim % 2 == 0);
  fwd_node_builder = LSTMBuilder(layers, input_dim, hidden_dim / 2, model);
  rev_node_builder = LSTMBuilder(layers, input_dim, hidden_dim / 2, model);
}

void BidirectionalTreeLSTMBuilder2::new_graph_impl(ComputationGraph& cg) {
  fwd_node_builder.new_graph(cg);
  rev_node_builder.new_graph(cg);
}

// layout: 0..layers = c
//         layers+1..2*layers = h
void BidirectionalTreeLSTMBuilder2::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
  fwd_node_builder.start_new_sequence(hinit);
  rev_node_builder.start_new_sequence(hinit);
}

Expression BidirectionalTreeLSTMBuilder2::add_input(int id, vector<int> children, const Expression& x) {
  assert (id >= 0 && h.size() == (unsigned)id);

  RNNPointer prev = (RNNPointer)(-1);
  Expression fwd_embedding = fwd_node_builder.add_input(prev, x);
  prev = fwd_node_builder.state();
  for (unsigned child : children) {
    fwd_embedding = fwd_node_builder.add_input(prev, h[child]);
    prev = fwd_node_builder.state();
  }

  prev = (RNNPointer)(-1);
  Expression rev_embedding = rev_node_builder.add_input(prev, x);
  prev = rev_node_builder.state();
  for (unsigned i = children.size(); i-- > 0;) {
    unsigned  child = children[i];
    rev_embedding = rev_node_builder.add_input(prev, h[child]);
    prev = rev_node_builder.state();
  }

  Expression embedding = concatenate({fwd_embedding, rev_embedding});
  h.push_back(embedding);

  return embedding;
}

DerpTreeLSTMBuilder::DerpTreeLSTMBuilder(Model* model) : cg(nullptr) {}

void DerpTreeLSTMBuilder::new_graph_impl(ComputationGraph& cg) {}

void DerpTreeLSTMBuilder::start_new_sequence_impl(const vector<Expression>& hinit) {
  h.clear();
}

Expression DerpTreeLSTMBuilder::add_input(int id, vector<int> children, const Expression& x) {
  assert (id >= 0 && h.size() == (unsigned)id);

  vector<Expression> child_vectors (children.size() + 1);
  for (unsigned i = 0; i < children.size(); ++i) {
    child_vectors[i] = h[children[i]];
  }
  child_vectors[children.size()] = x;
  Expression r = sum(child_vectors) / child_vectors.size();
  h.push_back(r);
  return r;
}


} // namespace cnn
