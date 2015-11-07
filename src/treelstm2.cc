#include "treelstm2.h"

#include <string>
#include <cassert>
#include <vector>
#include <iostream>

#include "cnn/nodes.h"

using namespace std;
using namespace cnn::expr;

namespace cnn {

enum { X2I, BI, X2F, BF, X2O, BO, X2C, BC };
enum { H2I, H2F, H2O, H2C, C2I, C2F, C2O };
// See "Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks"
// by Tai, Socher, and Manning (2015), section 3.2, for details on this model.
// http://arxiv.org/pdf/1503.00075v3.pdf
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

} // namespace cnn
