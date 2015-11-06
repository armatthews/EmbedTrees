#ifndef CNN_TREELSTM2_H_
#define CNN_TREELSTM2_H_

#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/expr.h"
#include "cnn/lstm.h"
#include "treelstm.h"

using namespace cnn::expr;

namespace cnn {

class Model;

struct TreeLSTMBuilder2 : public TreeLSTMBuilder {
  TreeLSTMBuilder2() = default;
  explicit TreeLSTMBuilder2(unsigned layers,
                       unsigned input_dim,
                       unsigned hidden_dim,
                       Model* model);

  Expression add_input(int id, std::vector<int> children, const Expression& x);
 protected:
  void new_graph_impl(ComputationGraph& cg) override;
  void start_new_sequence_impl(const std::vector<Expression>& h0) override;

 public:
  LSTMBuilder node_builder;
  std::vector<Expression> h;

private:
  ComputationGraph* cg;
};

} // namespace cnn

#endif
