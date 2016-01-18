#pragma once
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <unordered_map>
#include "cnn/cnn.h"
#include "cnn/lstm.h"
#include "treelstm.h"
#include "treelstm2.h"
#include "syntax_tree.h"

using namespace std;
using namespace cnn;
using namespace cnn::expr;

class TreeEmbedder {
public:
  TreeEmbedder();
  TreeEmbedder(Model& model, unsigned vocab_size);
  void InitializeParameters(Model& model, unsigned vocab_size);

  tuple<unsigned, Expression> BuildGraph(const SyntaxTree& tree, ComputationGraph& cg);
  tuple<unsigned, Expression> CalculateLoss(const SyntaxTree& tree, Expression embedding, ComputationGraph& cg);
  tuple<unsigned, Expression> CalculateContextLoss(const WordId prev, const WordId next, Expression embedding, ComputationGraph& cg);
  vector<Expression> BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& cg);
  vector<Expression> BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& cg);
  vector<Expression> BuildAnnotationVectors(const vector<Expression>& forward_annotations, const vector<Expression>& reverse_annotations, ComputationGraph& cg);
  vector<Expression> BuildLinearAnnotationVectors(const SyntaxTree& tree, ComputationGraph& cg);
  vector<Expression> BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg);
  vector<const SyntaxTree*> LinearizeNodes(const SyntaxTree& root);

 unsigned ComputeSpans(const SyntaxTree* root, unsigned start, unordered_map<const SyntaxTree*, tuple<unsigned, unsigned>>& spans);

private:
  void NewGraph(ComputationGraph& cg);
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  TreeLSTMBuilder* tree_builder;
  SimpleRNNBuilder output_builder;
  LookupParameters* p_E;

  Parameters* left_w;
  Parameters* left_b;
  Parameters* right_w;
  Parameters* right_b;

  Parameters* trans_w;
  Parameters* trans_b;
  Parameters* final_w;
  Parameters* final_b;

  Parameters* word_trans_w;
  Parameters* word_trans_b;

  Expression leftw;
  Expression leftb;
  Expression rightw;
  Expression rightb;

  Expression transw;
  Expression transb;
  Expression finalw;
  Expression finalb;

  unsigned lstm_layer_count = 2;
  unsigned word_embedding_dim = 100;
  unsigned node_embedding_dim = 98;
  unsigned final_hidden_dim = 50;

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & word_embedding_dim;
    ar & node_embedding_dim;
  }
};
