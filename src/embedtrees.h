#pragma once
#include <vector>
#include <boost/archive/text_oarchive.hpp>
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
  vector<Expression> BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& cg);
  vector<Expression> BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& cg);
  vector<Expression> BuildAnnotationVectors(const vector<Expression>& forward_annotations, const vector<Expression>& reverse_annotations, ComputationGraph& cg);
  vector<Expression> BuildLinearAnnotationVectors(const SyntaxTree& tree, ComputationGraph& cg);
  vector<Expression> BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg);
  vector<const SyntaxTree*> LinearizeNodes(const SyntaxTree& root);

private:
  LSTMBuilder forward_builder;
  LSTMBuilder reverse_builder;
  TreeLSTMBuilder* tree_builder;
  //SocherTreeLSTMBuilder tree_builder;
  //TreeLSTMBuilder2 tree_builder;
  SimpleRNNBuilder output_builder;
  LookupParameters* p_E;
  vector<cnn::real> zero_annotation;
  Parameters* final_w;
  Parameters* final_b;

  unsigned lstm_layer_count = 2;
  unsigned word_embedding_dim = 50;
  unsigned node_embedding_dim = 50;
  unsigned final_hidden_dim = 100;

  friend class boost::serialization::access;
  template<class Archive> void serialize(Archive& ar, const unsigned int) {
    ar & lstm_layer_count;
    ar & word_embedding_dim;
    ar & node_embedding_dim;
  }
};
