#include "embedtrees.h"

TreeEmbedder::TreeEmbedder() {
}

TreeEmbedder::TreeEmbedder(Model& model, unsigned vocab_size) {
  InitializeParameters(model, vocab_size);
}

void TreeEmbedder::InitializeParameters(Model& model, unsigned vocab_size) {
  assert (node_embedding_dim % 2 == 0);
  const unsigned half_node_embedding_dim = node_embedding_dim / 2;
  forward_builder = LSTMBuilder(lstm_layer_count, word_embedding_dim, half_node_embedding_dim, &model);
  reverse_builder = LSTMBuilder(lstm_layer_count, word_embedding_dim, half_node_embedding_dim, &model); 
  //tree_builder = new SocherTreeLSTMBuilder(5, lstm_layer_count, node_embedding_dim, node_embedding_dim, &model);
  //tree_builder = new TreeLSTMBuilder2(lstm_layer_count, node_embedding_dim, node_embedding_dim, &model);
  tree_builder = new BidirectionalTreeLSTMBuilder2(lstm_layer_count, node_embedding_dim, node_embedding_dim, &model);
  output_builder = SimpleRNNBuilder(lstm_layer_count, word_embedding_dim, final_hidden_dim, &model);

  left_w = model.add_parameters({vocab_size, node_embedding_dim});
  left_b = model.add_parameters({vocab_size});
  right_w = model.add_parameters({vocab_size, node_embedding_dim});
  right_b = model.add_parameters({vocab_size});

  trans_w = model.add_parameters({word_embedding_dim, node_embedding_dim});
  trans_b = model.add_parameters({word_embedding_dim});
  final_w = model.add_parameters({vocab_size, final_hidden_dim});
  final_b = model.add_parameters({vocab_size});

  word_trans_w = model.add_parameters({node_embedding_dim, word_embedding_dim});
  word_trans_b = model.add_parameters({node_embedding_dim});
  p_E = model.add_lookup_parameters(vocab_size, {word_embedding_dim});
  zero_annotation.resize(node_embedding_dim);
}

tuple<unsigned, Expression> TreeEmbedder::BuildGraph(const SyntaxTree& tree, ComputationGraph& cg) {
  const bool predict_context = true; // False = predict terminals within node, true = predict prev/next word
  const bool all_nodes = true; // False = roots only
  const WordId kBOS = 1;
  const WordId kEOS = 2;
  NewGraph(cg);
  output_builder.new_graph(cg);
  output_builder.start_new_sequence();

  vector<Expression> linear_annotations = BuildLinearAnnotationVectors(tree, cg);
  vector<Expression> tree_annotations = BuildTreeAnnotationVectors(tree, linear_annotations, cg);

  vector<const SyntaxTree*> nodes = LinearizeNodes(tree);
  assert (nodes.back() == &tree);
  assert (tree_annotations.size() == nodes.size());

  assert (!(predict_context && !all_nodes));

  if (predict_context) {
    vector<WordId> terminals = tree.GetTerminals();
    unordered_map<const SyntaxTree*, tuple<unsigned, unsigned>> spans;
    ComputeSpans(&tree, 0, spans);
    vector<unsigned> word_counts(nodes.size());
    vector<Expression> losses(nodes.size());
    unsigned total_word_count = 0;
    for (unsigned i = 0; i < nodes.size(); ++i) {
      const SyntaxTree* node = nodes[i];
      auto span = spans.find(node);
      assert (span != spans.end());
      unsigned s, e;
      tie(s, e) = span->second;
      assert (s >= 0 && s < terminals.size());
      assert (e > 0 && e <= terminals.size());
      WordId prev = (s > 0) ? terminals[s - 1] : kBOS;
      WordId next = (e < terminals.size() - 1) ? terminals[e + 1] : kEOS;
      assert (prev >= 0 && prev <= 20000);
      assert (next >= 0 && next <= 20000);
      tie(word_counts[i], losses[i]) = CalculateContextLoss(prev, next, tree_annotations[i], cg);
      total_word_count += word_counts[i];
    }
    return make_tuple(total_word_count, sum(losses));
  }
  else if (all_nodes) {
    vector<unsigned> word_counts(nodes.size());
    vector<Expression> losses(nodes.size());
    unsigned total_word_count = 0;
    for (unsigned i = 0; i < nodes.size(); ++i) {
      tie(word_counts[i], losses[i]) = CalculateLoss(*nodes[i], tree_annotations[i], cg);
      total_word_count += word_counts[i];
    }
    return make_tuple(total_word_count, sum(losses));
  }
  else {
    return CalculateLoss(*nodes.back(), tree_annotations.back(), cg);
  }
}

tuple<unsigned, Expression> TreeEmbedder::CalculateLoss(const SyntaxTree& tree, Expression embedding, ComputationGraph& cg) {
  vector<WordId> terminals = tree.GetTerminals();

  const unsigned kEOS = 2;
  terminals.push_back(kEOS);
  vector<Expression> word_losses(terminals.size());
  Expression next_input = affine_transform({transb, transw, embedding});
  
  RNNPointer prev_state = (RNNPointer)(-1);
  for (unsigned i = 0; i < terminals.size(); ++i) {
    Expression final_hidden_layer = output_builder.add_input(prev_state, next_input);
    Expression dist = affine_transform({finalb, finalw, final_hidden_layer});
    word_losses[i] = pickneglogsoftmax(dist, terminals[i]);
    prev_state = output_builder.state();
    next_input = lookup(cg, p_E, terminals[i]);
  } 
  return make_tuple(word_losses.size(), sum(word_losses));
}

tuple<unsigned, Expression> TreeEmbedder::CalculateContextLoss(const WordId prev, const WordId next, Expression embedding, ComputationGraph& cg) {
  Expression left_dist = affine_transform({leftb, leftw, embedding});
  Expression right_dist = affine_transform({rightb, rightw, embedding});

  Expression left_loss = pickneglogsoftmax(left_dist, prev);
  Expression right_loss = pickneglogsoftmax(right_dist, prev);

  Expression total_loss = left_loss + right_loss;
  return make_tuple(2, total_loss);
}

vector<Expression> TreeEmbedder::BuildForwardAnnotations(const vector<WordId>& sentence, ComputationGraph& cg) {
  forward_builder.new_graph(cg);
  forward_builder.start_new_sequence();
  vector<Expression> forward_annotations(sentence.size());
  for (unsigned t = 0; t < sentence.size(); ++t) {
    Expression i_x_t = lookup(cg, p_E, sentence[t]);
    Expression i_y_t = forward_builder.add_input(i_x_t);
    forward_annotations[t] = i_y_t;
  }
  return forward_annotations;
}

vector<Expression> TreeEmbedder::BuildReverseAnnotations(const vector<WordId>& sentence, ComputationGraph& cg) {
  reverse_builder.new_graph(cg);
  reverse_builder.start_new_sequence();
  vector<Expression> reverse_annotations(sentence.size());
  for (unsigned t = sentence.size(); t > 0; ) {
    t--;
    Expression i_x_t = lookup(cg, p_E, sentence[t]);
    Expression i_y_t = reverse_builder.add_input(i_x_t);
    reverse_annotations[t] = i_y_t;
  }
  return reverse_annotations;
}

vector<Expression> TreeEmbedder::BuildAnnotationVectors(const vector<Expression>& forward_annotations, const vector<Expression>& reverse_annotations, ComputationGraph& cg) {
  vector<Expression> annotations(forward_annotations.size());
  for (unsigned t = 0; t < forward_annotations.size(); ++t) {
    const Expression& i_f = forward_annotations[t];
    const Expression& i_r = reverse_annotations[t];
    Expression i_h = concatenate({i_f, i_r});
    annotations[t] = i_h;
  }
  return annotations;
}

vector<Expression> TreeEmbedder::BuildLinearAnnotationVectors(const SyntaxTree& tree, ComputationGraph& cg) {
  const bool use_bidirectional = false;
  if (use_bidirectional) {
    vector<Expression> forward_annotations = BuildForwardAnnotations(tree.GetTerminals(), cg);
    vector<Expression> reverse_annotations = BuildReverseAnnotations(tree.GetTerminals(), cg);
    vector<Expression> linear_annotations = BuildAnnotationVectors(forward_annotations, reverse_annotations, cg);
    return linear_annotations;
  }
  else {
    vector<Expression> linear_annotations;
    Expression transw = parameter(cg, word_trans_w);
    Expression transb = parameter(cg, word_trans_b);
    for (WordId w : tree.GetTerminals()) {
      Expression word_vector = lookup(cg, p_E, w);
      Expression node_embedding = affine_transform({transb, transw, word_vector});
      linear_annotations.push_back(node_embedding);
    }
    return linear_annotations;
  }
}

vector<Expression> TreeEmbedder::BuildTreeAnnotationVectors(const SyntaxTree& source_tree, const vector<Expression>& linear_annotations, ComputationGraph& cg) {
  tree_builder->new_graph(cg);
  tree_builder->start_new_sequence();
  vector<Expression> annotations;
  vector<Expression> tree_annotations;
  vector<const SyntaxTree*> node_stack = {&source_tree};
  vector<unsigned> index_stack = {0};
  unsigned terminal_index = 0;

  while (node_stack.size() > 0) {
    assert (node_stack.size() == index_stack.size());
    const SyntaxTree* node = node_stack.back();
    unsigned i = index_stack.back();
    if (i >= node->NumChildren()) {
      assert (tree_annotations.size() == node->id());
      vector<int> children(node->NumChildren());
      for (unsigned j = 0; j < node->NumChildren(); ++j) {
        unsigned child_id = node->GetChild(j).id();
        assert (child_id < tree_annotations.size());
        assert (child_id < (unsigned)INT_MAX);
        children[j] = (int)child_id;
      }

      Expression input_expr;
      if (node->NumChildren() == 0) {
        assert (terminal_index < linear_annotations.size());
        input_expr = linear_annotations[terminal_index];
        terminal_index++;
      }
      else {
        input_expr = input(cg, {(long)zero_annotation.size()}, &zero_annotation);
      }

      Expression node_annotation = tree_builder->add_input((int)node->id(), children, input_expr);
      tree_annotations.push_back(node_annotation);
      index_stack.pop_back();
      node_stack.pop_back();
    }
    else {
      index_stack[index_stack.size() - 1] += 1;
      node_stack.push_back(&node->GetChild(i));
      index_stack.push_back(0);
      ++i;
    }
  }
  assert (node_stack.size() == index_stack.size());

  return tree_annotations;
}

vector<const SyntaxTree*> TreeEmbedder::LinearizeNodes(const SyntaxTree& root) {
  vector<const SyntaxTree*> nodes;
  for (unsigned i = 0; i < root.NumChildren(); ++i) {
    const SyntaxTree& child = root.GetChild(i);
    vector<const SyntaxTree*> child_nodes = LinearizeNodes(child);
    nodes.insert(nodes.end(), child_nodes.begin(), child_nodes.end());
  }
  nodes.push_back(&root);
  return nodes;
}

unsigned TreeEmbedder::ComputeSpans(const SyntaxTree* root, unsigned start, unordered_map<const SyntaxTree*, tuple<unsigned, unsigned>>& spans) {
  if (root->NumChildren() == 0) {
    spans[root] = make_tuple(start, start + 1);
    return start + 1;
  }

  unsigned root_start = start;
  for (unsigned i = 0; i < root->NumChildren(); ++i) {
    const SyntaxTree& child = root->GetChild(i);
    start = ComputeSpans(&child, start, spans);
  }
  unsigned root_end = start;
  spans[root] = make_tuple(root_start, root_end);
  return root_end;
}

void TreeEmbedder::NewGraph(ComputationGraph& cg) {
  leftw = parameter(cg, left_w);
  leftb = parameter(cg, left_b);
  rightw = parameter(cg, right_w);
  rightb = parameter(cg, right_b);

  transw = parameter(cg, trans_w);
  transb = parameter(cg, trans_b);
  finalw = parameter(cg, final_w);
  finalb = parameter(cg, final_b);
}
