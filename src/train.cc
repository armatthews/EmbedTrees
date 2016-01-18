#include "cnn/cnn.h"
#include "cnn/training.h"

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/program_options.hpp>

#include <iostream>
#include <fstream>
#include <csignal>
#include <random>
#include <memory>
#include <algorithm>

#include "embedtrees.h"
#include "train.h"

using namespace cnn;
using namespace std;
namespace po = boost::program_options;

pair<cnn::real, unsigned> ComputeLoss(const vector<SyntaxTree>& data, TreeEmbedder& model) {
  cnn::real loss = 0.0;
  unsigned node_count = 0;
  for (unsigned i = 0; i < data.size(); ++i) {
    ComputationGraph cg;
    unsigned word_count;
    Expression loss_expr;
    tie(word_count, loss_expr) = model.BuildGraph(data[i], cg);
    node_count += word_count;
    double l = as_scalar(cg.forward());
    loss += l;
    if (ctrlc_pressed) {
      break;
    }
  }
  return make_pair(loss, node_count);
}

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("training_set", po::value<string>()->required(), "Training trees")
  ("dev_set", po::value<string>()->required(), "Dev trees, used for early stopping")
  ("num_iterations,i", po::value<unsigned>()->default_value(UINT_MAX), "Number of epochs to train for")
  ("batch_size,b", po::value<unsigned>()->default_value(1), "Size of minibatches")
  ("random_seed,r", po::value<unsigned>()->default_value(0), "Random seed. If this value is 0 a seed will be chosen randomly.")
  // Optimizer configuration
  ("sgd", "Use SGD for optimization")
  ("momentum", po::value<double>(), "Use SGD with this momentum value")
  ("adagrad", "Use Adagrad for optimization")
  ("adadelta", "Use Adadelta for optimization")
  ("rmsprop", "Use RMSProp for optimization")
  ("adam", "Use Adam for optimization")
  ("learning_rate", po::value<double>(), "Learning rate for optimizer (SGD, Adagrad, Adadelta, and RMSProp only)")
  ("alpha", po::value<double>(), "Alpha (Adam only)")
  ("beta1", po::value<double>(), "Beta1 (Adam only)")
  ("beta2", po::value<double>(), "Beta2 (Adam only)")
  ("rho", po::value<double>(), "Moving average decay parameter (RMSProp and Adadelta only)")
  ("epsilon", po::value<double>(), "Epsilon value for optimizer (Adagrad, Adadelta, RMSProp, and Adam only)")
  ("regularization", po::value<double>()->default_value(0.0), "L2 Regularization strength")
  ("eta_decay", po::value<double>()->default_value(0.05), "Learning rate decay rate (SGD only)")
  ("no_clipping", "Disable clipping of gradients")
  // End optimizer configuration
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("training_set", 1);
  positional_options.add("dev_set", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string train_filename = vm["training_set"].as<string>();
  const string dev_filename = vm["dev_set"].as<string>();
  const unsigned num_iterations = vm["num_iterations"].as<unsigned>();
  const unsigned random_seed = vm["random_seed"].as<unsigned>();
  const unsigned minibatch_size = vm["batch_size"].as<unsigned>();

  cnn::Initialize(argc, argv, random_seed);
  std::mt19937 rndeng(42);
  TreeEmbedder* tree_embedder = new TreeEmbedder();
  Model* cnn_model = new Model();
  Dict vocab;

  vocab.Convert("UNK");
  vocab.Convert("<s>");
  vocab.Convert("</s>");
  vector<SyntaxTree>* training_set = ReadTrees(train_filename, &vocab);
  if (training_set == nullptr) {
    return 1;
  }
  assert (minibatch_size <= training_set->size());
  //vocab.Freeze();
  vector<SyntaxTree>* dev_set = ReadTrees(dev_filename, &vocab);

  cerr << "Vocab size: " << vocab.size() << endl;
  tree_embedder->InitializeParameters(*cnn_model, vocab.size());
  Trainer* sgd = CreateTrainer(*cnn_model, vm);

  cerr << "Training model...\n";
  unsigned minibatch_count = 0;
  const unsigned report_frequency = 50;
  cnn::real best_dev_loss = numeric_limits<cnn::real>::max();
  for (unsigned iteration = 0; iteration < num_iterations; iteration++) {
    unsigned word_count = 0;
    unsigned tword_count = 0;
    random_shuffle(training_set->begin(), training_set->end());
    double loss = 0.0;
    double tloss = 0.0;
    for (unsigned i = 0; i < training_set->size(); ++i) {
      // These braces cause cg to go out of scope before we ever try to call
      // ComputeLoss() on the dev set. Without them, ComputeLoss() tries to
      // create a second ComputationGraph, which makes CNN quite unhappy.
      {
        ComputationGraph cg;
        SyntaxTree& example = training_set->at(i);
        unsigned sent_word_count;
        Expression loss_expr;
        tie(sent_word_count, loss_expr) = tree_embedder->BuildGraph(example, cg);
        word_count += sent_word_count;
        tword_count += sent_word_count;
        double sent_loss = as_scalar(cg.forward());
        loss += sent_loss;
        tloss += sent_loss;
        cg.backward();
      }
      if (i % report_frequency == report_frequency - 1) {
        float fractional_iteration = (float)iteration + ((float)(i + 1) / training_set->size());
        cerr << "--" << fractional_iteration << "     perp=" << exp(tloss/tword_count) <<  " (" << tloss << " over " << tword_count << " words)" << endl;
        cerr.flush();
        tloss = 0;
        tword_count = 0;
      }
      if (++minibatch_count == minibatch_size) {
        sgd->update(1.0 / minibatch_size);
        minibatch_count = 0;
      }
      if (ctrlc_pressed) {
        break;
      }
    }
    //sgd->update_epoch();
    cerr << "##" << (float)(iteration + 1) << "     perp=" << exp(loss / word_count) << " (" << loss << " over " << word_count << " words)" << endl;
    if (!ctrlc_pressed) {
      auto dev_loss = ComputeLoss(*dev_set, *tree_embedder);
      cnn::real dev_perp = exp(dev_loss.first / dev_loss.second);
      bool new_best = dev_loss.first <= best_dev_loss;
      cerr << "**" << iteration + 1 << " dev perp: " << dev_perp << (new_best ? " (New best!)" : "") << endl;
      cerr.flush();
      if (new_best) {
        Serialize(vocab, *tree_embedder, *cnn_model);
        best_dev_loss = dev_loss.first;
      }
    }

    if (ctrlc_pressed) {
      break;
    }
  }

  return 0;
}
