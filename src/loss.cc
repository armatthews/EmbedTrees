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

int main(int argc, char** argv) {
  signal (SIGINT, ctrlc_handler);

  po::options_description desc("description");
  desc.add_options()
  ("model", po::value<string>()->required(), "Model file")
  ("test_set", po::value<string>()->required(), "Test trees")
  // End optimizer configuration
  ("help", "Display this help message");

  po::positional_options_description positional_options;
  positional_options.add("model", 1);
  positional_options.add("test_set", 1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(positional_options).run(), vm);

  if (vm.count("help")) {
    cerr << desc;
    return 1;
  }

  po::notify(vm);

  const string model_filename = vm["model"].as<string>();
  const string test_filename = vm["test_set"].as<string>();

  cnn::Initialize(argc, argv);
  std::mt19937 rndeng(42);
  TreeEmbedder* tree_embedder = new TreeEmbedder();
  Model* cnn_model = new Model();
  Dict vocab;

  ifstream model_file(model_filename);
  boost::archive::text_iarchive ia(model_file);
  ia & vocab;
  cerr << "Vocab size: " << vocab.size() << endl;
  ia & *tree_embedder;
  tree_embedder->InitializeParameters(*cnn_model, vocab.size());
  ia & *cnn_model;
  model_file.close();

  vocab.Freeze();
  vector<SyntaxTree>* test_set = ReadTrees(test_filename, &vocab);
  if (test_set == nullptr) {
    return 1;
  }

  unsigned word_count = 0;
  double loss = 0.0;
  for (unsigned i = 0; i < test_set->size(); ++i) {
      ComputationGraph cg;
      SyntaxTree& example = test_set->at(i);
      unsigned sent_word_count;
      Expression loss_expr;
      tie(sent_word_count, loss_expr) = tree_embedder->BuildGraph(example, cg);
      word_count += sent_word_count;
      double sent_loss = as_scalar(cg.forward());
      loss += sent_loss;
      cout << i << " ||| loss=" << sent_loss << " words=" << sent_word_count << " perp=" << exp(sent_loss/sent_word_count) << endl;
    if (ctrlc_pressed) {
      break;
    }
  }

  cerr << "Totals: loss=" << loss << " words=" << word_count << " perp=" <<exp(loss/word_count) << endl;
  return 0;
}
