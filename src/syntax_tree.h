#pragma once
#include <vector>
#include <string>
#include "cnn/dict.h"
//#include "utils.h"

using namespace std;
using namespace cnn;

typedef int WordId;

class SyntaxTree {
public:
  SyntaxTree();
  SyntaxTree(string tree, Dict* dict, const SyntaxTree* parent = NULL);

  bool IsTerminal() const;
  unsigned NumChildren() const;
  unsigned NumNodes() const;
  unsigned MaxBranchCount() const;
  unsigned MinDepth() const;
  unsigned MaxDepth() const;
  WordId label() const;
  unsigned id() const;
  vector<WordId> GetTerminals() const;

  SyntaxTree& GetChild(unsigned i);
  const SyntaxTree& GetChild(unsigned i) const;
  const SyntaxTree* const GetParent() const;

  string ToString() const;
  unsigned AssignNodeIds(unsigned start = 0);
private:
  Dict* dict;
  WordId label_;
  unsigned id_;
  const SyntaxTree* parent;
  vector<SyntaxTree> children;
};

ostream& operator<< (ostream& stream, const SyntaxTree& tree);
