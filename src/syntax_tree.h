#pragma once
#include <vector>
#include <string>
#include "cnn/dict.h"

using namespace std;
using namespace cnn;

typedef int WordId;
class SyntaxTreeIterator;

class SyntaxTree {
friend class SyntaxTreeIterator;
public:
  SyntaxTree();
  SyntaxTree(string tree, Dict* dict, SyntaxTree* parent = nullptr);

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

  SyntaxTreeIterator begin();
  const SyntaxTreeIterator begin() const;
  SyntaxTreeIterator end();
  const SyntaxTreeIterator end() const;

  string ToString() const;
  unsigned AssignNodeIds(unsigned start = 0);
private:
  Dict* dict;
  WordId label_;
  unsigned id_;
  SyntaxTree* parent;
  SyntaxTree* next_sibling;
  vector<SyntaxTree> children;
};

class SyntaxTreeIterator {
public:
  SyntaxTreeIterator(const SyntaxTree* root, SyntaxTree* current = nullptr);
  bool operator==(const SyntaxTreeIterator& rhs) const;
  bool operator!=(const SyntaxTreeIterator& rhs) const;
  SyntaxTree& operator*() const;
  SyntaxTreeIterator& operator++(); //preincrement ++i
  SyntaxTreeIterator operator++(int); //postincrement i++
private:
  const SyntaxTree* root;
  SyntaxTree* current;
};

ostream& operator<< (ostream& stream, const SyntaxTree& tree);
