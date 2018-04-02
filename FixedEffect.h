#ifndef FIXEDEFFECT_H
#define FIXEDEFFECT_H
#include "TypeDefs.h"
#include <vector>

class FixedEffect {
  long N; // sample size
  short nCatX; //number of categories of X

  std::vector<std::pair<short, short> > localAllocation;
  sVec localCount;
  std::vector<sMap > localClusterDataIndex;
  std::vector<std::vector<double> > localClusterDataCount;

public:
  FixedEffect();
  FixedEffect(sVec& init_lAllo, sVec& xData, short init_C);

  std::pair<short,short>& operator[] (const short index);
  //if index = -1 returns current count, else decreases count by the data index
  sVec getSampleUpdateCount(short index = -1);
  sMap getLocalClusterDataIndex(short cluster);
  std::vector<double> getLocalClusterDataCount(short cluster, short index = -1);

  bool updateAllocation(short index, short allo);

  sVec relabelLocalClusters();
};

#endif
