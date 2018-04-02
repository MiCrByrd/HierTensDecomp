#ifndef TYPEDEFS_H
#define TYPEDEFS_H
#include <vector>
#include <map>
#include <utility>

typedef std::vector<short> sVec;
typedef std::vector<std::vector<short> > sMat;
typedef std::map<short, short> sMap;
typedef std::vector<std::pair<short, short> > pairVec;
typedef std::vector<long double> probVec;

struct PassProbs {
  //count of data associated with each cluster
  std::vector<long double> localCount;
  //log prob of data value belonging to each local cluster
  std::vector<long double> logLocalProb;

  //if data allocated to new local cluster, find which global cluster the
  // local cluster belongs
  //count of # local clusters associated with each global cluster
  std::vector<long double> globalCount;
  //log prob for the data value allocated to new local cluster belong to each
  // global cluster
  std::vector<long double> logGlobalProb;

  //which global cluster is each local cluster associated with
  std::vector<short> lclustAllo;
  //number of all allocated local clusters for every group
  short countAlloLclust;
};

struct UpdateParams {
  short lclust = -1;
  short gclust = -1;
};

struct PassGlobalProbs {
  //log count of the # of local clusters allocated to each global cluster
  std::vector<long double> globalCount;
  //which data pairs are associated with the local cluster
  sMap dataIndex;
  std::vector<long double> logGlobalProb;
};

#endif
