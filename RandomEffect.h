#ifndef RANDOMEFFECT_H
#define RANDOMEFFECT_H
#include "FixedEffect.h"
#include "TypeDefs.h"
#include "Generate.h"
#include <vector>
#include <map>

class RandomEffect {
  int nGroup; //# groups
  short nCatX; //# categories the variable can take
  std::vector<double> beta0;

  std::vector<FixedEffect> groupLocalAllocation; //[group][index]
  std::vector<std::vector<short> > globalAllocation; //[group][local cluster]
  std::vector<short> globalCount; //[global cluster]
  std::vector<std::vector<double> > globalClusterDataCount; //[globalcluster][xval]

public:
  RandomEffect();
  RandomEffect(std::vector<FixedEffect>& init_groupLocalAllo,
    sMat& init_globalAllo, int init_nGroup, short init_nCatX,
    std::vector<double>& cat_prior);

  std::vector<short>& operator[] (const short group);
  std::vector<short> getGlobalCount();

  //Updates for the local clusters
  PassProbs oneLocalUpdateGen(int group, short index);
  void oneLocalUpdate(int group, short index, short lclust, short gclust = -1);

  //Updates for the global clusters
  PassGlobalProbs oneGlobalUpdateGen(int group, short lCluster);
  void oneGlobalUpdate(int group, short lCluster, short gCluster);

  //Relabel local and global cluster to account for empty clusters
  void relabelAllLocalClusters();
  void relabelGlobalClusters();
};

#endif
