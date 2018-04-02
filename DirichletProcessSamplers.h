#ifndef DP_SAMPLERS
#define DP_SAMPLERS
#include "FixedEffect.h"
#include "RandomEffect.h"
#include "TypeDefs.h"
#include <vector>

//Dirichlet Process
FixedEffect DPSampler(sVec& xData, sVec& init_localAllo, int beta,
  std::vector<double>& beta0, long nIteration);

//Hierarchical Dirichlet Process
RandomEffect HDPSampler(std::vector<FixedEffect>& init_lCluster,
  sMat& init_globalAllo, double betastar, std::vector<double>& beta,
  std::vector<double>& beta0, long nIteration);

std::vector<short> genAllocation(short nSamp, short nCat, int nClust,
  long seed);

sMat genGroupAllocation(short nGroup, std::vector<short> nSamp, short nCat,
int nGClust, std::vector<int> nLClust, long seed);

#endif
