#ifndef TUCKERDECOMPSAMPLER_H
#define TUCKERDECOMPSAMPLER_H
#include "FixedEffect.h"
#include "RandomEffect.h"
#include "YKernalFixed.h"
#include "Generate.h"
#include "TypeDefs.h"
#include <vector>
#include <map>
#include <string>
#include <random>

class TuckerDecompSampler {
  std::mt19937_64 generator;

  //dimensions
  int nGroup; // number of groups J
  int nFeature; //number of features M
  sVec nSample; //[group] number of observations for each group N
  sVec nCatX; //[feature] number of categories in each dim, with C[0] #y cat C

  //data for each group, where first dimension is the group
  sMat yData; //[group][index]

  //priors (to be elaborated on more, but for now good enough)
  std::vector<std::vector<int> > beta0; //[feature][nCatX[feature+1]]
  std::vector<std::vector<int> > beta; //[feature][group]
  std::vector<int> betastar; //[feature]

  int prior = 1;

  std::vector<RandomEffect> featureAllocation;
  YKernalFixed kernalTracker;

  void oneFeatureUpdate(int feature); //feat
  void allFeatureUpdate();
  void updateKernals();
  void updateKernalProb();

public:
  GibbsSampler();
  GibbsSampler(sVec& init_C, std::vector<sMat >& features, sMat& response,
    std::vector<RandomEffect>& init_featAllo, short maxKernalClust);

  void update();
  long double getMarginalYProb();
  long double getPredictiveProb(std::vector<std::vector<short> >& newX);
};

#endif
