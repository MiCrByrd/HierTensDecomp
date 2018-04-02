#ifndef YKERNALFIXED
#define YKERNALFIXED
#include "KernalAttributes.h"
#include <vector>
#include <map>
#include <string>
#include <random>

class YKernalFixed {
  //constants
  int C0; //number of y categories
  short nYKernals; //truncation
  short nGroup; // number of groups in the data

  //allocation tracker
  std::vector<short> clusterCount; //clusterCount[i] = #data in clust i
  std::vector<std::vector<std::map<short, short> > > yClusterTracker;

  //Kernals
  std::map<std::string, KernalAttributes> activeKernals;

  //Probability a kernal belongs to a kernal
  std::vector<long double> clusterProbability;
  //Probability distribution over Y for each cluster
  std::vector<std::vector<long double> > clusterDistribution;

  //function to convert a feature allocation vector into a key
  std::string kernalKey(std::vector<short>& latentFeatures);

public:
  YKernalFixed();
  YKernalFixed(int possibleY, short possibleClusters, short the_nGroup);

  //add a Kernal that is associated with a y value
  void addActiveKernal(std::vector<short> alloc, short aCluster,
    short group, short index);
  //remove current allocation for update
  void clearActiveKernals();
  //update all cluster probabilities
  void updateAllKernalProb(std::vector<long double>& newClusterProbs);
  //update a single cluster distribution for vector prior over Dirichlet, alpha
  void updateAllClusterDist(std::vector<std::vector<long double> >& newProbMat);
  //get cluster counts

  std::vector<short> getClusterCount();
  std::vector<long double> getAllClusterProb(); //
  std::vector<long double> getResponseClustProb(int yVal); //
  std::vector<std::map<short, short> > getClusterData(short cluster);
  long double getAClusterDist(std::vector<short>& anAllocation, int yVal,
    std::mt19937_64& gen);
  //marginal as defined as P(Y|\mu) = sum_{i=1}^n pi_n P(Y|\mu_i)
  long double getMarginalDist(int yVal);


};

#endif
