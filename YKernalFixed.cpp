#include "YKernalFixed.h"
#include "KernalAttributes.h"
#include "Generate.h"
#include <vector>
#include <map>
#include <string>
#include <iterator>
#include <random>
#include <algorithm>
#include <utility>

std::string YKernalFixed::kernalKey(std::vector<short>& latentFeatures) {
  std::string key("");
  for (auto ii : latentFeatures) key += std::to_string(ii) + " ";
  return key;
}

YKernalFixed::YKernalFixed() {

}

YKernalFixed::YKernalFixed(int possibleY, short possibleClusters,
short the_nGroup) :
  C0(possibleY), nYKernals(possibleClusters), nGroup(the_nGroup),
  clusterCount(possibleClusters,0),
  yClusterTracker(the_nGroup,
    std::vector<std::map<short, short> >(possibleClusters)) {
}

void YKernalFixed::addActiveKernal(std::vector<short> alloc, short aCluster,
short group, short index) {
  //key for the kernal
  std::string aKey = kernalKey(alloc);

  //create if not present, update data values if it is
  if (activeKernals.count(aKey) == 0) {
    KernalAttributes tmpHolder (aCluster, group, index);
    activeKernals[aKey] = tmpHolder;
    clusterCount[aCluster] += 1;
  }
  else {
    activeKernals[aKey].addDataIndex(group, index);
  }

  yClusterTracker[group][aCluster][index] = index;
}

void YKernalFixed::clearActiveKernals() {
  activeKernals.clear();
  for (auto group : yClusterTracker)
    for (auto yClust : group) yClust.clear();
  for (short ii = 0; ii < nYKernals; ++ii) {
    clusterCount[ii] = 0;
  }
}

void YKernalFixed::updateAllKernalProb(std::vector<long double>& newClusterProbs)
{
    clusterProbability = newClusterProbs;
}

void YKernalFixed::updateAllClusterDist(std::vector<std::vector<long double> >&
newProbMat) {
  clusterDistribution = newProbMat;
}

std::vector<short> YKernalFixed::getClusterCount() {
  return clusterCount;
}

std::vector<long double> YKernalFixed::getAllClusterProb() {
  return clusterProbability;
}

std::vector<long double> YKernalFixed::getResponseClustProb(int yVal) {
  std::vector<long double> theOut(nYKernals);
  for (short kernal = 0; kernal < nYKernals; ++kernal)
    theOut[kernal] = clusterDistribution[kernal][yVal];
  return theOut;
}

std::vector<std::map<short, short> >YKernalFixed::getClusterData(short cluster){
  std::vector<std::map<short, short> > theOut(nGroup);
  for (short group = 0; group < nGroup; ++group)
    theOut[group] = yClusterTracker[group][cluster];
  return theOut;
}

long double YKernalFixed::getAClusterDist(std::vector<short>& anAllocation,
  int yVal, std::mt19937_64& gen) {

  std::string tmpKey = kernalKey(anAllocation);
  short cluster;
  if (activeKernals.count(tmpKey) == 0) {
    cluster = drawAllocation(clusterProbability, gen);
  }
  else  cluster = activeKernals[tmpKey].getCluster();

  return clusterDistribution[cluster][yVal];
}

long double YKernalFixed::getMarginalDist(int yVal) {
  long double marginalProb = 0;
  for (auto ii = 0; ii < clusterProbability.size(); ++ii)
    marginalProb += clusterProbability[ii] * clusterDistribution[ii][yVal];
  return marginalProb;
}
