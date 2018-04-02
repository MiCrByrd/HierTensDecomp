#include "RandomEffect.h"
#include "FixedEffect.h"
#include "Generate.h"
#include "TypeDefs.h"
#include <vector>
#include <map>
#include <algorithm>
#include <iterator>
#include <cmath>

RandomEffect::RandomEffect() {

}

RandomEffect::RandomEffect(std::vector<FixedEffect>& init_groupLocalAllo,
sMat& init_globalAllo, int init_nGroup, short init_nCatX,
std::vector<double>& cat_prior) :
  nGroup(init_nGroup), nCatX(init_nCatX) {

  beta0 = cat_prior;
  groupLocalAllocation = init_groupLocalAllo;
  globalAllocation = init_globalAllo;

  short maxCluster(0);
  for (auto group : globalAllocation) {
      auto tmpPtr = std::max_element(group.begin(),group.end());
      auto tmpMaxCluster = *tmpPtr;
      if (tmpMaxCluster > maxCluster) maxCluster = tmpMaxCluster;
  }

  globalCount.resize(maxCluster+1,0);
  for (auto group : globalAllocation) {
    for (auto index : group) {
      globalCount[index] += 1;
    }
  }

  globalClusterDataCount.resize(maxCluster+1, beta0);
  for (int group = 0; group < nGroup; ++group) {
    for (short lclust_index = 0; lclust_index < globalAllocation[group].size();
    ++lclust_index) {
      std::vector<double> tmpV =
        groupLocalAllocation[group].getLocalClusterDataCount(lclust_index);
      for (auto aCat = 0; aCat < nCatX; ++aCat)
        globalClusterDataCount[lclust_index][aCat] += tmpV[aCat];
    }
  }
}

std::vector<short>& RandomEffect::operator[] (const short group) {
  return globalAllocation[group];
}

std::vector<short> RandomEffect::getGlobalCount() {
  return globalCount;
}

PassProbs RandomEffect::oneLocalUpdateGen(int group, short index) {

  PassProbs output;

  //get local log counts for each possible cluster; 1000 means 0 prob
  auto tmpCount = groupLocalAllocation[group].getSampleUpdateCount(index);
  output.localCount = std::vector<long double>(tmpCount.begin(),tmpCount.end());
  short nlClust = tmpCount.size();

  //remove the data corresponding the the group and index
  auto tmpDatPair = groupLocalAllocation[group][index];
  globalClusterDataCount[globalAllocation[group][tmpDatPair.first]]
    [tmpDatPair.second] -= 1;

  //this would imply that the index was last element in the local cluster, and
  // thus would empty the local cluster and hence decrease the global by 1
  if (output.localCount[tmpDatPair.first] < 1)// i.e. == 0
    globalCount[globalAllocation[group][tmpDatPair.first]] -= 1;

  auto ngClust = globalCount.size();
  output.globalCount =
    std::vector<long double>(globalCount.begin(),globalCount.end());

  //log of index belonging to each global cluster (log(prob) < 0)
  output.logGlobalProb.resize(ngClust);
  for (auto a_gclust = 0; a_gclust < ngClust; ++a_gclust) {
    if (globalCount[a_gclust] == 0) output.logGlobalProb[a_gclust] = 1;
    else {
      output.logGlobalProb[a_gclust] =
        std::log(probDirCat(globalClusterDataCount[a_gclust],
          tmpDatPair.second));
    }
  }

  output.lclustAllo.resize(nlClust);
  output.logLocalProb.resize(nlClust);
  for (auto a_lclust = 0; a_lclust < nlClust; ++a_lclust) {
    if (output.localCount[a_lclust] < 1) {
      output.lclustAllo[a_lclust] = -1; //allocation can't be -1
      output.logLocalProb[a_lclust] = 1;
    }
    else {
      output.lclustAllo[a_lclust] = globalAllocation[group][a_lclust];
      output.logLocalProb[a_lclust] =
        output.logGlobalProb[output.lclustAllo[a_lclust]];
    }
  }

  output.countAlloLclust = 0;
  for (auto ii : globalCount) output.countAlloLclust += ii;

  return output;
}

void RandomEffect::oneLocalUpdate(int group, short index, short lclust,
short gclust) {

  bool newlCluster =
    groupLocalAllocation[group].updateAllocation(index, lclust);

  if (newlCluster) {
    globalAllocation[group].push_back(gclust);

    if (gclust == globalCount.size()) { //new global cluster
      globalCount.push_back(1);
      globalClusterDataCount.push_back(beta0);
    }
    else {
      globalCount[gclust] += 1;
    }

    globalClusterDataCount[gclust][
      groupLocalAllocation[group][index].second] += 1;
  }

  else {
    auto tmpGClust = globalAllocation[group][lclust];
    globalClusterDataCount[tmpGClust][
      groupLocalAllocation[group][index].second] += 1;
  }
}

//assumes lCluster is not empty
PassGlobalProbs RandomEffect::oneGlobalUpdateGen(int group, short lCluster) {

  auto L = globalCount.size(); //number of possible global clusters
  PassGlobalProbs output;

  globalCount[globalAllocation[group][lCluster]] -= 1;
  output.globalCount =
    std::vector<long double>(globalCount.begin(),globalCount.end());

  output.dataIndex =
    groupLocalAllocation[group].getLocalClusterDataIndex(lCluster);

  std::vector<double> tmpLClusterData =
    groupLocalAllocation[group].getLocalClusterDataCount(lCluster);
  for (auto a_xcat = 0; a_xcat < nCatX; ++a_xcat)
    globalClusterDataCount[globalAllocation[group][lCluster]][a_xcat] -=
      tmpLClusterData[a_xcat];

  output.logGlobalProb.resize(L,0);
  for (auto a_gclust = 0; a_gclust < L; ++a_gclust) {
    if (globalCount[a_gclust] == 0) output.logGlobalProb[a_gclust] = 1;
    else {
      for (auto a_xcat = 0; a_xcat < nCatX; ++a_xcat) {
        if (tmpLClusterData[a_xcat] > .5) //i.e. not 0
          output.logGlobalProb[a_gclust] += tmpLClusterData[a_xcat] *
            std::log(probDirCat(globalClusterDataCount[a_gclust],a_xcat));
      }
    }
  }

  return output;
}

void RandomEffect::oneGlobalUpdate(int group, short lCluster, short gCluster) {
  globalAllocation[group][lCluster] = gCluster;
  std::vector<double> tmpdat =
    groupLocalAllocation[group].getLocalClusterDataCount(lCluster);

  if (gCluster == globalCount.size()) {
    globalCount.push_back(1);
    std::vector<double> tmpdatwprior(nCatX);
    for (short cat = 0; cat < nCatX; ++cat)
      tmpdatwprior[cat] = tmpdat[cat] + beta0[cat];
    globalClusterDataCount.push_back(tmpdatwprior);
  }
  else {
    globalCount[gCluster] += 1;
    for (short a_xcat = 0; a_xcat < nCatX; ++a_xcat)
      globalClusterDataCount[gCluster][a_xcat] += tmpdat[a_xcat];
  }
}

//note all global counts/data allocation are accounted for in abover fns
void RandomEffect::relabelAllLocalClusters() {
  for (int group = 0; group < nGroup; ++group) {
    sVec relabelAGroup = groupLocalAllocation[group].relabelLocalClusters();
    for (auto empty_lclust : relabelAGroup) {
      globalAllocation[group].erase(globalAllocation[group].begin() +
        empty_lclust);
    }
  }
}

void RandomEffect::relabelGlobalClusters() {
  sVec globalRemove;
  auto L = globalCount.size();
  globalRemove.reserve(L);

  for (auto a_gclust = 0; a_gclust < L; ++a_gclust)
    if (globalCount[a_gclust] == 0) globalRemove.push_back(a_gclust);

  for (int group = 0; group < nGroup; ++group) {
    auto K = globalAllocation[group].size();
    for (short a_lclust = 0; a_lclust < K; ++a_lclust) {
      auto indReduce = lower_bound(globalRemove.begin(), globalRemove.end(),
        globalAllocation[group][a_lclust]);
      auto reduction = std::distance(globalRemove.begin(),indReduce);
      globalAllocation[group][a_lclust] -= reduction;
    }
  }

  for (auto ii = 0; ii < globalRemove.size(); ++ii)
    globalRemove[ii] -= ii;
  for (auto ii : globalRemove) {
    globalCount.erase(globalCount.begin() + ii);
    globalClusterDataCount.erase(globalClusterDataCount.begin() + ii);
  }
}
