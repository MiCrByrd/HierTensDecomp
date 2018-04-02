#include "FixedEffect.h"
#include "TypeDefs.h"
#include <vector>
#include <algorithm>
#include <utility>

FixedEffect::FixedEffect() {

}

FixedEffect::FixedEffect(sVec& init_lAllo, sVec& xData, short init_C) :
  N(xData.size()), nCatX(init_C) {
  localAllocation.resize(N);

  for (long index = 0; index < N; ++index) {
    localAllocation[index].first = init_lAllo[index];
    localAllocation[index].second = xData[index];
  }

  auto tmpPtr = std::max_element(init_lAllo.begin(),
    init_lAllo.end());
  short maxCluster = *tmpPtr;

  localCount.resize(maxCluster+1,0);
  localClusterDataIndex.resize(maxCluster+1);
  localClusterDataCount.resize(maxCluster+1, std::vector<double>(nCatX,0));

  for (long index = 0; index < N; ++index) {
    auto clust = localAllocation[index].first;
    auto dat = localAllocation[index].second;

    localClusterDataIndex[clust][index] = index;

    localCount[clust] += 1;
    localClusterDataCount[clust][dat] += 1;
  }
}

std::pair<short,short>& FixedEffect::operator[] (const short index) {
  return localAllocation[index];
}

sVec FixedEffect::getSampleUpdateCount(short index) {
  if (index == -1) return localCount;
  else { //for updating the sampler
    localCount[localAllocation[index].first] -= 1;
    return localCount;
  }
}

sMap FixedEffect::getLocalClusterDataIndex(short cluster) {
  return localClusterDataIndex[cluster];
}

std::vector<double> FixedEffect::getLocalClusterDataCount(short cluster,
short index) {
  if (index == -1) return localClusterDataCount[cluster];
  else {
    if (localAllocation[index].first == cluster)
      localClusterDataCount[cluster][localAllocation[index].second] -= 1;
    return localClusterDataCount[cluster];
  }
}

bool FixedEffect::updateAllocation(short index, short allo) {
  localClusterDataIndex[localAllocation[index].first].erase(index);
  localAllocation[index].first = allo;

  if (allo == localCount.size()) { //new cluster
    localCount.push_back(1);

    localClusterDataIndex.push_back(std::map<short, short> {{index, index}});

    localClusterDataCount.push_back(std::vector<double>(nCatX,0));
    localClusterDataCount[nCatX][localAllocation[index].second] += 1;

    return 1;
  }
  else {
    localCount[allo] += 1;
    localClusterDataIndex[allo][index] = index;
    localClusterDataCount[allo][localAllocation[index].second] += 1;

    return 0;
  }
}

sVec FixedEffect::relabelLocalClusters() {
  auto K = localCount.size();

  //find indices of the clusters to be removed
  sVec clusterRemoval;
  clusterRemoval.reserve(K);
  for (short ii = 0; ii < K; ++ii) {
    if (localCount[ii] == 0) clusterRemoval.push_back(ii);
  }

  if (clusterRemoval.size() != 0) {
    //relabel allocations
    for (auto ii = 0; ii < N; ++ii) {
      //find how much to reduce each cluster
      auto indReduce = lower_bound(clusterRemoval.begin(),
        clusterRemoval.end(),localAllocation[ii].first);
      auto reduction = std::distance(clusterRemoval.begin(),indReduce);
      localAllocation[ii].first -= reduction;
    }

    //remove index tracker and counts
    for (short ii = 0; ii < clusterRemoval.size(); ++ii)
      clusterRemoval[ii] -= ii; //for removal, as index will decrease each delete
    for (auto ii : clusterRemoval) {
      localCount.erase(localCount.begin() + ii);
      localClusterDataCount.erase(localClusterDataCount.begin() + ii);
    }
  }

  return clusterRemoval;
}
