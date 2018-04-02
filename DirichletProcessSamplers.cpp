#include "DirichletProcessSamplers.h"
#include "FixedEffect.h"
#include "RandomEffect.h"
#include "TypeDefs.h"
#include "Generate.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

FixedEffect DPSampler(sVec& xData, sVec& init_localAllo, int beta,
  std::vector<double>& beta0, long nIteration) {

  std::mt19937_64 generator(
    std::chrono::system_clock::now().time_since_epoch().count());

  FixedEffect theAllocation(init_localAllo, xData, beta0.size());
  short nCat = beta0.size();
  long N = xData.size();

  for (long iter = 0; iter < nIteration; ++iter) {
    for (long index = 0; index < N; ++index) {

      auto tmpData = theAllocation[index];
      sVec tmpCount = theAllocation.getSampleUpdateCount(index);
      auto nClust = tmpCount.size();

      std::vector<long double> tmpProb(nClust+1, 0);
      for (auto a_clust = 0; a_clust < nClust; ++a_clust) {
        if (tmpCount[a_clust] > 0) {
          std::vector<double> tmpParam=
            theAllocation.getLocalClusterDataCount(a_clust, index);
          for (short cat = 0; cat < nCat; ++cat) tmpParam[cat] += beta0[cat];
          tmpProb[a_clust] =
            tmpCount[a_clust] * probDirCat(tmpParam, tmpData.second);
        }
      }
      tmpProb[nClust] = beta / nCat;

      short newAllo = drawAllocation(tmpProb, generator);
      theAllocation.updateAllocation(index, newAllo);
    }

    theAllocation.relabelLocalClusters();
    if ((iter % 5000) == 0) std::cout << iter << " Done" << "\n";
  }

  return theAllocation;
}



RandomEffect HDPSampler(std::vector<FixedEffect>& init_lCluster,
sMat& init_globalAllo, double betastar, std::vector<double>& beta,
std::vector<double>& beta0, long nIteration) {

  std::mt19937_64 generator(
    std::chrono::system_clock::now().time_since_epoch().count());

  short nGroup = init_globalAllo.size();
  short nCat = beta0.size();

  std::vector<short> groupSize(nGroup);
  for (short group = 0; group < nGroup; ++group)
    groupSize[group] = init_globalAllo[group].size();

  RandomEffect groupAllocation(init_lCluster, init_globalAllo,
    init_globalAllo.size(), beta0.size(), beta0);

  for (long iter = 0; iter < nIteration; ++iter) {

    //update Local Clusters
    for (short group = 0; group < nGroup; ++group) {
      for (short index = 0; index < groupSize[group]; ++index) {
        //update counts and probabilities
        PassProbs tmpLocalUp = groupAllocation.oneLocalUpdateGen(group,index);
        short nlclust = tmpLocalUp.localCount.size();
        short ngclust = tmpLocalUp.globalCount.size();

        std::vector<long double> tmpUpdateParam(nlclust+1,0);
        //parameters to draw for established local clusters in the group
        for (auto a_lclust = 0; a_lclust < nlclust; ++a_lclust) {
          if (tmpLocalUp.localCount[a_lclust] > .5) {
            tmpUpdateParam[a_lclust] = tmpLocalUp.localCount[a_lclust] *
              std::exp(tmpLocalUp.logGlobalProb
                [tmpLocalUp.lclustAllo[a_lclust]]);
          }
        }

        //new local cluster
        std::vector<long double> tmpEachGlobalProb(ngclust+1,0);
        //probability for each global cluster
        for (auto a_gclust = 0; a_gclust < ngclust; ++a_gclust) {
          if (tmpLocalUp.globalCount[a_gclust] > .5) {
            tmpEachGlobalProb[a_gclust] = tmpLocalUp.globalCount[a_gclust] *
              std::exp(tmpLocalUp.logGlobalProb[a_gclust]);
          }
        }
        tmpEachGlobalProb[ngclust] = betastar / nCat;

        //proportional probability of new local cluster
        for (auto propProb : tmpEachGlobalProb)
          tmpUpdateParam[nlclust] += propProb;
        long double tmpCountnLocalClust;
        for (auto gclust_count : tmpLocalUp.globalCount)
          tmpCountnLocalClust += gclust_count;
        tmpUpdateParam[nlclust] *=
          beta[group] / (tmpCountnLocalClust + betastar);

        //draw local Allocation
        short localUpdate = drawAllocation(tmpUpdateParam, generator);

        // new local allocation, draw global allocation for it
        if (localUpdate == nlclust) {
          short globalUpdate = drawAllocation(tmpEachGlobalProb, generator);
          groupAllocation.oneLocalUpdate(
            group, index, localUpdate, globalUpdate);
        }
        else groupAllocation.oneLocalUpdate(group, index, localUpdate);
      }
    }
    groupAllocation.relabelAllLocalClusters();

    for (auto group = 0; group < nGroup; ++group) {
      for (auto a_lclust = 0; a_lclust < groupAllocation[group].size();
      ++a_lclust) {
        PassGlobalProbs tmpGlobalUp =
          groupAllocation.oneGlobalUpdateGen(group, a_lclust);
        short ngclust = tmpGlobalUp.globalCount.size();

        std::vector<long double> tmpUpdateParam(ngclust+1,0);
        for (short a_gclust = 0; a_gclust < ngclust; ++a_gclust) {
          if (tmpGlobalUp.globalCount[a_gclust] > .5) { //i.e. not 0
            tmpUpdateParam[a_gclust] = tmpGlobalUp.globalCount[a_gclust] *
              std::exp(tmpGlobalUp.logGlobalProb[a_gclust]);
          }
        }
        tmpUpdateParam[ngclust] = betastar / nCat;

        short globalUpdate = drawAllocation(tmpUpdateParam, generator);
        groupAllocation.oneGlobalUpdate(group, a_lclust, globalUpdate);
      }
    }
    groupAllocation.relabelGlobalClusters();

    if ((iter % 100) ==  0) {
      std::cout << iter << " Done" << "\n";
      std::vector<short> tmpClustCount = groupAllocation.getGlobalCount();
      for (short ii : tmpClustCount) std::cout << ii << " ";
      std::cout << "\n";
    }
  }

  return groupAllocation;
}

std::vector<short> genAllocation(short nSamp, short nCat, int nClust,
long seed) {

  std::mt19937_64 generator(seed);

  std::vector<double> tmpParam(nClust, 1);
  std::vector<long double> clusterProb = getRDirichlet(tmpParam, generator);

  std::vector<std::vector<long double> > clusterDist(nClust);
  for (int a_clust = 0; a_clust < nClust; ++a_clust) {
    std::vector<double> tmpDrawParam(nCat,1);
    clusterDist[a_clust] = getRDirichlet(tmpDrawParam, generator);
  }

  std::vector<short> dataGen(nSamp);
  for (short a_dat = 0; a_dat < nSamp; ++a_dat) {
    auto tmpCluster = drawAllocation(clusterProb, generator);
    dataGen[a_dat] = drawAllocation(clusterDist[tmpCluster],generator);
  }

  return dataGen;
}

sMat genGroupAllocation(short nGroup, std::vector<short> nSamp, short nCat,
int nGClust, std::vector<int> nLClust, long seed) {

  sMat dataGen;
  dataGen.resize(nGroup);
  std::mt19937_64 generator(seed);

  std::vector<double> tmpParam(nGClust, 1);
  std::vector<long double> globalClusterProb =
    getRDirichlet(tmpParam, generator);

  std::vector<std::vector<long double> > globalClusterDist(nGClust);
  std::vector<double> tmpCatParam(nCat, 1);
  for (auto gclust = 0; gclust < nGClust; ++gclust)
    globalClusterDist[gclust] = getRDirichlet(tmpCatParam, generator);

  for (auto group = 0; group < nGroup; ++group) {
    std::vector<double> tmpLocalParam(nLClust[group], 1);

    std::vector<long double> localClusterProb =
      getRDirichlet(tmpLocalParam, generator);

    std::vector<short> lClusterAllocation(nLClust[group]);
    for (auto lclust = 0; lclust < nLClust[group]; ++lclust)
      lClusterAllocation[lclust] = drawAllocation(globalClusterProb, generator);

    std::vector<short> tmpSampleAllocation(nSamp[group]);
    for (auto index = 0; index < nSamp[group]; ++index) {
      short tmpLAllo = drawAllocation(localClusterProb, generator);
      tmpSampleAllocation[index] =
        drawAllocation(globalClusterDist[lClusterAllocation[tmpLAllo]],
          generator);
    }
    dataGen[group] = tmpSampleAllocation;
  }

  return dataGen;
}
