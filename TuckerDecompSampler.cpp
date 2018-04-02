#include "TuckerDecompSampler.h"
#include "FixedEffect.h"
#include "RandomEffect.h"
#include "YKernalFixed.h"
#include "Generate.h"
#include "TypeDefs.h"
#include <vector>
#include <map>
#include <string>
#include <random>
#include <cmath>



void TuckerDecompSampler::oneFeatureUpdate(int feature) {

  for (short group = 0; group < nGroup; ++group) {
    for (short index = 0; index < nSample[group]; ++index) {

      //vector of the other feature's current allocation
      std::vector<short> tmpFeatureAllo(nFeature);
      for (int a_feature = 0; a_feature < nFeature; ++a_feature)
          tmpFeatureAllo[a_feature] =
            featureAllocation[a_feature][group][index];

      PassProbs tmpUpdate =
        featureAllocation[feature].oneLocalUpdateGen(group,index);
      auto nlclust = tmpUpdate.localCount.size();
      auto ngclust = tmpUpdate.globalCount.size();

      //Find probability for the allocation not a new cluster

      //prob of having observed Y for each possible allocation
      std::vector<long double> logKernalProbs(nlclust);
      for (short a_lclust = 0; a_lclust < nlclust; ++a_lclust) {
        if (tmpUpdate.lclustAllo[a_lclust] == -1) {
          //lcluster empty and hence 0 prob, log(prob) <= 0 for all prob
          logKernalProbs[a_lclust] = 1;
        }
        else {
          tmpFeatureAllo[feature] = tmpUpdate.lclustAllo[a_lclust];
          logKernalProbs[a_lclust] = std::log(kernalTracker.getAClusterDist(
            tmpFeatureAllo, yData[group][index], generator));
        }
      }

      std::vector<long double> sampPropProbs(nlclust+1,0);
      for (auto a_lclust = 0; a_lclust < nlclust; ++a_lclust) {
        if (tmpUpdate.localCount[a_lclust] < 1) sampPropProbs[a_lclust] = 0;
        else {
          sampPropProbs[a_lclust] = tmpUpdate.localCount * std::exp(
            tmpUpdate.logGlobalProb[tmpUpdate.lclustAllo[a_lclust]] +
            logKernalProbs[a_lclust]);
        }
      }

      //new local cluster probability
      std::vector<long double> newLocalProb(ngclust);

      //Find the probability for when a new local cluster appears, but not a
      // new global cluster
      std::vector<long double> logNewKernalProbs(ngclust);
      for (short a_gclust = 0; a_gclust < ngclust; ++a_gclust) {
        if (tmpUpdate.globalCount[a_gclust] < 1)
          logNewKernalProbs[a_gclust] = 1;
        else {
          tmpFeatureAllo[feature] = a_gclust;
          logNewKernalProbs[a_gclust] = std::log(kernalTracker.getAClusterDist(
            tmpFeatureAllo, yData[group][index], generator));
        }
      }

      for (auto a_gclust = 0; a_gclust < ngclust; ++a_gclust) {
        if (tmpUpdate.globalCount[a_gclust] < 1) newLocalProb[a_gclust] = 0;
        else {
          newLocalProb[a_gclust] = tmpUpdate.globalCount[a_gclust] * std::exp(
            tmpUpdate.logGlobalProb[a_gclust] + logNewKernalProbs[a_gclust]);
        }
      }

      //new local and global cluster
      long double newLocalnewGlobalProb = betastar[feature] *
        (1/nCatX[feature+1]) *
        kernalTracker.getMarginalDist(yData[group][index]);

      //new local cluster probability
      long double nLCluster = 0;
      for (long double gCount : tmpUpdate.globalCount) nLCluster += gCount;
      for (long double gclustProb : newLocalProb)
        sampPropProbs[nlclust] += gclustProb;
      sampPropProbs[nlclust] += newLocalnewGlobalProb;
      sampPropProbs[nlclust] *= beta[feature][group] /
        (nLCluster + betastar[feature]);

      //draw new allocation
      short newLocalAllo = drawAllocation(sampPropProbs,generator);

      //check if new, if yes then draw new global
      if (newLocalAllo == nlclust) {
        short newGlobalAllo = drawAllocation(newLocalProb.push_back(
          newLocalnewGlobalProb),generator);
        featureAllocation[feature].oneLocalUpdate(group, index, newLocalAllo,
          newGlobalAllo);

        //new global cluster and now need to allocate new kernal
        if (newGlobalAllo == ngclust) {
          std::vector<long double> tmpkProb = kernalTracker.getClusterProb();
          std::vector<long double> tmpkDist =
            kernalTracker.getResponseClustProb(yData[group][index]);

          for (short kClust = 0; kClust < tmpkProb.size(); ++kClust)
            tmpkProb[kClust] *= tmpkDist[kClust];

          short newKernalAllo = drawAllocation(tmpkProb, generator);
          tmpFeatureAllo[feature] = newGlobalAllo;
          kernalTracker.addActiveKernal(tmpFeatureAllo, newKernalAllo,
            group, index);
        }
      }
      else featureAllocation[feature].oneLocalUpdate(
        group, index, newLocalAllo);
    }
  }
  featureAllocation[feature].relabelAllLocalClusters();

  std::vector<short> tmpNLCluster(nGroup);
  for (auto group = 0; group < nGroup; ++group)
    tmpNLCLuster = featureAllocation[feature][group].size();

  for (auto group = 0; group < nGroup; ++group) {
    for (auto a_lclust = 0; a_lclust < tmpNLCluster[group]; ++a_lclust) {

      PassGlobalProbs tmpUpdate =
        featureAllocation.oneGlobalUpdateGen(group, a_lclust);
      short ngclust = tmpUpdate.globalCount.size();
      short ndat = tmpUpdate.dataIndex.size();

      std::map<short, std::vector<short> > tmpFeatureAllo;
      for (auto index : tmpFeatureAllo.dataIndex) {
        std::vector<short> tmpAllo(nFeature);
        for (int feat = 0; feat < nFeature; ++feat) {
          tmpAllo[feat] = featureAllocation[feat][]
          //need to get feature allocation for each response associated with th
          // table.  
        }
      }
    }
  }


}

void TuckerDecompSampler::allFeatureUpdate() {
  for (int feature = 0; feature < nFeature; ++feature)
    oneFeatureUpdate(feature);
  for (auto feature : featureAllocation) feature.relabelGlobalClusters();
}
