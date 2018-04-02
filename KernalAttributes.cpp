#include "KernalAttributes.h"
#include <vector>
#include <utility>


KernalAttributes::KernalAttributes() {

}

KernalAttributes::KernalAttributes(short aCluster, short group, short index) :
  cluster (aCluster) {
  std::pair<short,short> tmpPair {group, index};
  dataIndex.push_back(tmpPair);
}

void KernalAttributes::addDataIndex(short group, short index) {
  std::pair<short, short> tmpPair {group, index};
  dataIndex.push_back(tmpPair);
}

short KernalAttributes::getCluster() {
  return cluster;
}

std::vector<std::pair<short, short> > KernalAttributes::getIndex() {
  return dataIndex;
}
