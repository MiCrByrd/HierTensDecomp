#ifndef KERNALATTRIBUTES
#define KERNALATTRIBUTES
#include <vector>
#include <utility>

class KernalAttributes {
  short cluster;
  std::vector<std::pair<short,short> > dataIndex;

public:
  KernalAttributes();
  KernalAttributes(short aCluster, short group, short index);
  void addDataIndex(short group, short index);
  short getCluster();
  std::vector<std::pair<short, short> > getIndex();
};

#endif
