#ifndef GENERATE
#define GENERATE
#include <random>
#include <vector>

//draw beta rv
double getRBeta(std::vector<double> &alpha, std::mt19937_64 &generator);
//draw dirichlet rv
std::vector<long double> getRDirichlet(std::vector<double> &alpha,
  std::mt19937_64 &generator);
//prob of categorical distribution with params integrated out for one draw
long double probDirCat(std::vector<double>& params, short index);
//prob of n trials of a dirichlet multinomial distribution
long double probDirMnom(std::vector<double>& params, std::vector<short>& counts);
//categorical draw
short drawAllocation(std::vector<long double>& probVec, std::mt19937_64& gen);

#endif
