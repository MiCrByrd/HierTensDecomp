#include "Generate.h"
#include <random>
#include <vector>
#include <iostream>

// This Code is used to generate random beta and dirichlet variables

double getRBeta(std::vector<double> &alpha, std::mt19937_64 &generator) {
  double rBeta;
  if (alpha.size() > 2) {
    std::cout << "Error getRBeta: Beta random variables take only two parameters.";
    return -1;
  }

  if (alpha.size() < 2) {
    std::cout << "Error getRBeta: Beta random variables require two parameters.";
    return -1;
  }

  for (double ii : alpha) {
    if (ii <= 0) {
      std::cout << "Error getRbeta: parameters must be > 0.";
      return -1;
    }
  }


  if (alpha[0] == alpha[1]) {
    std::gamma_distribution<double> Gamma(alpha[0],1);
    double X, Y;
    X = Gamma(generator);
    Y = Gamma(generator);

    //rBeta ~ Beta(alphaParameters[0],alphaParameters[0])
    rBeta = X / (X + Y);
    return rBeta;
  }
  else {
    std::gamma_distribution<double> Gamma_alpha(alpha[0], 1);
    std::gamma_distribution<double> Gamma_beta(alpha[1],1);

    double X = Gamma_alpha(generator);
    double Y = Gamma_beta(generator);

    //rBeta ~ Beta(alphaParameters[0],alphaParameters[1])
    rBeta = X / (X + Y);
    return rBeta;
  }
}

std::vector<long double> getRDirichlet(std::vector<double> &alpha,
  std::mt19937_64 &generator) {
  if (alpha.size() < 2) {
    std::cout <<
    "Error getRDirichlet: Dirichlet random variables require at least 2 parameters.";
    std::vector<long double> error {-1};
    return error;
  }

  for (double ii : alpha) {
    if (ii <= 0) {
      std::cout << "Error getRDirichlet: parameters must be > 0.";
      std::vector<long double> error {-1};
      return error;
    }
  }

  //For i = 0,...,alpha.size()-1, draw X_i ~ gamma(alpha[i],1)
  std::vector<long double> gammaDraws;
  gammaDraws.reserve(alpha.size());
  for (double eachAlpha : alpha) {
    std::gamma_distribution<long double> Gamma(eachAlpha,1);
    gammaDraws.push_back( Gamma(generator) );
  }

  // Y = (Y_1,...,Y_alpha.size()) ~ Dir(alpha) => Y_i = X_i / sum(X_i)
  std::vector<long double> rDirichlet;
  rDirichlet.reserve(alpha.size());

  long double gammaSum {0};
  for (long double ii : gammaDraws) gammaSum += ii;

  for (long double eachGamma : gammaDraws) {
    rDirichlet.push_back(eachGamma / gammaSum);
  }

  return rDirichlet;
}

long double probDirCat(std::vector<double>& params, short index) {
  long double paramsSum = 0;
  for (auto ii : params) paramsSum += ii;

  return params[index] / paramsSum;
}

long double probDirMnom(std::vector<double>& params, std::vector<short>& counts) {
  long double n = 0;
  for (auto ii : counts) n += ii;
  long double paramsSum = 0;
  for (auto ii : params) paramsSum += ii;

  long double indSum = 0;
  for (auto ii = 0; ii < counts.size(); ++ii)
    indSum += std::lgamma(params[ii] + counts[ii]) - std::lgamma(counts[ii]+1) -
      std::lgamma(params[ii]);

  long double logVal = std::lgamma(n+1) + std::lgamma(paramsSum) -
    std::lgamma(n + paramsSum) +  indSum;

  long double output = std::exp(logVal);
  return output;
}

short drawAllocation(std::vector<long double>& probVec, std::mt19937_64& gen) {
  std::discrete_distribution<> draw(probVec.begin(), probVec.end());
  return draw(gen);
}
