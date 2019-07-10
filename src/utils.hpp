//
// Created by shankara on 9/13/18.
//

#ifndef PROJECT_UTILS_H
#define PROJECT_UTILS_H

#include <zonotope.h>
#include <Eigen/Dense>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <fstream>

class interval {
  public:
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;
    std::vector<int> posDims;

    interval();
    interval(Eigen::VectorXd, Eigen::VectorXd);

    elina_interval_t** get_elina_interval() const;
    void set_bounds(Eigen::VectorXd, Eigen::VectorXd);
    Eigen::VectorXd get_center() const;
};

interval::interval(Eigen::VectorXd l, Eigen::VectorXd u): lower(l), upper(u) {
  for(int i = 0; i < l.size(); i++) {
    if (upper(i) - lower(i) > 0)
      this->posDims.push_back(i);
  }
}

interval::interval(): lower(Eigen::VectorXd(0)), upper(Eigen::VectorXd(0)) {}

void interval::set_bounds(Eigen::VectorXd l, Eigen::VectorXd u) {
  lower = l;
  upper = u;
}

Eigen::VectorXd interval::get_center() const {
  Eigen::VectorXd ce(lower.size());
  for (int i = 0; i < lower.size(); i++) {
    double l = lower(i);
    double u = upper(i);
    ce(i) = (l + u) / 2.0;
  }
  return ce;
}

interval read_property1(std::string filename, int& dims) {
  std::vector<double> lower;
  std::vector<double> upper;
  std::ifstream in(filename.c_str());
  std::string line;
  while (getline(in, line)) {
    std::istringstream iss(line);
    double l, u;
    iss.get();
    iss >> l;
    iss.get();
    iss.get();
    iss >> u;
    lower.push_back(l);
    upper.push_back(u);
  }

  dims = lower.size();
  Eigen::VectorXd l(dims);
  Eigen::VectorXd u(dims);
  for (int i = 0; i < dims; i++) {
    l(i) = lower[i];
    u(i) = upper[i];
  }

  return interval(l, u);
}


#endif //PROJECT_UTILS_H
