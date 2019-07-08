#include <zonotope.h>
//#include <opt_pk.h>
#include <opt_oct.h>
#include <cstdlib>
#include <exception>
#include <random>
#include <Python.h>
#include <Eigen/Dense>
#include <time.h>
#include "network.hpp"
#include "interval.hpp"

#define POWERSET_SIZE 1

enum Domain {
    IntervalDomain,
    ZonotopeDomain,
};

extern Domain default_domain;

class timeout_exception: public std::exception {};



typedef struct abstractInput{
    int disjuncts;
    Domain domain;
    Interval property;
} AbstractInput;

typedef struct abstractResult {
    bool verified;
    bool falsified;
    Eigen::VectorXd counterexample;
    Interval interval;
    std::vector<Powerset> layerOutputs;
    int maxInd;
} AbstractResult;

typedef struct policyParams {
    double split_offset;
    int num_disjuncts_left;
    int num_disjuncts_right;
    Domain left_domain;
    Domain right_domain;
    uint dimension;
} PolicyParams;


uint back_prop(const Network &net, const Interval &input, bool &concretize, const int maxInd, const std::vector<Powerset> &layerOutputs);


class StrategyInterpretation {
  public:
    virtual ~StrategyInterpretation();
    virtual int domain_input_size() const = 0;
    virtual int domain_output_size() const = 0;
    virtual int split_input_size() const = 0;
    virtual int split_output_size() const = 0;
    virtual Eigen::VectorXd domain_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const = 0;
    virtual void domain_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const Eigen::VectorXd& counterexample,
        Domain& domain,
        int& num_disjuncts) const = 0;
    virtual Eigen::VectorXd split_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const = 0;
    virtual void split_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const AbstractResult& ar,
        double& split_offset,
        uint& dimension) const = 0;
};

class BayesianStrategy : public StrategyInterpretation {
public:
    constexpr static const float EPSILON = 0.01;

    int domain_input_size() const {
      return 7;
    }

    int domain_output_size() const {
      return 1;
    }

    int split_input_size() const {
      return 7;
    }

    int split_output_size() const {
      return 1;
    }
    Eigen::VectorXd domain_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const;
    void domain_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const Eigen::VectorXd& counterexample,
        Domain& domain,
        int& num_disjuncts) const;
    Eigen::VectorXd split_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const;
    void split_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const AbstractResult& ar,
        double& split_offset,
        uint& dimension) const;

};

class BisectStrategy : public StrategyInterpretation {
public:
    int domain_input_size() const {
      return 1;
    }

    int domain_output_size() const {
      return 1;
    }

    int split_input_size() const {
      return 1;
    }

    int split_output_size() const {
      return 1;
    }

    Eigen::VectorXd domain_featurize(const Network& net, const Interval& input_space,
                              const Eigen::VectorXd& counterexample) const {
      return Eigen::VectorXd(1);
    }

    Eigen::VectorXd split_featurize(const Network& net, const Interval& input_space,
                              const Eigen::VectorXd& counterexample) const {
      return Eigen::VectorXd(1);
    }

    void domain_extract(const Eigen::VectorXd& strategy_output, const Network& net,
                 const Eigen::VectorXd& counterexample, Domain& domain,
                 int& num_disjuncts) const {
        num_disjuncts = POWERSET_SIZE;
        domain = default_domain;
    }

    void split_extract(const Eigen::VectorXd& strategy_output, const Network& net,
        const AbstractResult& ar, double& split_offset, uint& dimension) const {
      const Interval &input_space = ar.interval;
      const Eigen::VectorXd &counterexample = ar.counterexample;
      Eigen::VectorXd center = input_space.get_center();
      uint longest_dim = 0;
      double length = 0; //changed to double

      for (int i = 0; i < net.get_input_size(); i++) {
        if (input_space.upper(i) - input_space.lower(i) > length) {
          longest_dim = i;
          length = input_space.upper(i) - input_space.lower(i);
        }
      }

      dimension = longest_dim;
      split_offset = center(dimension);
    }
};

class GradientStrategy : public StrategyInterpretation {
public:
    int domain_input_size() const {
      return 1;
    }

    int domain_output_size() const {
      return 1;
    }

    int split_input_size() const {
      return 1;
    }

    int split_output_size() const {
      return 1;
    }

    Eigen::VectorXd domain_featurize(const Network& net, const Interval& input_space,
                              const Eigen::VectorXd& counterexample) const {
      return Eigen::VectorXd(1);
    }

    Eigen::VectorXd split_featurize(const Network& net, const Interval& input_space,
        const Eigen::VectorXd& counterexample) const {
      return Eigen::VectorXd(1);
    }

    void domain_extract(const Eigen::VectorXd& strategy_output, const Network& net,
                 const Eigen::VectorXd& counterexample, Domain& domain,
                 int& num_disjuncts) const {
      domain = default_domain;
      num_disjuncts = POWERSET_SIZE;
    }

    void split_extract(const Eigen::VectorXd& strategy_output, const Network& net,
                 const AbstractResult &ar, double& split_offset, uint& dimension) const {
      const std::vector<Powerset> &layerOutputs = ar.layerOutputs;
      const Interval &input_space = ar.interval;
      bool concretize;
      Eigen::VectorXd center = input_space.get_center();
      dimension = back_prop(net, input_space, concretize, ar.maxInd, layerOutputs);
      split_offset = center(dimension);
    }
};

class CounterexampleStrategy : public StrategyInterpretation {
public:
    int domain_input_size() const {
      return 1;
    }

    int domain_output_size() const {
      return 1;
    }

    int split_input_size() const {
      return 1;
    }

    int split_output_size() const {
      return 1;
    }

    Eigen::VectorXd domain_featurize(const Network& net, const Interval& input_space,
                              const Eigen::VectorXd& counterexample) const {
      return Eigen::VectorXd(1);
    }

    Eigen::VectorXd split_featurize(const Network& net, const Interval& input_space,
                              const Eigen::VectorXd& counterexample) const {
      return Eigen::VectorXd(1);
    }

    void domain_extract(const Eigen::VectorXd& strategy_output, const Network& net,
                 const Eigen::VectorXd& counterexample, Domain& domain,
                 int& num_disjuncts) const {
      domain = default_domain;
      num_disjuncts = POWERSET_SIZE;
    }

    void split_extract(const Eigen::VectorXd& strategy_output, const Network& net,
                 const AbstractResult &ar, double& split_offset, uint& dimension) const {
      const Eigen::VectorXd counterexample = ar.counterexample;
      const Eigen::VectorXd center = ar.interval.get_center();
      const Interval input_space = ar.interval;

      bool concretize;
      dimension = back_prop(net, ar.interval, concretize, ar.maxInd, ar.layerOutputs);
      //uint longest_dim = 0;
      //double length = 0; //changed to double

      //for (int i = 0; i < net.get_input_size(); i++) {
      //  if (input_space.upper(i) - input_space.lower(i) > length) {
      //    longest_dim = i;
      //    length = input_space.upper(i) - input_space.lower(i);
      //  }
      //}

      //dimension = longest_dim;
      double width = ar.interval.upper(dimension) - ar.interval.lower(dimension);

      if (counterexample(dimension) > center(dimension)) {
        split_offset = ar.interval.upper(dimension) - width / 3.0;
      } else {
        split_offset = ar.interval.lower(dimension) + width / 3.0;
      }

      //Eigen::VectorXd out = net.evaluate(counterexample);
      //Eigen::VectorXd softmax(out.size());
      //double exp_total = 0.0;
      //for (int i = 0; i < out.size(); i++) {
      //  softmax(i) = std::exp(out(i));
      //  exp_total += softmax(i);
      //}
      //double max = 0;
      //for (int i = 0; i < out.size(); i++) {
      //  softmax(i) /= exp_total;
      //  if (softmax(i) > max && i != ar.maxInd) {
      //    max = softmax(i);
      //  }
      //}
      //double score = max / ar.maxInd;
      //split_offset = center(dimension) + score * (counterexample(dimension) - center(dimension));
    }
};

/*
class RandomStrategy: public StrategyInterpretation {
public:
    RandomStrategy() = default;

    int input_size() const {
      return 1;
    }

    int output_size() const {
      return 1;
    }

    Eigen::VectorXd featurize(const Network& net, const Interval& input_space,
                              const Eigen::VectorXd& counterexample) const {
      Eigen::VectorXd dummy(7);
      return dummy;
    }

    void extract(const Eigen::VectorXd& strategy_output, const Network& net,
                 const AbstractResult &ar, PolicyParams &params) const {
      const Interval &input_space = ar.interval;
      const Eigen::VectorXd &counterexample = ar.counterexample;
      //Find all the dimensions and store their index in posDims
      Eigen::VectorXd center = input_space.get_center();
      std::random_device rd; // obtain a random number from hardware
      std::mt19937 eng(rd()); // seed the generator
      int s = input_space.posDims.size()-1;
      std::uniform_int_distribution<> distr(0, s);
      params.dimension = input_space.posDims[distr(eng)];
      params.split_offset = center(params.dimension);
      params.num_disjuncts_left = POWERSET_SIZE;
      params.num_disjuncts_right = POWERSET_SIZE;
      params.left_domain = default_domain;
      params.right_domain = default_domain;
    }
};
*/

PyObject* eigen_vector_to_python_list(const Eigen::VectorXd&);
PyObject* eigen_matrix_to_python_list(const Eigen::MatrixXd&);
Eigen::VectorXd python_list_to_eigen_vector(PyObject*);

PyObject* create_attack_from_network(const Network&, PyObject*);

bool verify_with_strategy(const Eigen::VectorXd&,
    const Interval&, int max_ind, const Network& net,
    Eigen::VectorXd&, int&,
    const Eigen::MatrixXd&, const Eigen::MatrixXd&,
    const StrategyInterpretation&, double, PyObject*, PyObject*);

