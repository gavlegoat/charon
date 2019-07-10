#include <zonotope.h>

#include <cstdlib>
#include <exception>
#include <random>
#include <Python.h>
#include <Eigen/Dense>
#include <time.h>

#include "network.hpp"
#include "interval.hpp"

#define POWERSET_SIZE 1

/**
 * Represents the choice of underlying domains that can be used with the
 * `Powerset` class.
 */
enum Domain {
  IntervalDomain,
  ZonotopeDomain,
};

/** Set up a default domain to use */
extern Domain default_domain;

/** Used to signal a timeout without messing up a bunch of other code */
class timeout_exception: public std::exception {};

/** An input to the proof and counterexample search. */
typedef struct abstractInput{
  /** The number of disjuncts to use. */
  int disjuncts;
  /** The underlying domain for the powerset. */
  Domain domain;
  /** The input property to analyze. */
  Interval property;
} AbstractInput;

/** A result from the proof and counterexample search. */
typedef struct abstractResult {
  /** Whether or not the property was verified. */
  bool verified;
  /** Whether or not the property was falsified. */
  bool falsified;
  /** The counterexample if falsified is true. */
  Eigen::VectorXd counterexample;
  /** The interval which was analyzed. */
  Interval interval;
  /** The output of each layer of the network. */
  std::vector<Powerset> layerOutputs;
  /** The target class of the robustness property. */
  int maxInd;
} AbstractResult;

/** Parameters produced by a verification policy. */
typedef struct policyParams {
  /** The offset to split the input region at. */
  double split_offset;
  /** The number of disjuncts to use for the left half of the split. */
  int num_disjuncts_left;
  /** The number of disjuncts to use for the right half of the split. */
  int num_disjuncts_right;
  /** The underlying domain to use for the left half of the split. */
  Domain left_domain;
  /** The underlying domain to use for the right half of the split. */
  Domain right_domain;
  /** The dimension to split on. */
  uint dimension;
} PolicyParams;

uint back_prop(const Network &net,
               const Interval &input,
               bool &concretize,
               const int maxInd,
               const std::vector<Powerset> &layerOutputs);

/**
 * An abstract class used to represent the featurization and selection
 * selection functions. This is provided to allow easy changes to the meta-
 * strategy, although only one concrete implementation is given.
 */
class StrategyInterpretation {
  public:
    virtual ~StrategyInterpretation();
    /** The number of features used to choose a domain. */
    virtual int domain_input_size() const = 0;
    /** The number of outputs from the domain strategy. */
    virtual int domain_output_size() const = 0;
    /** The number of features used to choose a partition. */
    virtual int split_input_size() const = 0;
    /** The number of outputs from the partition strategy. */
    virtual int split_output_size() const = 0;

    /**
     * Extract features from the input to a verification policy.
     *
     * \param net The network being analyzed.
     * \param input_space The space being analyzed.
     * \param counterexample The value returned from the counterexample search.
     * \return A set of features to use when choosing a domain.
     */
    virtual Eigen::VectorXd domain_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const = 0;

    /**
     * Choose a domain from the output of the strategy.
     *
     * \param strategy_output The output of the strategy.
     * \param net The network being analyzed.
     * \param counterexample The result of the counterexample search.
     * \param domain A reference to be filled with the domain to use.
     * \param num_disjuncts A reference to be filled with the size of powerset.
     */
    virtual void domain_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const Eigen::VectorXd& counterexample,
        Domain& domain,
        int& num_disjuncts) const = 0;

    /**
     * Extract features from the input to a verification policy.
     *
     * \param net The network being analyzed.
     * \param input_space The space being analyzed.
     * \param counterexample The value returned from the counterexample search.
     * \return A set of features to use when choosing a domain.
     */
    virtual Eigen::VectorXd split_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const = 0;

    /**
     * Choose a partition from the output of the strategy.
     *
     * \param strategy_output The output of the strategy.
     * \param net The network being analyzed.
     * \param counterexample The result of the counterexample search.
     * \param domain A reference to be filled with the domain to use.
     * \param num_disjuncts A reference to be filled with the size of powerset.
     */
    virtual void split_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const AbstractResult& ar,
        double& split_offset,
        uint& dimension) const = 0;
};

/**
 * One particular strategy for using learning. This is the strategy used for
 * evaluation in the paper.
 */
class BayesianStrategy : public StrategyInterpretation {
  public:
    constexpr static const float EPSILON = 0.01;

    int domain_input_size() const {
      return 5;
    }

    int domain_output_size() const {
      return 2;
    }

    int split_input_size() const {
      return 5;
    }

    int split_output_size() const {
      return 2;
    }

    Eigen::VectorXd domain_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const override;
    void domain_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const Eigen::VectorXd& counterexample,
        Domain& domain,
        int& num_disjuncts) const override;
    Eigen::VectorXd split_featurize(
        const Network& net,
        const Interval& input_space,
        const Eigen::VectorXd& counterexample) const override;
    void split_extract(
        const Eigen::VectorXd& strategy_output,
        const Network& net,
        const AbstractResult& ar,
        double& split_offset,
        uint& dimension) const override;

};

/**
 * Convert an Eigen vector to a python list.
 *
 * \param v The vector to convert.
 * \return A python list representation of `v`.
 */
PyObject* eigen_vector_to_python_list(const Eigen::VectorXd& v);

/**
 * Convert an Eigen matrix to a python list.
 *
 * \param m The matrix to convert.
 * \return A python list representation of `m`.
 */
PyObject* eigen_matrix_to_python_list(const Eigen::MatrixXd& m);

/**
 * Convert a python list to an Eigen vector.
 *
 * \param o A python list.
 * \return An Eigen vector representation of `o`.
 */
Eigen::VectorXd python_list_to_eigen_vector(PyObject* o);

/**
 * Create an `IntervalPGDAttack` object from a given network. See interface.py
 * for details.
 *
 * \param n The network under attack.
 * \param o The `initialize_pgd_class` method from interface.py
 * \return An `IntervalPGDAttack` instance for the given network.
 */
PyObject* create_attack_from_network(const Network& n, PyObject* o);

/**
 * Verify a given robustness property using a given strategy.
 *
 * \param original The original point in the input region. Not always necessary.
 * \param property The input region we want to analyze.
 * \param max_ind The target class.
 * \param net The network to analyze.
 * \param counterexample A place to return the counterexample if one is found.
 * \param num_calls A place to return the number of calls to the proof search.
 * \param domain_strategy The strategy matrix for choosing a domain.
 * \param split_strategy The strategy matrix for choosing a partition.
 * \param interp An interpretation for the two strategy matrices.
 * \param timeout A limit to the amount of time used.
 * \param pgdAttack An `IntervalPGDAttack` object for the network.
 * \param pFunc A reference to `run_attack` from interface.py.
 * \exception timeout_exception The analysis timed out.
 * \return `true` if the robustness property holds.
 */
bool verify_with_strategy(const Eigen::VectorXd& original,
                          const Interval& property,
                          int max_ind,
                          const Network& net,
                          Eigen::VectorXd& counterexample,
                          int& num_calls,
                          const Eigen::MatrixXd& domain_strategy,
                          const Eigen::MatrixXd& split_strategy,
                          const StrategyInterpretation& interp,
                          double timeout,
                          PyObject* pgdAttack,
                          PyObject* pFunc);

