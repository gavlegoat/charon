/* Training module for Charon
 */

#include "strategy.hpp"
#include "powerset.hpp"
#include "network.hpp"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <time.h>
#include <map>
#include <Python.h>
#include <zonotope.h>
#include <Eigen/Dense>
#include <bayesopt/bayesopt.hpp>
#ifdef TACC
#include <mpi.h>
#else
#include <mpi/mpi.h>
#endif

#define TIMEOUT 1000
#define PENALTY 2

/** An enum describing all the networks used for training. */
namespace Networks {
  enum Nets {
    ACAS_XU,
  };
  static const Nets All[] = {ACAS_XU};
}

#ifdef CHARON_HOME
/** A map from network descriptors to network objects. */
static std::map<Networks::Nets, Network> networks = {
  {Networks::Nets::ACAS_XU,
    read_network(CHARON_HOME + std::string("example/acas_xu_1_1.txt"))}
};
#endif

/** A map from network descriptors to the associated PGD methods. */
static std::map<Networks::Nets, PyObject*> networkAttacks;

typedef struct property {
  Interval itv;
  Networks::Nets net;
} Property;

/** Holds information needed by the Bayesian optimization procedure. */
class CegarOptimizer: public bayesopt::ContinuousModel {
  private:
    /** A set of properties to train with. */
    std::vector<Property> properties;
    /** The meta-strategy we are training. */
    const StrategyInterpretation& strategy_interp;
    /** The number of MPI processes we're using. */
    int world_size;

  public:
    /**
     * Construct an optimizer.
     *
     * \param input_dimension The number of parameters to train.
     * \param params A set of parameters to use for training.
     * \param si The meta-strategy.
     * \param ws The number of MPI processes.
     * \param prop_file A file containing several training properties.
     */
    CegarOptimizer(size_t input_dimension, bayesopt::Parameters params,
        const StrategyInterpretation& si, int ws, std::string prop_file):
      bayesopt::ContinuousModel(input_dimension, params), strategy_interp(si),
      world_size(ws) {
        std::string line;
        std::ifstream fd(prop_file);
        // Load a set of properties. Each line in prop_file is a filename
        // for a file containing some training property. These filenames
        // should be relative to the Charon home directory.
        std::vector<std::string> prop_files;
        while(std::getline(fd, line))
           prop_files.push_back(line);

        for (std::string& s : prop_files) {
          std::vector<std::string> results;
          std::stringstream iss(s);
          for(std::string s; iss >> s;) {
            results.push_back(s);
          }
          Property p;
          p.itv = Interval(CHARON_HOME + std::string(results[0]));
          p.net = Networks::Nets::ACAS_XU;
          properties.push_back(p);
        }

      }

    /** Get the training properties of this class. */
    const std::vector<Property>& getProperties() const {
      return this->properties;
    }

    /**
     * Determine how good a given strategy is.
     *
     * \param query The strategy to evaluate.
     * \return A score for the strategy.
     */
    double evaluateSample(const boost::numeric::ublas::vector<double>& query) {
      // We should only get into this call when world_rank = 0
      // There might be a more efficient way to convert to an Eigen vector
      int numProperties, propertiesToAssign, propertiesEvaluated = 0;
      // Split the strategy vector into two matrices, one for choosing a domain
      // and one for choosing a partition.
      int dos = strategy_interp.domain_output_size();
      int dis = strategy_interp.domain_input_size();
      int sos = strategy_interp.split_output_size();
      int sis = strategy_interp.split_input_size();
      Eigen::MatrixXd domain_strat(dos, dis);
      Eigen::MatrixXd split_strat(sos, sis);
      for (int i = 0; i < query.size(); i++) {
        if (i < dos * dis) {
          domain_strat(i / dis, i % dis) = query(i);
        } else {
          split_strat((i - dos*dis) / sis, (i - dos*dis) % sis) = query(i);
        }
      }
      std::cout << "Evaluating: " << domain_strat << std::endl;
      std::cout << "and: " << split_strat << std::endl;
      // For some given time budget (per property), see how many properties
      // we can verify
      int count = 0;
      double total_time = 0.0;
      int i = 0;
      numProperties = properties.size();
      propertiesToAssign = numProperties;
      int N = (world_size-1 > numProperties) ? numProperties : world_size-1;
      for (i = 0; i < N; i++) {
        //Distribute properties to available workers
        this->sendProperty(this->properties[i].net, i, i+1, query);
        propertiesToAssign--;
      }
      while(propertiesEvaluated < numProperties) {
        int solved, source;
        MPI_Status status;
        MPI_Recv(&solved, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG,
            MPI_COMM_WORLD, &status);
        double elapsed;
        MPI_Recv(&elapsed, 1, MPI_DOUBLE, status.MPI_SOURCE, MPI_ANY_TAG,
            MPI_COMM_WORLD, &status);
        if (solved) {
          // The property was either verified or falsified.
          count++;
          total_time += elapsed;
        } else {
          // The verificatino timed out.
          total_time += PENALTY * TIMEOUT;
        }
        propertiesEvaluated++;
        if (propertiesToAssign > 0) {
          // If we still have properties to evaluate, send one to the thread
          // which we just got a result from.
          source = status.MPI_SOURCE;
          this->sendProperty(this->properties[i].net, i, source, query);
          i++, propertiesToAssign--;
        }
      }

      return total_time;
    }

    bool checkReachability(const boost::numeric::ublas::vector<double>& query) {
      // If we need constraints besides a bounding box we can put them here.
      return true;
    }

  private:
    void sendProperty(const Networks::Nets netId, int property,
        int worker, const boost::numeric::ublas::vector<double>& query) {
      std::cout << "Sending network " << netId << " and property: " << property
        << " to worker: " << worker << std::endl;
      MPI_Send(&netId, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
      int propertySize = properties[property].itv.lower.size();
      int querySize = query.size();
      int msgSize = querySize + 2 * propertySize;
      std::vector<double> strategyAndProperty(querySize + 2 * propertySize);
      for (int j = 0; j < querySize; j++) {
        strategyAndProperty[j] = query[j];
      }
      for (int j = 0; j < propertySize; j++) {
        strategyAndProperty[querySize + j] = properties[property].itv.lower[j];
        strategyAndProperty[querySize + propertySize + j] =
          properties[property].itv.upper[j];
      }
      std::cout << "Sending: " << std::endl;
      MPI_Send(&strategyAndProperty[0], msgSize, MPI_DOUBLE, worker, 0,
          MPI_COMM_WORLD);
      std::cout << "Sent!" << std::endl;
    }
};

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Usage: ./learn <property-file>" << std::endl;
    std::abort();
  }

  std::string benchmarks = argv[1];
  MPI_Init(NULL, NULL);
#ifdef CHARON_HOME
  std::string cwd = CHARON_HOME;
#else
  std::cout << "CHARON_HOME is undefined. If you compiled with the " <<
    "provided CMake file this shouldn't happen, otherwise set " <<
    "CHARON_HOME" << std::endl;
  std::abort();
#endif

  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::cout << "world size: " << world_size << std::endl;

  Py_Initialize();
  PyEval_InitThreads();

  PyGILState_STATE gstate = PyGILState_Ensure();

  char s[5] = "path";
  PyObject* sysPath = PySys_GetObject(s);
  PyObject* newElem = PyString_FromString((cwd + "/src").c_str());
  PyList_Append(sysPath, newElem);
  PySys_SetObject(s, sysPath);
  PyObject* pName = PyString_FromString("interface");
  PyObject* pModule = PyImport_Import(pName);
  Py_DECREF(pName);
  PyObject* pAttackInit = PyObject_GetAttrString(
      pModule, "initialize_pgd_class");
  if (!pAttackInit || !PyCallable_Check(pAttackInit)) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    Py_XDECREF(pAttackInit);
    Py_DECREF(pModule);
    throw std::runtime_error("Python error: Finding constructor");
  }

  for ( const auto e : Networks::All ) {
    Network net = networks[e];
    PyObject* pgdAttack;
    try {
      pgdAttack = create_attack_from_network(net, pAttackInit);
      networkAttacks[e] = pgdAttack;
    } catch (const std::runtime_error& e) {
      Py_DECREF(pAttackInit);
      Py_DECREF(pModule);
      throw e;
    }
  }

  Py_DECREF(pAttackInit);

  PyObject* pFunc = PyObject_GetAttrString(pModule, "run_attack");
  if (!pFunc || !PyCallable_Check(pFunc)) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    Py_XDECREF(pFunc);
    Py_DECREF(pModule);
    throw std::runtime_error("Python error: loading attack function");
  }

  PyGILState_Release(gstate);
  PyThreadState* tstate = PyEval_SaveThread();
  BayesianStrategy bi;

  bayesopt::Parameters params;
  params.n_iterations = 400;
  params.l_type = L_MCMC;
  params.n_iter_relearn = 20;
  params.load_save_flag = 2;

  if (world_rank == 0) {
    // The main process takes care of the Bayesian optimization stuff
    std::cout << "STARTING ROOT" << std::endl;
    //int dim = bi.input_size() * bi.output_size();
    int dim = bi.domain_input_size() * bi.domain_output_size() +
      bi.split_input_size() * bi.split_output_size();
    std::cout << "dim: " << dim << std::endl;

    boost::numeric::ublas::vector<double> best_point(dim);
    boost::numeric::ublas::vector<double> lower_bound(dim);
    boost::numeric::ublas::vector<double> upper_bound(dim);

    for (int i = 0; i < dim; i++) {
      lower_bound(i) = -1.0;
      upper_bound(i) = 1.0;
    }

    CegarOptimizer opt(dim, params, bi, world_size, benchmarks);
    opt.setBoundingBox(lower_bound, upper_bound);
    opt.optimize(best_point);

    int done = -1;
    for (int i = 1; i < world_size; i++) {
      MPI_Send(&done, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

  } else {
    struct timespec start, end;
    while(true) {
      //Receive properties from master process and then attempt to verify them.
      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      int numElements, solved, netId;
      MPI_Status status;

      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_INT, &numElements);
      MPI_Recv(&netId, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (netId < 0) {
        //Root process is done with optimization. We can exit
        std::cout << "Exiting" << std::endl;
        break;
      }
      Network net = networks[static_cast<Networks::Nets>(netId)];
      PyObject *pgdAttack = networkAttacks[static_cast<Networks::Nets>(netId)];

      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      MPI_Get_count(&status, MPI_DOUBLE, &numElements);
      std::vector<double> strategyAndProperty(numElements);
      MPI_Recv(&strategyAndProperty[0], numElements, MPI_DOUBLE, 0,
          MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      // Interpret the given strategy and property.
      int dos = bi.domain_output_size();
      int dis = bi.domain_input_size();
      int sos = bi.split_output_size();
      int sis = bi.split_input_size();
      Eigen::MatrixXd domain_strat(dos, dis);
      Eigen::MatrixXd split_strat(sos, sis);
      for (int i = 0; i < dos * dis + sos * sis; i++) {
        if (i < dos * dis) {
          domain_strat(i / dis, i % dis) = strategyAndProperty[i];
        } else {
          split_strat((i - dos*dis) / sis, (i - dos*dis) % sis) =
            strategyAndProperty[i];
        }
      }

      //Deserialize property
      int propertyStart = bi.domain_output_size() * bi.domain_input_size() +
        bi.split_output_size() * bi.split_input_size();
      Eigen::VectorXd lower((numElements-propertyStart)/2);
      Eigen::VectorXd upper((numElements-propertyStart)/2);
      int lowerStart = propertyStart, upperStart = propertyStart +
        (numElements - propertyStart)/2;
      for (int i = 0; i < (numElements-propertyStart)/2; i++) {
        lower(i) = strategyAndProperty[lowerStart+i];
        upper(i) = strategyAndProperty[upperStart+i];
      }

      // Verify property
      Interval itv(lower, upper);
      // We start by determining a target class. In order for the interval to
      // be robust, all points must have the same label, so we can evaluate any
      // point in the interval to get a target class.
      Eigen::VectorXd out = net.evaluate(lower);
      int max_ind = 0;
      double max = out(0);
      for (int i = 1; i < out.size(); i++) {
        if (out(i) > max) {
          max_ind = i;
          max = out(i);
        }
      }

      try {
        Eigen::VectorXd counterexample(lower.size());
        int num_calls = 0;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
        verify_with_strategy(lower, itv, max_ind, net,
                             counterexample, num_calls, domain_strat,
                             split_strat, bi, TIMEOUT, pgdAttack, pFunc);
        solved = 1;
      } catch (timeout_exception e) {
        // This exception indicates a timeout
        solved = 0;
      }
      clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
      MPI_Send(&solved, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      double elapsed = (double)(end.tv_sec - start.tv_sec);
      MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
  }

  PyEval_RestoreThread(tstate);
  gstate = PyGILState_Ensure();
  for (const auto e : Networks::All) {
    Py_DECREF(networkAttacks[e]);
  }
  Py_DECREF(pModule);
  Py_DECREF(pFunc);
  Py_Finalize();

  MPI_Finalize();

  return 0;
}
