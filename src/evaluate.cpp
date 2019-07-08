#include <zonotope.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
//#include <opt_pk.h>
//#include <opt_oct.h>
#include <random>
#include <unistd.h>
#include <set>
#include "network.hpp"
#include "strategy.hpp"
#include <sys/times.h>
#include <time.h>
#ifdef TACC
//MPI path is different on tacc
#include <mpi.h>
#else
#include <mpi/mpi.h>
#endif

#define EPSILON 0.01
#define TIMEOUT 1000

enum Strategy {
    Bisect,
    Gradient,
    Bayesian,
    Counterexample,
    Random,
};

typedef struct task {
    int propertyId;
    Interval itv;
    Strategy strat;
} Task;

typedef struct taskResult {
    int timeElapsed;
    int strategy;
    int verified;
    int propertyId;
    int numSplits;
} TaskResult;

std::string GetStatus(TaskResult &r) {
  if (r.timeElapsed >= TIMEOUT) {
    return std::string("timeout");
  }
  if (r.verified > 0) {
    return std::string("verified");
  }
  return std::string("falsified");
}


double dimension_to_double(int dimension, int highest) {
  return ((double) dimension) / highest;
}

int double_to_dimension(double d, int highest) {
  d = std::max(0.0, std::min(0.9999, d));
  return (int) (d * highest);
}

std::string currentDir() {
  char buf[FILENAME_MAX];
  if (!getcwd(buf, sizeof(buf))) {
    exit(1);
  }
  std::string cwd(buf);
  return cwd;
}


int verify(const Eigen::VectorXd& original,
           const Interval& property, int max_ind, const Network& net,
           Eigen::VectorXd& counterexample, int& num_calls,
           const Eigen::VectorXd& strategy, const StrategyInterpretation *interp,
           PyObject* pgdAttack, PyObject* pFunc, bool &verified) {
  std::cout << "max_ind: " << max_ind << std::endl;
  struct timespec start, end;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);
  int dos = interp->domain_output_size();
  int dis = interp->domain_input_size();
  int sos = interp->split_output_size();
  int sis = interp->split_input_size();
  Eigen::MatrixXd domain_strat(dos, dis);
  Eigen::MatrixXd split_strat(sos, sis);
  for (int i = 0; i < dos * dis; i++) {
    domain_strat(i / dis, i % dis) = strategy[i];
  }
  for (int i = 0; i < sos * sis; i++) {
    split_strat(i / sis, i % sis) = strategy[dos * dis + i];
  }
  try {
    verified = verify_with_strategy(original, property, max_ind, net,
                                           counterexample, num_calls,
                                           domain_strat, split_strat,
                                           *interp, TIMEOUT,
                                           pgdAttack, pFunc);
  } catch (const timeout_exception& e) {
    std::cout << "TIMEOUT" << std::endl;
  }
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
  return (int) (end.tv_sec - start.tv_sec);
}

void sendTask(Task &t, const int worker) {
  Interval itv = t.itv;
  int strategy = static_cast<int>(t.strat);
  int propertyId = t.propertyId;
  int itvSize = itv.lower.size();
  std::cout << "itvSize: " << itvSize << std::endl;
  std::vector<double> combinedInterval(2*itvSize);
  for (int i = 0; i < itvSize; i++) {
    combinedInterval[i] = itv.lower(i);
    combinedInterval[itvSize + i] = itv.upper(i);
  }
  std::cout << "Sending property: " << propertyId << " to worker: " << worker << std::endl;
  MPI_Send(&propertyId, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
  MPI_Send(&combinedInterval[0], 2*itvSize, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);
  MPI_Send(&strategy, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
}

int distribute(const int &worldSize, std::string prop_file) {
  int numTasks, tasksAssigned = 0, tasksCompleted = 0;
  int finished = 0;
  std::string cwd = currentDir();
  std::ifstream fd(prop_file);
  std::ofstream outfile("benchmarks.csv");
  std::vector<std::string> prop_files;
  std::string line;
 
  while(std::getline(fd, line))
    prop_files.push_back(line);

  std::cout << prop_files[0] << std::endl;
  std::cout << prop_files.back() << std::endl;

  int numProperties = prop_files.size();
  std::vector<TaskResult> bisectResults(numProperties);
  std::vector<TaskResult> bayesResults(numProperties);
  std::vector<TaskResult> gradResults(numProperties);
  std::vector<TaskResult> counterexampleResults(numProperties);
  const std::vector<Strategy> strategiesToEvaluate({Strategy::Bayesian});
  //const std::vector<Strategy> strategiesToEvaluate({Gradient, Counterexample});

  outfile << "Benchmark," <<
      "BayesStrategyStatus,BayesStrategyNumCalls,BayesStrategyTime" <<
      //"BisectStatus,BisectNumCalls,BisectTime," <<
      //"GradientStatus,GradientNumCalls,GradientTime," <<
      //"CounterexampleStatus,CounterexampleNumCalls,CounterexampleTime" <<
      std::endl;
  std::vector<Task> toVerify;
  for (int i = 0; i < numProperties; i++) {
    Interval benchmark(prop_files[i]);
    for (int j = 0; j < strategiesToEvaluate.size(); j++) {
      Task t = {i, benchmark, strategiesToEvaluate[j]};
      toVerify.push_back(t);
    }
  }
  numTasks = toVerify.size();
  int N = (worldSize-1 > numTasks) ? numTasks : worldSize-1;
  for (int i = 0; i < N; i++) {
    sendTask(toVerify[i], i+1);
    tasksAssigned++;
  }
  while (tasksCompleted < numTasks) {
    TaskResult t;
    MPI_Status status;
    MPI_Recv(&t, sizeof(TaskResult)/sizeof(int), MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    tasksCompleted++;
    std::cout << "RECEIVED VERIFICATION FROM: " << status.MPI_SOURCE << " with status: " << t.verified << " (strategy: " << t.strategy << ")" << std::endl;

    switch (static_cast<Strategy>(t.strategy)) {
      case Strategy::Bisect:
        bisectResults[t.propertyId] = t;
        break;
      case Strategy::Bayesian:
        bayesResults[t.propertyId] = t;
        break;
      case Strategy::Gradient:
        gradResults[t.propertyId] = t;
        break;
      case Strategy::Counterexample:
        counterexampleResults[t.propertyId] = t;
        break;
      default:
        std::cout << "Received result for unimplemented strategy: " << t.strategy << ". Exiting..." << std::endl;
        exit(1);
    }
    if ((numTasks - tasksAssigned) > 0) {
      sendTask(toVerify[tasksAssigned], status.MPI_SOURCE);
      tasksAssigned++;
    }
  }
  finished = -1;
  for (int i = 1; i < worldSize; i++)
    MPI_Send(&finished, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

  for (int i = 0; i < numProperties; i++) {
    outfile << prop_files[i] << "," <<
            //GetStatus(bisectResults[i]) << "," << bisectResults[i].numSplits << "," << bisectResults[i].timeElapsed << "," <<
            //GetStatus(gradResults[i]) << "," << gradResults[i].numSplits << "," << gradResults[i].timeElapsed << "," <<
            //GetStatus(counterexampleResults[i]) << "," << counterexampleResults[i].numSplits << "," << counterexampleResults[i].timeElapsed << std::endl;
            GetStatus(bayesResults[i]) <<  "," << bayesResults[i].numSplits << "," << bayesResults[i].timeElapsed  << std::endl;
            //status_random << "," << num_splits_random << "," << time_elapsed_random <<
  }
  outfile.close();
  return 0;
}



int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  int world_rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::string property_file = argv[1];
  std::string cwd = currentDir();
  //std::string network_filename = cwd + std::string("/benchmarks/cifar_3_100_80acc.txt");
  //std::string network_filename = cwd + std::string("/xor.txt");
  //std::string network_filename = "/home/ganderso/Documents/train/networks/xor.nnet";
  //std::string network_filename = "/home/ganderso/Documents/cegar-net/benchmarks/acas_xu_1_1.txt";
  std::string network_filename = argv[2];
  Network net = read_network(network_filename);
  Eigen::VectorXd strategyMat(14);
  strategyMat << 0.31788,-0.052716,-0.848361,0.00883906,0.774632,-0.132656,0.637842,0.254465,0.0464505,-0.329934,-0.235704,-0.0933458,-0.00911998,0.14479;
  Py_Initialize();
  PyEval_InitThreads();

  PyGILState_STATE gstate = PyGILState_Ensure();

  char s[5] = "path";
  PyObject* sysPath = PySys_GetObject(s);
  PyObject* newElem = PyString_FromString((cwd + std::string("/src")).c_str());
  PyList_Append(sysPath, newElem);
  PySys_SetObject(s, sysPath);
  PyObject* pName = PyString_FromString("interface");
  PyObject* pModule = PyImport_Import(pName);
  Py_DECREF(pName);

  if (pModule == NULL) {
    PyErr_Print();
    throw std::runtime_error("Python error: Loading module");
  }

  PyObject* pAttackInit = PyObject_GetAttrString(pModule, "initialize_pgd_class");
  if (!pAttackInit || !PyCallable_Check(pAttackInit)) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    Py_XDECREF(pAttackInit);
    Py_DECREF(pModule);
    throw std::runtime_error("Python error: Finding constructor");
  }

  PyObject* pgdAttack;
  try {
    pgdAttack = create_attack_from_network(net, pAttackInit);
  } catch (const std::runtime_error& e) {
    Py_DECREF(pAttackInit);
    Py_DECREF(pModule);
    throw e;
  }

  Py_DECREF(pAttackInit);
  if (pgdAttack == NULL) {
    Py_DECREF(pModule);
    PyErr_Print();
    throw std::runtime_error("Python error: Initializing attack");
  }
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
  if (world_rank == 0) {
    distribute(world_size, property_file);
  } else {
    while(true) {
      //struct tms start, end;
      //times(&start) start’ has incomplete type and cannot be defined
      //Receive properties from master process and then execute them
      TaskResult r;
      int world_rank, strategy, propertyId;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      std::cout << "worker: " << world_rank << " waiting...." << std::endl;
      int numElements;
      bool verified;
      MPI_Status status;

      //Wait for a property
      MPI_Recv(&propertyId, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

      if (propertyId < 0) {
        //All properties have been evaluated. We can exit.
        std::cout << "Worker: " << world_rank << " exiting..." << std::endl;
        break;
      }

      //Get size of property
      MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      std::cout << "Probed!" << std::endl;
      MPI_Get_count(&status, MPI_DOUBLE, &numElements);


      std::vector<double> property(numElements);

      MPI_Recv(&property[0], numElements, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      std::cout << "Num Elements: " << numElements << std::endl;
      MPI_Recv(&strategy, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      std::cout << "Received Strategy: " << strategy << std::endl;

      Eigen::VectorXd lower(numElements/2), upper(numElements/2);

      for (int i = 0; i < numElements/2; i++) {
        lower(i) = property[i];
        upper(i) = property[numElements/2+i];
      }


      //Verify property
      Interval itv(lower, upper);
      Eigen::VectorXd out = net.evaluate(lower);
      int max_ind = 0;
      double max = out(0);
      for (int i = 1; i < out.size(); i++) {
        if (out(i) > max) {
          max_ind = i;
          max = out(i);
        }
      }
      StrategyInterpretation *si;
      switch (static_cast<Strategy>(strategy)) {
        case Strategy::Gradient: {
          si = new GradientStrategy();
          break;
        }
        case Strategy::Bisect: {
          si = new BisectStrategy();
          break;
        }
        case Strategy::Counterexample: {
          si = new CounterexampleStrategy();
          break;
        }
        default: {
          si = new BayesianStrategy();
          break;
        }
        //default: {
        //  si = new RandomStrategy();
        //  break;
        //}
      }
      Eigen::VectorXd counterexample(lower.size());
      int numCalls = 0;
      int timeElapsed = verify(lower, itv, max_ind, net,
          counterexample, numCalls, strategyMat, si, pgdAttack, pFunc, verified);
      r.propertyId = propertyId;
      r.strategy = strategy;
      r.verified = verified;
      r.timeElapsed = timeElapsed;
      r.numSplits = numCalls;
      MPI_Send(&r, sizeof(r)/sizeof(int), MPI_INT, 0, 0, MPI_COMM_WORLD);
      delete si;
    }
  }


  PyEval_RestoreThread(tstate);
  gstate = PyGILState_Ensure();
  Py_DECREF(pgdAttack);
  Py_DECREF(pModule);
  Py_DECREF(pFunc);
  Py_Finalize();
  MPI_Finalize();

  return 0;
}
