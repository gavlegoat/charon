#include <zonotope.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <random>
#include <unistd.h>
#include <memory>
#include <set>
#include "network.hpp"
#include "strategy.hpp"

#define TIMEOUT 1000

int main(int argc, char** argv) {
#ifdef CHARON_HOME
  std::string cwd = CHARON_HOME;
#else
  std::cout << "CHARON_HOME is undefined. If you compiled with the " <<
    "provided CMake file this shouldn't happen, otherwise set " <<
    "CHARON_HOME" << std::endl;
  std::abort();
#endif
  if (argc != 5) {
    std::cout << "Usage: ./run <property-file> <network-file> <strategy-file>"
      << " <counterexample-file>" << std::endl;
    std::abort();
  }
  std::string property_file = argv[1];
  std::string network_filename = argv[2];
  std::string strategy_filename = argv[3];
  std::string counterexample_filename = argv[4];
  Network net = read_network(network_filename);
  Interval property = Interval(property_file);
  auto si = std::unique_ptr<StrategyInterpretation>(new BayesianStrategy());
  int dos = si->domain_output_size();
  int dis = si->domain_input_size();
  int sos = si->split_output_size();
  int sis = si->split_input_size();
  Eigen::VectorXd strategyMat(dis * dos + sis * sos);
  std::ifstream in(strategy_filename);
  for (int i = 0; i < dis * dos + sis * sos; i++) {
    in >> strategyMat(i);
  }
  Eigen::MatrixXd domain_strat(dos, dis);
  Eigen::MatrixXd split_strat(sos, sis);
  for (int i = 0; i < dos * dis; i++) {
    domain_strat(i / dis, i % dis) = strategyMat(i);
  }
  for (int i = 0; i < sos * sis; i++) {
    split_strat(i / sis, i % sis) = strategyMat(dos * dis + i);
  }

  Py_Initialize();
  PyEval_InitThreads();

  PyGILState_STATE gstate = PyGILState_Ensure();

  char s[5] = "path";
  PyObject* sysPath = PySys_GetObject(s);

  PyObject* newElem = PyString_FromString((cwd + std::string("src")).c_str());
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

  Eigen::VectorXd out = net.evaluate(property.lower);
  int max_ind = 0;
  double max = out(0); for (int i = 1; i < out.size(); i++) {
    if (out(i) > max) {
      max_ind = i;
      max = out(i);
    }
  }

  bool verified = false;
  bool timeout = false;
  Eigen::VectorXd counterexample(net.get_input_size());
  try {
    int num_calls = 0;
    verified = verify_with_strategy(
        property.lower, property, max_ind, net,
        counterexample, num_calls, domain_strat, split_strat, *si,
        TIMEOUT, pgdAttack, pFunc);
  } catch (const timeout_exception& e) {
    timeout = true;
  }

  if (verified) {
    std::cout << "Property verified" << std::endl;
  } else if (timeout) {
    std::cout << "Timed out after " << TIMEOUT << " seconds" << std::endl;
  } else {
    std::ofstream out(counterexample_filename);
    for (int i = 0; i < counterexample.size(); i++) {
      out << counterexample(i) << std::endl;
    }
    std::cout << "Property falsified, counterexample written to " <<
      counterexample_filename << std::endl;
  }

  PyEval_RestoreThread(tstate);
  gstate = PyGILState_Ensure();
  Py_DECREF(pgdAttack);
  Py_DECREF(pModule);
  Py_DECREF(pFunc);
  Py_Finalize();

  return 0;
}
