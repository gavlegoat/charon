# Charon
The Charon tool for analyzing neural network robustness. See our
[paper](https://arxiv.org/abs/1904.09959) at PLDI'19 for details.

# Dependencies
Charon uses [ELINA](http://elina.ethz.ch/) for abstract interpretation and
[BayesOpt](https://github.com/rmcantin/bayesopt) for Bayesian optimization.
Additionally, the training module uses MPI for parallelization to improve
training times. ELINA and BayesOpt have good installation instructions on the
linked websites. For MPI you can choose your favorite MPI implementation, or
use [Open MPI](https://www.open-mpi.org/) if you just want a quick start. Open
MPI can usually be found in the system package manager.

# Build
Once you have ELINA, BayesOpt, and an MPI implementation installed, you can
build Charon as follows:

```bash
$ git clone https://github.com/gavlegoat/charon.git .
$ cd charon
$ mkdir build && cd build
$ cmake ../
$ make learn
```

# Using Charon
Charon is split into two pieces: an off-line training phase and an on-line
deployment phase. Correspondingly, the provided CMake file generates two
executables: `learn` and `run`.

## Training
The training program, `learn`, takes a list of training properties and runs a
Bayesian optimization procedure to find a good strategy. The training procedure
uses MPI for parallelism, and should be run using `mpirun`. For example, to
train with the given example property, if you are in the build directory, you
can run

```bash
$ mpirun -n N ./learn ../example/training_properties.txt
```

where N is the number of MPI processes you wish to use. Notice that each MPI
process will spawn several separate threads, so N is *not* the number of
execution threads to use. The number of threads per MPI process is controlled
by the macro `NUM_THREADS` in `strategy.cpp`. Other parameters you may wish to
change for training include `TIMEOUT` and `PENALTY` in `bayesian_opt.cpp`.
These control the amount of time spent on each property and the penalty
applied when the system times out. Higher penalty values favor strategies which
can solve more benchmarks, even if they are slower on some benchmarks.
Conversely, smaller penalty values favor strategies which are fast on some
benchmarks, even if they time out on a greater number of benchmarks.

## Deployment
Once you have a strategy that works well, you can use the `run` executable to
run Charon on new properties. This program expects a property file, a network
file, a strategy file, and a counterexample file. The property file describes
a region to check for robustness, the network file holds the parameters and
architecture of a network, and the strategy file holds a (serialized) strategy.
The counterexample file will be used to store a counterexample to the
robustness property if one is found. For example, if you are in the build
directory, you can run

```bash
$ ./run ../example/acas_robustness.bmk ../example/acas_xu_1_1.txt ../example/basic_strategy.txt ../example/counterexample.txt
```

Notice that, as for the training phase, the verification may spawn up to
`NUM_THREADS` threads where `NUM_THREADS` is defined in strategy.cpp.
