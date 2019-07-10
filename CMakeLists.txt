cmake_minimum_required(VERSION 3.5)

execute_process(COMMAND python2.7-config --cflags OUTPUT_VARIABLE PY_C_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND python2.7-config --ldflags OUTPUT_VARIABLE PY_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O0 -g -Wall ${PY_C_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${PY_LD_FLAGS}")

link_directories(/usr/lib/python2.7/config-x86_64-linux-gnu)

add_library(cegar_net ${CMAKE_CURRENT_SOURCE_DIR}/src/network.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/powerset.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/interval.cpp)
target_include_directories(cegar_net PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)

add_executable(learn ${CMAKE_CURRENT_SOURCE_DIR}/src/bayesian_opt.cpp ${CMAKE_CURRENT_SOURCE_DIR}/src/strategy.cpp)
target_include_directories(learn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/eigen)
target_link_libraries(learn cegar_net bayesopt nlopt gmp mpfr zonotope elinaux python2.7 pthread dl util m gomp)

set(CMAKE_CXX_COMPILER mpicxx)
add_compile_definitions(CHARON_HOME="${CMAKE_CURRENT_SOURCE_DIR}/")

