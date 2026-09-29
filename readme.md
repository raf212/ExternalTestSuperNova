
```bash
g++ main.cpp \
    -std=c++20 -O3 -DNDEBUG -march=native -mtune=native -flto=auto -fopenmp \
    -I. -I./sortledton \
    -L./LiveGraph/build -llivegraph \
    ./sortledton/build/libsortledton.a \
    -L./SuperNova/build-release -latomiccim_core \
    -ltbb -pthread \
    -Wl,-rpath,'$ORIGIN/LiveGraph/build' \
    -o all_systems_benchmark
```