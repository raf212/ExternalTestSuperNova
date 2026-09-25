```bash
g++ main.cpp \
    -std=c++20 \
    -O3 \
    -march=native \
    -fopenmp \
    -I. \
    -L./LiveGraph/build \
    -llivegraph \
    -L./SuperNova/build-release \
    -latomiccim_core \
    -ltbb \
    -pthread \
    -Wl,-rpath,'$ORIGIN/LiveGraph/build' \
    -o livegraph_vs_supernova
```