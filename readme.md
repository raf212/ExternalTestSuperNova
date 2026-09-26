- LiveGraph
```bash
g++ main.cpp \
    -std=c++20 \
    -O3 \
    -DNDEBUG \
    -march=native \
    -mtune=native \
    -flto=auto \
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
- Sortledton
```bash
g++ main.cpp \
    -std=c++20 \
    -O3 \
    -DNDEBUG \
    -march=native \
    -mtune=native \
    -flto=auto \
    -fopenmp \
    -I. \
    -I./sortledton \
    -L./SuperNova/build-release \
    -latomiccim_core \
    ./sortledton/build/libsortledton.a \
    -ltbb \
    -pthread \
    -o sortledton_vs_supernova
```