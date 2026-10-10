
```bash
g++ main.cpp \
    -std=c++20 -O3 -DNDEBUG -march=native -mtune=native -flto=auto -fopenmp \
    -I. \
    -I./sortledton \
    -L./LiveGraph/build \
    -llivegraph \
    ./sortledton/build/libsortledton.a \
    -L./SuperNova/build-acpp-linux \
    -latomiccim_core \
    ./SuperNova/external/AdaptiveCpp/install/lib/libacpp-rt.so \
    ./SuperNova/external/AdaptiveCpp/install/lib/libacpp-common.so \
    -latomic \
    -lnuma \
    -ltbb \
    -pthread \
    -Wl,-rpath,'$ORIGIN/LiveGraph/build' \
    -Wl,-rpath,'$ORIGIN/SuperNova/external/AdaptiveCpp/install/lib' \
    -o all_systems_benchmark
```