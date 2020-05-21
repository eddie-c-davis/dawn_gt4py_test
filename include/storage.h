#ifndef DAWN_GT4PY_TEST_STORAGE_H
#define DAWN_GT4PY_TEST_STORAGE_H

#define NDIM 3
#define NHALO NDIM*2

using uint_t = unsigned;

enum Platform {CPU, GPU};

struct Domain {
  std::array<uint_t, NDIM> dims;
  std::array<uint_t, NHALO> halos;

  unsigned iminus() const { return halos[0]; }
  unsigned iplus() const { return halos[1]; }
  unsigned jminus() const { return halos[2]; }
  unsigned jplus() const { return halos[3]; }
  unsigned kminus() const { return halos[4]; }
  unsigned kplus() const { return halos[5]; }

  unsigned isize() const { return dims[0]; }
  unsigned jsize() const { return dims[1]; }
  unsigned ksize() const { return dims[2]; }
};

struct Storage {
  std::array<uint_t, NDIM> shape;
  std::array<uint_t, NDIM> strides;
  std::array<uint_t, NHALO> halos;
  double* ptr;
  Platform platform;

  // read-operator
  const double& operator()(int i, int j = 0, int k = 0) const {
    return ptr[((i)+halos[0])*strides[0]+((j)+halos[1])*strides[1]+((k)+halos[2])*strides[2]];
  }

  // write-operator
  double& operator()(int i, int j = 0, int k = 0) {
    return ptr[((i)+halos[0])*strides[0]+((j)+halos[1])*strides[1]+((k)+halos[2])*strides[2]];
  }
};

#endif // DAWN_GT4PY_TEST_STORAGE_H
