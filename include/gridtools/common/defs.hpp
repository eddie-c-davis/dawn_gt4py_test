
#ifndef DAWN_GT4PY_TEST_DEFS_HPP
#define DAWN_GT4PY_TEST_DEFS_HPP

#define NDIM 3
#define NHALO NDIM * 2

#ifndef GRIDTOOLS_DAWN_HALO_EXTENT
#define GRIDTOOLS_DAWN_HALO_EXTENT 3
#endif

#include <array>
#include <cassert>

namespace gridtools {

template <typename T, std::size_t N> using array = std::array<T, N>;
using uint_t = unsigned;

template <uint_t HaloSizeX, uint_t HaloSizeY, uint_t HaloSizeZ> struct halo {
  std::array<uint_t, NHALO> halos = {HaloSizeX, HaloSizeX, HaloSizeY,
                                     HaloSizeY, HaloSizeZ, HaloSizeZ};
};

/** tags specifying the backend to use */
namespace backend {
struct cuda {};
struct mc {};
struct x86 {};
struct naive {};
} // namespace backend

enum ownership { external_cpu, external_gpu };

namespace dawn {

typedef double float_type;

struct domain {
  std::array<uint_t, NDIM> dims;
  std::array<uint_t, NHALO> halos;

  void set_halos(int h0, int h1, int h2, int h3, int h4, int h5) {
    halos[0] = h0;
    halos[1] = h1;
    halos[2] = h2;
    halos[3] = h3;
    halos[4] = h4;
    halos[5] = h5;
  }

  void set_dims(int d0, int d1, int d2) {
    dims[0] = d0;
    dims[1] = d1;
    dims[2] = d2;
  }

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

namespace storage_traits_t {

// gridtools::dawn::storage_traits_t::storage_info_t<0, 3, gridtools::halo<3, 3, 0> >::storage_info_t(unsigned int, unsigned int, unsigned int)
template <int BeginIndex, uint_t Dim, typename HaloType> struct storage_info_t {
  int begin_index = BeginIndex;
  uint_t dim = Dim;
  HaloType halo;
  std::array<uint_t, NDIM> shape;

  storage_info_t(uint_t isize, uint_t jsize, uint_t ksize) :
                 shape{isize, jsize, ksize} {}
};

template <typename DataType, typename StorageType> struct data_store_t {
  DataType data_type;
  StorageType storage_type;
};

} // namespace storage_traits_t

struct meta_data {
  std::array<uint_t, NDIM> shape;
  std::array<uint_t, NDIM> strides;
  //std::array<int, NDIM> halos;

  meta_data(uint_t isize, uint_t jsize, uint_t ksize) :
            shape{isize, jsize, ksize} {}

  meta_data(std::array<uint_t, NDIM> shape, std::array<uint_t, NDIM> strides) :
            shape(shape), strides(strides) {}
};

typedef meta_data meta_data_t;
typedef meta_data meta_data_ijk_t;

template <typename DataType = float_type> struct storage {
  meta_data m_data;
  DataType* ptr;
  ownership platform;
  std::string name;

  storage(meta_data& meta, const std::string& name) :
          m_data(meta), name(name) {}

  storage(meta_data meta, DataType* ptr, ownership platform) :
      m_data(meta), ptr(ptr), platform(platform), name("") {}

  void sync() {
    // No-op...
  }
};

typedef storage<float_type> storage_t;
typedef storage<float_type> storage_ijk_t;

} // namespace dawn

template <typename StorageType, typename DataType = double> struct data_view {
  StorageType &storage;

  // read-operator
  const DataType& operator()(int i, int j = 0, int k = 0) const {
    return storage.ptr[(i * storage.m_data.strides[0]) +
                       (j * storage.m_data.strides[1]) +
                       (k * storage.m_data.strides[2])];
  }

  // write-operator
  DataType& operator()(int i, int j = 0, int k = 0) {
    return storage.ptr[(i * storage.m_data.strides[0]) +
                       (j * storage.m_data.strides[1]) +
                       (k * storage.m_data.strides[2])];
  }
};

template <typename StorageType>
data_view<StorageType> make_host_view(StorageType &storage) {
  return data_view<StorageType>{storage};
}

} // namespace gridtools

namespace dawn {

using float_type = gridtools::dawn::float_type;

namespace driver {

struct cartesian_extent {
  int begin_extent_x;
  int end_extent_x;
  int begin_extent_y;
  int end_extent_y;
  int begin_extent_z;
  int end_extent_z;
};

} // namespace driver
} // namespace dawn

#endif // DAWN_GT4PY_TEST_DEFS_HPP
