
#pragma once

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

template <class T, int dim>
using ETT = Eigen::Tensor<T, dim, Eigen::ColMajor>;
template <class T, int dim>
using ETMT = Eigen::TensorMap<ETT<T, dim>>;

template <int dim>
using Ti32 = Eigen::Tensor<int, dim, Eigen::ColMajor>;
template <int dim>
using TMi32 = Eigen::TensorMap<Ti32<dim>>;

template <int dim>
using cTi32 = const Eigen::Tensor<int, dim, Eigen::ColMajor>;
template <int dim>
using cTMi32 = const Eigen::TensorMap<cTi32<dim>>;

template <int dim>
using Tf64 = Eigen::Tensor<double, dim, Eigen::ColMajor>;
template <int dim>
using TMf64 = Eigen::TensorMap<Tf64<dim>>;

template <int dim>
using cTf64 = const Eigen::Tensor<double, dim, Eigen::ColMajor>;
template <int dim>
using cTMf64 = const Eigen::TensorMap<cTf64<dim>>;

template <typename T>
struct RemoveConst {
  using type = T;
};

template <typename T>
struct RemoveConst<const T> {
  using type = T;
};
template <typename T>
struct RemoveConst<const T *> {
  using type = T *;
};

template <typename T>
using RemoveConst_t = typename RemoveConst<T>::type;

#define TM_INIT(var, ...) var(var##_ptr, __VA_ARGS__)

// always assert
#define massert(cond)                                                                                                  \
  do {                                                                                                                 \
    if (!(cond)) {                                                                                                     \
      printf("%s:%d: Assertion failed: %s\n", __FILE__, __LINE__, #cond);                                              \
      assert(cond);                                                                                                    \
    }                                                                                                                  \
  } while (0)

template <class T>
inline T max(const T &a, const T &b) {
  return a > b ? a : b;
}

template <class T>
inline T min(const T &a, const T &b) {
  return a < b ? a : b;
}