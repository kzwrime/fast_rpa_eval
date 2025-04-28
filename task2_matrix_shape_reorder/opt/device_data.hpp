#pragma once

#include "common.hpp"
#include "device.hpp"
#include "setting.h"

#define TM_DEV_PS_INIT(var)                                                                                            \
  static_assert(                                                                                                       \
      std::is_same<RemoveConst_t<decltype(devPs.var.ptr)>, RemoveConst_t<decltype(var.data())>>::value,                \
      "Type mismatch between devPs.var.ptr and var.data() after removing const");                                      \
  devPs.var.size = var.size();                                                                                         \
  DEV_CHECK(DEV_MALLOC((void **)&(devPs.var.ptr), var.size() * sizeof(decltype(var(0)))));

// 从 host 到 device，大小参考 var.size()，即 host 端的 tensorMap 的 size，并且和 PtrContainer.byte_size() 比较大小
#define TM_DEV_PS_H2D_H(var)                                                                                           \
  static_assert(                                                                                                       \
      std::is_same<RemoveConst_t<decltype(devPs.var.ptr)>, RemoveConst_t<decltype(var.data())>>::value,                \
      "Type mismatch between devPs.var.ptr and var.data() after removing const");                                      \
  massert(devPs.var.byte_size() == var.size() * sizeof(decltype(var(0))));                                             \
  DEV_CHECK(DEV_MEMCPY_H2D(                                                                                            \
      devPs.var.ptr,                                                                                                   \
      (const void *)var.data(),                                                                                        \
      var.size() * sizeof(decltype(var(0))),                                                                           \
      DEV_MEMCPY_HOST_TO_DEVICE,                                                                                       \
      stream));

// 从 host 到 device，大小参考 PtrContainer.byte_size()
#define TM_DEV_PS_H2D_D(var)                                                                                           \
  static_assert(                                                                                                       \
      std::is_same<RemoveConst_t<decltype(devPs.var.ptr)>, RemoveConst_t<decltype(var.data())>>::value,                \
      "Type mismatch between devPs.var.ptr and var.data() after removing const");                                      \
  DEV_CHECK(DEV_MEMCPY_H2D(                                                                                            \
      devPs.var.ptr, (const void *)var##_ptr, devPs.var.byte_size(), DEV_MEMCPY_HOST_TO_DEVICE, stream));

// 从 device 到 host，大小参考 var.size()，即 host 端的 tensorMap 的 size，并且和 PtrContainer.byte_size() 比较大小
#define TM_DEV_PS_D2H_H(var)                                                                                           \
  static_assert(                                                                                                       \
      std::is_same<RemoveConst_t<decltype(devPs.var.ptr)>, RemoveConst_t<decltype(var.data())>>::value,                \
      "Type mismatch between devPs.var.ptr and var.data() after removing const");                                      \
  massert(devPs.var.byte_size() == var.size() * sizeof(decltype(var(0))));                                             \
  DEV_CHECK(DEV_MEMCPY_D2H(                                                                                            \
      (void *)var.data(), devPs.var.ptr, var.size() * sizeof(decltype(var(0))), DEV_MEMCPY_DEVICE_TO_HOST, stream));

// 从 device 到 host，大小参考 PtrContainer.byte_size()
#define TM_DEV_PS_D2H_D(var)                                                                                           \
  static_assert(                                                                                                       \
      std::is_same<RemoveConst_t<decltype(devPs.var.ptr)>, RemoveConst_t<decltype(var.data())>>::value,                \
      "Type mismatch between devPs.var.ptr and var.data() after removing const");                                      \
  DEV_CHECK(DEV_MEMCPY_D2H((void *)var##_ptr, devPs.var.ptr, devPs.var.byte_size(), DEV_MEMCPY_DEVICE_TO_HOST, stream));

#define TM_DEV_PS_FREE(var)                                                                                            \
  massert(devPs.var.byte_size() > 0);                                                                                  \
  devPs.var.size = 0;                                                                                                  \
  DEV_CHECK(DEV_FREE(devPs.var.ptr));

template <typename T>
struct PtrContainer {
  T *ptr = nullptr;
  size_t size = 0;
  size_t byte_size() const {
    return size * sizeof(T);
  }
};

template <typename T>
struct PtrContainerConst {
  const T *ptr = nullptr;
  size_t size = 0;
  size_t byte_size() const {
    return size * sizeof(T);
  }
};

struct DfptCommonScalars {
  int n_valid_batches;
};

struct HostFirstOrderRhoData {
  Ti32<1> work1_max_m;
  Ti32<1> work1_max_n;
  Ti32<1> work1_max_k;
};

struct DevicePtrs {
  // Const Arrays from fortran
  PtrContainer<int> basis_atom;
  PtrContainer<int> batch_sizes;
  PtrContainer<int> n_point_batches;
  PtrContainer<int> n_point_batches_prefix_sum;
  PtrContainer<int> i_valid_point_2_i_full_points_map;
  PtrContainer<int> n_compute_c_batches;
  PtrContainer<int> i_basis_batches;
  PtrContainer<int> atom_valid_n_compute_c_batches;
  PtrContainer<int> i_batch_2_wave_offset;
  PtrContainer<double> wave_batches_compress;
  PtrContainer<double> gradient_wave_batches_compress;
  PtrContainer<double> density_matrix;

  PtrContainer<int> n_radial;
  PtrContainer<int> species;
  PtrContainer<int> species_center;
  PtrContainer<double> coord_points;
  PtrContainer<int> l_hartree;
  PtrContainer<int> n_grid;
  PtrContainer<int> n_cc_lm_ijk;
  PtrContainer<int> centers_hartree_potential;
  PtrContainer<int> center_to_atom;
  PtrContainer<int> index_cc;
  PtrContainer<int> index_ijk_max_cc;
  PtrContainer<double> cc;
  PtrContainer<double> coords_center;
  PtrContainer<double> r_grid_min;
  PtrContainer<double> log_r_grid_inc;
  PtrContainer<double> scale_radial;
  PtrContainer<double> partition_tab;

  // Changeable Arrays from fortran
  PtrContainer<double> first_order_density_matrix;
  PtrContainer<double> first_order_rho;
  PtrContainer<double> first_order_rho_bias_part2;
  PtrContainer<double> first_order_gradient_rho;
  PtrContainer<double> first_order_gradient_rho_bias_batches_atoms;

  PtrContainer<int> l_hartree_max_far_distance;
  PtrContainer<double> multipole_radius_sq;
  PtrContainer<double> outer_potential_radius;
  PtrContainer<double> multipole_moments;
  PtrContainer<double> current_delta_v_hart_part_spl_tile;

  PtrContainer<double> delta_v_hartree;

  // Const Arrays initialized in cpp
  PtrContainer<int> i_valid_batch_2_i_batch;
  PtrContainer<int> n_point_valid_mul_3_batches;
  PtrContainer<int> n_compute_c_valid_batches;
  PtrContainer<int> n_compute_c_padding_valid_batches;
  PtrContainer<int> n_compute_c_mul_atom_tile_size_valid_batches;
  PtrContainer<int> first_order_density_matrix_compute_ldas;
  PtrContainer<int> work1_batches_ldas;
  PtrContainer<double *> first_order_density_matrix_compute_ptrs;
  PtrContainer<double *> wave_dev_ptrs;
  PtrContainer<double *> gradient_wave_dev_ptrs;
  PtrContainer<double *> work1_batches_ptrs;

  PtrContainer<int> index_lm;

  // 用于 Kernel 自身 / Kernel 之间使用的临时数组
  PtrContainer<double> density_matrix_compute_batches;
  PtrContainer<double> first_order_density_matrix_compute_batches;
  PtrContainer<double> work1_batches;
  PtrContainer<double> first_order_density_matrix_atom_inner_trans;

  PtrContainer<double> multipole_c;
  PtrContainer<double> current_delta_v_hart_part_spl_tile_trans;
};

extern DevicePtrs devPs;
extern DfptCommonScalars dfpt_common_scalars;
extern HostFirstOrderRhoData host_first_order_rho_data;

void init_dfpt_device_data(
    // scalars
    int *n_my_batches_work_,
    int *n_full_points_,
    int *n_valid_points_,
    // global scalars
    int *n_basis_,
    int *n_atoms_,
    int *n_max_compute_ham_,
    int *n_centers_basis_I_,
    int *n_max_batch_size_,
    int *l_pot_max_,
    int *n_max_radial_,
    int *n_hartree_grid_,
    int *n_centers_,
    int *n_centers_hartree_potential_,
    int *hartree_force_l_add_,
    int *n_species_,
    int *l_max_analytic_multipole_,
    int *index_cc_dim_0_,

    // Const Arrays from fortran
    int *basis_atom_ptr,
    int *batch_sizes_ptr,
    int *n_point_batches_ptr,
    int *n_point_batches_prefix_sum_ptr,
    int *i_valid_point_2_i_full_points_map_ptr,
    int *n_compute_c_batches_ptr,
    int *i_basis_batches_ptr,
    int *atom_valid_n_compute_c_batches_ptr,
    int *i_batch_2_wave_offset_ptr,
    double *wave_batches_compress_ptr,
    double *gradient_wave_batches_compress_ptr,
    double *density_matrix_ptr,

    int *n_radial_ptr,
    int *species_ptr,
    int *species_center_ptr,
    double *coord_points_ptr,
    int *l_hartree_ptr,
    int *n_grid_ptr,
    int *n_cc_lm_ijk_ptr,
    int *centers_hartree_potential_ptr,
    int *center_to_atom_ptr,
    int *index_cc_ptr,
    int *index_ijk_max_cc_ptr,
    double *cc_ptr,
    double *coords_center_ptr,
    double *r_grid_min_ptr,
    double *log_r_grid_inc_ptr,
    double *scale_radial_ptr,
    double *partition_tab_ptr,

    // Changeable Arrays from fortran
    double *first_order_density_matrix_ptr,
    double *first_order_rho_ptr,
    double *first_order_rho_bias_part2_ptr,
    double *first_order_gradient_rho_ptr,
    double *first_order_gradient_rho_bias_batches_atoms_ptr,

    int *l_hartree_max_far_distance_ptr,
    double *multipole_radius_sq_ptr,
    double *outer_potential_radius_ptr,
    double *multipole_moments_ptr,
    double *current_delta_v_hart_part_spl_tile_ptr,

    double *delta_v_hartree_ptr);

void free_dfpt_device_data();
