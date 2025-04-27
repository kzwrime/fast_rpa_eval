#include <cblas.h>
#include <vector>

#include "magma_v2.h"

#include "common.hpp"
#include "device.hpp"
#include "pass_mod_var.h"

#include "evaluate_first_order_rho_direct_test.hpp"

#include "device_data.hpp"

constexpr bool enable_profile_rho_each_kernel = false;
constexpr bool enable_profile_rho_end_to_end = true;

#define ALL_KERNELS_ON_GPU

// 一个 block 处理 16x16 的区域，即 i_compute_tile_size
template <int atom_tile_size, int block_size>
__global__ void global_first_order_density_to_local(
    const int batch_offset,
    const int n_basis,
    const int n_centers_basis_I,
    const int n_max_compute_ham,
    const int real_n_batches_tile,
    const int *__restrict__ __attribute__((aligned(16))) n_compute_c_batches_ptr,
    const int *__restrict__ __attribute__((aligned(16))) i_basis_batches_ptr,
    const int *__restrict__ __attribute__((aligned(16))) i_valid_batch_2_i_batch_ptr,
    const int *__restrict__ __attribute__((aligned(16))) first_order_density_matrix_compute_offsets_ptr, // 暂时没用上
    const double *__restrict__ __attribute__((aligned(16))) first_order_density_matrix_ptr,
    double *__restrict__ __attribute__((aligned(16))) first_order_density_matrix_compute_batches_ptr) {

  const int batch_inner_id = blockIdx.x;
  const int batch_id = i_valid_batch_2_i_batch_ptr[batch_inner_id + batch_offset];

  int i_compute = blockIdx.y * block_size + threadIdx.y;
  const int n_compute_c = n_compute_c_batches_ptr[batch_id];
  if (i_compute >= n_compute_c)
    return;

  cTMf64<3> TM_INIT(first_order_density_matrix, atom_tile_size, n_basis, n_basis);
  TMf64<4> TM_INIT(
      first_order_density_matrix_compute_batches,
      atom_tile_size,
      n_max_compute_ham,
      n_max_compute_ham,
      real_n_batches_tile);

  const int *i_basis_index_ptr = &i_basis_batches_ptr[n_centers_basis_I * batch_id];

  // for (int i_compute = threadIdx.x; i_compute < n_compute_c; i_compute += block_size) {
  int i_basis = i_basis_index_ptr[i_compute] - 1;
  for (int j_compute = 0; j_compute < n_compute_c; j_compute++) {
    int j_basis = i_basis_index_ptr[j_compute] - 1;

    if constexpr (atom_tile_size >= 4 && (atom_tile_size % 2) == 0) {
      XDEF_UNROLL
      for (int i = 0; i < 2; ++i) {
        first_order_density_matrix_compute_batches_ptr
            [i + threadIdx.x * 2 +
             atom_tile_size * (i_compute + n_max_compute_ham * (j_compute + n_max_compute_ham * batch_inner_id))] =
                first_order_density_matrix_ptr[i + threadIdx.x * 2 + atom_tile_size * (i_basis + j_basis * n_basis)];
      }
    } else {
      XDEF_UNROLL
      for (int i = 0; i < atom_tile_size; ++i) {
        first_order_density_matrix_compute_batches_ptr
            [i + atom_tile_size * (i_compute + n_max_compute_ham * (j_compute + n_max_compute_ham * batch_inner_id))] =
                first_order_density_matrix_ptr[i + atom_tile_size * (i_basis + j_basis * n_basis)];
      }
    }
  }
  // }

  // double mreg[block_size_j][block_size_i];

  // for (int j = 0; j < block_size_j; j++) {
  //   for (int i = 0; i < block_size_i; i++) {
  //     mreg[j][i] = first_order_density_matrix_ptr[i + atom_tile_size * (i_basis + j_basis * n_basis)];
  //   }
  // }

  // for (int j = 0; j < block_size_j; j++) {
  //   for (int i = 0; i < block_size_i; i++) {
  //     first_order_density_matrix_compute_batches_ptr
  //         [i + atom_tile_size * (i_compute + n_max_compute_ham * (j_compute + n_max_compute_ham * batch_inner_id))] =
  //             mreg[j][i];
  //   }
  // }
}

template <int atom_tile_size, int block_size>
__global__ void first_order_rho_ddot(
    const int i_batch_offset,
    // scalars
    const int n_full_points,
    const int n_max_batch_size,
    const int n_my_batches_work,
    const int n_batch_tile,
    const int n_max_compute_ham,
    // arrays
    const int *__restrict__ __attribute__((aligned(16))) i_valid_batch_2_i_batch_ptr,
    const int *__restrict__ __attribute__((aligned(16))) n_compute_c_batches_ptr,
    const int *__restrict__ __attribute__((aligned(16))) n_point_batches_ptr,
    const double *__restrict__ __attribute__((aligned(16))) wave_batches_compress_ptr,
    const double *__restrict__ __attribute__((aligned(16))) gradient_wave_batches_compress_ptr,
    const int *__restrict__ __attribute__((aligned(16))) i_batch_2_wave_offset_ptr,
    const int *__restrict__ __attribute__((aligned(16))) n_point_batches_prefix_sum_ptr,
    const int *__restrict__ __attribute__((aligned(16))) i_valid_point_2_i_full_points_map_ptr,
    double *__restrict__ __attribute__((aligned(16))) first_order_rho_ptr,
    const double *__restrict__ __attribute__((aligned(16))) first_order_rho_bias_part2_ptr,
    double *__restrict__ __attribute__((aligned(16))) first_order_gradient_rho_ptr,
    const double *__restrict__ __attribute__((aligned(16))) first_order_gradient_rho_bias_batches_atoms_ptr,
    const double *__restrict__ __attribute__((aligned(16))) work1_batches_ptr) {

  int i_batch_inner = blockIdx.x;
  // int i_point = threadIdx.x;
  int i_point = threadIdx.y * block_size + threadIdx.x;

  // for (int i_batch_inner = 0; i_batch_inner < n_batch_tile; i_batch_inner++) {

  const int i_my_batch = i_valid_batch_2_i_batch_ptr[i_batch_inner + i_batch_offset];

  const int n_compute_c = n_compute_c_batches_ptr[i_my_batch];
  const int n_compute_c_padding = ((n_compute_c + 3) / 4) * 4;

  const int n_points = n_point_batches_ptr[i_my_batch];

  if (i_point >= n_points) {
    return;
  }

  cTMf64<2> wave(wave_batches_compress_ptr + i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, n_points);
  cTMf64<3> gradient_basis_wave(
      gradient_wave_batches_compress_ptr + 3 * i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, 3, n_points);

  cTMf64<4> TM_INIT(
      first_order_gradient_rho_bias_batches_atoms, 3, n_max_batch_size, n_my_batches_work, atom_tile_size);
  cTMf64<3> TM_INIT(first_order_rho_bias_part2, n_max_batch_size, n_my_batches_work, atom_tile_size);

  TMf64<2> TM_INIT(first_order_rho, n_full_points, atom_tile_size);
  TMf64<4> TM_INIT(first_order_gradient_rho, 3, n_max_batch_size, n_my_batches_work, atom_tile_size);

  int i_valid_point_start = n_point_batches_prefix_sum_ptr[i_my_batch];
  const int *i_full_points_map_ptr = &i_valid_point_2_i_full_points_map_ptr[i_valid_point_start];
  cTMi32<1> TM_INIT(i_full_points_map, n_points);

  // TODO work1 有 n_batch_tile 份，目前测试时只有 1 份

  const double *work1_ptr = work1_batches_ptr + i_batch_inner * atom_tile_size * n_max_compute_ham * n_max_batch_size;
  cTMf64<3> TM_INIT(work1, atom_tile_size, n_max_compute_ham, n_points);

  // for (int i_point = threadIdx.x; i_point < n_points; i_point += block_size) {
  double acc[ATOM_TILE_SIZE] = { 0 };
  double acc_grad[3][ATOM_TILE_SIZE] = { 0 };
  for (int ic = 0; ic < n_compute_c; ic++) {
    XDEF_UNROLL
    for (int i = 0; i < ATOM_TILE_SIZE; i++) {
      acc[i] += wave(ic, i_point) * work1(i, ic, i_point);
    }
    XDEF_UNROLL
    for (int j = 0; j < 3; j++) {
      XDEF_UNROLL
      for (int i = 0; i < ATOM_TILE_SIZE; i++) {
        acc_grad[j][i] += gradient_basis_wave(ic, j, i_point) * work1(i, ic, i_point);
      }
    }
  }
  // INFO GGA only
  XDEF_UNROLL
  for (int i = 0; i < atom_tile_size; i++) {
    XDEF_UNROLL
    for (int j = 0; j < 3; j++) {
      first_order_gradient_rho(j, i_point, i_my_batch, i) =
          acc_grad[j][i] * 2 - first_order_gradient_rho_bias_batches_atoms(j, i_point, i_my_batch, i);
    }
  }

  XDEF_UNROLL
  for (int i = 0; i < atom_tile_size; i++) {
    first_order_rho(i_full_points_map(i_point), i) = acc[i] + first_order_rho_bias_part2(i_point, i_my_batch, i);
  }
  // }
  // }
}

extern "C" void evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_(
    // scalars
    int *n_my_batches_work_,
    int *j_atom_begin_,
    int *j_atom_end_,
    int *j_coord_,
    int *n_full_points_,
    // global scalars
    int *n_basis_,
    int *n_atoms_,
    int *n_max_compute_ham_,
    int *n_centers_basis_I_,
    int *n_max_batch_size_,
    // arrays
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
    double *first_order_density_matrix_ptr,
    double *first_order_rho_ptr,
    double *first_order_rho_bias_part2_ptr,
    double *first_order_gradient_rho_ptr,
    double *first_order_gradient_rho_bias_batches_atoms_ptr,
    // global arrays
    int *basis_atom_ptr,
    double *partition_tab_ptr,
    double *density_matrix_ptr) {

  // scalars
  [[maybe_unused]] int n_my_batches_work = *n_my_batches_work_;
  [[maybe_unused]] int j_atom_begin = *j_atom_begin_;
  [[maybe_unused]] int j_atom_end = *j_atom_end_;
  [[maybe_unused]] int j_coord = *j_coord_; // {1,2,3}
  [[maybe_unused]] int n_full_points = *n_full_points_;
  // global scalars
  [[maybe_unused]] int n_basis = *n_basis_;
  [[maybe_unused]] int n_atoms = *n_atoms_;
  [[maybe_unused]] int n_max_compute_ham = *n_max_compute_ham_;
  [[maybe_unused]] int n_centers_basis_I = *n_centers_basis_I_;
  [[maybe_unused]] int n_max_batch_size = *n_max_batch_size_;

  TMf64<1> TM_INIT(wave_batches_compress, i_batch_2_wave_offset_ptr[n_my_batches_work]);
  TMf64<1> TM_INIT(gradient_wave_batches_compress, 3 * i_batch_2_wave_offset_ptr[n_my_batches_work]);
  TMf64<3> TM_INIT(first_order_density_matrix, n_basis, n_basis, ATOM_TILE_SIZE);
  TMf64<2> TM_INIT(first_order_rho, n_full_points, ATOM_TILE_SIZE);
  TMf64<3> TM_INIT(first_order_rho_bias_part2, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  TMf64<4> TM_INIT(first_order_gradient_rho, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  TMf64<4> TM_INIT(first_order_gradient_rho_bias_batches_atoms, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  // global arrays
  TMf64<2> TM_INIT(density_matrix, n_basis, n_basis);

  first_order_rho.setZero();
  first_order_gradient_rho.setZero();

  DEV_STREAM_T &stream = devInfo.stream;
  magma_queue_t &magma_queue = devInfo.magma_queue;

  const int n_valid_batches = dfpt_common_scalars.n_valid_batches;

  EventHelper<enable_profile_rho_each_kernel> event_helper(stream);
  EventHelper<enable_profile_rho_end_to_end> event_helper_all(stream);

  event_helper_all.record_start();

  // Changeable Arrays from fortran
#ifndef ALL_KERNELS_ON_GPU
  TM_DEV_PS_H2D_H(first_order_density_matrix);
  TM_DEV_PS_H2D_H(first_order_rho_bias_part2);
  TM_DEV_PS_H2D_H(first_order_gradient_rho_bias_batches_atoms);
#endif

  {
    EventHelper<enable_profile_rho_each_kernel> event_helper(stream);
    event_helper.record_start();
    magmablas_dtranspose(
        n_basis * n_basis,
        ATOM_TILE_SIZE,
        devPs.first_order_density_matrix.ptr,
        n_basis * n_basis,
        devPs.first_order_density_matrix_atom_inner_trans.ptr,
        ATOM_TILE_SIZE,
        magma_queue);

    event_helper.elapsed_time("Kernel magmablas_dtranspose execution time in stream");
  }

  // WARNING 无论如何都算 ATOM_TILE_SIZE 大小，哪怕最后 atom_tile_size 会减少

  for (int i_my_batch_outer = 0; i_my_batch_outer < n_valid_batches; i_my_batch_outer += N_BATCHES_TILE) {
    const int real_n_batches_tile = std::min(N_BATCHES_TILE, n_valid_batches - i_my_batch_outer);
    {
      event_helper.record_start();
      const int block_size = 128;
      dim3 blockSizes;
      if constexpr (ATOM_TILE_SIZE >= 4 && (ATOM_TILE_SIZE % 2) == 0) {
        blockSizes = { ATOM_TILE_SIZE / 2, block_size, 1 };
      } else {
        blockSizes = { 1, block_size, 1 };
      }
      dim3 gridSizes(real_n_batches_tile, CDIV(n_max_compute_ham, block_size), 1);
      global_first_order_density_to_local<ATOM_TILE_SIZE, block_size><<<gridSizes, blockSizes, 0, stream>>>(
          i_my_batch_outer,
          n_basis,
          n_centers_basis_I,
          n_max_compute_ham,
          real_n_batches_tile,
          devPs.n_compute_c_batches.ptr,
          devPs.i_basis_batches.ptr,
          devPs.i_valid_batch_2_i_batch.ptr,
          nullptr,
          devPs.first_order_density_matrix_atom_inner_trans.ptr,
          devPs.first_order_density_matrix_compute_batches.ptr);
      event_helper.elapsed_time("Kernel global_first_order_density_to_local  execution time in stream");
    }

    {
      event_helper.record_start();
      magmablas_dgemm_vbatched_max_nocheck(
          MagmaNoTrans,
          MagmaNoTrans,
          &devPs.n_compute_c_mul_atom_tile_size_valid_batches.ptr[i_my_batch_outer],
          &devPs.n_point_valid_batches.ptr[i_my_batch_outer],
          &devPs.n_compute_c_valid_batches.ptr[i_my_batch_outer],
          1.0,
          &devPs.first_order_density_matrix_compute_ptrs.ptr[0],
          &devPs.first_order_density_matrix_compute_ldas.ptr[0],
          &devPs.wave_dev_ptrs.ptr[i_my_batch_outer],
          &devPs.n_compute_c_padding_valid_batches.ptr[i_my_batch_outer],
          0.0,
          &devPs.work1_batches_ptrs.ptr[0],
          &devPs.work1_batches_ldas.ptr[0],
          real_n_batches_tile,
          host_first_order_rho_data.work1_max_m(i_my_batch_outer / N_BATCHES_TILE),
          host_first_order_rho_data.work1_max_n(i_my_batch_outer / N_BATCHES_TILE),
          host_first_order_rho_data.work1_max_k(i_my_batch_outer / N_BATCHES_TILE),
          magma_queue);
      event_helper.elapsed_time("Kernel magmablas_dgemm_vbatched_max_nocheck execution time in stream");
    }

    {
      event_helper.record_start();
      constexpr int block_size = 256;
      dim3 blockSizes(block_size, CDIV(n_max_batch_size, block_size), 1);
      dim3 gridSizes(real_n_batches_tile, 1, 1);
      first_order_rho_ddot<ATOM_TILE_SIZE, block_size><<<gridSizes, blockSizes, 0, stream>>>(
          i_my_batch_outer,
          n_full_points,
          n_max_batch_size,
          n_my_batches_work,
          real_n_batches_tile,
          n_max_compute_ham,
          devPs.i_valid_batch_2_i_batch.ptr,
          devPs.n_compute_c_batches.ptr,
          devPs.n_point_batches.ptr,
          devPs.wave_batches_compress.ptr,
          devPs.gradient_wave_batches_compress.ptr,
          devPs.i_batch_2_wave_offset.ptr,
          devPs.n_point_batches_prefix_sum.ptr,
          devPs.i_valid_point_2_i_full_points_map.ptr,
          devPs.first_order_rho.ptr,
          devPs.first_order_rho_bias_part2.ptr,
          devPs.first_order_gradient_rho.ptr,
          devPs.first_order_gradient_rho_bias_batches_atoms.ptr,
          devPs.work1_batches.ptr);
      event_helper.elapsed_time("Kernel first_order_rho_ddot                 execution time in stream");
    }
    // printf("\n");
  }

#ifndef ALL_KERNELS_ON_GPU
  TM_DEV_PS_D2H_H(first_order_rho);
  TM_DEV_PS_D2H_H(first_order_gradient_rho);
  DEV_CHECK(DEV_STREAM_SYNCHRONIZE(stream));
#endif

  event_helper_all.elapsed_time(
      "evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_ execution time in stream");
}
