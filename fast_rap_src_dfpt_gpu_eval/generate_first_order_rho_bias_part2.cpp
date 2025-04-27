

#include <cblas.h>

#include "common.hpp"

#include "pass_mod_var.h"

#define ATOM_TILE_SIZE 4

extern "C" void generate_first_order_rho_bias_part2_(
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
    int *n_compute_c_batches_ptr,
    int *i_basis_batches_ptr,
    int *atom_valid_n_compute_c_batches_ptr,
    int* i_batch_2_wave_offset_ptr,
    double *wave_batches_compress_ptr,
    double *gradient_wave_batches_compress_ptr,
    double *first_order_rho_bias_part2_ptr,
    // global arrays
    double *partition_tab_ptr,
    double *density_matrix_ptr) {

  // scalars
  int n_my_batches_work = *n_my_batches_work_;
  int j_atom_begin = *j_atom_begin_;
  int j_atom_end = *j_atom_end_;
  int j_coord = *j_coord_; // {1,2,3}
  int n_full_points = *n_full_points_;

  // global scalars
  int n_basis = *n_basis_;
  int n_atoms = *n_atoms_;
  int n_max_compute_ham = *n_max_compute_ham_;
  int n_centers_basis_I = *n_centers_basis_I_;
  int n_max_batch_size = *n_max_batch_size_;

  // arrays
  TMi32<2> TM_INIT(i_basis_batches, n_centers_basis_I, n_my_batches_work);
  TMi32<2> TM_INIT(atom_valid_n_compute_c_batches, n_atoms + 1, n_my_batches_work);
  TMf64<3> TM_INIT(first_order_rho_bias_part2, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  // global arrays
  TMf64<2> TM_INIT(density_matrix, n_basis, n_basis);

  int i_full_points_DM_rho = 0;
  for (int i_my_batch = 0; i_my_batch < n_my_batches_work; i_my_batch++) {

    int n_points = n_point_batches_ptr[i_my_batch];
    int n_compute_c = n_compute_c_batches_ptr[i_my_batch];
    int local_batch_size = batch_sizes_ptr[i_my_batch];

    int n_compute_c_padding = ((n_compute_c + 3) / 4) * 4;

    // Temp Arrays
    Tf64<2> density_matrix_compute(n_compute_c, n_compute_c);
    Tf64<3> first_order_density_matrix_compute(n_compute_c, n_compute_c, ATOM_TILE_SIZE);
    Tf64<2> first_order_wave(n_compute_c, n_points);
    Tf64<3> work1(ATOM_TILE_SIZE, n_compute_c, n_points);
    Tf64<2> work(n_compute_c, n_points);
    Tf64<2> local_first_order_rho(n_max_batch_size, ATOM_TILE_SIZE);

    if (n_compute_c > 0) {

      int *i_basis_index_ptr = &i_basis_batches(0, i_my_batch);

      cTMf64<2> wave(wave_batches_compress_ptr + i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, n_points);
      cTMf64<3> gradient_basis_wave(
          gradient_wave_batches_compress_ptr + 3 * i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, 3, n_points);


      for (int j_compute = 0; j_compute < n_compute_c; j_compute++) {
        int j_basis = i_basis_index_ptr[j_compute] - 1;
        for (int i_compute = 0; i_compute < n_compute_c; i_compute++) {
          int i_basis = i_basis_index_ptr[i_compute] - 1;
          // TODO density_matrix_compute 仅会用到一个维度的全部和另一个维度的
          // valid_i_compute_start:valid_i_compute_end 可以再利用对称性，
          // 固定为 (:,valid_i_compute_start:valid_i_compute_end)
          density_matrix_compute(i_compute, j_compute) = density_matrix(i_basis, j_basis);
        }
      }

      for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
        int j_atom_inner = j_atom - j_atom_begin;

        // INFO first_order_wave 仅有 (valid_i_compute_start:valid_i_compute_end, :) 是有效的
        int valid_i_compute_start = atom_valid_n_compute_c_batches(j_atom - 1, i_my_batch);
        int valid_i_compute_end = atom_valid_n_compute_c_batches(j_atom, i_my_batch);
        int valid_n_compute_count = valid_i_compute_end - valid_i_compute_start;

        if (valid_n_compute_count > 0) {

          for (int i_point = 0; i_point < n_points; i_point++) {
            for (int i_compute = valid_i_compute_start; i_compute < valid_i_compute_end; i_compute++) {
              first_order_wave(i_compute, i_point) = -gradient_basis_wave(i_compute, j_coord - 1, i_point);
            }
          }

          cblas_dgemm(
              CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              n_compute_c,
              n_points,
              valid_n_compute_count,
              1.0,
              &density_matrix_compute(0, valid_i_compute_start),
              n_compute_c,
              &first_order_wave(valid_i_compute_start, 0),
              n_compute_c,
              0.0,
              work.data(),
              n_compute_c);

          for (int i_point = 0; i_point < n_points; i_point++) {
            double acc = 0;
            for (int i = 0; i < n_compute_c; i++) {
              acc += wave(i, i_point) * work(i, i_point);
            }
            // local_first_order_rho(i_point, j_atom_inner) += acc;
            first_order_rho_bias_part2(i_point, i_my_batch, j_atom_inner) = acc;
          }

          cblas_dgemm(
              CblasColMajor,
              CblasNoTrans,
              CblasNoTrans,
              valid_n_compute_count,
              n_points,
              n_compute_c,
              1.0,
              &density_matrix_compute(valid_i_compute_start, 0),
              n_compute_c,
              wave.data(),
              n_compute_c_padding,
              0.0,
              work.data(),
              n_compute_c);

          for (int i_point = 0; i_point < n_points; i_point++) {
            double acc = 0;
            for (int i_compute = valid_i_compute_start; i_compute < valid_i_compute_end; i_compute++) {
              acc += first_order_wave(i_compute, i_point) * work(i_compute - valid_i_compute_start, i_point);
            }
            first_order_rho_bias_part2(i_point, i_my_batch, j_atom_inner) += acc;
          }
        } else {
          for (int i_point = 0; i_point < n_points; i_point++) {
            first_order_rho_bias_part2(i_point, i_my_batch, j_atom_inner) = 0;
          }
        }
      }
    }
  }
}