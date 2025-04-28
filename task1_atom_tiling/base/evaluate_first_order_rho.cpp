#include <cblas.h>
#include <vector>

#include "common.hpp"
#include "setting.h"

#include "pass_mod_var.h"

#include "evaluate_first_order_rho_direct_test.hpp"
#include "evaluate_first_order_rho.h"

extern "C" void evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_(
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
  // arrays
  cTMi32<1> TM_INIT(batch_sizes, n_my_batches_work);
  cTMi32<1> TM_INIT(n_point_batches, n_my_batches_work);
  cTMi32<1> TM_INIT(n_point_batches_prefix_sum, n_my_batches_work + 1);
  cTMi32<1> TM_INIT(i_valid_point_2_i_full_points_map, n_full_points);
  cTMi32<1> TM_INIT(n_compute_c_batches, n_my_batches_work);
  cTMi32<2> TM_INIT(i_basis_batches, n_centers_basis_I, n_my_batches_work);
  cTMi32<2> TM_INIT(atom_valid_n_compute_c_batches, n_atoms + 1, n_my_batches_work);
  cTMi32<1> TM_INIT(i_batch_2_wave_offset, n_my_batches_work + 1);

  cTMf64<1> TM_INIT(wave_batches_compress, i_batch_2_wave_offset_ptr[n_my_batches_work]);
  cTMf64<1> TM_INIT(gradient_wave_batches_compress, 3 * i_batch_2_wave_offset_ptr[n_my_batches_work]);
  cTMf64<3> TM_INIT(first_order_density_matrix, n_basis, n_basis, ATOM_TILE_SIZE);
  TMf64<2> TM_INIT(first_order_rho, n_full_points, ATOM_TILE_SIZE);
  cTMf64<3> TM_INIT(first_order_rho_bias_part2, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  TMf64<4> TM_INIT(first_order_gradient_rho, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  cTMf64<4> TM_INIT(first_order_gradient_rho_bias_batches_atoms, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  // global arrays
  cTMi32<1> TM_INIT(basis_atom, n_basis);
  cTMf64<1> TM_INIT(partition_tab, n_full_points);
  cTMf64<2> TM_INIT(density_matrix, n_basis, n_basis);

#ifdef SAVE_DFPT_DATA_TEST

  FirstOrderRhoMetaData data;

  data.n_my_batches_work = n_my_batches_work;
  data.n_full_points = n_full_points;
  data.n_basis = n_basis;
  data.n_atoms = n_atoms;
  data.n_max_compute_ham = n_max_compute_ham;
  data.n_centers_basis_I = n_centers_basis_I;
  data.n_max_batch_size = n_max_batch_size;

  data.basis_atom = std::vector<int>(basis_atom_ptr, basis_atom_ptr+basis_atom.size());
  data.batch_sizes = std::vector<int>(batch_sizes_ptr, batch_sizes_ptr+batch_sizes.size());
  data.n_point_batches = std::vector<int>(n_point_batches_ptr, n_point_batches_ptr+n_point_batches.size());
  data.n_point_batches_prefix_sum = std::vector<int>(n_point_batches_prefix_sum_ptr, n_point_batches_prefix_sum_ptr+n_point_batches_prefix_sum.size());
  data.i_valid_point_2_i_full_points_map = std::vector<int>(i_valid_point_2_i_full_points_map_ptr, i_valid_point_2_i_full_points_map_ptr+i_valid_point_2_i_full_points_map.size());
  data.n_compute_c_batches = std::vector<int>(n_compute_c_batches_ptr, n_compute_c_batches_ptr+n_compute_c_batches.size());
  data.i_basis_batches = std::vector<int>(i_basis_batches_ptr, i_basis_batches_ptr+i_basis_batches.size());
  data.atom_valid_n_compute_c_batches = std::vector<int>(atom_valid_n_compute_c_batches_ptr, atom_valid_n_compute_c_batches_ptr+atom_valid_n_compute_c_batches.size());
  data.i_batch_2_wave_offset = std::vector<int>(i_batch_2_wave_offset_ptr, i_batch_2_wave_offset_ptr+i_batch_2_wave_offset.size());

  if (myid == 0) {
    char filename[1024];
    sprintf(filename, "first_order_rho.meta.test.nproc_%d.bin", n_tasks);
    printf("[DEBUG][SAVE_DFPT_DATA_TEST] write_first_order_rho_meta_data: %s\n", filename);
    write_first_order_rho_meta_data(filename, data);
  }

#endif

  first_order_rho.setZero();
  first_order_gradient_rho.setZero();

  for (int i_my_batch = 0; i_my_batch < n_my_batches_work; i_my_batch++) {

    int n_points = n_point_batches_ptr[i_my_batch];
    int n_compute_c = n_compute_c_batches_ptr[i_my_batch];
    int local_batch_size = batch_sizes_ptr[i_my_batch];

    int n_compute_c_padding = ((n_compute_c + 3) / 4) * 4;

    int i_valid_point_start = n_point_batches_prefix_sum_ptr[i_my_batch];
    int i_valid_point_end = n_point_batches_prefix_sum_ptr[i_my_batch + 1];
    int *i_full_points_map_ptr = &i_valid_point_2_i_full_points_map_ptr[i_valid_point_start];
    TMi32<1> TM_INIT(i_full_points_map, n_points);

    if (n_compute_c > 0) {
      const int *i_basis_index_ptr = &i_basis_batches(0, i_my_batch);

      // arrays
      cTMf64<2> wave(wave_batches_compress_ptr + i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, n_points);
      cTMf64<3> gradient_basis_wave(
          gradient_wave_batches_compress_ptr + 3 * i_batch_2_wave_offset_ptr[i_my_batch],
          n_compute_c_padding,
          3,
          n_points);

      // Temp Arrays
      Tf64<3> first_order_density_matrix_compute(ATOM_TILE_SIZE, n_compute_c, n_compute_c);
      Tf64<2> first_order_wave(n_compute_c, n_points);
      Tf64<3> work1(ATOM_TILE_SIZE, n_compute_c, n_points);
      Tf64<2> work(n_compute_c, n_points);
      Tf64<2> local_first_order_rho(n_max_batch_size, ATOM_TILE_SIZE);

      // first_order_wave.setZero();

      for (int j_compute = 0; j_compute < n_compute_c; j_compute++) {
        int j_basis = i_basis_index_ptr[j_compute] - 1;
        for (int i_compute = 0; i_compute < n_compute_c; i_compute++) {
          int i_basis = i_basis_index_ptr[i_compute] - 1;
          // TODO 最后将 first_order_density_matrix 整个改成 (ATOM_TILE_SIZE, n_basis, n_basis)
          for (int i = 0; i < ATOM_TILE_SIZE; i++) {
            first_order_density_matrix_compute(i, i_compute, j_compute) =
                first_order_density_matrix(i_basis, j_basis, i);
          }
        }
      }

      cblas_dgemm(
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          ATOM_TILE_SIZE * n_compute_c,
          n_points,
          n_compute_c,
          1.0,
          first_order_density_matrix_compute.data(),
          ATOM_TILE_SIZE * n_compute_c,
          wave.data(),
          n_compute_c_padding,
          0.0,
          work1.data(),
          ATOM_TILE_SIZE * n_compute_c);

      for (int i_point = 0; i_point < n_points; i_point++) {
        double acc[ATOM_TILE_SIZE] = { 0 };
        double acc_grad[3][ATOM_TILE_SIZE] = { 0 };
        for (int ic = 0; ic < n_compute_c; ic++) {
          for (int i = 0; i < ATOM_TILE_SIZE; i++) {
            acc[i] += wave(ic, i_point) * work1(i, ic, i_point);
          }
          for (int j = 0; j < 3; j++) {
            for (int i = 0; i < ATOM_TILE_SIZE; i++) {
              acc_grad[j][i] += gradient_basis_wave(ic, j, i_point) * work1(i, ic, i_point);
            }
          }
        }
        for (int i = 0; i < ATOM_TILE_SIZE; i++) {
          local_first_order_rho(i_point, i) = acc[i] + first_order_rho_bias_part2(i_point, i_my_batch, i);
        }
        // INFO GGA only
        for (int i = 0; i < ATOM_TILE_SIZE; i++) {
          for (int j = 0; j < 3; j++) {
            first_order_gradient_rho(j, i_point, i_my_batch, i) =
                acc_grad[j][i] * 2 - first_order_gradient_rho_bias_batches_atoms(j, i_point, i_my_batch, i);
          }
        }
      }

      for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
        int j_atom_inner = j_atom - j_atom_begin;
        for (int i_point = 0; i_point < n_points; i_point++) {
          first_order_rho(i_full_points_map(i_point), j_atom_inner) = local_first_order_rho(i_point, j_atom_inner);
        }
      }
    }
  }
}
