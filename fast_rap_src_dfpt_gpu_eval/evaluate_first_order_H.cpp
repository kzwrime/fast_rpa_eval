#include <cblas.h>

#include "common.hpp"

// clang-format off
const static int index_hessian[3][3] = {
    {0, 1, 2},
    {1, 3, 4},
    {2, 4, 5}
};
// clang-format on

extern "C" void evaluate_first_order_h_reduce_memory_c_v2_(
    int *j_atom_begin_,
    int *j_atom_end_,
    // scalars
    int *n_full_points_,
    int *j_coord_,
    // global scalars
    int *n_basis_,
    int *n_atoms_,
    int *n_max_compute_ham_,
    int *n_max_batch_size_,
    int *n_my_batches_work_,
    int *n_centers_basis_I_,
    // arrays
    int *n_point_batches_ptr,
    int *n_compute_c_batches_ptr,
    int *i_basis_batches_ptr,
    int *atom_valid_n_compute_c_batches_ptr,
    int *n_point_batches_prefix_sum_ptr,
    int *i_valid_point_2_i_full_points_map_ptr,
    double *vrho_batches_ptr,
    double *vsigma_batches_ptr,
    double *gradient_rho_ptr,
    double *H_times_psi_batches_compress_ptr,
    int *i_batch_2_wave_offset_ptr,
    double *wave_batches_compress_ptr,
    double *gradient_wave_batches_compress_ptr,
    double *hessian_wave_batches_compress_ptr,
    double *first_order_H_tile_ptr,
    double *H_prefactor1_batches_tile_ptr,
    double *first_order_gradient_rho_batches_tile_ptr,
    double *v_hartree_gradient_ptr,
    double *first_order_rho_ptr,
    double *v2rho2_batches_ptr,
    double *v2rhosigma_batches_ptr,
    double *v2sigma2_batches_ptr,

    // global arrays
    double *partition_tab_ptr,
    // final
    int *not_use) {

  int j_atom_begin = *j_atom_begin_;
  int j_atom_end = *j_atom_end_;
  int n_full_points = *n_full_points_;

  int j_coord = *j_coord_; // {1,2,3}

  int n_basis = *n_basis_;
  int n_atoms = *n_atoms_;
  int n_max_compute_ham = *n_max_compute_ham_;

  int n_max_batch_size = *n_max_batch_size_;
  int n_my_batches_work = *n_my_batches_work_;
  int n_centers_basis_I = *n_centers_basis_I_;

  TMi32<2> TM_INIT(i_basis_batches, n_centers_basis_I, n_my_batches_work);

  TMi32<2> TM_INIT(atom_valid_n_compute_c_batches, n_atoms + 1, n_my_batches_work);

  TMf64<3> TM_INIT(first_order_H_tile, n_basis, n_basis, ATOM_TILE_SIZE);
  TMf64<3> TM_INIT(H_prefactor1_batches_tile, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  TMf64<4> TM_INIT(first_order_gradient_rho_batches_tile, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);

  for (int i_my_batch = 0; i_my_batch < n_my_batches_work; i_my_batch++) {

    double *vrho_ptr = &vrho_batches_ptr[1 * n_max_batch_size * i_my_batch];
    double *vsigma_ptr = &vsigma_batches_ptr[3 * n_max_batch_size * i_my_batch];
    double *v2rho2_ptr = &v2rho2_batches_ptr[3 * n_max_batch_size * i_my_batch];
    double *v2rhosigma_ptr = &v2rhosigma_batches_ptr[6 * n_max_batch_size * i_my_batch];
    double *v2sigma2_ptr = &v2sigma2_batches_ptr[6 * n_max_batch_size * i_my_batch];

    for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
      int j_atom_inner = j_atom - j_atom_begin;

      double *first_order_H_ptr = &first_order_H_tile(0, 0, j_atom_inner);
      double *H_prefactor1_ptr = &H_prefactor1_batches_tile(0, i_my_batch, j_atom_inner);
      double *first_order_gradient_rho_ptr = &first_order_gradient_rho_batches_tile(0, 0, i_my_batch, j_atom_inner);

      int n_points = n_point_batches_ptr[i_my_batch];
      int n_compute_c = n_compute_c_batches_ptr[i_my_batch];

      int n_compute_c_padding = ((n_compute_c + 3) / 4) * 4;

      cTMf64<2> H_times_psi(H_times_psi_batches_compress_ptr + i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, n_points);
      cTMf64<2> wave(wave_batches_compress_ptr + i_batch_2_wave_offset_ptr[i_my_batch], n_compute_c_padding, n_points);
      cTMf64<3> gradient_basis_wave(
          gradient_wave_batches_compress_ptr + 3 * i_batch_2_wave_offset_ptr[i_my_batch],
          n_compute_c_padding,
          3,
          n_points);
      cTMf64<3> hessian_basis_wave(
          hessian_wave_batches_compress_ptr + 6 * i_batch_2_wave_offset_ptr[i_my_batch],
          n_compute_c_padding,
          6,
          n_points);

      if (n_compute_c > 0) {

        TMf64<1> TM_INIT(vrho, n_points);
        TMf64<2> TM_INIT(vsigma, 3, n_points);
        TMf64<2> TM_INIT(gradient_rho, 3, n_full_points);

        TMf64<2> TM_INIT(first_order_H, n_basis, n_basis);

        TMf64<1> TM_INIT(H_prefactor1, n_max_batch_size);
        TMf64<2> TM_INIT(first_order_gradient_rho, 3, n_points);
        TMf64<2> TM_INIT(v_hartree_gradient, n_full_points, ATOM_TILE_SIZE);
        TMf64<2> TM_INIT(first_order_rho, n_full_points, ATOM_TILE_SIZE);
        TMf64<2> TM_INIT(v2rho2, 3, n_points);
        TMf64<2> TM_INIT(v2rhosigma, 6, n_points);
        TMf64<2> TM_INIT(v2sigma2, 6, n_points);

        // global arrays
        TMf64<1> TM_INIT(partition_tab, n_full_points);

        //   Tf64<2> local_first_order_gradient_wave(3, n_compute_c);

        int valid_i_compute_start = atom_valid_n_compute_c_batches(j_atom - 1, i_my_batch);
        int valid_i_compute_end = atom_valid_n_compute_c_batches(j_atom, i_my_batch);
        int valid_n_compute_count = valid_i_compute_end - valid_i_compute_start;

        int i_valid_point_start = n_point_batches_prefix_sum_ptr[i_my_batch];
        int i_valid_point_end = n_point_batches_prefix_sum_ptr[i_my_batch + 1];
        int *i_full_points_map_ptr = &i_valid_point_2_i_full_points_map_ptr[i_valid_point_start];
        TMi32<1> TM_INIT(i_full_points_map, n_points);

        Tf64<2> contract_sum_1(n_compute_c, n_points);
        Tf64<2> contract_sum_2(n_compute_c, n_points);
        Tf64<2> local_first_order_H(n_compute_c, n_compute_c);

        for (int i_point = 0; i_point < n_points; ++i_point) {
          double prefactor1 = H_prefactor1(i_point);
          double local_first_order_sigma = 0;
          int i_full_point = i_full_points_map(i_point);

          for (int i = 0; i < 3; i++) {
            local_first_order_sigma += first_order_gradient_rho(i, i_point) * gradient_rho(i, i_full_point);
          }
          local_first_order_sigma *= 2.0;
          double prefactor2[3];
          for (int i = 0; i < 3; ++i) {
            prefactor2[i] =
                2.0 * partition_tab(i_full_point) * gradient_rho(i, i_full_point) *
                    (v2rhosigma(0, i_point) * first_order_rho(i_full_point, j_atom_inner) +
                     v2sigma2(0, i_point) * local_first_order_sigma) +
                2.0 * partition_tab(i_full_point) * vsigma(0, i_point) * first_order_gradient_rho(i, i_point);
          }
          double prefactor3 =
              partition_tab(i_full_point) * (v_hartree_gradient(i_full_point, j_atom_inner) +
                                             v2rho2(0, i_point) * first_order_rho(i_full_point, j_atom_inner) +
                                             v2rhosigma(0, i_point) * local_first_order_sigma);
          double prefactor5_1[3];
          for (int i = 0; i < 3; ++i) {
            prefactor5_1[i] = 2.0 * vsigma(0, i_point) * partition_tab(i_full_point) * gradient_rho(i, i_full_point);
          }

          for (int i_compute = 0; i_compute < n_compute_c; ++i_compute) {

            double contract_1 = (prefactor1 + prefactor3) * wave(i_compute, i_point);
            double contract_2 = 0;
            for (int i = 0; i < 3; i++) {
              contract_2 += prefactor2[i] * gradient_basis_wave(i_compute, i, i_point);
            }
            contract_sum_1(i_compute, i_point) = 0.5 * contract_1 + contract_2;
          }

          for (int i_compute = valid_i_compute_start; i_compute < valid_i_compute_end; i_compute++) {
            double contract_5 = 0;
            for (int i = 0; i < 3; i++) {
              // local_first_order_gradient_wave(i, i_compute) =
              double local_first_order_gradient_wave =
                  -hessian_basis_wave(i_compute, index_hessian[i][j_coord - 1], i_point);
              contract_5 += prefactor5_1[i] * local_first_order_gradient_wave;
            }
            contract_sum_1(i_compute, i_point) += contract_5;
          }
        }

        cblas_dsyr2k(
            CblasColMajor,
            CblasLower,
            CblasNoTrans,
            n_compute_c,
            n_points,
            1.0,
            contract_sum_1.data(),
            n_compute_c,
            wave.data(),
            n_compute_c_padding,
            0.0,
            local_first_order_H.data(),
            n_compute_c);

        for (int i_compute = 0; i_compute < n_compute_c; i_compute += 1) {
          int i_basis = i_basis_batches(i_compute, i_my_batch) - 1;
          first_order_H(i_basis, i_basis) = first_order_H(i_basis, i_basis) + local_first_order_H(i_compute, i_compute);
          for (int j_compute = i_compute + 1; j_compute < n_compute_c; j_compute += 1) {
            int j_basis = i_basis_batches(j_compute, i_my_batch) - 1;
            first_order_H(i_basis, j_basis) += local_first_order_H(j_compute, i_compute);
            first_order_H(j_basis, i_basis) += local_first_order_H(j_compute, i_compute);
          }
        }
      }
    }
  }
  // for(int i=0; i<n_basis * n_basis * (j_atom_end - j_atom_begin + 1); i++){
  //   first_order_H_tile_ptr[i] += first_order_H_bias_part2_tile_ptr[i];
  // }
}