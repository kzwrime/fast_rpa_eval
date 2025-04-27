#include "common.hpp"
#include "device.hpp"
#include "device_data.hpp"
#include "setting.h"

DevicePtrs devPs;
DfptCommonScalars dfpt_common_scalars;
HostFirstOrderRhoData host_first_order_rho_data;

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

    double *delta_v_hartree_ptr) {

  // scalars
  [[maybe_unused]] int n_my_batches_work = *n_my_batches_work_;
  [[maybe_unused]] int n_full_points = *n_full_points_;
  [[maybe_unused]] int n_valid_points = *n_valid_points_;
  // global scalars
  [[maybe_unused]] int n_basis = *n_basis_;
  [[maybe_unused]] int n_atoms = *n_atoms_;
  [[maybe_unused]] int n_max_compute_ham = *n_max_compute_ham_;
  [[maybe_unused]] int n_centers_basis_I = *n_centers_basis_I_;
  [[maybe_unused]] int n_max_batch_size = *n_max_batch_size_;
  [[maybe_unused]] int l_pot_max = *l_pot_max_;
  [[maybe_unused]] int n_max_radial = *n_max_radial_;
  [[maybe_unused]] int n_hartree_grid = *n_hartree_grid_;
  [[maybe_unused]] int n_centers = *n_centers_;
  [[maybe_unused]] int n_centers_hartree_potential = *n_centers_hartree_potential_;
  [[maybe_unused]] int hartree_force_l_add = *hartree_force_l_add_;
  [[maybe_unused]] int n_species = *n_species_;
  [[maybe_unused]] int l_max_analytic_multipole = *l_max_analytic_multipole_;
  [[maybe_unused]] int index_cc_dim_0 = *index_cc_dim_0_;

  const int l_pot_max_pow2 = (l_pot_max + 1) * (l_pot_max + 1);

  // Const Arrays from fortran
  TMi32<1> TM_INIT(basis_atom, n_basis);
  TMi32<1> TM_INIT(batch_sizes, n_my_batches_work);
  TMi32<1> TM_INIT(n_point_batches, n_my_batches_work);
  TMi32<1> TM_INIT(n_point_batches_prefix_sum, n_my_batches_work + 1);
  TMi32<1> TM_INIT(i_valid_point_2_i_full_points_map, n_full_points);
  TMi32<1> TM_INIT(n_compute_c_batches, n_my_batches_work);
  TMi32<2> TM_INIT(i_basis_batches, n_centers_basis_I, n_my_batches_work);
  TMi32<2> TM_INIT(atom_valid_n_compute_c_batches, n_atoms + 1, n_my_batches_work);
  TMi32<1> TM_INIT(i_batch_2_wave_offset, n_my_batches_work + 1);
  TMf64<1> TM_INIT(wave_batches_compress, i_batch_2_wave_offset_ptr[n_my_batches_work]);
  TMf64<1> TM_INIT(gradient_wave_batches_compress, 3 * i_batch_2_wave_offset_ptr[n_my_batches_work]);
  TMf64<2> TM_INIT(density_matrix, n_basis, n_basis);

  TMi32<1> TM_INIT(n_radial, n_species);
  TMi32<1> TM_INIT(species, n_atoms);
  TMi32<1> TM_INIT(species_center, n_centers);
  TMf64<2> TM_INIT(coord_points, 3, n_full_points);
  TMi32<1> TM_INIT(l_hartree, n_species);
  TMi32<1> TM_INIT(n_grid, n_species);
  TMi32<1> TM_INIT(n_cc_lm_ijk, l_max_analytic_multipole + 1);
  TMi32<1> TM_INIT(centers_hartree_potential, n_centers_hartree_potential);
  TMi32<1> TM_INIT(center_to_atom, n_centers);
  TMi32<2> TM_INIT(index_cc, index_cc_dim_0, 6);
  TMi32<2> TM_INIT(index_ijk_max_cc, 3, l_max_analytic_multipole + 1);
  TMf64<1> TM_INIT(cc, index_cc_dim_0);
  TMf64<2> TM_INIT(coords_center, 3, n_centers);
  TMf64<1> TM_INIT(r_grid_min, n_species);
  TMf64<1> TM_INIT(log_r_grid_inc, n_species);
  TMf64<1> TM_INIT(scale_radial, n_species);
  TMf64<1> TM_INIT(partition_tab, n_full_points);

  // Changeable Arrays from fortran
  TMf64<3> TM_INIT(first_order_density_matrix, n_basis, n_basis, ATOM_TILE_SIZE);
  TMf64<2> TM_INIT(first_order_rho, n_full_points, ATOM_TILE_SIZE);
  TMf64<3> TM_INIT(first_order_rho_bias_part2, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  TMf64<4> TM_INIT(first_order_gradient_rho, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);
  TMf64<4> TM_INIT(first_order_gradient_rho_bias_batches_atoms, 3, n_max_batch_size, n_my_batches_work, ATOM_TILE_SIZE);

  TMi32<2> TM_INIT(l_hartree_max_far_distance, n_atoms, ATOM_TILE_SIZE);
  TMf64<2> TM_INIT(multipole_radius_sq, n_atoms, ATOM_TILE_SIZE);
  TMf64<3> TM_INIT(outer_potential_radius, l_pot_max + 1, n_atoms, ATOM_TILE_SIZE);
  TMf64<3> TM_INIT(multipole_moments, l_pot_max_pow2, n_atoms, ATOM_TILE_SIZE);
  TMf64<5> TM_INIT(
      current_delta_v_hart_part_spl_tile, l_pot_max_pow2, n_coeff_hartree, n_hartree_grid, n_atoms, ATOM_TILE_SIZE);

  TMf64<2> TM_INIT(delta_v_hartree, n_full_points, ATOM_TILE_SIZE);

  // global arrays

  // ================================================

  // Const Arrays from fortran
  TM_DEV_PS_INIT(basis_atom);
  TM_DEV_PS_INIT(batch_sizes);
  TM_DEV_PS_INIT(n_point_batches);
  TM_DEV_PS_INIT(n_point_batches_prefix_sum);
  TM_DEV_PS_INIT(i_valid_point_2_i_full_points_map);
  TM_DEV_PS_INIT(n_compute_c_batches);
  TM_DEV_PS_INIT(i_basis_batches);
  TM_DEV_PS_INIT(atom_valid_n_compute_c_batches);
  TM_DEV_PS_INIT(i_batch_2_wave_offset);
  TM_DEV_PS_INIT(wave_batches_compress);
  TM_DEV_PS_INIT(gradient_wave_batches_compress);
  TM_DEV_PS_INIT(density_matrix);

  TM_DEV_PS_INIT(n_radial);
  TM_DEV_PS_INIT(species);
  TM_DEV_PS_INIT(species_center);
  TM_DEV_PS_INIT(coord_points);
  TM_DEV_PS_INIT(l_hartree);
  TM_DEV_PS_INIT(n_grid);
  TM_DEV_PS_INIT(n_cc_lm_ijk);
  TM_DEV_PS_INIT(centers_hartree_potential);
  TM_DEV_PS_INIT(center_to_atom);
  TM_DEV_PS_INIT(index_cc);
  TM_DEV_PS_INIT(index_ijk_max_cc);
  TM_DEV_PS_INIT(cc);
  TM_DEV_PS_INIT(coords_center);
  TM_DEV_PS_INIT(r_grid_min);
  TM_DEV_PS_INIT(log_r_grid_inc);
  TM_DEV_PS_INIT(scale_radial);
  TM_DEV_PS_INIT(partition_tab);

  // Changeable Arrays from fortran
  TM_DEV_PS_INIT(first_order_density_matrix);
  TM_DEV_PS_INIT(first_order_rho);
  TM_DEV_PS_INIT(first_order_rho_bias_part2);
  TM_DEV_PS_INIT(first_order_gradient_rho);
  TM_DEV_PS_INIT(first_order_gradient_rho_bias_batches_atoms);

  TM_DEV_PS_INIT(l_hartree_max_far_distance);
  TM_DEV_PS_INIT(multipole_radius_sq);
  TM_DEV_PS_INIT(outer_potential_radius);
  TM_DEV_PS_INIT(multipole_moments);
  TM_DEV_PS_INIT(current_delta_v_hart_part_spl_tile);

  TM_DEV_PS_INIT(delta_v_hartree);

  // 初始化用于 evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_ 的临时数组
  devPs.first_order_density_matrix_compute_batches.size =
      ATOM_TILE_SIZE * n_max_compute_ham * n_max_compute_ham * N_BATCHES_TILE;
  devPs.work1_batches.size = ATOM_TILE_SIZE * n_max_compute_ham * n_max_batch_size * N_BATCHES_TILE;
  DEV_CHECK(DEV_MALLOC(
      (void **)&devPs.first_order_density_matrix_compute_batches.ptr,
      devPs.first_order_density_matrix_compute_batches.byte_size()));
  DEV_CHECK(DEV_MALLOC((void **)&devPs.work1_batches.ptr, devPs.work1_batches.byte_size()));

  devPs.first_order_density_matrix_atom_inner_trans.size = ATOM_TILE_SIZE * n_basis * n_basis;
  DEV_CHECK(DEV_MALLOC(
      (void **)&devPs.first_order_density_matrix_atom_inner_trans.ptr,
      devPs.first_order_density_matrix_atom_inner_trans.byte_size()));

  devPs.multipole_c.size = n_cc_lm_ijk_ptr[l_pot_max] * n_atoms * ATOM_TILE_SIZE;
  DEV_CHECK(DEV_MALLOC((void **)&devPs.multipole_c.ptr, devPs.multipole_c.byte_size()));

  devPs.current_delta_v_hart_part_spl_tile_trans.size =
      l_pot_max_pow2 * n_coeff_hartree * n_hartree_grid * n_atoms * ATOM_TILE_SIZE;
  DEV_CHECK(DEV_MALLOC(
      (void **)&devPs.current_delta_v_hart_part_spl_tile_trans.ptr,
      devPs.current_delta_v_hart_part_spl_tile_trans.byte_size()));

  // ================================================

  // 计算用于 evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_ 中
  // magmablas_dgemm_vbatched_max_nocheck 的 ldas/ldbs/ldcs 以及 ptrs

  double *work1_dev = nullptr;
  DEV_CHECK(DEV_MALLOC((void **)&work1_dev, sizeof(double) * ATOM_TILE_SIZE * n_max_compute_ham * n_max_batch_size));

  Ti32<1> i_valid_batch_2_i_batch(n_my_batches_work + 1);
  Ti32<1> n_point_valid_batches(n_my_batches_work + 1);
  Ti32<1> n_compute_c_valid_batches(n_my_batches_work + 1);
  Ti32<1> n_compute_c_padding_valid_batches(ALIGN_UP(n_my_batches_work, N_BATCHES_TILE) + N_BATCHES_TILE + 1);
  Ti32<1> n_compute_c_mul_atom_tile_size_valid_batches(n_my_batches_work + 1);
  ETT<double *, 1> wave_dev_ptrs(n_my_batches_work);

  ETT<double *, 1> first_order_density_matrix_compute_ptrs(N_BATCHES_TILE);
  ETT<double *, 1> work1_batches_ptrs(N_BATCHES_TILE);
  Ti32<1> first_order_density_matrix_compute_ldas(N_BATCHES_TILE + 1);
  Ti32<1> work1_batches_ldas(N_BATCHES_TILE + 1);

  int i_valid_batch = 0;
  for (int i_batch = 0; i_batch < n_my_batches_work; i_batch++) {
    int n_compute_c = n_compute_c_batches_ptr[i_batch];
    int n_compute_c_padding = ALIGN_UP(n_compute_c, N_COMPUTE_C_PADDING_SIZE);

    if (n_compute_c > 0) {
      i_valid_batch_2_i_batch(i_valid_batch) = i_batch;
      n_point_valid_batches(i_valid_batch) = n_point_batches_ptr[i_batch];
      n_compute_c_valid_batches(i_valid_batch) = n_compute_c;
      n_compute_c_padding_valid_batches(i_valid_batch) = n_compute_c_padding;
      n_compute_c_mul_atom_tile_size_valid_batches(i_valid_batch) = n_compute_c * ATOM_TILE_SIZE;

      wave_dev_ptrs(i_valid_batch) = &devPs.wave_batches_compress.ptr[i_batch_2_wave_offset_ptr[i_batch]];

      i_valid_batch++;
    }
  }
  const int n_valid_batches = i_valid_batch;
  dfpt_common_scalars.n_valid_batches = n_valid_batches;

  host_first_order_rho_data.work1_max_m = Ti32<1>(ALIGN_UP(n_my_batches_work, N_BATCHES_TILE));
  host_first_order_rho_data.work1_max_n = Ti32<1>(ALIGN_UP(n_my_batches_work, N_BATCHES_TILE));
  host_first_order_rho_data.work1_max_k = Ti32<1>(ALIGN_UP(n_my_batches_work, N_BATCHES_TILE));

  for (int i_my_batch_outer = 0; i_my_batch_outer < n_valid_batches; i_my_batch_outer += N_BATCHES_TILE) {
    const int real_n_batches_tile = std::min(N_BATCHES_TILE, n_valid_batches - i_my_batch_outer);
    int max_m = 0;
    int max_n = 0;
    int max_k = 0;
    for (int i_batch_inner = 0; i_batch_inner < real_n_batches_tile; i_batch_inner++) {
      max_m = std::max(max_m, n_compute_c_mul_atom_tile_size_valid_batches(i_my_batch_outer + i_batch_inner));
      max_n = std::max(max_n, n_point_valid_batches(i_my_batch_outer + i_batch_inner));
      max_k = std::max(max_k, n_compute_c_valid_batches(i_my_batch_outer + i_batch_inner));
    }
    host_first_order_rho_data.work1_max_m(i_my_batch_outer / N_BATCHES_TILE) = max_m;
    host_first_order_rho_data.work1_max_n(i_my_batch_outer / N_BATCHES_TILE) = max_n;
    host_first_order_rho_data.work1_max_k(i_my_batch_outer / N_BATCHES_TILE) = max_k;
  }

  for (int i = 0; i < N_BATCHES_TILE; i++) {
    first_order_density_matrix_compute_ptrs(i) = &devPs.first_order_density_matrix_compute_batches
                                                      .ptr[ATOM_TILE_SIZE * n_max_compute_ham * n_max_compute_ham * i];
    work1_batches_ptrs(i) = &devPs.work1_batches.ptr[ATOM_TILE_SIZE * n_max_compute_ham * n_max_batch_size * i];

    first_order_density_matrix_compute_ldas(i) = ATOM_TILE_SIZE * n_max_compute_ham;
    work1_batches_ldas(i) = ATOM_TILE_SIZE * n_max_compute_ham;
  }

  TM_DEV_PS_INIT(i_valid_batch_2_i_batch);
  TM_DEV_PS_INIT(n_point_valid_batches);
  TM_DEV_PS_INIT(n_compute_c_valid_batches);
  TM_DEV_PS_INIT(n_compute_c_padding_valid_batches);
  TM_DEV_PS_INIT(n_compute_c_mul_atom_tile_size_valid_batches);
  TM_DEV_PS_INIT(first_order_density_matrix_compute_ldas);
  TM_DEV_PS_INIT(work1_batches_ldas);

  TM_DEV_PS_INIT(first_order_density_matrix_compute_ptrs);
  TM_DEV_PS_INIT(wave_dev_ptrs);
  TM_DEV_PS_INIT(work1_batches_ptrs);

  Ti32<2> index_lm(l_pot_max * 2 + 1, l_pot_max + 1); // index_lm(-l_pot_max:l_pot_max, 0:l_pot_max )
  int i_index = 0;
  for (int i_l = 0; i_l <= l_pot_max; i_l++) {
    for (int i_m = -i_l; i_m <= i_l; i_m++) {
      i_index++;
      index_lm(i_m + l_pot_max, i_l) = i_index;
    }
  }
  TM_DEV_PS_INIT(index_lm);

  // ================================================

  DEV_STREAM_T &stream = devInfo.stream;

  // Const Arrays from fortran
  TM_DEV_PS_H2D_H(basis_atom);
  TM_DEV_PS_H2D_H(batch_sizes);
  TM_DEV_PS_H2D_H(n_point_batches);
  TM_DEV_PS_H2D_H(n_point_batches_prefix_sum);
  TM_DEV_PS_H2D_H(i_valid_point_2_i_full_points_map);
  TM_DEV_PS_H2D_H(n_compute_c_batches);
  TM_DEV_PS_H2D_H(i_basis_batches);
  TM_DEV_PS_H2D_H(atom_valid_n_compute_c_batches);
  TM_DEV_PS_H2D_H(i_batch_2_wave_offset);
  TM_DEV_PS_H2D_H(wave_batches_compress);
  TM_DEV_PS_H2D_H(gradient_wave_batches_compress);
  TM_DEV_PS_H2D_H(density_matrix);

  TM_DEV_PS_H2D_H(n_radial);
  TM_DEV_PS_H2D_H(species);
  TM_DEV_PS_H2D_H(species_center);
  TM_DEV_PS_H2D_H(coord_points);
  TM_DEV_PS_H2D_H(l_hartree);
  TM_DEV_PS_H2D_H(n_grid);
  TM_DEV_PS_H2D_H(n_cc_lm_ijk);
  TM_DEV_PS_H2D_H(centers_hartree_potential);
  TM_DEV_PS_H2D_H(center_to_atom);
  TM_DEV_PS_H2D_H(index_cc);
  TM_DEV_PS_H2D_H(index_ijk_max_cc);
  TM_DEV_PS_H2D_H(cc);
  TM_DEV_PS_H2D_H(coords_center);
  TM_DEV_PS_H2D_H(r_grid_min);
  TM_DEV_PS_H2D_H(log_r_grid_inc);
  TM_DEV_PS_H2D_H(scale_radial);
  TM_DEV_PS_H2D_H(partition_tab);

  // Const Arrays initialized in cpp
  TM_DEV_PS_H2D_H(i_valid_batch_2_i_batch);
  TM_DEV_PS_H2D_H(n_point_valid_batches);
  TM_DEV_PS_H2D_H(n_compute_c_valid_batches);
  TM_DEV_PS_H2D_H(n_compute_c_padding_valid_batches);
  TM_DEV_PS_H2D_H(n_compute_c_mul_atom_tile_size_valid_batches);
  TM_DEV_PS_H2D_H(first_order_density_matrix_compute_ldas);
  TM_DEV_PS_H2D_H(work1_batches_ldas);
  TM_DEV_PS_H2D_H(first_order_density_matrix_compute_ptrs);
  TM_DEV_PS_H2D_H(wave_dev_ptrs);
  TM_DEV_PS_H2D_H(work1_batches_ptrs);

  TM_DEV_PS_H2D_H(index_lm);
}

void free_dfpt_device_data() {
  // Const Arrays from fortran
  TM_DEV_PS_FREE(basis_atom);
  TM_DEV_PS_FREE(batch_sizes);
  TM_DEV_PS_FREE(n_point_batches);
  TM_DEV_PS_FREE(n_point_batches_prefix_sum);
  TM_DEV_PS_FREE(i_valid_point_2_i_full_points_map);
  TM_DEV_PS_FREE(n_compute_c_batches);
  TM_DEV_PS_FREE(i_basis_batches);
  TM_DEV_PS_FREE(atom_valid_n_compute_c_batches);
  TM_DEV_PS_FREE(i_batch_2_wave_offset);
  TM_DEV_PS_FREE(wave_batches_compress);
  TM_DEV_PS_FREE(gradient_wave_batches_compress);
  TM_DEV_PS_FREE(density_matrix);

  TM_DEV_PS_FREE(n_radial);
  TM_DEV_PS_FREE(species);
  TM_DEV_PS_FREE(species_center);
  TM_DEV_PS_FREE(coord_points);
  TM_DEV_PS_FREE(l_hartree);
  TM_DEV_PS_FREE(n_grid);
  TM_DEV_PS_FREE(n_cc_lm_ijk);
  TM_DEV_PS_FREE(centers_hartree_potential);
  TM_DEV_PS_FREE(center_to_atom);
  TM_DEV_PS_FREE(index_cc);
  TM_DEV_PS_FREE(index_ijk_max_cc);
  TM_DEV_PS_FREE(cc);
  TM_DEV_PS_FREE(coords_center);
  TM_DEV_PS_FREE(r_grid_min);
  TM_DEV_PS_FREE(log_r_grid_inc);
  TM_DEV_PS_FREE(scale_radial);
  TM_DEV_PS_FREE(partition_tab);

  // Changeable Arrays from fortran
  TM_DEV_PS_FREE(first_order_density_matrix);
  TM_DEV_PS_FREE(first_order_rho);
  TM_DEV_PS_FREE(first_order_rho_bias_part2);
  TM_DEV_PS_FREE(first_order_gradient_rho);
  TM_DEV_PS_FREE(first_order_gradient_rho_bias_batches_atoms);

  TM_DEV_PS_FREE(l_hartree_max_far_distance);
  TM_DEV_PS_FREE(multipole_radius_sq);
  TM_DEV_PS_FREE(outer_potential_radius);
  TM_DEV_PS_FREE(multipole_moments);
  TM_DEV_PS_FREE(current_delta_v_hart_part_spl_tile);

  TM_DEV_PS_FREE(delta_v_hartree);

  // Const Arrays initialized in cpp
  TM_DEV_PS_FREE(i_valid_batch_2_i_batch);
  TM_DEV_PS_FREE(n_point_valid_batches);
  TM_DEV_PS_FREE(n_compute_c_valid_batches);
  TM_DEV_PS_FREE(n_compute_c_padding_valid_batches);
  TM_DEV_PS_FREE(n_compute_c_mul_atom_tile_size_valid_batches);
  TM_DEV_PS_FREE(first_order_density_matrix_compute_ldas);
  TM_DEV_PS_FREE(work1_batches_ldas);
  TM_DEV_PS_FREE(first_order_density_matrix_compute_ptrs);
  TM_DEV_PS_FREE(wave_dev_ptrs);
  TM_DEV_PS_FREE(work1_batches_ptrs);

  TM_DEV_PS_FREE(index_lm);

  // 用于 Kernel 自身 / Kernel 之间使用的临时数组
  TM_DEV_PS_FREE(first_order_density_matrix_compute_batches);
  TM_DEV_PS_FREE(work1_batches);
  TM_DEV_PS_FREE(first_order_density_matrix_atom_inner_trans);

  TM_DEV_PS_FREE(multipole_c);
  TM_DEV_PS_FREE(current_delta_v_hart_part_spl_tile_trans);
}