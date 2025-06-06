int __mpi_tasks_MOD_n_tasks = 1;
int __mpi_tasks_MOD_myid = 0;

#include "../common.hpp"
#include "../evaluate_first_order_rho_direct_test.hpp"
#include "evaluate_first_order_rho.h"

#include "../device.hpp"
#include "../device_data.hpp"

#include "../sum_up_direct_test.hpp"

void call_with_read_data(HartreePotentialData &sumup_data, FirstOrderRhoMetaData &rho_data) {

  int j_atom_begin_ = 1;
  int j_atom_end_ = j_atom_begin_ + ATOM_TILE_SIZE - 1;
  int j_coord_ = 1;

  // =======================================================
  // sumup random arrays, just for test ====================
  Tf64<1> delta_v_hartree(sumup_data.n_full_points * ATOM_TILE_SIZE);
  Tf64<1> delta_v_hartree_ref(sumup_data.n_full_points * ATOM_TILE_SIZE);

  const int l_pot_max_pow2 = (sumup_data.l_pot_max + 1) * (sumup_data.l_pot_max + 1);

  Tf64<5> current_delta_v_hart_part_spl_tile(
      l_pot_max_pow2, n_coeff_hartree, sumup_data.n_hartree_grid, sumup_data.n_atoms, ATOM_TILE_SIZE);

  current_delta_v_hart_part_spl_tile.setRandom();
  delta_v_hartree_ref.setRandom();
  delta_v_hartree = delta_v_hartree_ref;

  // =======================================================
  // rho & H random arrays, just for test ==================
  Tf64<2> first_order_rho_ref(rho_data.n_full_points, ATOM_TILE_SIZE);
  Tf64<4> first_order_gradient_rho_ref(3, rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);
  Tf64<2> first_order_rho(rho_data.n_full_points, ATOM_TILE_SIZE);
  Tf64<4> first_order_gradient_rho(3, rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);

  Tf64<1> wave_batches_compress(rho_data.i_batch_2_wave_offset[rho_data.n_my_batches_work]);
  Tf64<1> gradient_wave_batches_compress(3 * rho_data.i_batch_2_wave_offset[rho_data.n_my_batches_work]);
  Tf64<3> first_order_density_matrix(rho_data.n_basis, rho_data.n_basis, ATOM_TILE_SIZE);
  Tf64<3> first_order_rho_bias_part2(rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);
  Tf64<4> first_order_gradient_rho_bias_batches_atoms(
      3, rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);
  Ti32<1> basis_atom(rho_data.n_basis);
  Tf64<1> partition_tab(rho_data.n_full_points);
  Tf64<2> density_matrix(rho_data.n_basis, rho_data.n_basis);

  // =======================================================

  // Initialize tensors with random values between 0 and 1
  first_order_rho_ref.setZero();
  first_order_gradient_rho_ref.setZero();
  first_order_rho.setZero();
  first_order_gradient_rho.setZero();

  wave_batches_compress.setRandom();
  gradient_wave_batches_compress.setRandom();
  first_order_density_matrix.setRandom();
  first_order_rho_bias_part2.setRandom();
  first_order_gradient_rho_bias_batches_atoms.setRandom();
  basis_atom.setRandom();
  partition_tab.setRandom();
  density_matrix.setRandom();

  // CPU version, just for reference & verification
  evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_(
      &rho_data.n_my_batches_work,
      &j_atom_begin_,
      &j_atom_end_,
      &j_coord_,
      &rho_data.n_full_points,
      &rho_data.n_basis,
      &rho_data.n_atoms,
      &rho_data.n_max_compute_ham,
      &rho_data.n_centers_basis_I,
      &rho_data.n_max_batch_size,
      rho_data.batch_sizes.data(),
      rho_data.n_point_batches.data(),
      rho_data.n_point_batches_prefix_sum.data(),
      rho_data.i_valid_point_2_i_full_points_map.data(),
      rho_data.n_compute_c_batches.data(),
      rho_data.i_basis_batches.data(),
      rho_data.atom_valid_n_compute_c_batches.data(),
      rho_data.i_batch_2_wave_offset.data(),
      wave_batches_compress.data(),
      gradient_wave_batches_compress.data(),
      first_order_density_matrix.data(),
      first_order_rho_ref.data(),
      first_order_rho_bias_part2.data(),
      first_order_gradient_rho_ref.data(),
      first_order_gradient_rho_bias_batches_atoms.data(),

      basis_atom.data(),
      partition_tab.data(),
      density_matrix.data());

  init_dfpt_device_data(
      &rho_data.n_my_batches_work,
      &sumup_data.n_full_points,
      &sumup_data.n_valid_points,
      &rho_data.n_basis,
      &sumup_data.n_atoms,
      &rho_data.n_max_compute_ham,
      &rho_data.n_centers_basis_I,
      &rho_data.n_max_batch_size,
      &sumup_data.l_pot_max,
      &sumup_data.n_max_radial,
      &sumup_data.n_hartree_grid,
      &sumup_data.n_centers,
      &sumup_data.n_centers_hartree_potential,
      &sumup_data.hartree_force_l_add,
      &sumup_data.n_species,
      &sumup_data.l_max_analytic_multipole,
      &sumup_data.index_cc_dim_0,

      rho_data.basis_atom.data(),
      rho_data.batch_sizes.data(),
      rho_data.n_point_batches.data(),
      rho_data.n_point_batches_prefix_sum.data(),
      rho_data.i_valid_point_2_i_full_points_map.data(),
      rho_data.n_compute_c_batches.data(),
      rho_data.i_basis_batches.data(),
      rho_data.atom_valid_n_compute_c_batches.data(),
      rho_data.i_batch_2_wave_offset.data(),
      wave_batches_compress.data(),
      gradient_wave_batches_compress.data(),
      density_matrix.data(),

      sumup_data.n_radial.data(),
      sumup_data.species.data(),
      sumup_data.species_center.data(),
      sumup_data.coord_points.data(),
      sumup_data.l_hartree.data(),
      sumup_data.n_grid.data(),
      sumup_data.n_cc_lm_ijk.data(),
      sumup_data.centers_hartree_potential.data(),
      sumup_data.center_to_atom.data(),
      sumup_data.index_cc.data(),
      sumup_data.index_ijk_max_cc.data(),
      sumup_data.cc.data(),
      sumup_data.coords_center.data(),
      sumup_data.r_grid_min.data(),
      sumup_data.log_r_grid_inc.data(),
      sumup_data.scale_radial.data(),
      sumup_data.partition_tab.data(),

      first_order_density_matrix.data(),
      first_order_rho_ref.data(),
      first_order_rho_bias_part2.data(),
      first_order_gradient_rho_ref.data(),
      first_order_gradient_rho_bias_batches_atoms.data(),

      sumup_data.l_hartree_max_far_distance.data(),
      sumup_data.multipole_radius_sq.data(),
      sumup_data.outer_potential_radius.data(),
      sumup_data.multipole_moments.data(),
      current_delta_v_hart_part_spl_tile.data(),

      delta_v_hartree_ref.data());

  DEV_STREAM_T &stream = devInfo.stream;
  TM_DEV_PS_H2D_H(first_order_density_matrix);
  TM_DEV_PS_H2D_H(first_order_rho_bias_part2);
  TM_DEV_PS_H2D_H(first_order_gradient_rho_bias_batches_atoms);

  // GPU version
  evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_(
      &rho_data.n_my_batches_work,
      &j_atom_begin_,
      &j_atom_end_,
      &j_coord_,
      &rho_data.n_full_points,
      &rho_data.n_basis,
      &rho_data.n_atoms,
      &rho_data.n_max_compute_ham,
      &rho_data.n_centers_basis_I,
      &rho_data.n_max_batch_size,
      rho_data.batch_sizes.data(),
      rho_data.n_point_batches.data(),
      rho_data.n_point_batches_prefix_sum.data(),
      rho_data.i_valid_point_2_i_full_points_map.data(),
      rho_data.n_compute_c_batches.data(),
      rho_data.i_basis_batches.data(),
      rho_data.atom_valid_n_compute_c_batches.data(),
      rho_data.i_batch_2_wave_offset.data(),
      wave_batches_compress.data(),
      gradient_wave_batches_compress.data(),
      first_order_density_matrix.data(),
      first_order_rho.data(),
      first_order_rho_bias_part2.data(),
      first_order_gradient_rho.data(),
      first_order_gradient_rho_bias_batches_atoms.data(),

      basis_atom.data(),
      partition_tab.data(),
      density_matrix.data());

  // Verification
  TM_DEV_PS_D2H_H(first_order_rho);
  TM_DEV_PS_D2H_H(first_order_gradient_rho);

  int failed_count = 0;
  for (int i = 0; i < rho_data.n_full_points; i++) {
    if (std::abs(first_order_rho.data()[i] - first_order_rho_ref.data()[i]) > 1E-6) {
      failed_count++;
      printf(
          "first_order_rho error! i_full_point %9d, %.18f != %.18f\n",
          i,
          first_order_rho.data()[i],
          first_order_rho_ref.data()[i]);
    }
    if (failed_count > 10) {
      printf("Failed to many times (failed_count > %d), exit.\n", failed_count);
      break;
    }
  }
  printf("Check first_order_rho finished.\n");

  for (int i = 0; i < rho_data.n_full_points; i++) {
    if (std::abs(first_order_gradient_rho.data()[i] - first_order_gradient_rho_ref.data()[i]) > 1E-6) {
      failed_count++;
      printf(
          "first_order_gradient_rho error! i_full_point %9d, %.18f != %.18f\n",
          i,
          first_order_gradient_rho.data()[i],
          first_order_gradient_rho_ref.data()[i]);
    }
    if (failed_count > 10) {
      printf("Failed to many times (failed_count > %d), exit.\n", failed_count);
      break;
    }
  }
  printf("Check first_order_gradient_rho finished.\n");

  int warm_up = 10;
  for (int iter = 0; iter < warm_up; iter++) {
    evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_(
        &rho_data.n_my_batches_work,
        &j_atom_begin_,
        &j_atom_end_,
        &j_coord_,
        &rho_data.n_full_points,
        &rho_data.n_basis,
        &rho_data.n_atoms,
        &rho_data.n_max_compute_ham,
        &rho_data.n_centers_basis_I,
        &rho_data.n_max_batch_size,
        rho_data.batch_sizes.data(),
        rho_data.n_point_batches.data(),
        rho_data.n_point_batches_prefix_sum.data(),
        rho_data.i_valid_point_2_i_full_points_map.data(),
        rho_data.n_compute_c_batches.data(),
        rho_data.i_basis_batches.data(),
        rho_data.atom_valid_n_compute_c_batches.data(),
        rho_data.i_batch_2_wave_offset.data(),
        wave_batches_compress.data(),
        gradient_wave_batches_compress.data(),
        first_order_density_matrix.data(),
        first_order_rho.data(),
        first_order_rho_bias_part2.data(),
        first_order_gradient_rho.data(),
        first_order_gradient_rho_bias_batches_atoms.data(),

        basis_atom.data(),
        partition_tab.data(),
        density_matrix.data());
  }

  printf("Warm up finished.\n");

  printf("Start running...\n");
  int run_iters = 10;
  for (int iter = 0; iter < run_iters; iter++) {
    evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_(
        &rho_data.n_my_batches_work,
        &j_atom_begin_,
        &j_atom_end_,
        &j_coord_,
        &rho_data.n_full_points,
        &rho_data.n_basis,
        &rho_data.n_atoms,
        &rho_data.n_max_compute_ham,
        &rho_data.n_centers_basis_I,
        &rho_data.n_max_batch_size,
        rho_data.batch_sizes.data(),
        rho_data.n_point_batches.data(),
        rho_data.n_point_batches_prefix_sum.data(),
        rho_data.i_valid_point_2_i_full_points_map.data(),
        rho_data.n_compute_c_batches.data(),
        rho_data.i_basis_batches.data(),
        rho_data.atom_valid_n_compute_c_batches.data(),
        rho_data.i_batch_2_wave_offset.data(),
        wave_batches_compress.data(),
        gradient_wave_batches_compress.data(),
        first_order_density_matrix.data(),
        first_order_rho.data(),
        first_order_rho_bias_part2.data(),
        first_order_gradient_rho.data(),
        first_order_gradient_rho_bias_batches_atoms.data(),

        basis_atom.data(),
        partition_tab.data(),
        density_matrix.data());
  }
  printf("Run finished.\n");
  free_dfpt_device_data();
}

int main() {

  init_dfpt_device_info();

  try {
    char sumup_data_path[1024];
    char rho_data_path[1024];

    // char base_path[] = "../../../../dev_tests/water2_rpa_6_atoms";
    char base_path[] = "../../../../dev_tests/22_rpa_22_atoms";
    // char base_path[] = "../../../../dev_tests/61_rpa_36_atoms";
    int n_proc = 2;

    sprintf(sumup_data_path, "%s/sumup.test.nproc_%d.bin", base_path, n_proc);
    sprintf(rho_data_path, "%s/first_order_rho.meta.test.nproc_%d.bin", base_path, n_proc);

    // 3. Read sumup_data back from file
    HartreePotentialData read_sumup_data = read_hartree_potential_data(sumup_data_path);

    FirstOrderRhoMetaData read_rho_data = read_first_order_rho_meta_data(rho_data_path);

    // TODO 检查公共部分是否一致

    // 4. Call the original function with the read sumup_data
    call_with_read_data(read_sumup_data, read_rho_data);

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  free_dfpt_device_info();

  return 0;
}