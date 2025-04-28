
int __mpi_tasks_MOD_n_tasks = 1;
int __mpi_tasks_MOD_myid = 0;

#include "common.hpp"
#include "device.hpp"
#include "device_data.hpp"
#include "setting.h"

#include "evaluate_first_order_rho_direct_test.hpp"
#include "sum_up_direct_test.hpp"
#include "sum_up_whole_potential.h"

float call_with_read_data(HartreePotentialData &sumup_data, FirstOrderRhoMetaData &rho_data, const char *base_path) {

  printf("%d %d\n", sumup_data.n_full_points, (int)sumup_data.partition_tab.size());
  // printf("%d %d\n", sumup_data.n_full_points, (int)sumup_data.delta_v_hartree.size());
  // printf("%d %d\n", sumup_data.n_full_points, (int)sumup_data.delta_v_hartree_ref.size());

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

  Tf64<2> first_order_rho(rho_data.n_full_points, ATOM_TILE_SIZE);
  Tf64<4> first_order_gradient_rho(3, rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);

  Tf64<1> wave_batches_compress(rho_data.i_batch_2_wave_offset[rho_data.n_my_batches_work]);
  Tf64<1> gradient_wave_batches_compress(3 * rho_data.i_batch_2_wave_offset[rho_data.n_my_batches_work]);
  Tf64<3> first_order_density_matrix(rho_data.n_basis, rho_data.n_basis, ATOM_TILE_SIZE);
  Tf64<3> first_order_rho_bias_part2(rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);
  Tf64<4> first_order_gradient_rho_bias_batches_atoms(
      3, rho_data.n_max_batch_size, rho_data.n_my_batches_work, ATOM_TILE_SIZE);
  Tf64<2> density_matrix(rho_data.n_basis, rho_data.n_basis);

  // =======================================================

  printf("n_cc_lm_ijk_ptr[l_pot_max] = %d\n", sumup_data.n_cc_lm_ijk[sumup_data.l_pot_max]);
  printf("n_cc_lm_ijk[0:l_pot_max+1]\n");
  for (int i = 0; i < sumup_data.l_pot_max + 1; i++) {
    printf("%d\n", sumup_data.n_cc_lm_ijk[i]);
  }

  printf("\n");
  // printf("index_cc[?, 2:6]\n");
  // for (int i = 0; i < sumup_data.index_cc_dim_0; i++) {
  //   printf("%3d: ", i);
  //   for (int j = 2; j < 6; j++) {
  //     printf("%3d, ", sumup_data.index_cc[i + j * sumup_data.index_cc_dim_0]);
  //   }
  //   printf("\n");
  // }

  sum_up_whole_potential_c_v3_atoms_full_points_j_atom_tile_(
      &sumup_data.j_atom_begin,
      &sumup_data.j_atom_end,
      &sumup_data.n_full_points,
      &sumup_data.l_max_analytic_multipole,
      &sumup_data.index_cc_dim_0,
      &sumup_data.n_valid_points,
      &sumup_data.l_pot_max,
      &sumup_data.n_max_radial,
      &sumup_data.n_hartree_grid,
      &sumup_data.n_centers,
      &sumup_data.n_centers_hartree_potential,
      &sumup_data.n_atoms,
      &sumup_data.hartree_force_l_add,
      &sumup_data.n_species,
      sumup_data.n_radial.empty() ? nullptr : sumup_data.n_radial.data(),
      sumup_data.species.empty() ? nullptr : sumup_data.species.data(),
      sumup_data.species_center.empty() ? nullptr : sumup_data.species_center.data(),
      sumup_data.l_hartree_max_far_distance.empty() ? nullptr : sumup_data.l_hartree_max_far_distance.data(),
      sumup_data.i_valid_point_2_i_full_points_map.empty() ? nullptr
                                                           : sumup_data.i_valid_point_2_i_full_points_map.data(),
      sumup_data.coord_points.empty() ? nullptr : sumup_data.coord_points.data(),
      sumup_data.multipole_radius_sq.empty() ? nullptr : sumup_data.multipole_radius_sq.data(),
      sumup_data.outer_potential_radius.empty() ? nullptr : sumup_data.outer_potential_radius.data(),
      sumup_data.multipole_moments.empty() ? nullptr : sumup_data.multipole_moments.data(),
      current_delta_v_hart_part_spl_tile.data(),
      sumup_data.l_hartree.empty() ? nullptr : sumup_data.l_hartree.data(),
      sumup_data.n_grid.empty() ? nullptr : sumup_data.n_grid.data(),
      sumup_data.n_cc_lm_ijk.empty() ? nullptr : sumup_data.n_cc_lm_ijk.data(),
      sumup_data.centers_hartree_potential.empty() ? nullptr : sumup_data.centers_hartree_potential.data(),
      sumup_data.center_to_atom.empty() ? nullptr : sumup_data.center_to_atom.data(),
      sumup_data.index_cc.empty() ? nullptr : sumup_data.index_cc.data(),
      sumup_data.index_ijk_max_cc.empty() ? nullptr : sumup_data.index_ijk_max_cc.data(),
      sumup_data.cc.empty() ? nullptr : sumup_data.cc.data(),
      sumup_data.coords_center.empty() ? nullptr : sumup_data.coords_center.data(),
      sumup_data.r_grid_min.empty() ? nullptr : sumup_data.r_grid_min.data(),
      sumup_data.log_r_grid_inc.empty() ? nullptr : sumup_data.log_r_grid_inc.data(),
      sumup_data.scale_radial.empty() ? nullptr : sumup_data.scale_radial.data(),
      sumup_data.partition_tab.empty() ? nullptr : sumup_data.partition_tab.data(),
      delta_v_hartree_ref.data());

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
      first_order_rho.data(),
      first_order_rho_bias_part2.data(),
      first_order_gradient_rho.data(),
      first_order_gradient_rho_bias_batches_atoms.data(),

      sumup_data.l_hartree_max_far_distance.data(),
      sumup_data.multipole_radius_sq.data(),
      sumup_data.outer_potential_radius.data(),
      sumup_data.multipole_moments.data(),
      current_delta_v_hart_part_spl_tile.data(),

      delta_v_hartree_ref.data());

  DEV_STREAM_T &stream = devInfo.stream;

  sum_up_whole_potential_c_v3_atoms_full_points_j_atom_tile_cu_host_(
      sumup_data.j_atom_begin,
      sumup_data.j_atom_end,
      sumup_data.n_full_points,
      sumup_data.l_max_analytic_multipole,
      sumup_data.index_cc_dim_0,
      sumup_data.n_valid_points,
      sumup_data.l_pot_max,
      sumup_data.n_max_radial,
      sumup_data.n_hartree_grid,
      sumup_data.n_centers,
      sumup_data.n_centers_hartree_potential,
      sumup_data.n_atoms,
      sumup_data.hartree_force_l_add,
      sumup_data.n_species,
      sumup_data.n_radial.empty() ? nullptr : sumup_data.n_radial.data(),
      sumup_data.species.empty() ? nullptr : sumup_data.species.data(),
      sumup_data.species_center.empty() ? nullptr : sumup_data.species_center.data(),
      sumup_data.l_hartree_max_far_distance.empty() ? nullptr : sumup_data.l_hartree_max_far_distance.data(),
      sumup_data.i_valid_point_2_i_full_points_map.empty() ? nullptr
                                                           : sumup_data.i_valid_point_2_i_full_points_map.data(),
      sumup_data.coord_points.empty() ? nullptr : sumup_data.coord_points.data(),
      sumup_data.multipole_radius_sq.empty() ? nullptr : sumup_data.multipole_radius_sq.data(),
      sumup_data.outer_potential_radius.empty() ? nullptr : sumup_data.outer_potential_radius.data(),
      sumup_data.multipole_moments.empty() ? nullptr : sumup_data.multipole_moments.data(),
      current_delta_v_hart_part_spl_tile.data(),
      sumup_data.l_hartree.empty() ? nullptr : sumup_data.l_hartree.data(),
      sumup_data.n_grid.empty() ? nullptr : sumup_data.n_grid.data(),
      sumup_data.n_cc_lm_ijk.empty() ? nullptr : sumup_data.n_cc_lm_ijk.data(),
      sumup_data.centers_hartree_potential.empty() ? nullptr : sumup_data.centers_hartree_potential.data(),
      sumup_data.center_to_atom.empty() ? nullptr : sumup_data.center_to_atom.data(),
      sumup_data.index_cc.empty() ? nullptr : sumup_data.index_cc.data(),
      sumup_data.index_ijk_max_cc.empty() ? nullptr : sumup_data.index_ijk_max_cc.data(),
      sumup_data.cc.empty() ? nullptr : sumup_data.cc.data(),
      sumup_data.coords_center.empty() ? nullptr : sumup_data.coords_center.data(),
      sumup_data.r_grid_min.empty() ? nullptr : sumup_data.r_grid_min.data(),
      sumup_data.log_r_grid_inc.empty() ? nullptr : sumup_data.log_r_grid_inc.data(),
      sumup_data.scale_radial.empty() ? nullptr : sumup_data.scale_radial.data(),
      sumup_data.partition_tab.empty() ? nullptr : sumup_data.partition_tab.data(),
      delta_v_hartree.data());

  int failed_count = 0;
  for (int i = 0; i < sumup_data.n_full_points; i++) {
    if (std::abs(delta_v_hartree(i) - delta_v_hartree_ref(i)) > 1E-8) {
      failed_count++;
      printf("Error! i_full_point %9d, %.18f != %.18f\n", i, delta_v_hartree(i), delta_v_hartree_ref(i));
    }
    if (failed_count > 10) {
      printf("Failed to many times (failed_count > %d), exit.\n", failed_count);
      break;
    }
  }
  printf("Check delta_v_hartree finished.\n");

  int warm_up = 10;
  for (int iter = 0; iter < warm_up; iter++) {
    sum_up_whole_potential_c_v3_atoms_full_points_j_atom_tile_cu_host_(
        sumup_data.j_atom_begin,
        sumup_data.j_atom_end,
        sumup_data.n_full_points,
        sumup_data.l_max_analytic_multipole,
        sumup_data.index_cc_dim_0,
        sumup_data.n_valid_points,
        sumup_data.l_pot_max,
        sumup_data.n_max_radial,
        sumup_data.n_hartree_grid,
        sumup_data.n_centers,
        sumup_data.n_centers_hartree_potential,
        sumup_data.n_atoms,
        sumup_data.hartree_force_l_add,
        sumup_data.n_species,
        sumup_data.n_radial.empty() ? nullptr : sumup_data.n_radial.data(),
        sumup_data.species.empty() ? nullptr : sumup_data.species.data(),
        sumup_data.species_center.empty() ? nullptr : sumup_data.species_center.data(),
        sumup_data.l_hartree_max_far_distance.empty() ? nullptr : sumup_data.l_hartree_max_far_distance.data(),
        sumup_data.i_valid_point_2_i_full_points_map.empty() ? nullptr
                                                             : sumup_data.i_valid_point_2_i_full_points_map.data(),
        sumup_data.coord_points.empty() ? nullptr : sumup_data.coord_points.data(),
        sumup_data.multipole_radius_sq.empty() ? nullptr : sumup_data.multipole_radius_sq.data(),
        sumup_data.outer_potential_radius.empty() ? nullptr : sumup_data.outer_potential_radius.data(),
        sumup_data.multipole_moments.empty() ? nullptr : sumup_data.multipole_moments.data(),
        current_delta_v_hart_part_spl_tile.data(),
        sumup_data.l_hartree.empty() ? nullptr : sumup_data.l_hartree.data(),
        sumup_data.n_grid.empty() ? nullptr : sumup_data.n_grid.data(),
        sumup_data.n_cc_lm_ijk.empty() ? nullptr : sumup_data.n_cc_lm_ijk.data(),
        sumup_data.centers_hartree_potential.empty() ? nullptr : sumup_data.centers_hartree_potential.data(),
        sumup_data.center_to_atom.empty() ? nullptr : sumup_data.center_to_atom.data(),
        sumup_data.index_cc.empty() ? nullptr : sumup_data.index_cc.data(),
        sumup_data.index_ijk_max_cc.empty() ? nullptr : sumup_data.index_ijk_max_cc.data(),
        sumup_data.cc.empty() ? nullptr : sumup_data.cc.data(),
        sumup_data.coords_center.empty() ? nullptr : sumup_data.coords_center.data(),
        sumup_data.r_grid_min.empty() ? nullptr : sumup_data.r_grid_min.data(),
        sumup_data.log_r_grid_inc.empty() ? nullptr : sumup_data.log_r_grid_inc.data(),
        sumup_data.scale_radial.empty() ? nullptr : sumup_data.scale_radial.data(),
        sumup_data.partition_tab.empty() ? nullptr : sumup_data.partition_tab.data(),
        delta_v_hartree.data());
  }

  printf("Warm up finished.\n");

  printf("Start running...\n");

  EventHelper<true> event_helper_all(stream);
  event_helper_all.record_start();

  int run_iters = 10;
  for (int iter = 0; iter < run_iters; iter++) {
    sum_up_whole_potential_c_v3_atoms_full_points_j_atom_tile_cu_host_(
        sumup_data.j_atom_begin,
        sumup_data.j_atom_end,
        sumup_data.n_full_points,
        sumup_data.l_max_analytic_multipole,
        sumup_data.index_cc_dim_0,
        sumup_data.n_valid_points,
        sumup_data.l_pot_max,
        sumup_data.n_max_radial,
        sumup_data.n_hartree_grid,
        sumup_data.n_centers,
        sumup_data.n_centers_hartree_potential,
        sumup_data.n_atoms,
        sumup_data.hartree_force_l_add,
        sumup_data.n_species,
        sumup_data.n_radial.empty() ? nullptr : sumup_data.n_radial.data(),
        sumup_data.species.empty() ? nullptr : sumup_data.species.data(),
        sumup_data.species_center.empty() ? nullptr : sumup_data.species_center.data(),
        sumup_data.l_hartree_max_far_distance.empty() ? nullptr : sumup_data.l_hartree_max_far_distance.data(),
        sumup_data.i_valid_point_2_i_full_points_map.empty() ? nullptr
                                                             : sumup_data.i_valid_point_2_i_full_points_map.data(),
        sumup_data.coord_points.empty() ? nullptr : sumup_data.coord_points.data(),
        sumup_data.multipole_radius_sq.empty() ? nullptr : sumup_data.multipole_radius_sq.data(),
        sumup_data.outer_potential_radius.empty() ? nullptr : sumup_data.outer_potential_radius.data(),
        sumup_data.multipole_moments.empty() ? nullptr : sumup_data.multipole_moments.data(),
        current_delta_v_hart_part_spl_tile.data(),
        sumup_data.l_hartree.empty() ? nullptr : sumup_data.l_hartree.data(),
        sumup_data.n_grid.empty() ? nullptr : sumup_data.n_grid.data(),
        sumup_data.n_cc_lm_ijk.empty() ? nullptr : sumup_data.n_cc_lm_ijk.data(),
        sumup_data.centers_hartree_potential.empty() ? nullptr : sumup_data.centers_hartree_potential.data(),
        sumup_data.center_to_atom.empty() ? nullptr : sumup_data.center_to_atom.data(),
        sumup_data.index_cc.empty() ? nullptr : sumup_data.index_cc.data(),
        sumup_data.index_ijk_max_cc.empty() ? nullptr : sumup_data.index_ijk_max_cc.data(),
        sumup_data.cc.empty() ? nullptr : sumup_data.cc.data(),
        sumup_data.coords_center.empty() ? nullptr : sumup_data.coords_center.data(),
        sumup_data.r_grid_min.empty() ? nullptr : sumup_data.r_grid_min.data(),
        sumup_data.log_r_grid_inc.empty() ? nullptr : sumup_data.log_r_grid_inc.data(),
        sumup_data.scale_radial.empty() ? nullptr : sumup_data.scale_radial.data(),
        sumup_data.partition_tab.empty() ? nullptr : sumup_data.partition_tab.data(),
        delta_v_hartree.data());
  }

  float milliseconds = event_helper_all.elapsed_time("Run evaluate_first_order_gradient_rho finished.");
  printf("Run %s evaluate_first_order_gradient_rho avg time (ms): %f\n", base_path, milliseconds / run_iters);

  printf("Run finished.\n");
  free_dfpt_device_data();

  return milliseconds / run_iters;
}

float run(const char *base_path, int n_proc) {
  try {
    char sumup_data_path[1024];
    char rho_data_path[1024];

    sprintf(sumup_data_path, "%s/sumup.test.nproc_%d.bin", base_path, n_proc);
    sprintf(rho_data_path, "%s/first_order_rho.meta.test.nproc_%d.bin", base_path, n_proc);

    HartreePotentialData read_sumup_data = read_hartree_potential_data(sumup_data_path);
    FirstOrderRhoMetaData read_rho_data = read_first_order_rho_meta_data(rho_data_path);

    return call_with_read_data(read_sumup_data, read_rho_data, base_path);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    exit(1);
  }
}

int main() {

  init_dfpt_device_info();

  int n_proc = 4;

  float time_18 = run("../../fragment_save_bin/case_18", n_proc);
  float time_20 = run("../../fragment_save_bin/case_20", n_proc);
  float time_22 = run("../../fragment_save_bin/case_22", n_proc);
  float time_24 = run("../../fragment_save_bin/case_24", n_proc);
  float time_26 = run("../../fragment_save_bin/case_26", n_proc);
  float time_28 = run("../../fragment_save_bin/case_28", n_proc);
  float time_30 = run("../../fragment_save_bin/case_30", n_proc);
  float time_32 = run("../../fragment_save_bin/case_32", n_proc);

  printf("[[Result]], %d, %d, case_18, %f\n", int(ATOM_TILE_SIZE), n_proc, time_18);
  printf("[[Result]], %d, %d, case_20, %f\n", int(ATOM_TILE_SIZE), n_proc, time_20);
  printf("[[Result]], %d, %d, case_22, %f\n", int(ATOM_TILE_SIZE), n_proc, time_22);
  printf("[[Result]], %d, %d, case_24, %f\n", int(ATOM_TILE_SIZE), n_proc, time_24);
  printf("[[Result]], %d, %d, case_26, %f\n", int(ATOM_TILE_SIZE), n_proc, time_26);
  printf("[[Result]], %d, %d, case_28, %f\n", int(ATOM_TILE_SIZE), n_proc, time_28);
  printf("[[Result]], %d, %d, case_30, %f\n", int(ATOM_TILE_SIZE), n_proc, time_30);
  printf("[[Result]], %d, %d, case_32, %f\n", int(ATOM_TILE_SIZE), n_proc, time_32);

  free_dfpt_device_info();

  return 0;
}
