#include <cblas.h>
#include <math.h>

#include "setting.h"
#include "common.hpp"
#include "pass_mod_var.h"
#include "sheval.hpp"
#include "sum_up_whole_potential.h"

#include "sum_up_direct_test.hpp"

double spline_vector_v2_n2_c_fused_ddot(
    double r_output,
    int n_vector,
    int n_l_dim,
    int n_grid_dim,
    int n_points,
    const double *spl_param_ptr,
    const double *ddot_factors) {

  // double r_output = *r_output_;
  // int n_vector = *n_vector_;

  // int n_points = *n_points_;

  // int n_l_dim = *n_l_dim_;
  // int n_grid_dim = *n_grid_dim_;

  constexpr int n_coeff = 2;
  cTMf64<3> TM_INIT(spl_param, n_l_dim, n_coeff, n_grid_dim);

  int i_spl = int(r_output);
  i_spl = max(1, i_spl);
  i_spl = min(n_points - 1, i_spl);
  double t = r_output - i_spl;

  double ta = (t - 1) * (t - 1) * (1 + 2 * t);
  double tb = (t - 1) * (t - 1) * t;
  double tc = t * t * (3 - 2 * t);
  double td = t * t * (t - 1);

  double acc = 0;
  for (int i = 0; i < n_vector; i++) {
    acc += (spl_param(i, 0, i_spl - 1) * ta + // spl_param(i, 0, i_spl - 1)
            spl_param(i, 1, i_spl - 1) * tb + // spl_param(i, 1, i_spl - 1)
            spl_param(i, 0, i_spl) * tc +     // spl_param(i, 0, i_spl)
            spl_param(i, 1, i_spl) * td       // spl_param(i, 1, i_spl)
            ) *
           ddot_factors[i];
  }
  return acc;
}

void spline_vector_v2_n4_c_(
    double r_output,
    int n_vector,
    int n_l_dim,
    int n_grid_dim,
    int n_points,
    double *spl_param_ptr,
    double *out_result_ptr) {

  // double r_output = *r_output_;
  // int n_vector = *n_vector_;

  // int n_points = *n_points_;

  // int n_l_dim = *n_l_dim_;
  // int n_grid_dim = *n_grid_dim_;

  constexpr int n_coeff = 4;
  TMf64<3> TM_INIT(spl_param, n_l_dim, n_coeff, n_grid_dim);
  TMf64<1> TM_INIT(out_result, n_vector);

  int i_spl = int(r_output);
  i_spl = max(1, i_spl);
  i_spl = min(n_points - 1, i_spl);
  double t = r_output - i_spl;

  double t2 = t * t;
  double t3 = t * t2;

  for (int i = 0; i < n_vector; i++) {
    out_result(i) = spl_param(i, 0, i_spl - 1) +      //
                    spl_param(i, 1, i_spl - 1) * t +  //
                    spl_param(i, 2, i_spl - 1) * t2 + //
                    spl_param(i, 3, i_spl - 1) * t3;  //
  }
}

void far_distance_real_hartree_potential_single_atom_p2_fused_fp_c(
    // scalars
    int l_max,
    int l_max_analytic_multipole,
    int i_center,
    double dist_tab,
    // global scalars
    int l_pot_max,
    int n_centers,
    int n_centers_hartree_potential,
    int n_atoms,
    int hartree_force_l_add,
    // arrays
    const double *coord_current,
    // global arrays
    const int *n_cc_lm_ijk_ptr,
    const int *center_to_atom_ptr,
    const int *index_cc_ptr,
    const int *index_ijk_max_cc_ptr,
    const double *coords_center_ptr,
    const double *multipole_c_ptr,
    // output
    double *potential) {

  // // scalars
  // // int l_atom_max = *l_atom_max_;
  // int l_max_analytic_multipole = *l_max_analytic_multipole_;
  // int i_center = *i_center_;
  // double dist_tab = *dist_tab_;
  // // global scalars
  // int l_pot_max = *l_pot_max_;
  // int n_centers = *n_centers_;
  // int n_centers_hartree_potential = *n_centers_hartree_potential_;
  // int n_atoms = *n_atoms_;
  // int hartree_force_l_add = *hartree_force_l_add_;

  // //   int l_max = l_atom_max;
  // int l_max = *l_max_;

  cTMi32<1> TM_INIT(n_cc_lm_ijk, l_max_analytic_multipole + 1);

  const int index_cc_dim_0 = n_cc_lm_ijk(l_max_analytic_multipole);
  const int multipole_c_dim_0 = n_cc_lm_ijk(l_pot_max);

  cTMi32<1> TM_INIT(center_to_atom, n_centers);
  cTMi32<2> TM_INIT(index_cc, index_cc_dim_0, 6);
  cTMi32<2> TM_INIT(index_ijk_max_cc, 3, l_max_analytic_multipole + 1);
  cTMf64<2> TM_INIT(coords_center, 3, n_centers);
  cTMf64<2> TM_INIT(multipole_c, multipole_c_dim_0, n_atoms);

  // far_distance_hartree_Fp_cluster_single_atom_p2
  Tf64<1> Fp(l_pot_max + 2);
  double dist_sq = dist_tab * dist_tab;
  int one_minus_2l = 1;
  Fp(0) = 1.0 / dist_tab;
  for (int i_l = 1; i_l <= l_max + hartree_force_l_add; i_l++) {
    one_minus_2l -= 2;
    Fp(i_l) = Fp(i_l - 1) * one_minus_2l / dist_sq;
  }

  // far_distance_real_hartree_potential_single_atom_p2

  double coord_c[3][(l_pot_max + 1)];
  double dir[3];

  coord_c[0][0] = 1.0;
  coord_c[1][0] = 1.0;
  coord_c[2][0] = 1.0;

  dir[0] = coord_current[0] - coords_center(0, i_center - 1);
  dir[1] = coord_current[1] - coords_center(1, i_center - 1);
  dir[2] = coord_current[2] - coords_center(2, i_center - 1);

  int maxval = -1;
  for (int i = 0; i < 3; i++)
    maxval = maxval > index_ijk_max_cc(i, l_max) ? maxval : index_ijk_max_cc(i, l_max);
  for (int i_l = 1; i_l <= maxval; i_l++) {
    coord_c[0][i_l] = dir[0] * coord_c[0][i_l - 1];
    coord_c[1][i_l] = dir[1] * coord_c[1][i_l - 1];
    coord_c[2][i_l] = dir[2] * coord_c[2][i_l - 1];
  }

  double dpot = 0.0;

  int n_end = n_cc_lm_ijk(l_max);

  for (int n = 0; n < n_end; n++) {
    int ii = index_cc(n, 3 - 1);
    int jj = index_cc(n, 4 - 1);
    int kk = index_cc(n, 5 - 1);
    int nn = index_cc(n, 6 - 1);
    dpot += coord_c[0][ii] * coord_c[1][jj] * coord_c[2][kk] * //
            Fp(nn) * multipole_c(n, center_to_atom(i_center - 1) - 1);
  }
  //   delta_v_hartree[i_full_points - 1] += dpot;
  *potential += dpot;
}

extern "C" void
increment_ylm_(const double *, const double *, const double *, const double *, const int *, const int *, double *);

// 你是一名 c++
// 专家，请你把以下这些变量和数组导出到一份文件二进制中，并编写一个写入的函数，和一个读取并调用该函数的代码，

extern "C" void sum_up_whole_potential_c_v3_atoms_full_points_j_atom_tile_(
    // scalars
    const int *j_atom_begin_,
    const int *j_atom_end_,
    const int *n_full_points_,
    const int *l_max_analytic_multipole_,
    const int *index_cc_dim_0_,
    const int *n_valid_points_,
    // global scalars
    const int *l_pot_max_,
    const int *n_max_radial_,
    const int *n_hartree_grid_,
    const int *n_centers_,
    const int *n_centers_hartree_potential_,
    const int *n_atoms_,
    const int *hartree_force_l_add_,
    const int *n_species_,
    // arrays
    const int *n_radial_ptr,                              // (n_species)
    const int *species_ptr,                               // (n_atoms)
    const int *species_center_ptr,                        // (n_centers)
    const int *l_hartree_max_far_distance_ptr,            // (n_atoms, ATOM_TILE_SIZE)
    const int *i_valid_point_2_i_full_points_map_ptr,     // (n_valid_points)
    const double *coord_points_ptr,                       // (3, n_full_points)
    const double *multipole_radius_sq_ptr,                // (n_atoms, ATOM_TILE_SIZE)
    const double *outer_potential_radius_ptr,             // (l_pot_max + 1, n_atoms, ATOM_TILE_SIZE)
    const double *multipole_moments_ptr,                  // (l_pot_max_pow2, n_atoms, ATOM_TILE_SIZE)
    const double *current_delta_v_hart_part_spl_tile_ptr, // (l_pot_max_pow2, n_coeff_hartree, n_hartree_grid, n_atoms,
                                                          // ATOM_TILE_SIZE)
    // global arrays
    const int *l_hartree_ptr,                 // (n_species)
    const int *n_grid_ptr,                    // (n_species)
    const int *n_cc_lm_ijk_ptr,               // (l_max_analytic_multipole + 1)
    const int *centers_hartree_potential_ptr, // (n_centers_hartree_potential)
    const int *center_to_atom_ptr,            // (n_centers)
    const int *index_cc_ptr,                  // (index_cc_dim_0, 6)
    const int *index_ijk_max_cc_ptr,          // (3, l_max_analytic_multipole + 1)
    const double *cc_ptr,                     // (index_cc_dim_0)
    const double *coords_center_ptr,          // (3, n_centers)
    const double *r_grid_min_ptr,             // (n_species)
    const double *log_r_grid_inc_ptr,         // (n_species)
    const double *scale_radial_ptr,           // (n_species)
    const double *partition_tab_ptr,          // (n_full_points)
    // output
    double *delta_v_hartree_ptr // (n_full_points, ATOM_TILE_SIZE)
) {

  const int j_atom_begin = *j_atom_begin_;
  const int j_atom_end = *j_atom_end_;

  const int n_full_points = *n_full_points_;
  const int l_max_analytic_multipole = *l_max_analytic_multipole_;
  const int index_cc_dim_0 = *index_cc_dim_0_;
  const int n_valid_points = *n_valid_points_;

  const int l_pot_max = *l_pot_max_;
  const int n_max_radial = *n_max_radial_;
  const int n_hartree_grid = *n_hartree_grid_;
  const int n_centers = *n_centers_;
  const int n_centers_hartree_potential = *n_centers_hartree_potential_;
  const int n_atoms = *n_atoms_;
  const int hartree_force_l_add = *hartree_force_l_add_;
  const int n_species = *n_species_;

  const int l_pot_max_pow2 = (l_pot_max + 1) * (l_pot_max + 1);

  // 以下一些数组可能通过 xx_ptr 直接访问

  // arrays
  cTMi32<1> TM_INIT(n_radial, n_species);
  cTMi32<1> TM_INIT(species, n_atoms);
  cTMi32<1> TM_INIT(species_center, n_centers);
  cTMi32<2> TM_INIT(l_hartree_max_far_distance, n_atoms, ATOM_TILE_SIZE);
  cTMi32<1> TM_INIT(i_valid_point_2_i_full_points_map, n_valid_points);
  cTMf64<2> TM_INIT(coord_points, 3, n_full_points);
  cTMf64<2> TM_INIT(multipole_radius_sq, n_atoms, ATOM_TILE_SIZE);
  cTMf64<3> TM_INIT(outer_potential_radius, l_pot_max + 1, n_atoms, ATOM_TILE_SIZE);
  cTMf64<3> TM_INIT(multipole_moments, l_pot_max_pow2, n_atoms, ATOM_TILE_SIZE);
  cTMf64<5> TM_INIT(
      current_delta_v_hart_part_spl_tile, l_pot_max_pow2, n_coeff_hartree, n_hartree_grid, n_atoms, ATOM_TILE_SIZE);

  // global arrays
  cTMi32<1> TM_INIT(l_hartree, n_species);
  cTMi32<1> TM_INIT(n_grid, n_species);
  cTMi32<1> TM_INIT(n_cc_lm_ijk, l_max_analytic_multipole + 1);
  cTMi32<1> TM_INIT(centers_hartree_potential, n_centers_hartree_potential);
  cTMi32<1> TM_INIT(center_to_atom, n_centers);
  cTMi32<2> TM_INIT(index_cc, index_cc_dim_0, 6);
  cTMi32<2> TM_INIT(index_ijk_max_cc, 3, l_max_analytic_multipole + 1);
  cTMf64<1> TM_INIT(cc, index_cc_dim_0);
  cTMf64<2> TM_INIT(coords_center, 3, n_centers);
  cTMf64<1> TM_INIT(r_grid_min, n_species);
  cTMf64<1> TM_INIT(log_r_grid_inc, n_species);
  cTMf64<1> TM_INIT(scale_radial, n_species);
  cTMf64<1> TM_INIT(partition_tab, n_full_points);

  // output
  TMf64<2> TM_INIT(delta_v_hartree, n_full_points, ATOM_TILE_SIZE);

#ifdef SAVE_DFPT_DATA_TEST

  // clang-format off
  HartreePotentialData data;

  data.j_atom_begin = j_atom_begin;
  data.j_atom_end = j_atom_end;
  data.n_full_points = n_full_points;
  data.l_max_analytic_multipole = l_max_analytic_multipole;
  data.index_cc_dim_0 = index_cc_dim_0;
  data.n_valid_points = n_valid_points;

  data.l_pot_max = l_pot_max;
  data.n_max_radial = n_max_radial;
  data.n_hartree_grid = n_hartree_grid;
  data.n_centers = n_centers;
  data.n_centers_hartree_potential = n_centers_hartree_potential;
  data.n_atoms = n_atoms;
  data.hartree_force_l_add = hartree_force_l_add;
  data.n_species = n_species;

  data.n_radial = std::vector<int>(n_radial.data(), n_radial.data()+n_radial.size());
  data.species = std::vector<int>(species.data(), species.data()+species.size());
  data.species_center = std::vector<int>(species_center.data(), species_center.data()+species_center.size());
  data.l_hartree_max_far_distance = std::vector<int>(l_hartree_max_far_distance.data(), l_hartree_max_far_distance.data()+l_hartree_max_far_distance.size());
  data.i_valid_point_2_i_full_points_map = std::vector<int>(i_valid_point_2_i_full_points_map.data(), i_valid_point_2_i_full_points_map.data()+i_valid_point_2_i_full_points_map.size());
  data.coord_points = std::vector<double>(coord_points.data(), coord_points.data()+coord_points.size());
  data.multipole_radius_sq = std::vector<double>(multipole_radius_sq.data(), multipole_radius_sq.data()+multipole_radius_sq.size());
  data.outer_potential_radius = std::vector<double>(outer_potential_radius.data(), outer_potential_radius.data()+outer_potential_radius.size());
  data.multipole_moments = std::vector<double>(multipole_moments.data(), multipole_moments.data()+multipole_moments.size());
  // data.current_delta_v_hart_part_spl_tile = std::vector<double>(current_delta_v_hart_part_spl_tile.data(), current_delta_v_hart_part_spl_tile.data()+current_delta_v_hart_part_spl_tile.size());

  data.l_hartree = std::vector<int>(l_hartree.data(), l_hartree.data()+l_hartree.size());
  data.n_grid = std::vector<int>(n_grid.data(), n_grid.data()+n_grid.size());
  data.n_cc_lm_ijk = std::vector<int>(n_cc_lm_ijk.data(), n_cc_lm_ijk.data()+n_cc_lm_ijk.size());
  data.centers_hartree_potential = std::vector<int>(centers_hartree_potential.data(), centers_hartree_potential.data()+centers_hartree_potential.size());
  data.center_to_atom = std::vector<int>(center_to_atom.data(), center_to_atom.data()+center_to_atom.size());
  data.index_cc = std::vector<int>(index_cc.data(), index_cc.data()+index_cc.size());
  data.index_ijk_max_cc = std::vector<int>(index_ijk_max_cc.data(), index_ijk_max_cc.data()+index_ijk_max_cc.size());
  data.cc = std::vector<double>(cc.data(), cc.data()+cc.size());
  data.coords_center = std::vector<double>(coords_center.data(), coords_center.data()+coords_center.size());
  data.r_grid_min = std::vector<double>(r_grid_min.data(), r_grid_min.data()+r_grid_min.size());
  data.log_r_grid_inc = std::vector<double>(log_r_grid_inc.data(), log_r_grid_inc.data()+log_r_grid_inc.size());
  data.scale_radial = std::vector<double>(scale_radial.data(), scale_radial.data()+scale_radial.size());
  data.partition_tab = std::vector<double>(partition_tab.data(), partition_tab.data()+partition_tab.size());

  // data.delta_v_hartree = std::vector<double>(delta_v_hartree.data(), delta_v_hartree.data()+delta_v_hartree.size());
  // clang-format on

#endif

  Tf64<1> ylm_tab((l_pot_max + 1) * (l_pot_max + 1));

  // multipole_moments, outer_potential_radius, multipole_radius_sq 与 j_atom 相关
  // Initialize index_lm
  Ti32<2> index_lm(l_pot_max * 2 + 1, l_pot_max + 1); // index_lm(-l_pot_max:l_pot_max, 0:l_pot_max )
  int i_index = 0;
  for (int i_l = 0; i_l <= l_pot_max; i_l++) {
    for (int i_m = -i_l; i_m <= i_l; i_m++) {
      i_index++;
      index_lm(i_m + l_pot_max, i_l) = i_index;
    }
  }
  // hartree_potential_real_coeff
  Tf64<3> multipole_c(n_cc_lm_ijk_ptr[l_pot_max], n_atoms, ATOM_TILE_SIZE);
  for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
    int j_atom_inner = j_atom - j_atom_begin;
    for (int i_atom = 0; i_atom < n_atoms; i_atom++) {

      // n_cc_lm_ijk(0:l_max_analytic_multipole)
      for (int n = 0; n < n_cc_lm_ijk_ptr[l_hartree_max_far_distance(i_atom, j_atom_inner)]; n++) {

        int i_l = index_cc(n, 0);             // index_cc(n, 1)
        int i_m = index_cc(n, 1) + l_pot_max; // index_cc(n, 2)

        if (i_l <= l_hartree_max_far_distance(i_atom, j_atom_inner) && //
            abs(multipole_moments(index_lm(i_m, i_l) - 1, i_atom, j_atom_inner)) > 1e-10) {
          multipole_c(n, i_atom, j_atom_inner) =
              cc_ptr[n] * multipole_moments(index_lm(i_m, i_l) - 1, i_atom, j_atom_inner);
        } else {
          multipole_c(n, i_atom, j_atom_inner) = 0;
        }
      }
    }
  }

  std::vector<int> count_each_case((l_pot_max + 1) * 2);

  for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
    int j_atom_inner = j_atom - j_atom_begin;

    for (int i_center = 0; i_center < n_centers_hartree_potential; i_center++) {
      int current_center = centers_hartree_potential_ptr[i_center] - 1;
      int current_spl_atom = center_to_atom_ptr[current_center] - 1;

      for (int i_valid_points = 0; i_valid_points < n_valid_points; i_valid_points++) {
        int i_full_points = i_valid_point_2_i_full_points_map(i_valid_points);

        // if (partition_tab_ptr[i_full_points] > 0) {

        // tab_single_atom_centered_coords_p0_c_(&current_center, coord_current, &dist_tab_sq, dir_tab);
        double dist_tab_sq = 0.0;
        double dir_tab[3];
        {
          for (int i_coord = 0; i_coord < 3; i_coord++) {
            dir_tab[i_coord] = coord_points(i_coord, i_full_points) - coords_center(i_coord, current_center);
            dist_tab_sq += dir_tab[i_coord] * dir_tab[i_coord];
          }
        }

        int l_atom_max = l_hartree_ptr[species_ptr[current_spl_atom] - 1];
        while (outer_potential_radius(l_atom_max, current_spl_atom, j_atom_inner) < dist_tab_sq && l_atom_max > 0) {
          l_atom_max--;
        }
        int l_h_dim = (l_atom_max + 1) * (l_atom_max + 1);

        // printf("atom %d, valid_points %d, l_atom_max: %d\n", current_spl_atom, i_valid_points, l_atom_max);
        if (dist_tab_sq < multipole_radius_sq(current_spl_atom, j_atom_inner)) {
        } else if (dist_tab_sq < 1E8) {
          count_each_case[l_atom_max + (l_pot_max + 1)]++;
          double dist_tab_out = sqrt(dist_tab_sq);
          far_distance_real_hartree_potential_single_atom_p2_fused_fp_c(
              l_atom_max,
              l_max_analytic_multipole,
              i_center + 1,
              dist_tab_out,
              l_pot_max,
              n_centers,
              n_centers_hartree_potential,
              n_atoms,
              hartree_force_l_add,
              &coord_points(0, i_full_points),
              n_cc_lm_ijk_ptr,
              center_to_atom_ptr,
              index_cc_ptr,
              index_ijk_max_cc_ptr,
              coords_center_ptr,
              &multipole_c(0, 0, j_atom_inner),
              &delta_v_hartree(i_full_points, j_atom_inner));
        }
        // }
      }
    }
  }

  // int sum = 0;
  // for (int i = 0; i < (l_pot_max + 1); i++) {
  //   sum += count_each_case[i];
  //   printf("%9d ", count_each_case[i]);
  // }
  // printf(" total: %d\n", sum);
  // sum = 0;
  // for (int i = 0; i < (l_pot_max + 1); i++) {
  //   sum += count_each_case[i + (l_pot_max + 1)];
  //   printf("%9d ", count_each_case[i + (l_pot_max + 1)]);
  // }
  // printf(" total: %d\n", sum);

#ifdef SAVE_DFPT_DATA_TEST
  // data.delta_v_hartree_ref =
  //     std::vector<double>(delta_v_hartree.data(), delta_v_hartree.data() + delta_v_hartree.size());
  if (myid == 0) {
    // 这里传入的是 string 类型
    char filename[1024];
    sprintf(filename, "sumup.test.nproc_%d.bin", n_tasks);
    printf("[DEBUG][SAVE_DFPT_DATA_TEST] write_hartree_potential_data: %s\n", filename);
    write_hartree_potential_data(filename, data);
  }
#endif
}
