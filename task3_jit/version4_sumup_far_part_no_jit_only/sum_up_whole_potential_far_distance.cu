#include <chrono>
#include <iostream>
#include <math.h>
#include <type_traits>

#include "common.hpp"
#include "device.hpp"
#include "device_data.hpp"
#include "pass_mod_var.h"
#include "setting.h"
#include "sum_up_direct_test.hpp"
#include "sum_up_whole_potential.h"

#include "sheval.hpp"

constexpr bool enable_profile_sumup_each_kernel = false;
constexpr bool enable_profile_sumup_end_to_end = false;

__device__ double spline_vector_v2_n2_c_fused_ddot_cu(
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
  cTMf64<3> TM_INIT(spl_param, n_coeff, n_l_dim, n_grid_dim);

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
    acc += (spl_param(0, i, i_spl - 1) * ta + // spl_param(i, 0, i_spl - 1)
            spl_param(1, i, i_spl - 1) * tb + // spl_param(i, 1, i_spl - 1)
            spl_param(0, i, i_spl) * tc +     // spl_param(i, 0, i_spl)
            spl_param(1, i, i_spl) * td       // spl_param(i, 1, i_spl)
            ) *
           ddot_factors[i];
  }
  return acc;
}

template <int l_pot_max>
__device__ void far_distance_real_hartree_potential_single_atom_p2_fused_fp_cu_inner(
    // scalars
    int l_max,
    int l_max_analytic_multipole,
    int i_center,
    int index_cc_dim_0,
    int multipole_c_dim_0,
    double dist_tab,
    // global scalars
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

  cTMi32<1> TM_INIT(n_cc_lm_ijk, l_max_analytic_multipole + 1);

  cTMi32<1> TM_INIT(center_to_atom, n_centers);
  cTMi32<2> TM_INIT(index_cc, index_cc_dim_0, 6);
  cTMi32<2> TM_INIT(index_ijk_max_cc, 3, l_max_analytic_multipole + 1);
  cTMf64<2> TM_INIT(coords_center, 3, n_centers);
  cTMf64<2> TM_INIT(multipole_c, multipole_c_dim_0, n_atoms);

  // far_distance_hartree_Fp_cluster_single_atom_p2
  double Fp[l_pot_max + 2];
  double dist_sq = dist_tab * dist_tab;
  int one_minus_2l = 1;
  Fp[0] = 1.0 / dist_tab;
  for (int i_l = 1; i_l <= l_max + hartree_force_l_add; i_l++) {
    one_minus_2l -= 2;
    Fp[i_l] = Fp[i_l - 1] * one_minus_2l / dist_sq;
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
            Fp[nn] * multipole_c(n, center_to_atom(i_center - 1) - 1);
  }
  //   delta_v_hartree[i_full_points - 1] += dpot;

  *potential += dpot;
}

template <int l_pot_max, int BLOCK_SIZE, int POINTS_PER_THREAD>
__global__ void sum_up_close_part_cu(
    // scalars
    const int j_atom_begin,
    const int j_atom_end,
    const int n_full_points,
    const int l_max_analytic_multipole,
    const int index_cc_dim_0,
    const int n_valid_points,
    // global scalars
    const int n_max_radial,
    const int n_hartree_grid,
    const int n_centers,
    const int n_centers_hartree_potential,
    const int n_atoms,
    const int hartree_force_l_add,
    const int n_species,
    // arrays
    const int *n_radial_ptr,
    const int *species_ptr,
    const int *species_center_ptr,
    const int *l_hartree_max_far_distance_ptr,
    const int *i_valid_point_2_i_full_points_map_ptr,
    const double *coord_points_ptr,
    const double *multipole_radius_sq_ptr,
    const double *outer_potential_radius_ptr,
    const double *multipole_moments_ptr,
    const double *current_delta_v_hart_part_spl_tile_trans_ptr,
    const double *multipole_c_ptr,
    // global arrays
    const int *l_hartree_ptr,
    const int *n_grid_ptr,
    const int *n_cc_lm_ijk_ptr,
    const int *centers_hartree_potential_ptr,
    const int *center_to_atom_ptr,
    const int *index_cc_ptr,
    const int *index_ijk_max_cc_ptr,
    const double *cc_ptr,
    const double *coords_center_ptr,
    const double *r_grid_min_ptr,
    const double *log_r_grid_inc_ptr,
    const double *scale_radial_ptr,
    const double *partition_tab_ptr,
    // output
    double *delta_v_hartree_ptr) {

  constexpr int l_pot_max_pow2 = (l_pot_max + 1) * (l_pot_max + 1);
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
      current_delta_v_hart_part_spl_tile_trans,
      n_coeff_hartree,
      l_pot_max_pow2,
      n_hartree_grid,
      n_atoms,
      ATOM_TILE_SIZE);
  cTMf64<3> TM_INIT(multipole_c, n_cc_lm_ijk_ptr[l_pot_max], n_atoms, ATOM_TILE_SIZE);

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

  // double ylm_tab[(l_pot_max + 1) * (l_pot_max + 1)];

  int point_offset = blockIdx.x * BLOCK_SIZE * POINTS_PER_THREAD;
  [[maybe_unused]] int multipole_c_dim_0 = n_cc_lm_ijk_ptr[l_pot_max];

  for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
    int j_atom_inner = j_atom - j_atom_begin;

    for (int i_center = 0; i_center < n_centers_hartree_potential; i_center++) {
      int current_center = centers_hartree_potential_ptr[i_center] - 1;
      int current_spl_atom = center_to_atom_ptr[current_center] - 1;

      for (int iter_point = 0; iter_point < POINTS_PER_THREAD; iter_point++) {

        int i_valid_points = point_offset + iter_point * BLOCK_SIZE + threadIdx.x;
        if (i_valid_points >= n_valid_points) {
          break;
        }
        int i_full_points = i_valid_point_2_i_full_points_map_ptr[i_valid_points];

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

        if (dist_tab_sq < multipole_radius_sq(current_spl_atom, j_atom_inner)) {

          // tab_single_atom_centered_coords_radial_log_p0_c_(&current_center, &dist_tab_sq, dir_tab, &dist_tab_in,
          // &i_r, &i_r_log, dir_tab_in);
          // double i_r;
          double i_r_log;
          double dir_tab_in[3];
          {
            double dist_tab_in = sqrt(dist_tab_sq);
            dir_tab_in[0] = dir_tab[0] / dist_tab_in;
            dir_tab_in[1] = dir_tab[1] / dist_tab_in;
            dir_tab_in[2] = dir_tab[2] / dist_tab_in;
            int i_species = species_center_ptr[current_center];
            i_r_log = 1.0 + log(dist_tab_in / r_grid_min_ptr[i_species - 1]) / log_r_grid_inc_ptr[i_species - 1];
          }

          // tab_single_trigonom_p0_c_(dir_tab_in, trigonom_tab);
          double trigonom_tab[4];
          {
            double abmax, abcmax, ab, abc;
            abmax = fmax(fabs(dir_tab_in[0]), fabs(dir_tab_in[1]));
            if (abmax > 1.0e-36) {
              ab = sqrt(dir_tab_in[0] * dir_tab_in[0] + dir_tab_in[1] * dir_tab_in[1]);
              trigonom_tab[3] = dir_tab_in[0] / ab;
              trigonom_tab[2] = dir_tab_in[1] / ab;
            } else {
              trigonom_tab[3] = 1.0;
              trigonom_tab[2] = 0.0;
              ab = 0.0;
            }
            abcmax = fmax(abmax, fabs(dir_tab_in[2]));
            if (abcmax > 1.0e-36) {
              abc = sqrt(ab * ab + dir_tab_in[2] * dir_tab_in[2]);
              trigonom_tab[1] = dir_tab_in[2] / abc;
              trigonom_tab[0] = ab / abc;
            } else {
              trigonom_tab[1] = 0.0;
              trigonom_tab[0] = 1.0;
            }
          }

          // 针对小体系是较大概率的，体系较大时需要重新确认

          delta_v_hartree(i_full_points, j_atom_inner) += SHEval_spline_vector_v2_n2_ddot_fused<-1>(
              l_atom_max,
              trigonom_tab[0],
              trigonom_tab[1],
              trigonom_tab[2],
              trigonom_tab[3],
              i_r_log,
              l_h_dim,
              (l_pot_max + 1) * (l_pot_max + 1),
              n_hartree_grid,
              n_grid_ptr[species_center_ptr[current_center] - 1],
              &current_delta_v_hart_part_spl_tile_trans(0, 0, 0, current_spl_atom, j_atom_inner));

        } else {
          // double dist_tab_out = sqrt(dist_tab_sq);
          // far_distance_real_hartree_potential_single_atom_p2_fused_fp_cu_inner<l_pot_max>(
          //     l_atom_max,
          //     l_max_analytic_multipole,
          //     i_center + 1,
          //     index_cc_dim_0,
          //     multipole_c_dim_0,
          //     dist_tab_out, // dist_tab_sq
          //     n_centers,
          //     n_centers_hartree_potential,
          //     n_atoms,
          //     hartree_force_l_add,
          //     &coord_points(0, i_full_points),
          //     n_cc_lm_ijk_ptr,
          //     center_to_atom_ptr,
          //     index_cc_ptr,
          //     index_ijk_max_cc_ptr,
          //     coords_center_ptr,
          //     &multipole_c(0, 0, j_atom_inner),
          //     &delta_v_hartree(i_full_points, j_atom_inner));
        }
      }
    }
  }
}

template <int l_pot_max, int BLOCK_SIZE, int POINTS_PER_THREAD>
__global__ void far_distance_real_hartree_potential_single_atom_p2_fused_fp_cu(
    // scalars
    const int j_atom_begin,
    const int j_atom_end,
    const int n_full_points,
    const int l_max_analytic_multipole,
    const int index_cc_dim_0,
    const int n_valid_points,
    // global scalars
    const int n_centers,
    const int n_centers_hartree_potential,
    const int n_atoms,
    const int hartree_force_l_add,
    // arrays
    const int *species_ptr,
    const int *i_valid_point_2_i_full_points_map_ptr,
    const double *coord_points_ptr,
    const double *multipole_radius_sq_ptr,
    const double *outer_potential_radius_ptr,
    const double *multipole_c_ptr,
    // global arrays
    const int *l_hartree_ptr,
    const int *n_cc_lm_ijk_ptr,
    const int *centers_hartree_potential_ptr,
    const int *center_to_atom_ptr,
    const int *index_cc_ptr,
    const int *index_ijk_max_cc_ptr,
    const double *coords_center_ptr,
    // output
    double *potential_ptr) {

  cTMf64<2> TM_INIT(coord_points, 3, n_full_points);
  cTMf64<2> TM_INIT(multipole_radius_sq, n_atoms, ATOM_TILE_SIZE);
  cTMf64<3> TM_INIT(outer_potential_radius, l_pot_max + 1, n_atoms, ATOM_TILE_SIZE);
  cTMf64<3> TM_INIT(multipole_c, n_cc_lm_ijk_ptr[l_pot_max], n_atoms, ATOM_TILE_SIZE);

  cTMf64<2> TM_INIT(coords_center, 3, n_centers);

  TMf64<2> TM_INIT(potential, n_full_points, ATOM_TILE_SIZE);

  int point_offset = blockIdx.x * BLOCK_SIZE * POINTS_PER_THREAD;

  int multipole_c_dim_0 = n_cc_lm_ijk_ptr[l_pot_max];

  for (int j_atom = j_atom_begin; j_atom <= j_atom_end; j_atom++) {
    int j_atom_inner = j_atom - j_atom_begin;

    // constexpr int center_tile_size = 8;
    // __shared__ double outer_potential_radius_shared[center_tile_size * l_pot_max];

    for (int i_center = 0; i_center < n_centers_hartree_potential; i_center++) {
      int current_center = centers_hartree_potential_ptr[i_center] - 1;
      int current_spl_atom = center_to_atom_ptr[current_center] - 1;

      const int l_atom_const_max = l_hartree_ptr[species_ptr[current_spl_atom] - 1];

      for (int iter_point = 0; iter_point < POINTS_PER_THREAD; iter_point++) {

        int i_valid_points = point_offset + iter_point * BLOCK_SIZE + threadIdx.x;
        if (i_valid_points >= n_valid_points) {
          break;
        }
        int i_full_points = i_valid_point_2_i_full_points_map_ptr[i_valid_points];

        // tab_single_atom_centered_coords_p0_c_(&current_center, coord_current, &dist_tab_sq, dir_tab);
        double dist_tab_sq = 0.0;
        double dir_tab[3];
        {
          for (int i_coord = 0; i_coord < 3; i_coord++) {
            dir_tab[i_coord] = coord_points(i_coord, i_full_points) - coords_center(i_coord, current_center);
            dist_tab_sq += dir_tab[i_coord] * dir_tab[i_coord];
          }
        }

        // l_atom_max <= l_pot_max
        int l_atom_max = l_atom_const_max;
        // #pragma unroll
        // for (int i = l_pot_max; i > 0; i--) {
        //   l_atom_max = (outer_potential_radius(i, current_spl_atom, j_atom_inner) < dist_tab_sq) ? i - 1 :
        //   l_atom_max;
        // }
        while (outer_potential_radius(l_atom_max, current_spl_atom, j_atom_inner) < dist_tab_sq && l_atom_max > 0) {
          l_atom_max--;
        }

        if (!(dist_tab_sq < multipole_radius_sq(current_spl_atom, j_atom_inner)) && dist_tab_sq < 1E8) {
          double dist_tab_out = sqrt(dist_tab_sq);
          far_distance_real_hartree_potential_single_atom_p2_fused_fp_cu_inner<l_pot_max>(
              l_atom_max,
              l_max_analytic_multipole,
              i_center + 1,
              index_cc_dim_0,
              multipole_c_dim_0,
              dist_tab_out,
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
              &potential(i_full_points, j_atom_inner));
        }
      }
    }
  }
}

// (dim0, dim1, dim2) -> (dim1, dim0, dim2)
template <int dim1>
__global__ void transpose_inner_constexpr_3d_array_cu(const int dim0, const int dim2, const double *src, double *dst) {
  int d0 = blockIdx.x * blockDim.x + threadIdx.x;
  int d2 = blockIdx.y * blockDim.y + threadIdx.y;

  if (d0 >= dim0 || d2 >= dim2) {
    return;
  }

  for (int d1 = 0; d1 < dim1; d1++) {
    dst[d1 + d0 * dim1 + d2 * dim1 * dim0] = src[d0 + d1 * dim0 + d2 * dim0 * dim1];
  }
}

__global__ void init_multipole_c_cu(
    const int n_atoms,
    const int l_max_analytic_multipole,
    const int index_cc_dim_0,
    const int l_pot_max,
    const int multipole_c_dim_0,
    const int *index_cc_ptr,
    const int *n_cc_lm_ijk_ptr,
    const int *l_hartree_max_far_distance_ptr,
    const double *multipole_moments_ptr,
    const double *cc_ptr,
    const int *index_lm_ptr,
    double *multipole_c_ptr) {

  const int l_pot_max_pow2 = (l_pot_max + 1) * (l_pot_max + 1);

  cTMi32<2> TM_INIT(l_hartree_max_far_distance, n_atoms, ATOM_TILE_SIZE);
  cTMi32<2> TM_INIT(index_cc, index_cc_dim_0, 6);
  cTMi32<2> TM_INIT(index_lm, l_pot_max * 2 + 1, l_pot_max + 1);

  cTMf64<3> TM_INIT(multipole_moments, l_pot_max_pow2, n_atoms, ATOM_TILE_SIZE);

  // output
  TMf64<3> TM_INIT(multipole_c, multipole_c_dim_0, n_atoms, ATOM_TILE_SIZE);

  int j_atom_inner = blockIdx.y;
  int i_atom = blockIdx.x;

  int n = threadIdx.x;
  if (n >= n_cc_lm_ijk_ptr[l_hartree_max_far_distance(i_atom, j_atom_inner)]) {
    return;
  }

  int i_l = index_cc(n, 0);             // index_cc(n, 1)
  int i_m = index_cc(n, 1) + l_pot_max; // index_cc(n, 2)

  if (i_l <= l_hartree_max_far_distance(i_atom, j_atom_inner) && //
      abs(multipole_moments(index_lm(i_m, i_l) - 1, i_atom, j_atom_inner)) > 1e-10) {
    multipole_c(n, i_atom, j_atom_inner) = cc_ptr[n] * multipole_moments(index_lm(i_m, i_l) - 1, i_atom, j_atom_inner);
  } else {
    multipole_c(n, i_atom, j_atom_inner) = 0;
  }
}

// 你是一名 c++
// 专家，请你把以下这些变量和数组导出到一份文件二进制中，并编写一个写入的函数，和一个读取并调用该函数的代码，

void sum_up_whole_potential_c_v3_atoms_full_points_j_atom_tile_cu_host_(
    // scalars
    const int j_atom_begin,
    const int j_atom_end,
    const int n_full_points,
    const int l_max_analytic_multipole,
    const int index_cc_dim_0,
    const int n_valid_points,
    // global scalars
    const int l_pot_max,
    const int n_max_radial,
    const int n_hartree_grid,
    const int n_centers,
    const int n_centers_hartree_potential,
    const int n_atoms,
    const int hartree_force_l_add,
    const int n_species,
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

  Tf64<1> ylm_tab((l_pot_max + 1) * (l_pot_max + 1));

  const int multipole_c_dim_0 = n_cc_lm_ijk_ptr[l_pot_max];

  // multipole_moments, outer_potential_radius, multipole_radius_sq 与 j_atom 相关

  DEV_STREAM_T &stream = devInfo.stream;

  EventHelper<enable_profile_sumup_each_kernel> event_helper(stream);
  EventHelper<enable_profile_sumup_end_to_end> event_helper_all(stream);

  // input & output arrays
  TM_DEV_PS_H2D_H(l_hartree_max_far_distance);
  TM_DEV_PS_H2D_H(multipole_radius_sq);
  TM_DEV_PS_H2D_H(outer_potential_radius);
  TM_DEV_PS_H2D_H(multipole_moments);
  TM_DEV_PS_H2D_H(current_delta_v_hart_part_spl_tile);

  TM_DEV_PS_H2D_H(delta_v_hartree);

  {
    event_helper.record_start();

    massert(multipole_c_dim_0 <= 1024);
    dim3 blockSizes(multipole_c_dim_0, 1, 1);
    dim3 gridSizes(n_atoms, (j_atom_end - j_atom_begin + 1), 1);
    init_multipole_c_cu<<<gridSizes, blockSizes, 0, stream>>>(
        n_atoms,
        l_max_analytic_multipole,
        index_cc_dim_0,
        l_pot_max,
        multipole_c_dim_0,
        devPs.index_cc.ptr,
        devPs.n_cc_lm_ijk.ptr,
        devPs.l_hartree_max_far_distance.ptr,
        devPs.multipole_moments.ptr,
        devPs.cc.ptr,
        devPs.index_lm.ptr,
        devPs.multipole_c.ptr);
    event_helper.elapsed_time("Kernel init_multipole_c_cu execution time in stream");
  }

  static constexpr int block_size = 128;
  static constexpr int point_per_thread = 1;
  static constexpr int point_per_block = block_size * point_per_thread;
  int block_dim = (n_valid_points + point_per_block - 1) / point_per_block;

  // 交换维度
  // current_delta_v_hart_part_spl_tile, l_pot_max_pow2, n_coeff_hartree, n_hartree_grid, n_atoms, ATOM_TILE_SIZE);
  // current_delta_v_hart_part_spl_tile, n_coeff_hartree, l_pot_max_pow2, n_hartree_grid, n_atoms, ATOM_TILE_SIZE);
  event_helper.record_start();
  {
    // n_coeff_hartree 很小，目前是常量 2
    // 假设 l_pot_max = 6，则 l_pot_max_pow2 = 36
    dim3 blockSizes(36, 16, 1);
    dim3 gridSizes(
        CDIV(l_pot_max_pow2, blockSizes.x), CDIV(n_hartree_grid * n_atoms * ATOM_TILE_SIZE, blockSizes.y), 1);
    transpose_inner_constexpr_3d_array_cu<n_coeff_hartree><<<gridSizes, blockSizes, 0, stream>>>(
        l_pot_max_pow2,
        n_hartree_grid * n_atoms * ATOM_TILE_SIZE,
        devPs.current_delta_v_hart_part_spl_tile.ptr,
        devPs.current_delta_v_hart_part_spl_tile_trans.ptr);
  }
  event_helper.elapsed_time("Kernel transpose current_delta_v_hart_part_spl_tile execution time in stream");

  event_helper.record_start();
  sum_up_close_part_cu<6, block_size, point_per_thread><<<block_dim, block_size, 0, stream>>>(
      j_atom_begin,
      j_atom_end,
      n_full_points,
      l_max_analytic_multipole,
      index_cc_dim_0,
      n_valid_points,
      n_max_radial,
      n_hartree_grid,
      n_centers,
      n_centers_hartree_potential,
      n_atoms,
      hartree_force_l_add,
      n_species,
      devPs.n_radial.ptr,
      devPs.species.ptr,
      devPs.species_center.ptr,
      devPs.l_hartree_max_far_distance.ptr,
      devPs.i_valid_point_2_i_full_points_map.ptr,
      devPs.coord_points.ptr,
      devPs.multipole_radius_sq.ptr,
      devPs.outer_potential_radius.ptr,
      devPs.multipole_moments.ptr,
      devPs.current_delta_v_hart_part_spl_tile_trans.ptr,
      devPs.multipole_c.ptr,
      devPs.l_hartree.ptr,
      devPs.n_grid.ptr,
      devPs.n_cc_lm_ijk.ptr,
      devPs.centers_hartree_potential.ptr,
      devPs.center_to_atom.ptr,
      devPs.index_cc.ptr,
      devPs.index_ijk_max_cc.ptr,
      devPs.cc.ptr,
      devPs.coords_center.ptr,
      devPs.r_grid_min.ptr,
      devPs.log_r_grid_inc.ptr,
      devPs.scale_radial.ptr,
      devPs.partition_tab.ptr,
      devPs.delta_v_hartree.ptr);
  event_helper.elapsed_time("Kernel sum_up_close_part_cu execution time in stream");

  event_helper.record_start();

  far_distance_real_hartree_potential_single_atom_p2_fused_fp_cu<6, block_size, point_per_thread>
      <<<block_dim, block_size, 0, stream>>>(
          j_atom_begin,
          j_atom_end,
          n_full_points,
          l_max_analytic_multipole,
          index_cc_dim_0,
          n_valid_points,
          n_centers,
          n_centers_hartree_potential,
          n_atoms,
          hartree_force_l_add,
          devPs.species.ptr,
          devPs.i_valid_point_2_i_full_points_map.ptr,
          devPs.coord_points.ptr,
          devPs.multipole_radius_sq.ptr,
          devPs.outer_potential_radius.ptr,
          devPs.multipole_c.ptr,
          devPs.l_hartree.ptr,
          devPs.n_cc_lm_ijk.ptr,
          devPs.centers_hartree_potential.ptr,
          devPs.center_to_atom.ptr,
          devPs.index_cc.ptr,
          devPs.index_ijk_max_cc.ptr,
          devPs.coords_center.ptr,
          devPs.delta_v_hartree.ptr);

  event_helper.elapsed_time(
      "Kernel far_distance_real_hartree_potential_single_atom_p2_fused_fp_cu execution time in stream");

  TM_DEV_PS_D2H_H(delta_v_hartree);
  DEV_CHECK(DEV_STREAM_SYNCHRONIZE(stream));
}