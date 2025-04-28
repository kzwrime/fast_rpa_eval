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

#include "sum_up_far_dist_potential_dot_auto_generated_by_index_cc_aot.hpp"

constexpr bool enable_profile_sumup_each_kernel = false;
constexpr bool enable_profile_sumup_end_to_end = false;

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

  double dpot = 0.0;

  // int n_end = n_cc_lm_ijk(l_max);

  // for (int n = 0; n < n_end; n++) {
  //   int ii = index_cc(n, 3 - 1);
  //   int jj = index_cc(n, 4 - 1);
  //   int kk = index_cc(n, 5 - 1);
  //   int nn = index_cc(n, 6 - 1);
  //   dpot += coord_c[0][ii] * coord_c[1][jj] * coord_c[2][kk] * //
  //           Fp[nn] * multipole_c(n, center_to_atom(i_center - 1) - 1);
  // }
  //   delta_v_hartree[i_full_points - 1] += dpot;

  {
    int one_minus_2l = 1;
    double inv_dist_sq = 1.0 / dist_sq;
    Fp[0] = 1.0 / dist_tab;
    for (int i = 1; i < (l_pot_max + 2); i++) {
      one_minus_2l -= 2;
      Fp[i] = Fp[i - 1] * one_minus_2l * inv_dist_sq;
    }
    dpot += sum_up_far_dist_potential_dot_spec(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1), l_max);

    // if (l_max == 1) {
    //   dpot += sum_up_far_dist_potential_dot_spec<1>(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1));
    // } else if (l_max == 2) {
    //   dpot += sum_up_far_dist_potential_dot_spec<2>(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1));
    // } else if (l_max == 3) {
    //   dpot += sum_up_far_dist_potential_dot_spec<3>(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1));
    // } else if (l_max == 4) {
    //   dpot += sum_up_far_dist_potential_dot_spec<4>(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1));
    // } else if (l_max == 5) {
    //   dpot += sum_up_far_dist_potential_dot_spec<5>(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1));
    // } else if (l_max == 6) {
    //   dpot += sum_up_far_dist_potential_dot_spec<6>(dir, Fp, &multipole_c(0, center_to_atom(i_center - 1) - 1));
    // }
  }

  *potential += dpot;
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

}