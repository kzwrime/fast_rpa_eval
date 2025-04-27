#pragma once

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
    const double *multipole_radius_sq_ptr,                // (n_atoms, n_full_points)
    const double *outer_potential_radius_ptr,             // (l_pot_max + 1, n_atoms, n_full_points)
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
);

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
);
