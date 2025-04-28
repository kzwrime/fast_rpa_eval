#pragma once

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
    double *density_matrix_ptr);

extern "C" void evaluate_first_order_rho_reduce_memory_c_v3_batches_atoms_cu_host_(
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
    double *density_matrix_ptr);
