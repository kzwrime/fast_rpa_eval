#pragma once

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
    int *not_use);
