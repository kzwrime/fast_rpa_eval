#ifndef SETTING_H
#define SETTING_H

#define N_BATCHES_TILE 256

#define N_COMPUTE_C_PADDING_SIZE 4

#if defined(__cplusplus) || defined(__STDC_VERSION__)
// only define these constants when compiling as C or C++
constexpr int n_coeff_hartree = 2;
constexpr int n_max_spline = 4;
// constexpr int n_max_spline = 4;
#endif

#endif

// #define SAVE_DFPT_DATA_TEST

#ifndef ATOM_TILE_SIZE
#define ATOM_TILE_SIZE 1
#endif
