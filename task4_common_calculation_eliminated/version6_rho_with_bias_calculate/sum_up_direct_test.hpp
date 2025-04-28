#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// Structure to hold all the data
struct HartreePotentialData {
  // Scalars
  int j_atom_begin;
  int j_atom_end;
  int n_full_points;
  int l_max_analytic_multipole;
  int index_cc_dim_0;
  int n_valid_points;

  // Global scalars
  int l_pot_max;
  int n_max_radial;
  int n_hartree_grid;
  int n_centers;
  int n_centers_hartree_potential;
  int n_atoms;
  int hartree_force_l_add;
  int n_species;

  // Arrays
  std::vector<int> n_radial;
  std::vector<int> species;
  std::vector<int> species_center;
  std::vector<int> l_hartree_max_far_distance;
  std::vector<int> i_valid_point_2_i_full_points_map;
  std::vector<double> coord_points;
  std::vector<double> multipole_radius_sq;
  std::vector<double> outer_potential_radius;
  std::vector<double> multipole_moments;
  // std::vector<double> current_delta_v_hart_part_spl_tile;

  // Global arrays
  std::vector<int> l_hartree;
  std::vector<int> n_grid;
  std::vector<int> n_cc_lm_ijk;
  std::vector<int> centers_hartree_potential;
  std::vector<int> center_to_atom;
  std::vector<int> index_cc;
  std::vector<int> index_ijk_max_cc;
  std::vector<double> cc;
  std::vector<double> coords_center;
  std::vector<double> r_grid_min;
  std::vector<double> log_r_grid_inc;
  std::vector<double> scale_radial;
  std::vector<double> partition_tab;

  // // Output
  // std::vector<double> delta_v_hartree;
  // std::vector<double> delta_v_hartree_ref;
};

// Function to write data to binary file
template <class T = void>
void write_hartree_potential_data(const std::string &filename, const HartreePotentialData &data) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Write scalar values
  outfile.write(reinterpret_cast<const char *>(&data.j_atom_begin), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.j_atom_end), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_full_points), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.l_max_analytic_multipole), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.index_cc_dim_0), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_valid_points), sizeof(int));

  outfile.write(reinterpret_cast<const char *>(&data.l_pot_max), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_max_radial), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_hartree_grid), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_centers), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_centers_hartree_potential), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_atoms), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.hartree_force_l_add), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_species), sizeof(int));

  // Write array sizes and data
  auto write_vector = [&outfile](const auto &vec) {
    size_t size = vec.size();
    if (size <= 0) {
      printf("%s:%d vector size should > 0, check your code!\n", __FILE__, __LINE__);
      exit(-1);
    }
    outfile.write(reinterpret_cast<const char *>(&size), sizeof(size_t));
    outfile.write(
        reinterpret_cast<const char *>(vec.data()),
        size * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type));
  };

  write_vector(data.n_radial);
  write_vector(data.species);
  write_vector(data.species_center);
  write_vector(data.l_hartree_max_far_distance);
  write_vector(data.i_valid_point_2_i_full_points_map);
  write_vector(data.coord_points);
  write_vector(data.multipole_radius_sq);
  write_vector(data.outer_potential_radius);
  write_vector(data.multipole_moments);
  // write_vector(data.current_delta_v_hart_part_spl_tile);

  write_vector(data.l_hartree);
  write_vector(data.n_grid);
  write_vector(data.n_cc_lm_ijk);
  write_vector(data.centers_hartree_potential);
  write_vector(data.center_to_atom);
  write_vector(data.index_cc);
  write_vector(data.index_ijk_max_cc);
  write_vector(data.cc);
  write_vector(data.coords_center);
  write_vector(data.r_grid_min);
  write_vector(data.log_r_grid_inc);
  write_vector(data.scale_radial);
  write_vector(data.partition_tab);

  // write_vector(data.delta_v_hartree);
  // write_vector(data.delta_v_hartree_ref);

  if (!outfile) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

// Function to read data from binary file
template <class T = void>
HartreePotentialData read_hartree_potential_data(const std::string &filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  HartreePotentialData data;

  // Read scalar values
  infile.read(reinterpret_cast<char *>(&data.j_atom_begin), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.j_atom_end), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_full_points), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.l_max_analytic_multipole), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.index_cc_dim_0), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_valid_points), sizeof(int));

  infile.read(reinterpret_cast<char *>(&data.l_pot_max), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_max_radial), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_hartree_grid), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_centers), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_centers_hartree_potential), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_atoms), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.hartree_force_l_add), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_species), sizeof(int));

  printf("j_atom_begin = %d\n", data.j_atom_begin);
  printf("j_atom_end = %d\n", data.j_atom_end);
  printf("n_full_points = %d\n", data.n_full_points);
  printf("l_max_analytic_multipole = %d\n", data.l_max_analytic_multipole);
  printf("index_cc_dim_0 = %d\n", data.index_cc_dim_0);
  printf("n_valid_points = %d\n", data.n_valid_points);
  printf("l_pot_max = %d\n", data.l_pot_max);
  printf("n_max_radial = %d\n", data.n_max_radial);
  printf("n_hartree_grid = %d\n", data.n_hartree_grid);
  printf("n_centers = %d\n", data.n_centers);
  printf("n_centers_hartree_potential = %d\n", data.n_centers_hartree_potential);
  printf("n_atoms = %d\n", data.n_atoms);
  printf("hartree_force_l_add = %d\n", data.hartree_force_l_add);
  printf("n_species = %d\n", data.n_species);

  // Helper function to read vectors
  auto read_vector = [&infile](auto &vec) {
    size_t size;
    infile.read(reinterpret_cast<char *>(&size), sizeof(size_t));
    if (size <= 0) {
      printf("%s:%d vector size should > 0, check your code!\n", __FILE__, __LINE__);
      exit(-1);
    }
    vec.resize(size);
    infile.read(
        reinterpret_cast<char *>(vec.data()),
        size * sizeof(typename std::remove_reference<decltype(vec)>::type::value_type));
  };

  // Read arrays
  read_vector(data.n_radial);
  read_vector(data.species);
  read_vector(data.species_center);
  read_vector(data.l_hartree_max_far_distance);
  read_vector(data.i_valid_point_2_i_full_points_map);
  read_vector(data.coord_points);
  read_vector(data.multipole_radius_sq);
  read_vector(data.outer_potential_radius);
  read_vector(data.multipole_moments);
  // read_vector(data.current_delta_v_hart_part_spl_tile);

  read_vector(data.l_hartree);
  read_vector(data.n_grid);
  read_vector(data.n_cc_lm_ijk);
  read_vector(data.centers_hartree_potential);
  read_vector(data.center_to_atom);
  read_vector(data.index_cc);
  read_vector(data.index_ijk_max_cc);
  read_vector(data.cc);
  read_vector(data.coords_center);
  read_vector(data.r_grid_min);
  read_vector(data.log_r_grid_inc);
  read_vector(data.scale_radial);
  read_vector(data.partition_tab);

  // read_vector(data.delta_v_hartree);
  // read_vector(data.delta_v_hartree_ref);

  if (!infile) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return data;
}