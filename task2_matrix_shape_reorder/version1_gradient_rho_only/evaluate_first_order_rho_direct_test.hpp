#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

struct FirstOrderRhoMetaData {
  int n_my_batches_work;
  int n_full_points;
  int n_basis;
  int n_atoms;
  int n_max_compute_ham;
  int n_centers_basis_I;
  int n_max_batch_size;

  std::vector<int> basis_atom;
  std::vector<int> batch_sizes;
  std::vector<int> n_point_batches;
  std::vector<int> n_point_batches_prefix_sum;
  std::vector<int> i_valid_point_2_i_full_points_map;
  std::vector<int> n_compute_c_batches;
  std::vector<int> i_basis_batches;
  std::vector<int> atom_valid_n_compute_c_batches;
  std::vector<int> i_batch_2_wave_offset;
};

// Function to write FirstOrderRhoMetaData to binary file
template <class T = void>
void write_first_order_rho_meta_data(const std::string &filename, const FirstOrderRhoMetaData &data) {
  std::ofstream outfile(filename, std::ios::binary);
  if (!outfile) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  // Write scalar values
  outfile.write(reinterpret_cast<const char *>(&data.n_my_batches_work), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_full_points), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_basis), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_atoms), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_max_compute_ham), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_centers_basis_I), sizeof(int));
  outfile.write(reinterpret_cast<const char *>(&data.n_max_batch_size), sizeof(int));

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

  write_vector(data.basis_atom);
  write_vector(data.batch_sizes);
  write_vector(data.n_point_batches);
  write_vector(data.n_point_batches_prefix_sum);
  write_vector(data.i_valid_point_2_i_full_points_map);
  write_vector(data.n_compute_c_batches);
  write_vector(data.i_basis_batches);
  write_vector(data.atom_valid_n_compute_c_batches);
  write_vector(data.i_batch_2_wave_offset);

  if (!outfile) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

// Function to read FirstOrderRhoMetaData from binary file
template <class T = void>
FirstOrderRhoMetaData read_first_order_rho_meta_data(const std::string &filename) {
  std::ifstream infile(filename, std::ios::binary);
  if (!infile) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  FirstOrderRhoMetaData data;

  // Read scalar values
  infile.read(reinterpret_cast<char *>(&data.n_my_batches_work), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_full_points), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_basis), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_atoms), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_max_compute_ham), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_centers_basis_I), sizeof(int));
  infile.read(reinterpret_cast<char *>(&data.n_max_batch_size), sizeof(int));

  printf("n_my_batches_work = %d\n", data.n_my_batches_work);
  printf("n_full_points = %d\n", data.n_full_points);
  printf("n_basis = %d\n", data.n_basis);
  printf("n_atoms = %d\n", data.n_atoms);
  printf("n_max_compute_ham = %d\n", data.n_max_compute_ham);
  printf("n_centers_basis_I = %d\n", data.n_centers_basis_I);
  printf("n_max_batch_size = %d\n", data.n_max_batch_size);

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
  read_vector(data.basis_atom);
  read_vector(data.batch_sizes);
  read_vector(data.n_point_batches);
  read_vector(data.n_point_batches_prefix_sum);
  read_vector(data.i_valid_point_2_i_full_points_map);
  read_vector(data.n_compute_c_batches);
  read_vector(data.i_basis_batches);
  read_vector(data.atom_valid_n_compute_c_batches);
  read_vector(data.i_batch_2_wave_offset);

  if (!infile) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return data;
}