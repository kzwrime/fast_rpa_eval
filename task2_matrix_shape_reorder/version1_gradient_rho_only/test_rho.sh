#!/bin/bash

# Array of tile sizes to test
tile_sizes=(1 2 4 8)

for size in "${tile_sizes[@]}"; do
    echo "Testing with ATOM_TILE_SIZE=$size"
    
    # Get current CXXFLAGS from Makefile and update ATOM_TILE_SIZE
    original_line=$(grep "^CXXFLAGS :=" Makefile)
    new_line=$(echo "$original_line" | sed "s/-DATOM_TILE_SIZE=[0-9]\+/-DATOM_TILE_SIZE=$size/")
    
    # Update the Makefile
    sed -i "s|^CXXFLAGS :=.*|$new_line|" Makefile
    
    # Clean and rebuild
    make clean
    make -j8
    
    echo "Running test with ATOM_TILE_SIZE=$size"
    ./build/evaluate_first_order_rho_run.out
    echo "----------------------------------------"
done

# Restore original CXXFLAGS by getting it from the backup
original_line=$(grep "^CXXFLAGS :=" Makefile | sed "s/-DATOM_TILE_SIZE=[0-9]\+/-DATOM_TILE_SIZE=4/")
sed -i "s|^CXXFLAGS :=.*|$original_line|" Makefile
