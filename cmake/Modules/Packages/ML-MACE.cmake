cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# find mace_ops
find_library(MACE_OPS_LIBRARY mace_ops
    HINTS /home/wcw28/work/mace-cuda/mace_ops/build/lib.linux-x86_64-cpython-310/mace_ops/lib 
    REQUIRED)
target_link_libraries(lammps PRIVATE -Wl,--no-as-needed ${MACE_OPS_LIBRARY})

# find sphericart
find_library(SPHERICART_LIBRARY sphericart_torch
    HINTS /home/wcw28/work/mace-cuda/sphericart/sphericart-torch/build/lib.linux-x86_64-cpython-310/sphericart/torch/lib 
    REQUIRED)
target_link_libraries(lammps PRIVATE ${SPHERICART_LIBRARY})

# find torch
find_program(Python_EXECUTABLE python)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
    RESULT_VARIABLE TORCH_CMAKE_PATH_RESULT
    OUTPUT_VARIABLE TORCH_CMAKE_PATH_OUTPUT
    ERROR_VARIABLE TORCH_CMAKE_PATH_ERROR
)
string(STRIP ${TORCH_CMAKE_PATH_OUTPUT} TORCH_CMAKE_PATH_OUTPUT)
if (NOT ${TORCH_CMAKE_PATH_RESULT} EQUAL 0)
    message(FATAL_ERROR "Failed to find pytorch with error: ${TORCH_CMAKE_PATH_ERROR}\n")
endif()
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${TORCH_CMAKE_PATH_OUTPUT}")
find_package(Torch 1.13 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(lammps PRIVATE "${TORCH_LIBRARIES}")
