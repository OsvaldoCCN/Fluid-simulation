cmake_minimum_required(VERSION 3.17)
project(GOL_mimir LANGUAGES CXX CUDA)

# Incluir FetchContent para obtener Mimir desde el repositorio
include(FetchContent)

# Set default build type to 'Release'
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# Añadir ruta al CMake Module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMake")

# Encontrar OpenMP si está disponible
find_package(OpenMP)

# FetchContent para clonar la biblioteca Mimir si no está disponible
FetchContent_Declare(
    mimir
    GIT_REPOSITORY https://github.com/temporal-hpc/mimir.git
    GIT_TAG        279e2ccea5f76fc806f5d867e8fc873b83ce4875 # master
    FIND_PACKAGE_ARGS
)
FetchContent_MakeAvailable(mimir)

# Añadir subdirectorios relevantes si son necesarios para el proyecto
# Aquí se pueden incluir otros directorios si los tuvieras

# Compilar el archivo GOL_mimir.cu
add_executable(fluid_simulation fluid_simulation.cu)

# Vincular Mimir y OpenMP (si se encuentra)
target_link_libraries(fluid_simulation mimir)

# Especificar las propiedades de compilación CUDA
set_target_properties(fluid_simulation PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "all"
)