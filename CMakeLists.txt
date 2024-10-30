add_executable(simulador simulador.cu)
target_link_libraries(simulador PRIVATE mimir OpenMP::OpenMP_CXX)
set_target_properties(simulador PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/samples
)
