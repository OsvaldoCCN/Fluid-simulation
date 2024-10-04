#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda

using namespace std;

#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                    \
    }                                                               \
}

/*
* Kernel para actualizar las velocidades en la matriz de velocidades
* u: Matriz de velocidades en x
* v: Matriz de velocidades en y
* u_new: Matriz de velocidades actualizadas
* nx: Tamaño en x
* ny: Tamaño en y
* dt: Paso de tiempo
* dx: Espaciado en x
* dy: Espaciado en y
* nu: Viscosidad cinemática
*/
__global__ void updateVelocities(float* u, float* v, float* u_new, int nx, int ny, float dt, float dx, float dy, float nu) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < ny - 1 && j > 0 && j < nx - 1) {
        int idx = i * nx + j;

        // Advección y difusión para u
        u_new[idx] = u[idx]
            - dt * u[idx] * (u[idx + 1] - u[idx - 1]) / (2.0 * dx)
            - dt * v[idx] * (u[(i + 1) * nx + j] - u[(i - 1) * nx + j]) / (2.0 * dy)
            + nu * dt * ((u[(i + 1) * nx + j] - 2.0 * u[idx] + u[(i - 1) * nx + j]) / (dy * dy)
                        + (u[i * nx + (j + 1)] - 2.0 * u[idx] + u[i * nx + (j - 1)]) / (dx * dx));
    }
}

/*
* Kernel para aplicar condiciones de frontera en la matriz de velocidades
* u: Matriz de velocidades
* nx: Tamaño en x
* ny: Tamaño en y
*/
__global__ void applyBoundaryConditions(float* u, int nx, int ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < ny) {
        u[i * nx] = 1.0;          // Frontera izquierda
        u[i * nx + (nx - 1)] = 0.0; // Frontera derecha
    }
}

void printMatrix(const float* array, int nx, int ny) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            cout << array[i * nx + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    const int nx = 10;    // Tamaño en x
    const int ny = 10;    // Tamaño en y
    const int nit = 100000;   // Número de iteraciones
    const float dt = 0.1; // Tiempo de simulación
    const float dx = 1.0; // Espaciado en x
    const float dy = 1.0; // Espaciado en y
    const float nu = 0.1; // Viscosidad cinemática

    // Inicializar Mimir
    MimirEngine engine;
    ViewerOptions options;
    options.window_size = {800, 600}; // Tamaño de la ventana
    options.target_fps = 60;          // FPS objetivo
    engine.init(options);

    size_t size = nx * ny * sizeof(float);
    cout << "Tamaño de datos: " << nx << "x" << ny << endl; // DEPURACIÓN

    // Reservar memoria en dispositivo
    float *d_u, *d_v, *d_u_new;
    cudaMalloc(&d_u, size);
    cudaCheckError();
    cudaMalloc(&d_v, size);
    cudaCheckError();
    cudaMalloc(&d_u_new, size);
    cudaCheckError();

    // Inicializar condiciones de frontera directamente en la GPU
    applyBoundaryConditions<<<(ny + 15) / 16, 16>>>(d_u, nx, ny);
    cudaDeviceSynchronize();
    cudaCheckError();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Crear buffer de visualización
    MemoryParams m;
    m.layout          = DataLayout::Layout2D;
    m.element_count.x = nx;
    m.element_count.y = ny;
    m.component_type  = ComponentType::Float;
    m.channel_count   = 1;
    m.resource_type   = ResourceType::Buffer;
    auto buf = engine.createBuffer((void**)&d_u, m);

    // Crear vista para la visualización
    ViewParams params;
    params.element_count = nx * ny;
    params.data_domain   = DataDomain::Domain2D;
    params.view_type     = ViewType::Markers; // Cambia el tipo de vista según lo que quieras visualizar
    params.attributes[AttributeType::Position] = *buf;
    engine.createView(params);


    for (int it = 0; it < nit; ++it) {
        // Actualizar velocidades en GPU
        updateVelocities<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_u_new, nx, ny, dt, dx, dy, nu);
        cudaDeviceSynchronize();
        cudaCheckError();
        
        // Aplicar condiciones de frontera en la GPU
        applyBoundaryConditions<<<(ny + 15) / 16, 16>>>(d_u_new, nx, ny);
        cudaDeviceSynchronize();
        cudaCheckError();

        // Copiar resultados de d_u_new a d_u
        cudaMemcpy(d_u, d_u_new, size, cudaMemcpyDeviceToDevice);
        cudaCheckError();

        // Actualizar la vista en Mimir
        engine.updateViews();

        // Imprimir la matriz de velocidades (opcional)
        // printMatrix(d_u, nx, ny); // Esta función no puede ser llamada sin copia a host
    }

    // Liberar memoria
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u_new);

    // Salir de Mimir
    engine.exit();

    cout << "Simulación completada." << endl;
    return 0;
}
