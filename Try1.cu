#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std;

const int nx = 10;    // Tamaño en x
const int ny = 10;    // Tamaño en y
const int nit = 50;   // Número de iteraciones
const double dt = 0.1; // Tiempo de simulación
const double dx = 1.0; // Espaciado en x
const double dy = 1.0; // Espaciado en y
const double nu = 0.1; // Viscosidad cinemática

#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                    \
    }                                                               \
}

__global__ void updateVelocities(double* u, double* v, double* u_new, int nx, int ny, double dt, double dx, double dy, double nu) {
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

void applyBoundaryConditions(double* u, int nx, int ny) {
    // Aplicar condiciones de frontera: u = 1 en la frontera izquierda y u = 0 en la frontera derecha
    for (int i = 0; i < ny; ++i) {
        u[i * nx] = 1.0;          // Frontera izquierda
        u[i * nx + (nx - 1)] = 0.0; // Frontera derecha
    }
}

void printMatrix(const double* array, int nx, int ny) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            cout << array[i * nx + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    size_t size = nx * ny * sizeof(double);

    // Reservar memoria en host
    double* h_u = new double[nx * ny]();
    double* h_v = new double[nx * ny]();
    double* h_u_new = new double[nx * ny]();

    // Inicializar condiciones de frontera
    applyBoundaryConditions(h_u, nx, ny);

    // Imprimir la matriz inicial
    cout << "Matriz inicial:" << endl;
    printMatrix(h_u, nx, ny);

    // Reservar memoria en dispositivo
    double *d_u, *d_v, *d_u_new;
    cudaMalloc(&d_u, size);
    cudaCheckError();
    cudaMalloc(&d_v, size);
    cudaCheckError();
    cudaMalloc(&d_u_new, size);
    cudaCheckError();

    // Copiar datos iniciales al dispositivo
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
    cudaCheckError();

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int it = 0; it < nit; ++it) {
        // Actualizar velocidades en GPU
        updateVelocities<<<numBlocks, threadsPerBlock>>>(d_u, d_v, d_u_new, nx, ny, dt, dx, dy, nu);

        // Sincronizar para esperar que la GPU termine
        cudaDeviceSynchronize();
        cudaCheckError();

        // Copiar resultados al host
        cudaMemcpy(h_u_new, d_u_new, size, cudaMemcpyDeviceToHost);
        cudaCheckError();

        // Aplicar condiciones de frontera en el host
        applyBoundaryConditions(h_u_new, nx, ny);
        cudaCheckError();

        // Copiar las matrices actualizadas de vuelta a la GPU
        cudaMemcpy(d_u, h_u_new, size, cudaMemcpyHostToDevice);
        cudaCheckError();

        // Imprimir la matriz de velocidades
        cout << "Iteración " << it + 1 << ":" << endl;
        printMatrix(h_u_new, nx, ny);

        // Esperar un tiempo antes de imprimir la siguiente matriz
        this_thread::sleep_for(chrono::milliseconds(500));
    }

    // Liberar memoria
    delete[] h_u;
    delete[] h_v;
    delete[] h_u_new;
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_u_new);

    cout << "Simulación completada." << endl;
    return 0;
}
