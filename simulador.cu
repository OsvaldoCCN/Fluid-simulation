#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <thread>



// incluir Mimir para la visualizacion de la simulacion en GPU
#include <mimir/mimir.hpp>
#include <mimir/validation.hpp> // checkCuda
using namespace mimir;
using namespace mimir::validation; // checkCuda
using namespace std;

// Definición del tamaño del grid
#define NX 500
#define NY 500
#define NT 500
#define BLOCK_SIZE 16  // Tamaño del bloque CUDA



__global__ void compute_curl(float *u, float *v, float *curl, int nx, int ny, float dx, float dy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        // Calcular las derivadas
        float du_dy = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2 * dy);
        float dv_dx = (v[j * nx + (i + 1)] - v[j * nx + (i - 1)]) / (2 * dx);

        // Calcular el curl
        curl[j * nx + i] = du_dy - dv_dx;
    }
}

__global__ void apply_boundary_conditions(float *u, float *v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Condiciones de no deslizamiento en paredes superior e inferior (y = 0 y y = ny-1)
    if (i < nx) {
        u[i] = 0.0f;                  // Borde inferior (y = 0)
        v[i] = 0.0f;
        u[(ny - 1) * nx + i] = 0.0f;  // Borde superior (y = ny-1)
        v[(ny - 1) * nx + i] = 0.0f;
    }

    // Condiciones de no deslizamiento en los extremos izquierdo y derecho (x = 0 y x = nx-1)
    if (i < ny) {
        u[i * nx] = 0.0f;                  // Borde izquierdo (x = 0)
        v[i * nx] = 0.0f;
        u[i * nx + (nx - 1)] = 0.0f;       // Borde derecho (x = nx-1)
        v[i * nx + (nx - 1)] = 0.0f;
    }
}


__global__ void initialize_laminar_flow(float *u, float *v, int nx, int ny, float u_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int idx = j * nx + i;

        // Condiciones de flujo laminar (Poiseuille) en la dirección x
        float y_position = static_cast<float>(j) / static_cast<float>(ny - 1);  // Normalizado entre 0 y 1
        u[idx] = 4 * u_max * y_position * (1.0f - y_position);  // Perfil parabólico de velocidad
        v[idx] = 0.0f;  // No hay velocidad en la dirección y
    }
}


/*
__global__ void border_condition(float *u, float *v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Asegúrate de que i y j están dentro de los límites
    if (i < nx && j < ny) {
        // Condiciones periódicas en el eje x
        if (i == 0) {
            u[j * nx + nx - 1] = u[j * nx + 0]; // Pared izquierda -> derecha
            v[j * nx + nx - 1] = v[j * nx + 0];
        } else if (i == nx - 1) {
            u[j * nx + 0] = u[j * nx + nx - 1]; // Pared derecha -> izquierda
            v[j * nx + 0] = v[j * nx + nx - 1];
        }

        // Condiciones periódicas en el eje y
        if (j == ny -1) {
            u[0 * nx + i] = u[(ny - 1) * nx + i]; // Pared inferior -> superior
            v[0 * nx + i] = v[(ny - 1) * nx + i];
        } else if (j == 0) {
            u[(ny - 1) * nx + i] = u[0 * nx + i]; // Pared superior -> inferior
            v[(ny - 1) * nx + i] = v[0 * nx + i];
        }
    }
}
*/



__global__ void initialize_perturbacion(float *u, float *v, int nx, int ny, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Inicializar el generador de números aleatorios para cada hilo
    curandState state;
    curand_init(seed, j * nx + i, 0, &state);

    if (i < nx && j < ny) {
        int idx = j * nx + i;

        // Generar perturbaciones aleatorias
        float perturbation_u = curand_uniform(&state) * 1; // Perturbación entre 0 y 0.1
        float perturbation_v = curand_uniform(&state) * 1; // Perturbación entre 0 y 0.1

        // Inicializar velocidades con perturbaciones aleatorias
        u[idx] = perturbation_u;
        v[idx] = perturbation_v;
    }
}

__global__ void initialize_impulse(float *u, float *v, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx && j < ny) {
        int idx = j * nx + i;

        // Inicializar el impulso en una posición específica, por ejemplo, (1, 1)
        if (i == 30 && j == 30) {
            u[idx] = 10.0f;  // Valor del impulso en u
            v[idx] = 10.0f;  // Valor del impulso en v
        } else {
            u[idx] = 0.0f;
            v[idx] = 0.0f;
        }
    }
}





// Función para la actualización de la velocidad (método de diferencias finitas)
__global__ void update_velocity(float *u, float *v, float *u_new, float *v_new, float nu, float dt, float dx, float dy, int nx, int ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;


    if (i < nx && j < ny) {
        int idx = j * nx + i;

        // Términos convectivos
        float du_dx = (u[j * nx + (i + 1) % nx] - u[j * nx + (i - 1 + nx) % nx]) / (2 * dx);
        float du_dy = (u[((j + 1) % ny) * nx + i] - u[((j - 1 + ny) % ny) * nx + i]) / (2 * dy);
        float dv_dx = (v[j * nx + (i + 1) % nx] - v[j * nx + (i - 1 + nx) % nx]) / (2 * dx);
        float dv_dy = (v[((j + 1) % ny) * nx + i] - v[((j - 1 + ny) % ny) * nx + i]) / (2 * dy);

        u_new[idx] = u[idx] 
                    - dt * (u[idx] * du_dx + v[idx] * du_dy) 
                    + nu * dt / (dx * dx) * (u[j * nx + (i + 1) % nx] - 2 * u[idx] + u[j * nx + (i - 1 + nx) % nx]) 
                    + nu * dt / (dy * dy) * (u[((j + 1) % ny) * nx + i] - 2 * u[idx] + u[((j - 1 + ny) % ny) * nx + i]);


        v_new[idx] = v[idx] 
                    - dt * (u[idx] * dv_dx + v[idx] * dv_dy) 
                    + nu * dt / (dx * dx) * (v[j * nx + (i + 1) % nx] - 2 * v[idx] + v[j * nx + (i - 1 + nx) % nx]) 
                    + nu * dt / (dy * dy) * (v[((j + 1) % ny) * nx + i] - 2 * v[idx] + v[((j - 1 + ny) % ny) * nx + i]);

    }
}


int main() {
    // Tamaño de la simulación
    int nx = NX;
    int ny = NY;

    // Parámetros de simulación
    float dx = 2.0f / (nx - 1);
    float dy = 2.0f / (ny - 1);
    float nu = 0.01f;  // Viscosidad
    float sigma = 0.2f;
    float dt = sigma * dx * dy / nu;  // Paso de tiempo




    // parametros
    size_t iter_count = 10000;
    unsigned long long seed = time(nullptr); // O cualquier otro método para generar un seed

    // Reservar memoria para los campos de velocidad en la CPU y la GPU
    float *u        = nullptr;
    float *v        = nullptr;
    float *u_new    = nullptr;
    float *v_new    = nullptr;
    float *curl     = nullptr;
    checkCuda(cudaMalloc((void**)&u, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&v, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&u_new, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&v_new, nx * ny * sizeof(float)));
    checkCuda(cudaMalloc((void**)&curl, nx * ny * sizeof(float)));

    
    // Buffer para la visualización de la simulación

    MimirEngine engine;
    engine.init(1920, 1080);


    MemoryParams m1;
    m1.layout = DataLayout::Layout2D;
    m1.element_count = {(unsigned)(nx), (unsigned)(ny)};
    m1.component_type = ComponentType::Float;
    m1.channel_count = 1;// copilot dijo que era 2
    m1.resource_type = ResourceType::Buffer;
    auto points = engine.createBuffer((void**)&u, m1);

    ViewParams p1;
    p1.element_count = nx * ny;
    p1.extent = {(unsigned)(nx), (unsigned)(ny),1};
    p1.data_domain = DataDomain::Domain2D;
    p1.domain_type = DomainType::Structured;
    p1.view_type = ViewType::Voxels; // o puede ser
    p1.attributes[AttributeType::Color] = *points;
    p1.options.default_color = {256,256,256}; // no sirve segun Isaias
    p1.options.default_size = 1;

    engine.createView(p1);
 
    // -------------------------------------------------------
    /*
    
    MemoryParams curlParams;
    curlParams.layout = DataLayout::Layout2D;
    curlParams.element_count = {(unsigned)(nx), (unsigned)(ny)};
    curlParams.component_type = ComponentType::Float;
    curlParams.channel_count = 1; // Puede ser 1 o más dependiendo de cómo quieras representar el curl
    curlParams.resource_type = ResourceType::Buffer;
    auto curlBuffer = engine.createBuffer((void**)&curl, curlParams);

    ViewParams curlViewParams;
    curlViewParams.element_count = nx * ny;
    curlViewParams.extent = {(unsigned)(nx), (unsigned)(ny), 1};
    curlViewParams.data_domain = DataDomain::Domain2D;
    curlViewParams.domain_type = DomainType::Structured;
    curlViewParams.view_type = ViewType::Voxels; // O lo que mejor se adapte
    curlViewParams.attributes[AttributeType::Color] = *curlBuffer;
    curlViewParams.options.default_color = {256, 256, 256};
    curlViewParams.options.default_size = 1;

    engine.createView(curlViewParams);
       */
    // -------------------------------------------------------


    engine.displayAsync();

    checkCuda(cudaDeviceSynchronize());

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((nx + BLOCK_SIZE - 1) / BLOCK_SIZE, (ny + BLOCK_SIZE - 1) / BLOCK_SIZE);


    initialize_laminar_flow<<<numBlocks, threadsPerBlock>>>(u, v, nx, ny, 1.0f);
    checkCuda(cudaDeviceSynchronize());  // Esperar a que los hilos terminen


    for(size_t i = 0; i < iter_count; i++){
        

        // Condiciones de contorno periódicas
        apply_boundary_conditions<<<numBlocks, threadsPerBlock>>>(u, v, nx, ny);
        checkCuda(cudaDeviceSynchronize());


        // Calcular el curl
        compute_curl<<<numBlocks, threadsPerBlock>>>(u, v, curl, nx, ny, dx, dy);
        checkCuda(cudaDeviceSynchronize());


        // Actualizar los campos de velocidad
        update_velocity<<<numBlocks, threadsPerBlock>>>(u, v, u_new, v_new, nu, dt, dx, dy, nx, ny);
        //update_velocity_with_barrier<<<numBlocks, threadsPerBlock>>>(u, v, u_new, v_new, nu, dt, dx, dy, nx, ny);
        checkCuda(cudaDeviceSynchronize());





        // Intercambiar punteros para el siguiente paso de tiempo
        std::swap(u, u_new);
        std::swap(v, v_new);

        engine.updateViews();
        this_thread::sleep_for(chrono::milliseconds(500));
    }

    engine.showMetrics();
    engine.exit();


    checkCuda(cudaFree(u));
    checkCuda(cudaFree(v));
    checkCuda(cudaFree(u_new));
    checkCuda(cudaFree(v_new));
    checkCuda(cudaFree(curl));



    return EXIT_SUCCESS;
}
