#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <thread>

using namespace std;

const int nx = 10;    // Tamaño en x
const int ny = 10;    // Tamaño en y
const int nit = 50;    // Número de iteraciones
const double dt = 0.1; // Tiempo de simulación
const double dx = 1.0; // Espaciado en x
const double dy = 1.0; // Espaciado en y
const double nu = 0.1; // Viscosidad cinemática

// Función para imprimir la matriz
void printMatrix(const double* array, int nx, int ny) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            cout << array[i * nx + j] << " "; // Acceso mediante cálculo de índice
        }
        cout << endl;
    }
    cout << endl; // Espacio entre iteraciones
}

int main() {
    // Crear arrays unidimensionales para velocidad y presión
    double* u = new double[nx * ny](); // Velocidad en x
    double* v = new double[nx * ny](); // Velocidad en y
    double* p = new double[nx * ny](); // Presión

    // Inicializar condiciones de frontera
    for (int i = 0; i < ny; ++i) {
        u[i * nx] = 1.0;   // Velocidad constante en la frontera izquierda
        u[i * nx + (nx - 1)] = 0.0; // Velocidad en la frontera derecha
    }

    // Imprimir la matriz inicial
    cout << "Matriz inicial:" << endl;
    printMatrix(u, nx, ny);

    for (int it = 0; it < nit; ++it) {
        double* u_new = new double[nx * ny](); // Nuevo array para la velocidad

        // Actualización de las velocidades usando las ecuaciones de Navier-Stokes
        for (int i = 1; i < ny - 1; ++i) {
            for (int j = 1; j < nx - 1; ++j) {
                int idx = i * nx + j; // Índice en el array unidimensional
                u_new[idx] = u[idx] - dt * (
                    u[idx] * (u[idx + 1] - u[idx - 1]) / (2.0 * dx) +
                    v[(i - 1) * nx + j] - v[(i + 1) * nx + j]) / (2.0 * dy) +
                    nu * dt * (
                    (u[(i + 1) * nx + j] - 2 * u[idx] + u[(i - 1) * nx + j]) / (dy * dy) +
                    (u[i * nx + (j + 1)] - 2 * u[idx] + u[i * nx + (j - 1)]) / (dx * dx));
            }
        }

        // Aplicar condiciones de frontera
        for (int i = 0; i < ny; ++i) {
            u_new[i * nx] = 1.0;   // Mantener la frontera izquierda en 1.0
            u_new[i * nx + (nx - 1)] = 0.0; // Mantener la frontera derecha en 0.0
        }

        // Copiar los valores de las nuevas velocidades a las originales
        delete[] u; // Liberar memoria del array anterior
        u = u_new; // Asignar el nuevo array a u

        // Imprimir la matriz de velocidades
        cout << "Iteración " << it + 1 << ":" << endl;
        printMatrix(u, nx, ny);

        // Esperar un tiempo antes de imprimir la siguiente matriz
        this_thread::sleep_for(chrono::milliseconds(500));
    }

    // Liberar la memoria utilizada
    delete[] u;
    delete[] v;
    delete[] p;

    cout << "Simulación completada." << endl;
    return 0;
}
