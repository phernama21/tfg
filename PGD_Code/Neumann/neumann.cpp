#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <random>

//Settings
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// CONSTANTS
const int M = 101, max_p = 30, max_n = 20; // Mesh points
const double epsilon = 1e-6; // Error tolerance
const double q = 3;
const Vector f (M,0); // f function

// Global variables
int N = -1; // Current iteration of the solution
const int a_y = 0, b_y = 1, a_x = 0, b_x = 1;

// Create and initialize the matrix with zeros
Matrix createMatrix(int m, int n) {
    Matrix matrix(m, Vector(n, 0.0));
    return matrix;
}

// Print the given matrix
void printMatrix(const Matrix& matrix, const std::string& name) {
    std::cout << "Matrix: " << name << std::endl;
    for (const auto& row : matrix) {
        for (double value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

// Print the given vector
void printVector(const Vector& myVector, const std::string& name) {
    std::cout << "Vector: " << name << std::endl;
    for (const auto& element : myVector) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

// Compute the outer product of two vectors
Matrix outerProduct(const Vector& vector1, const Vector& vector2) {
    // Get the sizes of the vectors
    size_t size1 = vector1.size();
    size_t size2 = vector2.size();

    // Create an MxM matrix filled with zeros
    Matrix resultMatrix(size1, Vector(size2, 0));

    // Compute the outer product
    for (size_t i = 0; i < size1; ++i) {
        for (size_t j = 0; j < size2; ++j) {
            resultMatrix[i][j] = vector1[i] * vector2[j];
        }
    }

    return resultMatrix;
}

// Compute the 1-norm of the given matrix
double computeNorm(const Matrix& matrix) {
    // Get the dimensions of the matrix
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // Initialize the 1-norm to a negative value
    double norm = -1.0;

    // Iterate over columns and calculate the absolute column sum
    for (size_t j = 0; j < cols; ++j) {
        double columnSum = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            columnSum += std::abs(matrix[i][j]);
        }
        // Update the 1-norm if the current column sum is larger
        norm = std::max(norm, columnSum);
    }
    return norm;
}

// Check if the solution has converged based on a tolerance criterion
bool checkSolutionConvergence(const Matrix& X, const Matrix& Y) {
    if (N == -1) return false;
    Matrix m1 = outerProduct(X[N], Y[N]);
    Matrix m2 = outerProduct(X[0], Y[0]);

    return (computeNorm(m1) / computeNorm(m2)) < epsilon;
}

// Compute the integral of a function
double computeIntegral(const Vector& function, int a, int b) {
    double h = static_cast<double>(b - a) / (2.0 * (M - 1));
    double integral = 0.0;
    for (int i = 0; i < M; ++i) {
        if (i == 0 || i == M - 1) {
            integral += function[i];
        } else {
            integral += 2 * function[i];
        }
    }
    return integral * h;
}

// Compute the derivative of a function
Vector computeDerivative(const Vector& function, int a, int b) {
    Vector der(M, 0.0);
    double h = static_cast<double>(b - a) / (M - 1);
    for (int i = 0; i < M; ++i) {
        if (i == 0) {
            der[i] = (function[i] - 2 * function[i + 1] + function[i + 2]) / (h * h);
        } else if (i == M - 1) {
            der[i] = (function[i - 1] - 2 * function[i] + function[i + 1]) / (h * h);
        } else {
            der[i] = (function[i - 2] - 2 * function[i - 1] + function[i]) / (h * h);
        }
    }
    return der;
}

// Compute the product of two vectors element-wise
Vector vectorProduct(const Vector& v1, const Vector& v2) {
    Vector product(M, 0.0);
    for (int i = 0; i < M; ++i) {
        product[i] = v1[i] * v2[i];
    }
    return product;
}

// Compute the product of a scalar and a vector
Vector scalarProduct(double scalar, const Vector& v1) {
    Vector product(M, 0.0);
    for (int i = 0; i < M; ++i) {
        product[i] = v1[i] * scalar;
    }
    return product;
}

// Compute the sum of two vectors
Vector vectorSum(const Vector& v1, const Vector& v2) {
    Vector sum(M, 0.0);
    for (int i = 0; i < M; ++i) {
        sum[i] = v1[i] + v2[i];
    }
    return sum;
}

// Compute the difference between two matrices
Matrix matrixSub(const Matrix& m1, const Matrix& m2) {
    Matrix sub = createMatrix(M, M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            sub[i][j] = m1[i][j] - m2[i][j];
        }
    }
    return sub;
}

// Compute the sum of two matrices
Matrix matrixSum(const Matrix& m1, const Matrix& m2) {
    Matrix sum = createMatrix(M, M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            sum[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return sum;
}

// Compute the sum used as an EDO parameter
Vector computeSum(const Matrix& m1, const Matrix& m2, const Vector& previous,const Vector& previous_derivative , int a1, int b1, int a2, int b2) {
    double gamma, delta;
    Vector test_derivative(M, -(b2-a2)/static_cast<double>(M-1));
    Vector suma(M, 0.0);
    Vector integralVector(M, 0.0);
    for (int i = 0; i < N; ++i) {
        gamma = computeIntegral(vectorProduct(previous, m1[i]), a1, b1);
        delta = computeIntegral(vectorProduct(previous_derivative, computeDerivative(m1[i], a1, b1)), a1, b1);

        std::fill(integralVector.begin(), integralVector.end(),computeIntegral(computeDerivative(computeDerivative(m2[i],a2,b2),a2,b2),a2,b2 ));
        suma = vectorSum(suma, vectorSum(scalarProduct(gamma, computeDerivative(m2[i], a2, b2)), scalarProduct(-gamma, integralVector)));
        std::fill(integralVector.begin(), integralVector.end(),computeIntegral(scalarProduct(delta, m2[i]),a2,b2 ));
        suma = vectorSum(suma, integralVector);
    }
    return suma;
}

// Solve the system of equations using the given parameters
Vector solveSystem(double alpha, double beta, double xi, Vector& summation, int a, int c) {
    Eigen::MatrixXd A(M, M);
    Vector xiVector(M, xi);
    A.setZero();
    double h = (c - a) / static_cast<double>(M - 1);
    Eigen::Map<Eigen::VectorXd> b(summation.data(), M);
    for (int i = 0; i < M; ++i) {
        if (i == 0 || i == M - 1) {
            b(i) = 0.0;
        } else {
            b(i) -= computeIntegral(xiVector,a,c);
        }

        for (int j = 0; j < M; ++j) {
            if (i == j) {
                if (i == 0 || i == M - 1) {
                    A(i, j) = 1.0;
                } else {
                    A(i, j) = 2 * alpha / h + beta * h;
                }
            } else if ((i == j - 1 || i == j + 1) && i != 0 && i != M - 1) {
                if (j != 0 && j != M - 1) {
                    A(i, j) = -1.0f * alpha / h;
                }
            }
        }
    }

    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    Vector solution(x.data(), x.data() + x.size());

    return solution;
}

// Compute the EDO (Ordinary Differential Equation) for each alternating direction step
Vector computeEDO(Vector& previous, const Matrix& m1, const Matrix& m2, int a1, int b1, int a2, int b2) {
    Vector squared_previous;
    Vector previous_derivative = computeDerivative(previous,a1,b1);
    Vector squared_previous_derivative;
    std::transform(previous.begin(), previous.end(), std::back_inserter(squared_previous),
                    [](double x) { return x * x; });
    std::transform(previous_derivative.begin(), previous_derivative.end(), std::back_inserter(squared_previous_derivative),
                    [](double x) { return x * x; });
    double alpha = computeIntegral(squared_previous, a1, b1);
    double beta = computeIntegral(squared_previous_derivative, a1, b1);
    double xi = computeIntegral(vectorProduct(previous, f), a1, b1);
    double mu = previous[M-1] * q;
    Vector summation = computeSum(m1, m2, previous,previous_derivative, a1, b1, a2, b2);

    return solveSystem(alpha, beta, xi, summation, a2, b2);
}

// Check if the tolerance criterion is met
bool checkTolerance(const Vector& current_X, const Vector& current_Y, const Vector& previous_X, const Vector& previous_Y) {
    double numerator = computeNorm(matrixSub(outerProduct(current_X, current_Y), outerProduct(previous_X, previous_Y)));
    double denominator = computeNorm(outerProduct(previous_X, previous_Y));
    return (numerator / denominator) < epsilon;
}

// Generate a random vector to start the alternatingdirection process
Vector generateRandomVector(int m, double minVal, double maxVal) {
        std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(minVal, maxVal);

    Vector randomVector;
    for (int i = 0; i < m; ++i) {
        double randomValue = dis(gen);
        randomVector.push_back(randomValue);
    }
    return randomVector;
}

// Alternating Direction Method to solve the system
void alternatingDirection(Matrix& X, Matrix& Y, int iteration) {
    Vector previous_Y(M, 1.0);
    previous_Y[0] = 0.0;
    previous_Y[M - 1] = 0.0;
    Vector current_Y(M, 0.0);
    Vector previous_X(M, 0.0);
    Vector current_X(M, 0.0);
    current_X = computeEDO(previous_Y, Y, X, a_y, b_y, a_x, b_x);
    current_Y = computeEDO(current_X, X, Y, a_x, b_x, a_y, b_y);
    previous_X = current_X;
    int p = 1;
    
    while (!checkTolerance(current_X, current_Y, previous_X, previous_Y) && p < max_p) {
        previous_Y = current_Y;
        previous_X = current_X;
        current_X = computeEDO(previous_Y, Y, X, a_y, b_y, a_x, b_x);
        current_Y = computeEDO(current_X, X, Y, a_x, b_x, a_y, b_y);
        p += 1;
    }
    X[N] = current_X;
    for (int i = 0; i < M; ++i) {
        X[N] = current_X;
        Y[N] = current_Y;
    }
}
// Export the matrix to a CSV file
void exportMatrixToCSV(const Matrix& matrix, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    for (size_t j = 0; j < matrix[0].size(); ++j) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            if (i == matrix.size() - 1) {
                outFile << matrix[i][j] << "\n";
            } else {
                outFile << matrix[i][j] << ",";
            }
        }
    }

    outFile.close();
}

// Compute the gradient fields of the solution u(x,y)
std::tuple<Matrix, Matrix> computeGradient(const Matrix& function) {
    int rows = function.size();
    int cols = function[0].size();
    Matrix gradient_x = createMatrix(rows, cols);
    Matrix gradient_y = createMatrix(rows, cols);

    // Compute the gradient using central differences
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            // Compute partial derivatives with respect to x and y
            double df_dx = (function[i][j + 1] - function[i][j - 1]) / 2.0; // Central difference for x
            double df_dy = (function[i + 1][j] - function[i - 1][j]) / 2.0; // Central difference for y

            // Assign the derivatives to gradient matrices
            gradient_x[i][j] = df_dx;
            gradient_y[i][j] = df_dy;
        }
    }
    return { gradient_x, gradient_y };
}

// Main function
int main() {
    Matrix solution = createMatrix(M, M);
    Matrix X = createMatrix(max_n, M);
    Matrix Y = createMatrix(max_n, M);

    // Loop until convergence
    while (N < (max_n - 1) && !checkSolutionConvergence(X, Y)) {
        N += 1;
        alternatingDirection(X, Y, N);
    }

    // COMPUTE SOLUTION
    for (int i = 0; i < N; ++i) {
        solution = matrixSum(solution, outerProduct(X[i], Y[i]));
    }

    return 0;
}