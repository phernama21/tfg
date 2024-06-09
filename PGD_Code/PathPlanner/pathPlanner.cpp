#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <random>

//SETTINGS
using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;
struct Point {
    double x;
    double y;
};

//CONSTANTS
const int M = 101, max_p = 30, max_n = 20,max_f=50; //Mesh points
const double epsilon = 1e-6; //Error tolerance
const double variance = 0.1; //Variance r of the Gaussian model
Point source, target; //Source and target points
int N = -1, F=-1; //current iteration of the solution
const int a_y = 0, b_y = 5, a_x = 0, b_x = 7; //Domain

//Create and initialise the matrix with zeros
Matrix createMatrix(int m, int n) {
    Matrix matrix(m, Vector(n, 0.0f));
    return matrix;
}

//Print the given matrix
void printMatrix(const Matrix& matrix,const std::string &name ) {
    std::cout << "Matrix: " << name << std::endl;
    for (const auto& row : matrix) {
        for (double value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

//Print the given vector
void printVector(Vector &myVector,const std::string &name){
    std::cout << "Vector: "<< name << std::endl;
    for (const auto &element : myVector) {
        std::cout << element << " ";
    }

    std::cout << std::endl;
}

//Compute the outer product of two vectors
Matrix outerProduct(const Vector& vector1, const Vector& vector2) {
    // Get the sizes of the vectors
    size_t size1 = vector1.size();
    size_t size2 = vector2.size();

    // Create an MxM matrix filled with zeros
    Matrix resultMatrix(size1, Vector(size2, 0));

    // Compute the outer product
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            resultMatrix[i][j] = vector1[i] * vector2[j];
        }
    }
    return resultMatrix;
}

//Compute the 1-norm of the given matrix
double computeNorm(const Matrix& matrix) {
    // Get the dimensions of the matrix
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    // Initialize the 1-norm to a negative value
    double norm = -1.0f;

    // Iterate over columns and calculate the absolute column sum
    for (int j = 0; j < M; ++j) {
        double columnSum = 0.0f;
        for (int i = 0; i < M; ++i) {
            columnSum += std::abs(matrix[i][j]);
        }
        // Update the 1-norm if the current column sum is larger
        norm = std::max(norm, columnSum);
    }
    return norm;
}

//Check if the solution has converged based on the stopping criterion
bool checkSolutionConvergence(const Matrix& X, const Matrix& Y) {
    if(N==-1)return false;
    Matrix m1 = outerProduct(X[N],Y[N]);
    Matrix m2 = outerProduct(X[0],Y[0]);

    return (computeNorm(m1) / computeNorm(m2)) < epsilon;
}

//Compute the discrete integral of a function
double computeIntegral(const Vector &function, int a, int b) {
    double h =  static_cast<float>(b - a) / (double)(2.0f *(M-1));
    double integral = 0.0f;
    for (int i = 0; i < M; ++i){
        if (i == 0 or i == M-1){
            integral += function[i];
        }else{
            integral += 2 * function[i];
        }

    }
    return integral * h;
}

//Compute the derivative of a function
Vector computeDerivative(const Vector &function, int a, int b){
    Vector der(M, 0.0f);
    double h = static_cast<float>(b - a)/(M-1);
    for (int i = 0; i < M; ++i){
        if ( i == 0 ){
            der[i] = (function[i] - 2*function[i + 1] + function[i + 2] ) / (h*h);
        }else if(i == M-1){
            der[i] = (function[i-1] - 2*function[i] + function[i + 1] ) / (h*h);
        }else{
            der[i] = (function[i-2] - 2*function[i-1] + function[i] ) / (h*h);
        }
    }
    return der;
}

//Compute the product of two vectors element-wise
Vector vectorProduct(const Vector &v1,const Vector &v2 ) {
    Vector product(M, 0.0f);
    for (int i = 0; i < M; ++i){
        product[i] = v1[i] * v2[i];

    }
    return product;
}

//Compute the scalar product of two vectors
double scalarVectorProduct(const Vector& v1, const Vector& v2) {
    double result;
    for (int i = 0; i < M; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

//Compute the product of an scalar and a vector
Vector scalarProduct(double scalar,const Vector &v1) {
    Vector product(M, 0.0f);
    for (int i = 0; i < M; ++i){
        product[i] = v1[i] * scalar;

    }
    return product;
}

//Compute the sum of two vectors
Vector vectorSum(const Vector &v1,const Vector &v2) {
    Vector sum(M, 0.0f);
    for (int i = 0; i < M; ++i){
        sum[i] = v1[i] + v2[i];

    }
    return sum;
}

//Compute the difference between two vectors
Vector vectorSub(const Vector& v1, const Vector& v2) {
    Vector sum(M, 0.0);
    for (int i = 0; i < M; ++i) {
        sum[i] = v1[i] - v2[i];
    }
    return sum;
}

//Compute the difference between two matrices
Matrix matrixSub(const Matrix& m1,const Matrix& m2) {
    Matrix sub = createMatrix(M,M);
    for (int i = 0; i < M; ++i){
        for(int j = 0; j < M; ++j){
            sub[i][j] = m1[i][j] - m2[i][j];
        }    
    }
    return sub;
}

//Compute the sum of two matrices
Matrix matrixSum(const Matrix& m1,const Matrix& m2) {
    Matrix sum = createMatrix(M,M);
    for (int i = 0; i < M; ++i){
        for(int j = 0; j < M; ++j){
            sum[i][j] = m1[i][j] + m2[i][j];
        }
    }
    return sum;
}

//Compute the sum element of the EDO
Vector computeSum(const Matrix& m1, const Matrix& m2, const Vector &previous, int a1, int b1,int a2, int b2){
    double gamma, delta;
    Vector suma (M, 0.0f);
    for (int i = 0; i < N; ++i){
        gamma = computeIntegral(vectorProduct(previous, m1[i]), a1, b1);
        delta = computeIntegral(vectorProduct(previous, computeDerivative(m1[i],a1,b1)), a1, b1);
        suma = vectorSum(suma ,vectorSum(scalarProduct(gamma, computeDerivative(m2[i],a2,b2)) , scalarProduct(delta, m2[i])));
    }
    return suma;
}

//Solve the system of equations using the given parameters
Vector solveSystem(double alpha, double beta, const Vector& xi, const Vector &summation,int a, int c ){
    Eigen::MatrixXd A(M,M);
    A.setZero();
    double h = (c-a)/(double)(M-1);
    Eigen::VectorXd b(M);
    b.setConstant(0);
    for (int i = 0; i < M ; ++i){
        if(i==0 or i == M-1){
            b(i) = 0.0f;
        }else{
            b(i) += xi[i] - summation[i];
        }
        for (int j = 0; j < M ; ++j){
            if(i==j){
                if(i == 0 or i == M-1 ){
                    A(i,j) = 1.0f;
                }else{
                    A(i,j) = -2*alpha/(h*h) + beta;
                }
            }
            else if((i == j-1 or i == j+1) and i != 0 and i!=M-1 ){
                if (j!=0 and j != M-1){
                    A(i,j) = alpha/(h*h);
                }  
            }
        }
    }
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

    Vector solution(x.data(), x.data() + x.size());

    return solution;

}

//Solve the one-dimensional EDO for each alternating direction step
Vector computeEDO(Vector &previous, const Matrix& m1,const Matrix& m2,
                                const Matrix& function1,const Matrix& function2, int a1, int b1, int a2, int b2){
    Vector squared_previous;
    std::transform(previous.begin(), previous.end(), std::back_inserter(squared_previous),
                    [](double x) { return x * x; });
    double alpha = computeIntegral(squared_previous, a1,b1);
    double beta = computeIntegral(vectorProduct(previous, computeDerivative(previous,a1,b1)),a1,b1);
    Vector vectorXi(M, 0.0);
    for(int i = 0; i < F; ++i){
        vectorXi = vectorSum(vectorXi, scalarProduct(computeIntegral(vectorProduct(previous, function2[i]),a1,b1),function1[i]));
    }
    Vector summation = computeSum(m1, m2, previous, a1 , b1, a2, b2);

    return solveSystem(alpha, beta, vectorXi, summation, a2,b2);


}

//Check the error of the alternating direction step
bool checkTolerance(const Vector &current_X, const Vector &current_Y, const Vector &previous_X, const Vector &previous_Y){
    double numerator = computeNorm(matrixSub(outerProduct(current_X, current_Y) , outerProduct(previous_X,previous_Y)));
    double denominator = computeNorm(outerProduct(previous_X, previous_Y));
    return (numerator/denominator) < epsilon;
}

//Generate a random vector to start the alternating direction process
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

//Execute the alternating direction process
void alternatingDirection(Matrix& X, Matrix& Y,
                            Matrix& f_x,Matrix& f_y,int iteration) {
    Vector previous_Y = generateRandomVector(M,-3,3);
    previous_Y[0] = 0.0f;
    previous_Y[M-1] = 0.0f;
    Vector current_Y(M, 0.0f);
    Vector previous_X(M, 0.0f);
    Vector current_X(M, 0.0f);
    current_X = computeEDO(previous_Y, Y, X,f_x,f_y, a_y, b_y, a_x, b_x);
    current_Y = computeEDO(current_X, X, Y,f_y,f_x, a_x, b_x, a_y, b_y);
    previous_X = current_X;
    int p = 1;
    
    while(!checkTolerance(current_X, current_Y, previous_X, previous_Y) and p < max_p ){
        previous_Y = current_Y;
        previous_X = current_X;
        current_X = computeEDO(previous_Y, Y, X,f_x,f_y, a_y, b_y, a_x, b_x);
        current_Y = computeEDO(current_X, X, Y,f_y,f_x, a_x, b_x, a_y, b_y);
        p+=1;

    }
    X[N] = current_X;
    Y[N] = current_Y;
}

//Compute product between a matrix and a vector
Vector productMatrixVector(Matrix& matrix,Vector& vector, bool transposed){
    Vector result(M, 0.0);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            if (transposed){
                result[i] += matrix[j][i] * vector[j];
            }else{
                result[i] += matrix[i][j] * vector[j];
            }
        }
    }
    return result;
}

//Compute the adittion of the previous computed steps of the alternating direction strategy
Vector sumPrevious( Matrix& previousMatrixA,Matrix& previousMatrixB,Vector& previousVector ){
    Vector result(M, 0.0);
    for(int i = 0; i < F; ++i){
        result = vectorSum(result, scalarProduct(scalarVectorProduct(previousVector,previousMatrixB[i]), previousMatrixA[i]));
    }
    return result;
}

//ompute the alternating direction matrices for the source term f
void alternatingDirectionSourceTerm(Matrix& function,Matrix& X, Matrix& Y, int iteration){
    Vector previous_Y(M, 1.0);
    previous_Y[0] = 0.0;
    previous_Y[M - 1] = 0.0;
    Vector current_Y(M, 0.0);
    Vector previous_X(M, 0.0);
    Vector current_X(M, 0.0);
    current_X = scalarProduct(1.0 / scalarVectorProduct(previous_Y,previous_Y), vectorSub( productMatrixVector(function, previous_Y,false),sumPrevious(X,Y,previous_Y)));
    current_Y = scalarProduct(1.0 / scalarVectorProduct(current_X,current_X),
    vectorSub(productMatrixVector(function,current_X,true),sumPrevious(Y,X,current_X)));
    previous_X = current_X;
    int p = 1;
    while (!checkTolerance(current_X, current_Y, previous_X, previous_Y) && p < max_p) {
        previous_Y = current_Y;
        previous_X = current_X;
        current_X = scalarProduct(1.0 / scalarVectorProduct(previous_Y,previous_Y), vectorSub( productMatrixVector(function, previous_Y,false),sumPrevious(X,Y,previous_Y) ));
        current_Y = scalarProduct(1.0 / scalarVectorProduct(current_X,current_X),vectorSub(productMatrixVector(
        function, current_X,true),
        sumPrevious(Y,X,current_X)));
        p += 1;
    }
    X[F] = current_X;
    Y[F] = current_Y;

}

//Gaussian Model for a particular point, a mean and a variance
double gaussian2D(double x, double y, double mean_x, double mean_y) {
    double exponent = -((x - mean_x) * (x - mean_x) / (2 * variance * variance) +
                        (y - mean_y) * (y - mean_y) / (2 * variance * variance));

    return exp(exponent) / (2 * M_PI * variance * variance);
}

//Compute an specific ource term modelled by the Gaussian distribution
void computeUniqueF(Matrix& matrix){
    double h_x = (b_x - a_x) / (double)(M-1);
    double h_y = (b_y - a_y) / (double)(M-1);
    std::cout << "Enter X-component of the Source: \n";
    std::cin >> source.x;

    std::cout << "Enter Y component of the Source: \n";
    std::cin >> source.y;
    std::cout << "\n\nYou entered: (" << source.x << "," <<  source.y<< ")" << std::endl;
    // Ask the user to input values
    std::cout << "Enter X component of the Target: \n";
    std::cin >> target.x;

    std::cout << "Enter Y component of the Target: \n";
    std::cin >> target.y;

    std::cout << "\n\nYou entered: (" << target.x << "," << target.y<< ")" << std::endl;
    for(int i = 0; i < M; ++i){
        for (int j = 0; j < M; ++j) {
            matrix[i][j] = gaussian2D(a_x + i*h_x,a_y + j*h_y,source.x,source.y) -
                    gaussian2D(a_x + i*h_x,a_y + j*h_y,target.x,target.y);
        }
    }
}

//Compute the source term for all possible combinations of source terms
void computeF(Matrix& matrix) {
    int x,y,s1,s2;
    double h_x = (b_x - a_x) / (double)(M-1);
    double h_y = (b_y - a_y) / (double)(M-1);
    for (int i = 0; i < M*M; ++i) {
        for (int j = 0; j < M*M; ++j) {
            x = j / M;
            y = j % M;

            s1 = i / M;
            s2 = i % M;

            matrix[i][j] = gaussian2D(a_x + x*h_x,a_y + y*h_y, a_x + s1*h_x,a_y + s2*h_y );
        }
    }
}

//Compute the two matrices F and G of all possible goals and target combinations
void  computeFlow(Matrix& function) {
    Matrix F = createMatrix(M*M, M*M);
    Matrix G = createMatrix(M*M, M*M);

    // Ask the user to input values
    std::cout << "Enter X component of the Source: \n";
    std::cin >> source.x;

    std::cout << "Enter Y component of the Source: \n";
    std::cin >> source.y;
    std::cout << "\n\nYou entered: " << source.x << " and " << source.y << std::endl;
    // Ask the user to input values
    std::cout << "Enter X component of the Target: \n";
    std::cin >> target.x;

    std::cout << "Enter Y component of the Target: \n";
    std::cin >> target.y;

    std::cout << "\n\nYou entered: " << target.x << " and " << target.y << std::endl;
    computeF(F);
    computeF(G);

    int s_position = source.x * M + source.y;
    int t_position = target.x * M + target.y;
    int x,y;

    // Loop over the matrix
    double h_x = (b_x - a_x) / (double)(M-1);
    double h_y = (b_y - a_y) / (double)(M-1);
    for (int i = 0; i < M*M; ++i){
        x = i / M;
        y = i % M;
        function[x][y] = F[s_position][i] - G[t_position][i];
    }
}

//Export a given matrix to .csv
void exportMatrixToCSV(const Matrix& matrix, const std::string& filename) {
    std::ofstream outFile(filename);
    double h_x = (b_x - a_x) / (double)(M-1);
    double h_y = (b_y - a_y) / (double)(M-1);
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    // Transpose the matrix while writing to the file
    for (size_t j = 0; j < matrix[0].size(); ++j) {
        for (size_t i = 0; i < matrix.size(); ++i) {
            outFile << (a_x + i*h_x) << " " << (a_y + j*h_y) << " " << matrix[i][j]/10 <<"\n";
        }
        outFile <<"\n";
    }

    outFile.close();
}

//Compute the gradient fields generated by the solution u(x,y)
std::pair<Matrix, Matrix> computeGradient(const Matrix& function){
    double h_x = (b_x - a_x) / (double)(M-1);
    double h_y = (b_y - a_y) / (double)(M-1);
    Matrix gradient_x = createMatrix(M,M);
    Matrix gradient_y = createMatrix(M,M);

    // Compute the gradient using central differences
    for (int i = 1; i < M-1; ++i) {
        for (int j = 1; j < M-1; ++j) {
            // Compute partial derivatives with respect to x and y
            double df_dx = (function[i - 1][j] - function[i + 1][j]) / (2.0*h_x); // Central difference for x
            double df_dy = (function[i][j - 1] -function[i][j + 1] ) / (2.0*h_y); // Central difference for y

            // Assign the derivatives to gradient matrices
            gradient_x[i][j] = df_dx;
            gradient_y[i][j] = df_dy;
            double norm = std::sqrt(df_dx * df_dx + df_dy * df_dy);
        }
    }
    return std::make_pair(gradient_x, gradient_y);
}

//Unify the separated form of the source term for testing purposes
void unifyFunction(Matrix& function,Matrix& f_x,Matrix& f_y){
    for (int i = 0; i < F ; ++i){
        function = matrixSum(function, outerProduct(f_x[i], f_y[i]));
    }

}

//Get the separated form of the non-constant source term
void separateF(Matrix& function,Matrix& f_x, Matrix& f_y ){
    while (F < (max_f - 1) && !checkSolutionConvergence(f_x, f_y)) {
        F += 1;
        alternatingDirectionSourceTerm(function,f_x, f_y, F);
    }
}

//Calculate the distance of two 2-D points
double calculateDistance(const Point& P1, const Point& P2) {
    return std::sqrt((P2.x - P1.x) * (P2.x - P1.x) + (P2.y - P1.y) * (P2.y - P1.y));
}

//Get the first point of the path
Point findPointAtDistance(const Point& P1, const Point& P2, double h) {
    double distance = calculateDistance(P1, P2);
    double unitVectorX = (P2.x - P1.x) / distance;
    double unitVectorY = (P2.y - P1.y) / distance;
    Point P3;
    P3.x = P1.x + h * unitVectorX;
    P3.y = P1.y + h * unitVectorY;
    return P3;
}

//Interpolate the path
std::vector<Point> interpolatePath(const Matrix& gradient_x, const Matrix& gradient_y, Point start, double step_size, int num_steps) {
    std::vector<Point> path;
    path.push_back({source.x,source.y});
    path.push_back(start);

    double x = start.x;
    double y = start.y;
    double h_x = (b_x - a_x) / (double)(M-1);
    double h_y = (b_y - a_y) / (double)(M-1);

    for (int step = 0; step < num_steps; ++step) {
        int i = static_cast<int>(x / h_x);
        int j = static_cast<int>(y / h_y);

        if (i < 0 || i >= M || j < 0 || j >= M) {
            break; // Out of bounds
        }

        double dx = gradient_x[i][j];
        double dy = gradient_y[i][j];

        x += step_size * dx;
        y += step_size * dy;

        path.push_back({x, y});
        if (std::sqrt((x - target.x) * (x - target.x) + (y - target.y) * (y - target.y)) < step_size) {
            path.push_back({target.x, target.y});
            break;
        }
    }

    return path;
}

//Export the path to an .yml file
void exportPath(std::vector<Point> path) {
    std::ofstream outFile("../waypoints.yaml");
    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file " << "vectorfield" << std::endl;
        //return;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        outFile <<"goal"<< i <<": {\"x\":"<< path[i].x << ", \"y\": " << path[i].y << ", \"w\": 90}"<< "\n";

    }
    outFile.close();

}

//Plot vector field for graphic examples
void plotVectorField(const Matrix& matrix){
    std::pair<Matrix, Matrix> fields = computeGradient(matrix);
    // Define the step size and the first point
    double stepSize = 0.5;
    Point firstPoint = findPointAtDistance(source,target,stepSize);
        
    int maxIterations = 300;
    std::vector<Point> path = interpolatePath(fields.first, fields.second, firstPoint, stepSize, maxIterations);
    //Export our computed path to .yaml file to work with ROS
    exportPath(path);
    
}

int main() {

    Matrix function = createMatrix(M, M);
    Matrix new_function = createMatrix(M, M);
    computeUniqueF(function);
    Matrix f_x = createMatrix(M,M);
    Matrix f_y = createMatrix(M,M);
    separateF(function,f_x,f_y);
    //Test the new function is the one we desire
    unifyFunction(new_function, f_x, f_y);
    Matrix solution = createMatrix(M, M);
    Matrix X = createMatrix(max_n, M);
    Matrix Y = createMatrix(max_n, M);

    // Loop until convergence
    while (N < (max_n - 1) && !checkSolutionConvergence(X, Y)) {
        N += 1;
        alternatingDirection(X, Y,f_x,f_y, N);
    }

    // COMPUTE SOLUTION
    for (int i = 0; i < N; ++i) {
        solution = matrixSub(solution, outerProduct(X[i], Y[i]));
    }
    //Plot vector field and export path
    exportMatrixToCSV(solution,"../path_planner_solution.csv")
    plotVectorField(solution);

    return 0;
}