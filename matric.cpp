#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include<iostream>
#include<fstream>
#include<vector>
#include<random>
#include<ctime>
#include<cmath>
#include<algorithm>
#include <cstdlib> // for rand() and RAND_MAX
#include "matric.hpp"


double random_uniform(double lower_bound, double upper_bound) {
    // Generate random numbers between lower_bound and upper_bound
    return lower_bound + (upper_bound - lower_bound) * (rand() / (double)RAND_MAX);
}



Matrix::Matrix():n(0), m(0), a(nullptr){} 
Matrix::Matrix(int rows, int cols){
    n=rows;
    m=cols;
    a = new double* [n];

    for(int i=0; i<n; i++){
        a[i] = new double [m];
        for(int j=0; j<m; j++){
            a[i][j] == 0.0;
        }
    }
}
Matrix::Matrix(const Matrix &x){
    n = x.n;
    m = x.m;

    a = new double* [n];
    for(int i=0; i<n; i++){
        a[i] = new double [m];

        for(int j = 0; j<m; j++){
            a[i][j] = x.a[i][j];
        }
    }
}
void Matrix::operator=(const Matrix &x){
    n = x.n;
    m = x.m;

    a = new double* [n];
    for(int i=0; i<n; i++){
        a[i] = new double [m];

        for(int j = 0; j<m; j++){
            a[i][j] = x.a[i][j];
        }
    }
}

void Matrix::randInit(){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            a[i][j] = randDouble() - 0.5;
        }
    }
}

void Matrix::zero(){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            a[i][j] = 0.0;
        }
    }
}

void Matrix::add(Matrix x){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            a[i][j] += x.a[i][j];
        }
    }
}

void Matrix::print(){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            std::cout << a[i][j] << ", ";
        }
        std::cout << "\n";
    }
    std::cout << ".............\n";
}

Matrix::~Matrix(){
    for(int i=0; i<n; i++){
        delete[] a[i];
    }
    delete[] a;
}


Matrix initialize_weights_xavier(int fan_in, int fan_out) {
    double limit = sqrt(1.0 / fan_in); // Xavier scaling factor
    Matrix W(fan_in, fan_out);
    for (int i = 0; i < fan_in; i++) {
        for (int j = 0; j < fan_out; j++) {
            W.a[i][j] = random_uniform(-limit, limit); // Xavier initialization
        }
    }
    return W;

}

Matrix add(Matrix M1, Matrix M2){
    Matrix temp(M1.n, M1.m);
    for(int i=0; i<M1.n; i++){
        for(int j=0; j<M1.m; j++){
            temp.a[i][j] = M1.a[i][j] + M2.a[i][j];
        }
    }
    return temp;
}

Matrix subtract(Matrix M1, Matrix M2){
    Matrix temp(M1.n, M1.m);
    for(int i=0; i<M1.n; i++){
        for(int j=0; j<M1.m; j++){
            temp.a[i][j] = M1.a[i][j] - M2.a[i][j];
        }
    }
    return temp;
}

Matrix term_by_term(Matrix M1, Matrix M2){
    Matrix temp(M1.n, M1.m);
    for(int i=0; i<M1.n; i++){
        for(int j=0; j<M1.m; j++){
            temp.a[i][j] = M1.a[i][j] * M2.a[i][j];
        }
    }
    return temp;
}

Matrix transpose(Matrix M){
    Matrix temp(M.m, M.n);
    for(int i=0; i<M.n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[j][i] = M.a[i][j];
        }
    }
    return temp;
}

Matrix multiply(Matrix M1, Matrix M2){
    Matrix temp(M1.n, M2.m);
    for(int i=0; i<M1.n; i++){
        for(int j=0; j<M2.m; j++){
            for(int z=0; z<M1.m; z++){
                temp.a[i][j] += M1.a[i][z] * M2.a[z][j]; 
            }
        }
        
    }
    return temp;
}

Matrix relu(Matrix M){
    Matrix temp(M.n, M.m);
    for(int i=0; i<M.n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[i][j] =relu(M.a[i][j]);
        }
    }
    return temp;
}

Matrix reluDeriv(Matrix M){
    Matrix temp(M.n, M.m);
    for(int i=0; i<M.n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[i][j] = reluDeriv(M.a[i][j]);
        }
    }
    return temp;
}


Matrix softmax(Matrix M) {
    Matrix temp(M.n, M.m);
    
    // Iterate over each row
    for(int i = 0; i < M.n; i++) {
        double sum = 0.0;
        
        // Step 1: Find the maximum value in the row for numerical stability (optional)
        double maxVal = M.a[i][0];
        for (int j = 1; j < M.m; j++) {
            if (M.a[i][j] > maxVal) {
                maxVal = M.a[i][j];
            }
        }

        // Step 2: Calculate the sum of exponentials for the row
        for(int j = 0; j < M.m; j++) {
            sum += exp(M.a[i][j] - maxVal);  // Subtracting maxVal for stability
        }

        // Step 3: Compute softmax values for the row
        for(int j = 0; j < M.m; j++) {
            temp.a[i][j] = exp(M.a[i][j] - maxVal) / sum;
        }
    }
    
    return temp;
}

double randDouble(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution distrFloat(0.0, 1.0);
    double randFloat = distrFloat(gen);
    return randFloat;
}


Matrix sumRow(Matrix M){
    Matrix temp(1, M.m);
    for(int j=0; j<M.m; j++){
        double sum = 0.0;
        for(int i=0; i<M.n; i++){
            sum += M.a[i][j];
        }
        temp.a[0][j] = sum;
    }
    return temp;
}

Matrix extendRow(Matrix M, int n){
    Matrix temp(n, M.m);
    for(int i=0; i<n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[i][j] = M.a[0][j];
        }
    }
    return temp;
}


Matrix scalarDivide(Matrix M, double d){
    Matrix temp(M.n, M.m);
    for(int i=0; i<M.n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[i][j] = M.a[i][j] / (double)d;
        }
    }
    return temp;
}

double relu(double x){
    if (x >= 0.0) return x;
    else return 0;
    //return 1.0/(1.0 + exp(-x));
}

double reluDeriv(double x){
    if (x >= 0.0) return 1.0;
    else return 0.0;
    // return x * (1.0 - x);
}

Matrix inputMatrix(std::vector <Matrix> inputs){
    Matrix input((int)(inputs.size()), 784);
    for(int i=0; i<(int)(inputs.size()); i++){
        for(int j=0; j<784; j++){
            input.a[i][j] = inputs[i].a[0][j]; 
        }
    }
    return input;
}

Matrix outputMatrix(std::vector <Matrix> outputs){
    Matrix output((int)(outputs.size()), 10);
    for(int i=0; i<(int)(outputs.size()); i++){
        for(int j=0; j<10; j++){
            output.a[i][j] = outputs[i].a[0][j]; 
        }
    }
    return output;
}

void saveToCSV(Matrix temp, const std::string& filename) {
    std::ofstream file(filename); // Open file in write mode
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    for (int row=0; row < temp.n; row++){
        for (int col = 0; col < temp.m; ++col) {
            file << temp.a[row][col]; // Write value
            if (col < temp.m-1) {
                file << ','; // Add a comma except for the last element
            }
        }
        file << '\n'; // End of the row
    }

    file.close(); // Close the file
    std::cout << "Data saved to " << filename << std::endl;
}

Matrix ReadFile(std::string filename){
    std::ifstream file(filename);

    // Check if the file opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open the file " << filename << std::endl;
    }

    // Temporary vector to determine matrix dimensions
    std::vector<std::vector<double>> data;
    std::string line;

    int rows = 0, cols = 0;
    // Read file line by line to count rows and columns
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string value;
        
        // Count columns in the first row
        int colCount = 0;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
            colCount++;
        }
        cols = colCount;  // Update columns count
        rows++;           // Increment row count
        data.push_back(row);  // Store row data in data vector
    }

    file.close();

    // Initialize Matrix object with determined rows and columns
    Matrix temp(rows, cols);

    // Fill Matrix with values from data
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            temp.a[i][j] = data[i][j];
        }
    }

    // Print matrix to verify
    return temp;
}

Matrix readImage(const std::string filename = "image.pgm"){
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }

    // Read PGM header
    std::string magicNumber;
    int width, height, maxVal;
    file >> magicNumber >> width >> height >> maxVal;
    file.ignore(); // Skip the newline after maxVal
    
    Matrix temp(1, width*height);

    if (magicNumber != "P5") {
        std::cerr << "Error: Unsupported PGM format!" << std::endl;
    }

    // Read pixel data
    std::vector<unsigned char> pixels(width * height);
    file.read(reinterpret_cast<char*>(pixels.data()), pixels.size());

    if (!file) {
        std::cerr << "Error: Could not read pixel data!" << std::endl;
    }

    file.close();

    for(int i=0; i<width*height; i++){
        int value = 255 - static_cast<int>(pixels[i]);
        temp.a[0][i] = value/(double)255;
    }
    return temp;
}