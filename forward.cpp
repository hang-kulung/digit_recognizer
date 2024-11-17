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

class Matrix;
double randDouble();
double sigmoid(double);
double sigmoidDeriv(double);

double random_uniform(double lower_bound, double upper_bound) {
    // Generate random numbers between lower_bound and upper_bound
    return lower_bound + (upper_bound - lower_bound) * (rand() / (double)RAND_MAX);
}


class Matrix{
    public:
    int n, m;
    double **a;

    Matrix():n(0), m(0), a(nullptr){} 
    Matrix(int rows, int cols){
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
    Matrix(const Matrix &x){
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
    void operator=(const Matrix &x){
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

    void randInit(){
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                a[i][j] = randDouble() - 0.5;
            }
        }
    }

    void zero(){
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                a[i][j] = 0.0;
            }
        }
    }

    void add(Matrix x){
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                a[i][j] += x.a[i][j];
            }
        }
    }

    void print(){
        for(int i=0; i<n; i++){
            for(int j=0; j<m; j++){
                std::cout << a[i][j] << ", ";
            }
            std::cout << "\n";
        }
        std::cout << ".............\n";
    }

    ~Matrix(){
        for(int i=0; i<n; i++){
            delete[] a[i];
        }
        delete[] a;
    }
};

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

Matrix sigmoid(Matrix M){
    Matrix temp(M.n, M.m);
    for(int i=0; i<M.n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[i][j] = sigmoid(M.a[i][j]);
        }
    }
    return temp;
}

Matrix sigmoidDeriv(Matrix M){
    Matrix temp(M.n, M.m);
    for(int i=0; i<M.n; i++){
        for(int j=0; j<M.m; j++){
            temp.a[i][j] = sigmoidDeriv(M.a[i][j]);
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

double sigmoid(double x){
    if (x >= 0.0) return x;
    else return 0;
    //return 1.0/(1.0 + exp(-x));
}

double sigmoidDeriv(double x){
    if (x >= 0.0) return 1.0;
    else return 0.0;
    // return x * (1.0 - x);
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

std::vector <Matrix> train_input, train_output;

std::vector <int> split(std::string s){
    int curr = 0;
    std::vector <int> ans;

    for(int i=0; i<(int)(s.size()); i++){
        if(s[i]==','){
            ans.push_back(curr);
            curr = 0;
        }
        else{
            curr *= 10;
            curr += s[i] -'0';
        }
    }
    ans.push_back(curr);
    return ans;
}

void parseTrainingData(){
    std::ifstream IN("data/train.csv");
    std::string trash;
    std::vector <int> v;
    Matrix input(1, 784), output(1, 10);
    train_input.reserve(42000);
    train_output.reserve(42000);

    IN >> trash;

    for(int i=0; i<42000; i++){
        IN >> trash;

        v = split(trash);

        output.zero();
        output.a[0][v[0]] = 1;
        for(int j=1; j<785; j++){
            input.a[0][j-1] = v[j]/255.0;
        }
        train_input.push_back(input);
        train_output.push_back(output);
    }
    std::cout << "Training Data loaded!! \n";
}



Matrix feedForward(Matrix input, Matrix W0, Matrix b0, Matrix W1, Matrix b1){
        input = multiply(input, W0);
        input = add(input, extendRow(b0, input.n));
        input = sigmoid(input);
        
        input = multiply(input, W1);
        input = add(input, extendRow(b1, input.n));
        input = softmax(input);

        return input;
}

int main() {
    parseTrainingData();
    
    Matrix W0, b0, W1, b1;

    b0 = ReadFile("data/b1.csv");
    b1 = ReadFile("data/b2.csv");
    W0 = ReadFile("data/W1.csv");
    W1 = ReadFile("data/W2.csv");

    Matrix result, input;
    result = feedForward(train_input[6], W0, b0, W1, b1);
    result.print();

    return 0;
}