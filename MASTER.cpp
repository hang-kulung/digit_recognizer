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

void time_stamp(){
    std::cout << "Time: " << (int)(clock() * 1000/ CLOCKS_PER_SEC) << " ms\n";
}


class NeuralNetwork{
    public:
    int n;
    std::vector <int> size;
    std::vector <Matrix> w, b, z, delta_w, delta_b, delta_z;
    double learning_rate;
    NeuralNetwork(){}
    NeuralNetwork(std::vector <int> sz, double alpha){
        n = (int)(sz.size());
        size = sz;
        w.resize(n-1);
        b.resize(n-1);
        z.resize(n-1);
        delta_z.resize(n-1);
        delta_w.resize(n-1);
        delta_b.resize(n-1);

        for(int i=0; i<n-1; i++){
            w[i] = Matrix(size[i], size[i+1]);
            b[i] = Matrix(1, size[i+1]);
            delta_w[i] = Matrix(size[i], size[i+1]);
            delta_b[i] = Matrix(1, size[i+1]);

            b[i] = initialize_weights_xavier(1, size[i+1]);
            w[i] = initialize_weights_xavier(size[i], size[i+1]);
        }
        learning_rate = alpha;
    }

    Matrix feedForward(Matrix inputs){
        // for(int i=0; i<n-1; i++){
        //     input = multiply(input, w[i]);
        //     input = add(input, b[i]);
        //     input = sigmoid(input);
        // }
        Matrix input;
        for(int i=0; i < n-2; i++){
            input = multiply(inputs, w[i]);
            input = add(input, extendRow(b[i], input.n));
            input = sigmoid(input);
        }
        
        input = multiply(input, w[n-2]);
        input = add(input, extendRow(b[n-2], input.n));
        input = softmax(input);

        return input;
    }

    void backPropagation(Matrix input, Matrix output){
        std::vector <Matrix> l;
        Matrix delta;
        int size_of_inputs = input.n;
        l.push_back(input);


        for(int i=0; i < n-2; i++){        
            z[i] = add(multiply(input, w[i]), extendRow(b[i], input.n));
            input = sigmoid(z[i]);
            l.push_back(input);
        }
        z[n-2] = add(multiply(input, w[n-2]), extendRow(b[n-2], input.n));
        input = softmax(z[n-2]);
        l.push_back(input);

        delta_z[n-2] = subtract(input, output);
        delta_b[n-2] = scalarDivide(sumRow(delta_z[n-2]), size_of_inputs);
        delta_w[n-2] = scalarDivide(multiply(transpose(l[n-2]), delta_z[n-2]), size_of_inputs);
        for(int i=n-3; i >=0; i--){
            delta_z[i] = term_by_term(multiply(delta_z[i+1], transpose(w[i+1])), sigmoidDeriv(z[i]));
            delta_b[i] = scalarDivide(sumRow(delta_z[i]), size_of_inputs);
            delta_w[i] = scalarDivide(multiply(transpose(l[0]), delta_z[i]), size_of_inputs);
        }
    }

    double findError(Matrix input, Matrix output){
        Matrix err;
        double error = 0.0;
        // for(int i=0; i < n-1; i++){
        //     z[i] = add(multiply(inputs, w[i]), extendRow(b[i], inputs.n));
        //     inputs = sigmoid(z[i]);
        // }
        for(int i=0; i < n-2; i++){
            input = multiply(input, w[i]);
            input = add(input, extendRow(b[i], input.n));
            input = sigmoid(input);
        }
        
        input = multiply(input, w[n-2]);
        input = add(input, extendRow(b[n-2], input.n));
        input = softmax(input);

        err = term_by_term(subtract(input, output), subtract(input, output));
        err = sumRow(err);
        for(int i=0; i<10; i++){
            error += err.a[0][i];
        }
        error = error / (double)(input.n);
        error = error / 10;
        return error;
    }

    void train(Matrix inputs,Matrix outputs){
        
        double err = 0.0;
        for(int i=0; i<n-1; i++){
            delta_w[i].zero();
            delta_b[i].zero();
        }
        for(int iteration=0; iteration <5; iteration++){
            std::cout << "Iteration " << iteration+1 << " starting!! \n";

            backPropagation(inputs, outputs);
            //delta_w[0].print();
            for(int i=0; i<n-1;i++){
                for(int j=0; j<delta_w[i].n; j++){
                    for(int z=0; z<delta_w[i].m; z++){
                        w[i].a[j][z] -= learning_rate * delta_w[i].a[j][z];
                    }
                }
                for(int j=0; j<delta_b[i].n; j++){
                    for(int z=0; z<delta_b[i].m; z++){
                        b[i].a[j][z] -= learning_rate * delta_b[i].a[j][z];
                    }
                }
            }
            
            err = findError(inputs, outputs);
            std::cout << "Error: " << err << std::endl;

            time_stamp();
        }
    }
};






NeuralNetwork net({784, 16, 10}, 1);
std::vector <Matrix> train_input, train_output;

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
    std::ifstream IN("train.csv");
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
    time_stamp();
}



int main(){
    parseTrainingData();
    Matrix input, output, result;
    std::vector<Matrix> sliced_input(train_input.begin() , train_input.begin() + 42000);
    std::vector<Matrix> sliced_output(train_output.begin() , train_output.begin() + 42000);

    input = inputMatrix(sliced_input);
    output = outputMatrix(sliced_output);
    net.train(input, output);
    result = net.feedForward(train_input[6]);
    result.print();
    result = net.feedForward(train_input[7]);
    result.print();
    return 0;
}