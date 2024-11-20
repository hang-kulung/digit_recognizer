#include<iostream>
#include<fstream>
#include<vector>
#include<random>
#include<ctime>
#include<cmath>
#include<algorithm>
#include <cstdlib> // for rand() and RAND_MAX
// #include "matric.hpp"
#include "network.hpp"

void time_stamp(){
    std::cout << "Time: " << (int)(clock() * 1000/ CLOCKS_PER_SEC) << " ms\n";
}


NeuralNetwork::NeuralNetwork(){}
NeuralNetwork::NeuralNetwork(std::vector <int> sz, double alpha){
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

Matrix NeuralNetwork::feedForward(Matrix input){
    // for(int i=0; i<n-1; i++){
    //     input = multiply(input, w[i]);
    //     input = add(input, b[i]);
    //     input = sigmoid(input);
    // }
    for(int i=0; i < n-2; i++){
        input = multiply(input, w[i]);
        input = add(input, extendRow(b[i], input.m));
        input = relu(input);
    }
    input = multiply(input, w[n-2]);
    input = add(input, extendRow(b[n-2], input.m));
    input = softmax(input);

    return input;
}

void NeuralNetwork::backPropagation(Matrix input, Matrix output){
    std::vector <Matrix> l;
    Matrix delta;
    int size_of_inputs = input.n;
    l.push_back(input);


    for(int i=0; i < n-2; i++){        
        z[i] = add(multiply(input, w[i]), extendRow(b[i], input.n));
        input = relu(z[i]);
        l.push_back(input);
    }
    z[n-2] = add(multiply(input, w[n-2]), extendRow(b[n-2], input.n));
    input = softmax(z[n-2]);
    l.push_back(input);

    delta_z[n-2] = subtract(input, output);
    delta_b[n-2] = scalarDivide(sumRow(delta_z[n-2]), size_of_inputs);
    delta_w[n-2] = scalarDivide(multiply(transpose(l[n-2]), delta_z[n-2]), size_of_inputs);
    for(int i=n-3; i >=0; i--){
        delta_z[i] = term_by_term(multiply(delta_z[i+1], transpose(w[i+1])), reluDeriv(z[i]));
        delta_b[i] = scalarDivide(sumRow(delta_z[i]), size_of_inputs);
        delta_w[i] = scalarDivide(multiply(transpose(l[0]), delta_z[i]), size_of_inputs);
    }
}

double NeuralNetwork::findError(Matrix input, Matrix output){
    Matrix err;
    double error = 0.0;
    // for(int i=0; i < n-1; i++){
    //     z[i] = add(multiply(inputs, w[i]), extendRow(b[i], inputs.n));
    //     inputs = sigmoid(z[i]);
    // }
    for(int i=0; i < n-2; i++){
        input = multiply(input, w[i]);
        input = add(input, extendRow(b[i], input.n));
        input = relu(input);
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


void NeuralNetwork::train(Matrix inputs,Matrix outputs){
    
    double err = 0.0;
    for(int i=0; i<n-1; i++){
        delta_w[i].zero();
        delta_b[i].zero();
    }
    for(int iteration=0; iteration <100; iteration++){
        //std::cout << "Iteration " << iteration+1 << " starting!! \n";
        
        backPropagation(inputs, outputs);

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
        
        // err = findError(inputs, outputs);
        // std::cout << "Error: " << err << std::endl;

        time_stamp();
    }
}

std::vector <Matrix> train_input;
std::vector <Matrix> train_output;

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
    time_stamp();
}
