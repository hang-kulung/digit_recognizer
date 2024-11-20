#include<iostream>
#include <vector>
#include <string>
#include "forward.hpp"
#include "matric.hpp"
#include "network.hpp"

NeuralNetwork net({784, 16, 10}, 0.1);

std::vector <double> predict(std::string image) {

    Matrix W0, b0, W1, b1;

    b0 = ReadFile("data/b1.csv");
    b1 = ReadFile("data/b2.csv");
    W0 = ReadFile("data/W1.csv");
    W1 = ReadFile("data/W2.csv");

    net.b[0] = b0;
    net.b[1] = b1;
    net.w[0] = W0;
    net.w[1] = W1;

    Matrix result, input;
    input = readImage(image);
    //input.print();

    result = net.feedForward(input);
    // train_input[41210].print();
    result.print();

    int guess = 0;
    double maxPercent = 0.0;

    for(int i=0; i<10; i++){
        if(result.a[0][i] > maxPercent){
            guess = i;
            maxPercent = result.a[0][i];
        }
    }
    std::vector <double> values;
    values.push_back(static_cast<double>(guess));
    values.push_back(maxPercent * 100);
    return values;
}
