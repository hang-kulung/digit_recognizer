#include<vector>
#include<algorithm>
#include "matric.hpp"
#include "network.hpp"

NeuralNetwork net({784, 16, 10}, 0.1);

int main(){
    parseTrainingData();

    Matrix input, output, result;

    std::vector<Matrix> sliced_input(train_input.begin() , train_input.begin() + 1000);
    std::vector<Matrix> sliced_output(train_output.begin() , train_output.begin() + 1000);

    input = inputMatrix(sliced_input);
    output = outputMatrix(sliced_output);

    net.train(input, output);

    saveToCSV(net.w[0], "_datA/w0.csv");
    saveToCSV(net.w[1], "_datA/w1.csv");
    saveToCSV(net.b[0], "_datA/b0.csv");
    saveToCSV(net.b[1], "_datA/b1.csv");

    result = net.feedForward(train_input[0]);
    result.print();
    char c;
    std::cin >> c;
    
    return 0;
}
