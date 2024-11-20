#include<vector>
#include "matric.hpp"

void time_stamp();

class NeuralNetwork{
    public:
    int n;
    std::vector <int> size;
    std::vector <Matrix> w, b, z, delta_w, delta_b, delta_z;
    double learning_rate;
    NeuralNetwork();
    NeuralNetwork(std::vector <int> sz, double alpha);

    Matrix feedForward(Matrix input);

    void backPropagation(Matrix input, Matrix output);

    double findError(Matrix input, Matrix output);

    void train(Matrix inputs,Matrix outputs);
};

extern std::vector <Matrix> train_input;
extern std::vector <Matrix> train_output;

std::vector <int> split(std::string s);

void parseTrainingData();



