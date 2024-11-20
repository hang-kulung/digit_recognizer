#ifndef MATRIC_HPP
#define MATRIC_HPP

#include <vector>
#include<iostream>

double randDouble();
double relu(double);
double reluDeriv(double);

double random_uniform(double lower_bound, double upper_bound);


class Matrix{
    public:
    int n, m;
    double **a;

    Matrix(); 
    Matrix(int rows, int cols);
    Matrix(const Matrix &x);
    void operator=(const Matrix &x);

    void randInit();

    void zero();

    void add(Matrix x);
    void print();
    ~Matrix();
};

Matrix initialize_weights_xavier(int fan_in, int fan_out);

Matrix add(Matrix M1, Matrix M2);

Matrix subtract(Matrix M1, Matrix M2);

Matrix term_by_term(Matrix M1, Matrix M2);

Matrix transpose(Matrix M);

Matrix multiply(Matrix M1, Matrix M2);

Matrix relu(Matrix M);

Matrix reluDeriv(Matrix M);

Matrix softmax(Matrix M);

double randDouble();

Matrix sumRow(Matrix M);

Matrix extendRow(Matrix M, int n);

Matrix scalarDivide(Matrix M, double d);

Matrix inputMatrix(std::vector <Matrix> inputs);

Matrix outputMatrix(std::vector <Matrix> outputs);

void saveToCSV(Matrix temp, const std::string& filename);

Matrix ReadFile(std::string filename);

Matrix readImage(const std::string filename);

#endif