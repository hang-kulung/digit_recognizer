#include <iostream>
#include <vector>
#include "forward.hpp"
// #include "matric.hpp"
// #include "network.hpp"

int main() {
    std::vector <double> guesses = predict("image.pgm");
    
    std::cout << "guess: " << guesses[0] << " with" << guesses[1];

    char a;
    std::cin >> a;
    return 0;
}