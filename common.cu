// implementation of common code used by all parallel sorting implementations
// author: Shrihan Dadi (sdadi2)
#include <cstddef>
#include <random>
#include <thrust/host_vector.h>
#include "common.h"
using namespace std;

// lowest number that can be generated
#define LOWEST -1

// highest number that can be generated
#define HIGHEST 1

// function for generating a random vector of floats
thrust::host_vector<float> genVec(size_t size) {
    // create seeded generator
    mt19937 gen(SEED);
    uniform_real_distribution<float> dist(LOWEST, HIGHEST);

    // create and fill vec with random values
    cout << "Generating random vector of size " << size << "..." << endl;
    thrust::host_vector<float> vec(size);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dist(gen);
    }

    return vec;
}

// function for checking if a vector of floats is sorted or not
// only used in DEBUG mode
#ifdef DEBUG
    bool sorted(const thrust::host_vector<float>& vec) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            if (vec[i] > vec[i + 1]) {
                cout << "Index: " << i << "; Value: " << vec[i] << endl;
                cout << "Index: " << i + 1 << "; Value: " << vec[i + 1] << endl;
                return false;
            }
        }
        return true;
    }
#endif
