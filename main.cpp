#include <iostream>

#include <cmath>
#include <utility>
#include <vector>

double poission_pmf(int i, double lambda) {
    return pow(M_E, i * log(lambda) - lambda - lgamma(i + 1.0));
}

double expected_truncated(int maxEx, double lambda, bool weighted) {
    double exp = 0.0;
    for (int i = 0; i < maxEx; ++i) {
        exp += poission_pmf(i, lambda) * (weighted ? i : 1.0);
    }
    return exp;
}


std::pair<double, double> alphaAndRelSize(int k, double lambda) {
    double alphaNotFull = expected_truncated(k, lambda, true) / k;
    double prob = expected_truncated(k, lambda, false);
    return {1.0 - prob + alphaNotFull, -log(prob)};
}

void buildRec(std::vector<std::pair<double, double>> &samples, double deltaX, double llamb, double rlamb, int k) {
    std::pair<double, double> left = alphaAndRelSize(k, llamb);
    std::pair<double, double> right = alphaAndRelSize(k, rlamb);

    //double area = (left.second + right.second) / 2.0 * (right.first - left.first);
    if (right.first - left.first < deltaX)
        return;
    double midPoint = (rlamb + llamb) / 2.0;
    std::pair<double, double> midSample = alphaAndRelSize(k, midPoint);
    buildRec(samples, deltaX, llamb, midPoint, k);
    if (std::isnormal(midSample.second)) {
        samples.push_back(midSample);
        buildRec(samples, deltaX, midPoint, rlamb, k);
    }
}

std::vector<uint64_t> getBucketFunctionFulcrums(int k, int fulcs, uint64_t bucketCount) {
    std::vector<std::pair<double, double>> samples;
    double leftLimit = k * 0.0001;
    double rightLimit = k * 10;
    buildRec(samples, 0.1 / fulcs, leftLimit, rightLimit, k);

    std::vector<std::pair<double, double>> integral;
    std::pair<double, double> lastPair = {0, 0};
    integral.push_back(lastPair);
    for (auto pair: samples) {
        pair.second = std::max(lastPair.second, lastPair.second + pair.second * (pair.first - lastPair.first));
        integral.push_back(pair);
        lastPair = pair;
    }
    for (auto &pair: integral) {
        pair.second /= lastPair.second;
    }

    std::vector<uint64_t> fulcrums;
    int index = 0;
    fulcrums.push_back(0);
    for (int i = 1; i < fulcs - 1; ++i) {
        double x = i / (double) (fulcs - 1);
        while (integral[index].first < x) {
            index++;
        }
        fulcrums.push_back(integral[index].second * double(bucketCount << 32U));
    }
    fulcrums.push_back(bucketCount << 32U);

    return fulcrums;
}

uint64_t queryFulcrums(const std::vector<uint64_t> &fulcs, uint64_t hash) {
    __uint128_t z = hash * __uint128_t(fulcs.size() - 1);
    uint64_t index = z >> 64;
    uint64_t part = z;
    uint64_t v1 = (fulcs[index + 0] * __uint128_t(uint64_t(-1) - part)) >> 64;
    uint64_t v2 = (fulcs[index + 1] * __uint128_t(part)) >> 64;
    return (v1 + v2) >> 32;
}


int main() {
    std::vector<uint64_t> fulcrumtest = getBucketFunctionFulcrums(1, 1000, 1000);
    uint64_t bucket1 = queryFulcrums(fulcrumtest, 0);
    uint64_t bucket2 = queryFulcrums(fulcrumtest, uint64_t(-1) / 2);
    uint64_t bucket3 = queryFulcrums(fulcrumtest, uint64_t(-1));
    std::cout << bucket1 << " " << bucket2 << " " << bucket3 << std::endl;
    return 0;
}
