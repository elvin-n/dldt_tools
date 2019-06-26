// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <stdint.h>
#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <string>

#include "data_stats.h"
#include "user_exception.hpp"

TensorStatistic::TensorStatistic(float* data, size_t count, size_t nbuckets) {
    _min = std::numeric_limits<float>::max();
    _max = std::numeric_limits<float>::min();
    for (size_t i = 0; i < count; i++) {
        float val = static_cast<float>(data[i]);
        if (_min > val) {
            _min = val;
        }

        if (_max < val) {
            _max = val;
        }
    }

    if (_min == _max) {
        return;
    }
}

float TensorStatistic::getMaxValue() const {
    return _max;
}


float TensorStatistic::getMinValue() const {
    return _min;
}

std::vector<std::string> AggregatedDataStats::registeredLayers() {
    std::vector<std::string> layers;
    for (auto l : _data) {
        layers.push_back(l.first);
    }
    return layers;
}

void AggregatedDataStats::registerLayer(std::string layer) {
    _data[layer];
}

void AggregatedDataStats::addTensorStatistics(const std::string& name, size_t channel, float* data, size_t count) {
    auto&& byChannel = _data[name];
    byChannel[channel].push_back(TensorStatistic(data, count));
}

void AggregatedDataStats::addTensorStatistics(const std::string &name, size_t channel, uint8_t *data, size_t count) {
    std::vector<float> intermediate;
    for (size_t i = 0; i < count; i++) {
        intermediate.push_back(data[i]);
    }
    addTensorStatistics(name, channel, intermediate.data(), count);
}

size_t AggregatedDataStats::getNumberChannels(const std::string& name) const {
    auto it = _data.find(name);
    if (it != _data.end()) {
        return it->second.size();
    }
    return 0;
}

void AggregatedDataStats::getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold) {
    // take data by name
    auto it = _data.find(name);
    if (it != _data.end()) {
        auto stats = it->second[channel];
        // having absolute min/max values, we can create new statistic
        std::vector<float> maxValues;
        std::vector<float> minValues;
        for (size_t i = 0; i < stats.size(); i++) {
            const TensorStatistic& tsS = stats[i];
            maxValues.push_back(tsS.getMaxValue());
            minValues.push_back(tsS.getMinValue());
        }
        // define number of elements to throw out
        size_t elementToTake = static_cast<size_t>(maxValues.size() * (threshold / 100));
        int elementsToThrow = maxValues.size() - elementToTake;
        std::sort(maxValues.begin(), maxValues.end());
        std::sort(minValues.begin(), minValues.end());

        min = minValues[elementsToThrow];
        max = maxValues[elementToTake - 1];
    } else {
        min = max = 0.f;
    }
}




TensorHistogram::TensorHistogram(float minV, float maxV, size_t buckets) :
    _min(minV),
    _max(maxV),
    _buckets(buckets) {
    _histogram.resize(_buckets, 0);
}

void TensorHistogram::addValues(float* data, size_t size) {
    float step = (_max - _min)/_buckets;
    for (size_t i = 0; i < size; i++) {
        if (data[i]) {
            size_t bucket = (data[i] - _min) / step;
            if (bucket >= _buckets) {
                bucket = _buckets -1;
            }
            _histogram[bucket]++;
        }
    }
}

size_t TensorHistogram::buckets() const {
    return _buckets;
}

float TensorHistogram::minValue() const {
    return _min;
}
float TensorHistogram::maxValue() const {
    return _max;
}
size_t TensorHistogram::elementsInBucket(size_t bucket) const {
    if (bucket < _buckets) {
        return _histogram[bucket];
    }
    return -1;
}

size_t TensorHistogram::sumInBuckets(size_t start, size_t end) const {
    if (start >= 0 && start < _buckets && end >= 0 && end < _buckets &&
        start < end) {
        size_t sum = 0;
        for (size_t i = start; i <= end; i++) {
            sum += _histogram[i];
        }
        return sum;
    }
    return 0.f;
}

TensorDistribution TensorHistogram::createClampedDistribution(size_t startBucket, size_t endBucket) const {
    // calculate new values for min and max
    float originalStep = (_max - _min)/_buckets;
    float newMin = originalStep * startBucket + _min;
    float newMax = originalStep * endBucket + _min;
    TensorDistribution clampedDist(newMin, newMax, endBucket - startBucket + 1);

    std::vector<size_t> clampedHist;
    clampedHist.resize(clampedDist.buckets());
    clampedHist[0] = sumInBuckets(0, startBucket - 1);
    clampedHist[clampedHist.size() - 1] = sumInBuckets(endBucket + 1, _buckets - 1);
    for (size_t i = 0; i < endBucket - startBucket + 1 ; i++) {
        clampedHist[i] += _histogram[startBucket + i];
    }
    double sumAll = static_cast<double>(sumInBuckets(0, buckets() - 1));


    for (size_t i = 0; i < clampedDist.buckets(); i++) {
        clampedDist.setProbability(i, static_cast<double>(clampedHist[i])/sumAll);
    }
    return clampedDist;
}
TensorDistribution TensorHistogram::createQuantizedClampedDistribution(size_t startBucket, size_t endBucket) const {
    float originalStep = (_max - _min)/_buckets;
    float newMin = originalStep * startBucket + _min;
    float newMax = originalStep * endBucket + _min;
    TensorDistribution clampedDist(newMin, newMax, endBucket - startBucket + 1);

    std::vector<size_t> clampedHist;
    clampedHist.resize(clampedDist.buckets());
    clampedHist[0] = sumInBuckets(0, startBucket - 1);
    clampedHist[clampedHist.size() - 1] = sumInBuckets(endBucket + 1, _buckets - 1);
    for (size_t i = 0; i < endBucket - startBucket + 1 ; i++) {
        clampedHist[i] += _histogram[startBucket + i];
    }

    std::vector<double> clampedQHist;
    clampedQHist.resize(clampedDist.buckets());
    size_t qStep = clampedDist.buckets() / 256;
    if (qStep == 0) {
        qStep ++;
    }
    double sumQHist = 0.f;
    for (size_t i = 0; i < clampedDist.buckets(); i += qStep) {
        size_t tmpQ = 0;
        size_t nonZBuckets = 0;
        for (size_t j = 0; j < qStep && i + j < clampedDist.buckets(); j++) {
            if (clampedHist[i + j]) {
                tmpQ += clampedHist[i + j];
                nonZBuckets++;
            }
        }
        if (nonZBuckets) {
            float qv = static_cast<double>(tmpQ) / static_cast<double>(nonZBuckets);
            for (size_t j = 0; j < qStep && i + j < clampedDist.buckets(); j++) {
                if (clampedHist[i + j]) {
                    clampedQHist[i + j] = qv;
                    sumQHist += qv;
                }
            }
        }
    }

    for (size_t i = 0; i < clampedDist.buckets(); i++) {
        clampedDist.setProbability(i, clampedQHist[i]/sumQHist);
    }

    return clampedDist;
}

void TensorHistogram::getZSymmetric(float ratio, size_t& minBucket, size_t& maxBucket) const {
    if (ratio > 1.0f) {
        THROW_USER_EXCEPTION(2) << "invalid ratio parameter in TensorHistogram::getZSymmetric";
    }
    // find zero point
    size_t zeroBucket = findZeroBucket();
    size_t sumAll = sumInBuckets(0, buckets() - 1);
    size_t sumArea = _histogram[zeroBucket];
    size_t leftBucket, rightBucket;
    leftBucket = rightBucket = zeroBucket;
    while (static_cast<float>(sumArea) / static_cast<float>(sumAll) < ratio) {
        if (leftBucket > 0) {
            sumArea += _histogram[--leftBucket];
        }
        if (rightBucket < buckets() - 1) {
            sumArea += _histogram[++rightBucket];
        }
    }
    minBucket = leftBucket;
    maxBucket = rightBucket;
}

size_t TensorHistogram::findZeroBucket() const {
    if (_min >= 0) {
        return _min;
    } else if (_max <= 0) {
        return _max;
    } else {
        size_t b = _buckets * (0.f - _min) / (_max - _min);
        return b;
    }
}

float TensorHistogram::valueForBucket(size_t bucket) const {
    float step = (_max - _min)/_buckets;
    return _min + bucket * step;
}

TensorHistogramHelper::TensorHistogramHelper(const TensorHistogram& th) :
    _th(th) {

}

void TensorHistogramHelper::minimizeKlDivergence(std::pair<float, float>& effective) {
    effective.first = _th.maxValue();
    effective.second = _th.minValue();

    size_t minBucket, maxBucket;
    _th.getZSymmetric(0.6f, minBucket, maxBucket);
    size_t iMinBucket = minBucket, iMaxBucket = maxBucket;
    float kldivMin = std::numeric_limits<float>::max();

    while (iMinBucket > 0 || iMaxBucket < _th.buckets() - 1) {
        if (iMinBucket > 0) {
            iMinBucket--;
        }
        if (iMaxBucket < _th.buckets() - 1) {
            iMaxBucket++;
        }

        size_t q1 = iMaxBucket - iMinBucket + 1;
        size_t step = q1 / 256;

        // IMPORTANT! below verifications are extrimely important to get steady and monotonic values of
        // KL divergence value over changing the set of distribution set
        // 1. number of buckets in quant should be more than one
        // 2. the border quants should contain full number of buckets (step == latest quant number of bucket)
        if (iMaxBucket - iMinBucket > 1200 && q1 % step == 0) {
            TensorDistribution cH = _th.createClampedDistribution(iMinBucket, iMaxBucket);
            TensorDistribution cQ = _th.createQuantizedClampedDistribution(iMinBucket, iMaxBucket);
            float kldiv = cH.KLDivergence(cQ);
            if (kldivMin > kldiv) {
                kldivMin = kldiv;
                effective.first = _th.valueForBucket(iMinBucket);
                effective.second = _th.valueForBucket(iMaxBucket);
            }
        }
    }
}

TensorDistribution::TensorDistribution(float minV, float maxV, size_t buckets) :
    _min(minV),
    _max(maxV),
    _buckets(buckets) {
    _distribution.resize(_buckets, 0);
}

void TensorDistribution::setProbability(size_t bucket, double probability) {
    if (bucket >= 0 && bucket < _buckets) {
        _distribution[bucket] = probability;
    } else {
        THROW_USER_EXCEPTION(2) << "TensorDistribution::setProbability bucket is out of range";
    }
}

double TensorDistribution::TensorDistribution::getProbability(size_t bucket) const {
    if (bucket >= 0 && bucket < _buckets) {
        return _distribution[bucket];
    } else {
        THROW_USER_EXCEPTION(2) << "TensorDistribution::setProbability bucket is out of range";
    }
}

size_t TensorDistribution::TensorDistribution::buckets() const {
    return _buckets;
}

float TensorDistribution::minValue() const {
    return _min;
}

float TensorDistribution::maxValue() const {
    return _max;
}


float TensorDistribution::KLDivergence(const TensorDistribution& q) const {
    // compare current and q tensor
    if (buckets() != q.buckets()) {
        THROW_USER_EXCEPTION(2) << "TensorDistribution::KLDivergence, number of buckets are different";
    }

    // verify that distributions are normalized
    double sumP = 0.f, sumQ = 0.f;
    for (size_t i = 0; i < _buckets; i++) {
        sumP += _distribution[i];
        sumQ += q.getProbability(i);
    }

    if (fabs(sumP - 1.0f) > 0.1) {
        std::cout << "sum of probabilities for P distribution in TensorDistribution::KLDivergence is not eq 1 but eq " << sumP;
        THROW_USER_EXCEPTION(2) << "sum of probabilities for current distribution in TensorDistribution::KLDivergence is not eq 1";
    }

    if (fabs(sumQ - 1.0f) > 0.1) {
        std::cout << "sum of probabilities for Q distribution in TensorDistribution::KLDivergence is not eq 1 but eq " << sumQ;
        THROW_USER_EXCEPTION(2) << "sum of probabilities for Q distribution in TensorDistribution::KLDivergence is not eq 1";
    }

    double tmpSum = 0;
    for (size_t i = 0; i < buckets(); i++) {
        if (_distribution[i] != 0) {
            if (q.getProbability(i) == 0) {
                THROW_USER_EXCEPTION(2) << "TensorDistribution::KLDivergence probability value in q distribution must not be eq to 0 if it is not 0 in base distribution";
            }
            tmpSum += _distribution[i] * log(_distribution[i]/q.getProbability(i));
        }
    }

    return tmpSum;
}


