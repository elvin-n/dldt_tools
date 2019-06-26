// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <string>

struct TensorStatistic {
    TensorStatistic(float* data, size_t count, size_t nbuckets = 1000);
    float getMaxValue() const;
    float getMinValue()const;
protected:
    float _min;
    float _max;
};

class AggregatedDataStats {
public:
    void addTensorStatistics(const std::string& name, size_t channel, float* data, size_t count);
    void addTensorStatistics(const std::string &name, size_t channel, uint8_t *data, size_t count);
    void getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold);
    size_t getNumberChannels(const std::string& name) const;
    std::vector <std::string> registeredLayers();
    void registerLayer(std::string layer);
protected:
    std::map<std::string, std::map<size_t, std::vector<TensorStatistic> > > _data;
};


class TensorDistribution {
public:
    TensorDistribution(float minV, float maxV, size_t buckets = 2048);
    void setProbability(size_t bucket, double probability);
    double getProbability(size_t bucket) const;
    float minValue() const;
    float maxValue() const;
    size_t buckets() const;
    float KLDivergence(const TensorDistribution& q) const;
private:
    float _min;
    float _max;
    size_t _buckets;
    std::vector<double> _distribution;
};

class TensorHistogram {
public:
    TensorHistogram(float minV, float maxV, size_t buckets = 2048);
    void addValues(float* data, size_t size);
    size_t buckets() const;
    float minValue() const;
    float maxValue() const;
    size_t elementsInBucket(size_t bucket) const;
    float valueForBucket(size_t bucket) const;

    TensorDistribution createClampedDistribution(size_t startBucket, size_t endBucket) const;
    TensorDistribution createQuantizedClampedDistribution(size_t startBucket, size_t endBucket) const;
    void getZSymmetric(float ratio, size_t& minBucket, size_t& maxBucket) const;

    size_t findZeroBucket() const;
private:
    size_t sumInBuckets(size_t start, size_t end) const;
    float _min;
    float _max;
    size_t _buckets;
    std::vector<size_t> _histogram;
};

class TensorHistogramHelper {
public:
    TensorHistogramHelper(const TensorHistogram& th);
    void minimizeKlDivergence(std::pair<float, float>& effective);
private:


    const TensorHistogram& _th;
};
