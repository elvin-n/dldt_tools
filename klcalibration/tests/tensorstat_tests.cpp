#include <gtest/gtest.h>
#include<data_stats.h>
#include <cmath>


class TensorHistogramTest : public ::testing::Test {
};

class TensorDistributionTest : public ::testing::Test {
};

class TensorHistogramHelperTest : public ::testing::Test {
};

TEST_F(TensorHistogramTest, addValues) {
    std::vector<float> t1;
    t1.push_back(-4.5f);
    t1.push_back(-4.3f);
    t1.push_back(-4.2f);
    t1.push_back(-0.5f);
    t1.push_back(5.5f);
    t1.push_back(5.6f);
    t1.push_back(5.7f);
    t1.push_back(5.8f);
    t1.push_back(5.9f);
    TensorHistogram ts(-5.0f, 15.0f, 4);
    ts.addValues(t1.data(), t1.size());
    ASSERT_EQ(static_cast<size_t>(4), ts.buckets());
    ASSERT_EQ(-5.f, ts.minValue());
    ASSERT_EQ(15.f, ts.maxValue());
    ASSERT_EQ(static_cast<size_t>(4), ts.elementsInBucket(0));
    ASSERT_EQ(static_cast<size_t>(0), ts.elementsInBucket(1));
    ASSERT_EQ(static_cast<size_t>(5), ts.elementsInBucket(2));
    ASSERT_EQ(static_cast<size_t>(0), ts.elementsInBucket(3));


    ts.addValues(t1.data(), t1.size());
    ASSERT_EQ(static_cast<size_t>(4), ts.buckets());
    ASSERT_EQ(-5.f, ts.minValue());
    ASSERT_EQ(15.f, ts.maxValue());
    ASSERT_EQ(static_cast<size_t>(8), ts.elementsInBucket(0));
    ASSERT_EQ(static_cast<size_t>(0), ts.elementsInBucket(1));
    ASSERT_EQ(static_cast<size_t>(10), ts.elementsInBucket(2));
    ASSERT_EQ(static_cast<size_t>(0), ts.elementsInBucket(3));

}

TEST_F(TensorHistogramTest, findZeroBucket) {
    std::vector<float> t1;
    t1.push_back(-4.5f);
    t1.push_back(-4.3f);
    t1.push_back(-4.2f);
    t1.push_back(-0.5f);
    t1.push_back(5.5f);
    t1.push_back(5.6f);
    t1.push_back(5.7f);
    t1.push_back(5.8f);
    t1.push_back(5.9f);

    TensorHistogram ts1(-5.0f, 15.0f, 4);
    ts1.addValues(t1.data(), t1.size());
    ASSERT_EQ(static_cast<size_t>(1), ts1.findZeroBucket());

    TensorHistogram ts2(-5.0f, 15.0f, 5);
    ts2.addValues(t1.data(), t1.size());
    ASSERT_EQ(static_cast<size_t>(1), ts2.findZeroBucket());

    TensorHistogram ts3(-5.0f, 15.0f, 3);
    ts3.addValues(t1.data(), t1.size());
    ASSERT_EQ(static_cast<size_t>(0), ts3.findZeroBucket());

    TensorHistogram ts4(-5.0f, 15.0f, 2000);
    ts4.addValues(t1.data(), t1.size());
    ASSERT_EQ(static_cast<size_t>(500), ts4.findZeroBucket());
}

TEST_F(TensorHistogramTest, getZSymmetric) {
    std::vector<float> t1;
    for (float i = 0; i < 1000; i++) {
        t1.push_back(i - 500.0f);
    }

    TensorHistogram ts1(-500.0f, 500.0f, 1000);
    ts1.addValues(t1.data(), t1.size());
    size_t minBucket, maxBucket;
    ts1.getZSymmetric(0.5f, minBucket, maxBucket);
    ASSERT_EQ(static_cast<size_t>(250), minBucket);
    ASSERT_EQ(static_cast<size_t>(750), maxBucket);

    ts1.getZSymmetric(0.6f, minBucket, maxBucket);
    ASSERT_EQ(static_cast<size_t>(200), minBucket);
    ASSERT_EQ(static_cast<size_t>(800), maxBucket);
}

TEST_F(TensorHistogramTest, createClampedHistogram) {
    std::vector<float> t1;
    for (float i = 0; i < 1000; i++) {
        t1.push_back(i - 500.0f);
    }

    TensorHistogram ts1(-500.0f, 500.0f, 1000);
    ts1.addValues(t1.data(), t1.size());
    TensorDistribution td1 = ts1.createClampedDistribution(100, 900);

    ASSERT_EQ(static_cast<size_t>(801), td1.buckets());
    ASSERT_EQ(101.f/1000.f, td1.getProbability(0));
    ASSERT_EQ(1.f/1000.f, td1.getProbability(1));
    ASSERT_EQ(1.f/1000.f, td1.getProbability(799));
    ASSERT_EQ(100.f/1000.f, td1.getProbability(800));
}

TEST_F(TensorHistogramTest, createQuantizedClampedDistribution) {

}

TEST_F(TensorDistributionTest, KLDivergence) {
    TensorDistribution td1(0,2,3);
    TensorDistribution td2(0,2,3);
    td1.setProbability(0, 0.36);
    td1.setProbability(1, 0.48);
    td1.setProbability(2, 0.16);
    td2.setProbability(0, 0.33333333);
    td2.setProbability(1, 0.33333333);
    td2.setProbability(2, 0.33333333);
    ASSERT_TRUE(fabs(0.0852996 - td1.KLDivergence(td2)) < 0.0001);
    ASSERT_TRUE(fabs(0.097455 - td2.KLDivergence(td1)) < 0.0001);
}

TEST_F(TensorHistogramTest, aggregatedHistogram) {
    std::map<std::string, std::vector<TensorHistogram > > _data;

    TensorHistogram ts1(-5.0f, 15.0f, 4);
    TensorHistogram ts2(-15.0f, 15.0f, 10);
    std::vector<TensorHistogram> th;
    th.push_back(ts1);
    th.push_back(ts2);
    _data["input"] = th;

    std::vector<float> t1;
    t1.push_back(-4.5f);
    t1.push_back(-4.3f);
    t1.push_back(-4.2f);
    t1.push_back(-0.5f);
    t1.push_back(5.5f);
    t1.push_back(5.6f);
    t1.push_back(5.7f);
    t1.push_back(5.8f);
    t1.push_back(5.9f);

    _data["input"][0].addValues(t1.data(), t1.size());
}

TEST_F(TensorHistogramHelperTest, minimizeKlDivergence) {
    std::vector<float> t1;
    for (float i = 0; i < 1000; i++) {
        t1.push_back(i - 500.0f);
    }

    TensorHistogram ts1(-500.0f, 500.0f, 1000);
    ts1.addValues(t1.data(), t1.size());

    TensorHistogramHelper thh(ts1);
    std::pair<float, float> effective;
    try {
        thh.minimizeKlDivergence(effective);
    } catch (std::string q) {
        std::cout << q << std::endl;
    }
    ASSERT_EQ(-499.f, effective.first);
    ASSERT_EQ(499.f, effective.second);
}


