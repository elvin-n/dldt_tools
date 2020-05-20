#pragma once

#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <list>
#include <functional>
#include <chrono>
#include <fstream>

#include "image_description.h"

class AveragePrecisionCalculator {
private:
    enum MatchKind {
        TruePositive, FalsePositive
    };

    /**
     * Here we count all TP and FP matches for all the classes in all the images.
     */
    std::map<int, std::vector<std::pair<double, MatchKind>>> matches;

    std::map<int, int> N;

    double threshold;

    static bool SortBBoxDescend(const DetectedObject &bbox1, const DetectedObject &bbox2);
    static bool SortPairDescend(const std::pair<double, MatchKind> &p1, const std::pair<double, MatchKind> &p2);

public:
    explicit AveragePrecisionCalculator(double _threshold) : threshold(_threshold) { }

    // gt_bboxes -> des
    // bboxes -> det

    void consumeImage(const ImageDescription &detectedObjects, const ImageDescription &desiredObjects);

    std::map<int, double> calculateAveragePrecisionPerClass() const;
};
