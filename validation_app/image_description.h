#pragma once

#include <string>
#include <algorithm>
#include <map>
#include <vector>
#include <list>
#include <functional>
#include <chrono>
#include <fstream>

/**
 * @brief This class represents an object that is found by an object detection net
 */
class DetectedObject {
public:
    int objectType;
    float xmin, xmax, ymin, ymax, prob;
    bool difficult;

    DetectedObject(int _objectType, float _xmin, float _ymin, float _xmax, float _ymax, float _prob, bool _difficult = false)
        : objectType(_objectType), xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax), prob(_prob), difficult(_difficult) {
    }

    DetectedObject(const DetectedObject &other) = default;

    static float ioU(const DetectedObject &detectedObject1_, const DetectedObject &detectedObject2_);

    DetectedObject scale(float scale_x, float scale_y) const;
};

class ImageDescription {
public:
    const std::list<DetectedObject> alist;
    const bool check_probs;

    explicit ImageDescription(const std::list<DetectedObject> &_alist, bool _check_probs = false)
        : alist(_alist), check_probs(_check_probs) {
    }

    static float ioUMultiple(const ImageDescription &detectedObjects, const ImageDescription &desiredObjects);

    ImageDescription scale(float scale_x, float scale_y) const;
};
