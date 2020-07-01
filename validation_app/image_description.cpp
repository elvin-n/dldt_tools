#include "image_description.h"

float DetectedObject::ioU(const DetectedObject &detectedObject1_, const DetectedObject &detectedObject2_) {
    // Add small space to eliminate empty squares
    float epsilon = 0;  // 1e-5f;

    DetectedObject detectedObject1(detectedObject1_.objectType,
                                    (detectedObject1_.xmin - epsilon),
                                    (detectedObject1_.ymin - epsilon),
                                    (detectedObject1_.xmax - epsilon),
                                    (detectedObject1_.ymax - epsilon), detectedObject1_.prob);
    DetectedObject detectedObject2(detectedObject2_.objectType,
                                    (detectedObject2_.xmin + epsilon),
                                    (detectedObject2_.ymin + epsilon),
                                    (detectedObject2_.xmax),
                                    (detectedObject2_.ymax), detectedObject2_.prob);

    if (detectedObject1.objectType != detectedObject2.objectType) {
        // objects are different, so the result is 0
        return 0.0f;
    }

    if (detectedObject1.xmax < detectedObject1.xmin) return 0.0;
    if (detectedObject1.ymax < detectedObject1.ymin) return 0.0;
    if (detectedObject2.xmax < detectedObject2.xmin) return 0.0;
    if (detectedObject2.ymax < detectedObject2.ymin) return 0.0;


    float xmin = (std::max)(detectedObject1.xmin, detectedObject2.xmin);
    float ymin = (std::max)(detectedObject1.ymin, detectedObject2.ymin);
    float xmax = (std::min)(detectedObject1.xmax, detectedObject2.xmax);
    float ymax = (std::min)(detectedObject1.ymax, detectedObject2.ymax);

    // Caffe adds 1 to every length if the box isn't normalized. So do we...
    float addendum;
    if (xmax > 1 || ymax > 1) addendum = 1;
    else addendum = 0;

    // intersection
    float intr;
    if ((xmax >= xmin) && (ymax >= ymin)) {
        intr = (addendum + xmax - xmin) * (addendum + ymax - ymin);
    } else {
        intr = 0.0f;
    }

    // union
    float square1 = (addendum + detectedObject1.xmax - detectedObject1.xmin) * (addendum + detectedObject1.ymax - detectedObject1.ymin);
    float square2 = (addendum + detectedObject2.xmax - detectedObject2.xmin) * (addendum + detectedObject2.ymax - detectedObject2.ymin);

    float unn = square1 + square2 - intr;

    return static_cast<float>(intr) / unn;
}

DetectedObject DetectedObject::scale(float scale_x, float scale_y) const {
    return DetectedObject(objectType, xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y, prob, difficult);
}

// ---------------------------------------------------------------------------------------------------------

float ImageDescription::ioUMultiple(const ImageDescription &detectedObjects, const ImageDescription &desiredObjects) {
    const ImageDescription *detectedObjectsSmall, *detectedObjectsBig;
    bool check_probs = desiredObjects.check_probs;

    if (detectedObjects.alist.size() < desiredObjects.alist.size()) {
        detectedObjectsSmall = &detectedObjects;
        detectedObjectsBig = &desiredObjects;
    } else {
        detectedObjectsSmall = &desiredObjects;
        detectedObjectsBig = &detectedObjects;
    }

    std::list<DetectedObject> doS = detectedObjectsSmall->alist;
    std::list<DetectedObject> doB = detectedObjectsBig->alist;

    float fullScore = 0.0f;
    while (doS.size() > 0) {
        float score = 0.0f;
        std::list<DetectedObject>::iterator bestJ = doB.end();
        for (auto j = doB.begin(); j != doB.end(); j++) {
            float curscore = DetectedObject::ioU(*doS.begin(), *j);
            if (score < curscore) {
                score = curscore;
                bestJ = j;
            }
        }

        float coeff = 1.0;
        if (check_probs) {
            if (bestJ != doB.end()) {
                float mn = std::min((*bestJ).prob, (*doS.begin()).prob);
                float mx = std::max((*bestJ).prob, (*doS.begin()).prob);

                coeff = mn / mx;
            }
        }

        doS.pop_front();
        if (bestJ != doB.end()) doB.erase(bestJ);
        fullScore += coeff * score;
    }
    fullScore /= detectedObjectsBig->alist.size();


    return fullScore;
}

ImageDescription ImageDescription::scale(float scale_x, float scale_y) const {
    std::list<DetectedObject> slist;
    for (auto &dob : alist) {
        slist.push_back(dob.scale(scale_x, scale_y));
    }
    return ImageDescription(slist, check_probs);
}
