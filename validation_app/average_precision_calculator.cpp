#include "average_precision_calculator.h"

bool AveragePrecisionCalculator::SortBBoxDescend(const DetectedObject &bbox1, const DetectedObject &bbox2) {
    return bbox1.prob > bbox2.prob;
}

bool AveragePrecisionCalculator::SortPairDescend(const std::pair<double, MatchKind> &p1, const std::pair<double, MatchKind> &p2) {
    return p1.first > p2.first;
}

void AveragePrecisionCalculator::consumeImage(
    const ImageDescription &detectedObjects,
    const ImageDescription &desiredObjects) {
    // Collecting IoU values
    std::vector<bool> visited(desiredObjects.alist.size(), false);
    std::vector<DetectedObject> bboxes{std::begin(detectedObjects.alist), std::end(detectedObjects.alist)};
    std::sort(bboxes.begin(), bboxes.end(), SortBBoxDescend);


    for (auto &&detObj : bboxes) {
        // Searching for the best match to this detection
        // Searching for desired object
        float overlap_max = -1;
        int jmax = -1;
        auto desmax = desiredObjects.alist.end();

        int j = 0;
        for (auto desObj = desiredObjects.alist.begin(); desObj != desiredObjects.alist.end(); desObj++, j++) {
            double iou = DetectedObject::ioU(detObj, *desObj);
            if (iou > overlap_max) {
                overlap_max = static_cast<float>(iou);
                jmax = j;
                desmax = desObj;
            }
        }

        MatchKind mk;
        if (overlap_max >= threshold) {
            if (!desmax->difficult) {
                if (!visited[jmax]) {
                    mk = TruePositive;
                    visited[jmax] = true;
                } else {
                    mk = FalsePositive;
                }
                matches[detObj.objectType].push_back(std::make_pair(detObj.prob, mk));
            }
        } else {
            mk = FalsePositive;
            matches[detObj.objectType].push_back(std::make_pair(detObj.prob, mk));
        }
    }

    for (auto desObj = desiredObjects.alist.begin(); desObj != desiredObjects.alist.end(); desObj++) {
        if (!desObj->difficult) {
            N[desObj->objectType]++;
        }
    }
}

std::map<int, double> AveragePrecisionCalculator::calculateAveragePrecisionPerClass() const {
    /**
     * Precision-to-TP curve per class (a variation of precision-to-recall curve without dividing into N)
     */
    std::map<int, std::map<int, double>> precisionToTP;


    std::map<int, double> res;

    for (auto m : matches) {
        // Sorting
        std::sort(m.second.begin(), m.second.end(), SortPairDescend);

        int clazz = m.first;
        int TP = 0, FP = 0;

        std::vector<double> prec;
        std::vector<double> rec;

        for (auto mm : m.second) {
            // Here we are descending in a probability value
            MatchKind mk = mm.second;
            if (mk == TruePositive) TP++;
            else if (mk == FalsePositive) FP++;

            double precision = static_cast<double>(TP) / (TP + FP);
            double recall = 0;
            if (N.find(clazz) != N.end()) {
                recall = static_cast<double>(TP) / N.at(clazz);
            }

            prec.push_back(precision);
            rec.push_back(recall);
        }

        int num = rec.size();

        // 11point from Caffe
        double ap = 0;
        std::vector<float> max_precs(11, 0.);
        int start_idx = num - 1;
        for (int j = 10; j >= 0; --j) {
            for (int i = start_idx; i >= 0; --i) {
                if (rec[i] < j / 10.) {
                    start_idx = i;
                    if (j > 0) {
                        max_precs[j - 1] = max_precs[j];
                    }
                    break;
                } else {
                    if (max_precs[j] < prec[i]) {
                        max_precs[j] = static_cast<float>(prec[i]);
                    }
                }
            }
        }
        for (int j = 10; j >= 0; --j) {
            ap += max_precs[j] / 11;
        }
        res[clazz] = ap;
    }

    return res;
}
