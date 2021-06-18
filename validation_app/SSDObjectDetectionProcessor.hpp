// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <list>
#include <utility>

#include "ObjectDetectionProcessor.hpp"

using namespace std;

class SSDObjectDetectionProcessor : public ObjectDetectionProcessor {
protected:
    std::map<std::string, std::list<DetectedObject>> processResult(std::vector<std::string> files) {
        std::map<std::string, std::list<DetectedObject>> detectedObjects;

        std::string firstOutputName = this->_outputInfo.begin()->first;
        const auto detectionOutArray = _backend->getBlob(firstOutputName);
        const float *box = static_cast<const float *>(detectionOutArray->_data);

        if (_outputInfo.size() == 1) {
            VShape outputDims = _outputInfo.begin()->second._shape;

            const size_t maxProposalCount = outputDims[2];
            const size_t objectSize = outputDims[3];

            for (size_t b = 0; b < batch; b++) {
                string fn = files[b];
                std::list<DetectedObject> dr = std::list<DetectedObject>();
                detectedObjects.insert(std::pair<std::string, std::list<DetectedObject>>(fn, dr));
            }

            for (size_t i = 0; i < maxProposalCount; i++) {
                float image_id = box[i * objectSize + 0];
                float label = box[i * objectSize + 1];
                float confidence = box[i * objectSize + 2];
                float xmin = box[i * objectSize + 3] * inputDims[3];
                float ymin = box[i * objectSize + 4] * inputDims[2];
                float xmax = box[i * objectSize + 5] * inputDims[3];
                float ymax = box[i * objectSize + 6] * inputDims[2];

                if (image_id < 0 /* better than check == -1 */) {
                    break;  // Finish
                }

                detectedObjects[files[static_cast<size_t>(image_id)]].push_back(
                    DetectedObject(static_cast<int>(label), xmin, ymin, xmax, ymax, confidence));
            }
        } else if (_outputInfo.size() == 4) {
            std::string scoresName = "Postprocessor/BatchMultiClassNonMaxSuppression_scores";
            std::string classesName = "detection_classes:0";
            std::string boxesName = "Postprocessor/BatchMultiClassNonMaxSuppression_boxes";
            if (_outputInfo.find(scoresName) == _outputInfo.end() ||
                _outputInfo.find(classesName) == _outputInfo.end() ||
                _outputInfo.find(boxesName) == _outputInfo.end()) {
                // TFLite
                scoresName = "TFLite_Detection_PostProcess:2";
                classesName = "TFLite_Detection_PostProcess:1";
                boxesName = "TFLite_Detection_PostProcess";
                if (_outputInfo.find(scoresName) == _outputInfo.end() ||
                    _outputInfo.find(classesName) == _outputInfo.end() ||
                    _outputInfo.find(boxesName) == _outputInfo.end()) {
                    THROW_USER_EXCEPTION(1) << "We expect model converted by SNPE or TFLite with certain outputs, but cannot get them";
                }
            }

            const auto scoresBlob = _backend->getBlob(scoresName);
            const auto classesBlob = _backend->getBlob(classesName);
            const auto boxesBlob = _backend->getBlob(boxesName);
            const float *oScores = static_cast<const float *>(scoresBlob->_data);
            const float *oClasses = static_cast<const float *>(classesBlob->_data);
            const float *oBoxes = static_cast<const float *>(boxesBlob->_data);


            for (size_t curProposal = 0; curProposal < scoresBlob->_shape[1]; curProposal++) {
                float confidence = oScores[curProposal];
                float label = static_cast<int>(oClasses[curProposal]);
                if (classesName == "TFLite_Detection_PostProcess:1") {
                 label += 1;
                }
                // boxes have follow layout top, left, bottom, right
                // according to this link: https://www.tensorflow.org/lite/models/object_detection/overview
                auto ymin = static_cast<int>(oBoxes[4 * curProposal] * inputDims[1]);
                auto xmin = static_cast<int>(oBoxes[4 * curProposal + 1] * inputDims[2]);
                auto ymax = static_cast<int>(oBoxes[4 * curProposal + 2] * inputDims[1]);
                auto xmax = static_cast<int>(oBoxes[4 * curProposal + 3] * inputDims[2]);

                if (ymin == 0.f && xmin == 0.f && ymax == 0.f && xmax == 0.f && confidence == 0.f)
                    break;
                size_t image_id = 0; // no support for batch so far
                detectedObjects[files[static_cast<size_t>(image_id)]].push_back(
                    DetectedObject(static_cast<int>(label), xmin, ymin, xmax, ymax, confidence));
            }
        } else {
            THROW_USER_EXCEPTION(1) << "This app accepts networks having only one output";
        }

        return detectedObjects;
    }

public:
  SSDObjectDetectionProcessor(Backend *backend,
                              const VLauncher *launcher,
                              const std::vector<std::string> &outputs,
                              const std::string &flags_i,
                              const std::string &subdir,
                              int flags_b,
                              double threshold,
                              CsvDumper &dumper,
                              const std::string &flags_a,
                              const std::string &classes_list_file,
                              const VDataset *dataset) :
    ObjectDetectionProcessor(backend, launcher, outputs, flags_i, subdir, flags_b,
                             threshold, dumper, flags_a, classes_list_file, dataset, true) { }
};
