// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <vector>

#include "PreprocessingOptions.hpp"
#include "backend.hpp"
#include "ValidationConfig.h"

using namespace cv;

class ImageDecoder {
public:
    /**
     * @brief Insert image data to blob at specified batch position.
     *        Does no checks if blob has sufficient space
     * @param name - image file name
     * @param batch_pos - batch position image should be loaded to
     * @param blob - blob object to load image data to
     * @return original image size
     */
     Size insertIntoBlob(std::string name, int batch_pos, std::shared_ptr<VBlob> blob,
                         const std::vector<VPreprocessingStep> &preprocessingOptions);
};
