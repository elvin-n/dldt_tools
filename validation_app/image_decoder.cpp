// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>

#include "image_decoder.hpp"
#include "user_exception.hpp"
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

int getLoadModeForChannels(int channels, int base) {
    switch (channels) {
    case 1:
        return base | IMREAD_GRAYSCALE;
    case 3:
        return base | IMREAD_COLOR;
    }
    return base | IMREAD_UNCHANGED;
}

template <class T>
cv::Size addToBlob(std::string name, int batch_pos, VBlob* blob, PreprocessingOptions preprocessingOptions) {
    VShape blobSize = blob->_shape;
    int width = blob->_layout == "NCHW" ? static_cast<int>(blobSize[3]) : static_cast<int>(blobSize[2]);
    int height = blob->_layout == "NCHW" ? static_cast<int>(blobSize[2]) : static_cast<int>(blobSize[1]);
    int channels = blob->_layout == "NCHW" ? static_cast<int>(blobSize[1]) : static_cast<int>(blobSize[3]);
    T* blob_data = static_cast<T*>(blob->_data);
    Mat orig_image, result_image;
    int loadMode = getLoadModeForChannels(channels, 0);

    std::string tryName = name;

    // TODO This is a dirty hack to support VOC2007 (where no file extension is put into annotation).
    //      Rewrite.
    if (name.find('.') == std::string::npos) tryName = name + ".JPEG";

    orig_image = imread(tryName, loadMode);

    if (orig_image.empty()) {
        THROW_USER_EXCEPTION(1) << "Cannot open image file: " << tryName;
    }

    // Preprocessing the image
    Size res = orig_image.size();

    if (preprocessingOptions.resizeCropPolicy == ResizeCropPolicy::Resize) {
        cv::resize(orig_image, result_image, Size(width, height));
    } else if (preprocessingOptions.resizeCropPolicy == ResizeCropPolicy::ResizeThenCrop) {
        Mat resized_image;

        cv::resize(orig_image, resized_image, Size(preprocessingOptions.resizeBeforeCropX, preprocessingOptions.resizeBeforeCropY));

        size_t cx = preprocessingOptions.resizeBeforeCropX / 2;
        size_t cy = preprocessingOptions.resizeBeforeCropY / 2;

        cv::Rect cropRect(cx - width / 2, cy - height / 2, width, height);
        result_image = resized_image(cropRect);
    } else if (preprocessingOptions.resizeCropPolicy == ResizeCropPolicy::DoNothing) {
        // No image preprocessing to be done here
        result_image = orig_image;
    } else {
        THROW_USER_EXCEPTION(1) << "Unsupported ResizeCropPolicy value";
    }

    float scaleFactor = preprocessingOptions.scaleValuesTo01 ? 255.0f : 1.0f;

    if (blob->_layout == "NCHW") {
        for (int c = 0; c < channels; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    blob_data[batch_pos * channels * width * height + c * width * height + h * width + w] =
                        static_cast<T>(result_image.at<cv::Vec3b>(h, w)[c] / scaleFactor);
                }
            }
        }
        if (blob->_colourFormat == "RGB") {

          // TODO: to implement different way
          // reverse channels & do normalization
          for (int c = 0; c < channels; c++) {
            float sub = 0.f;
            float div = 1.f;
            if (c == 2) {
                sub = 123.6f;
                div = 58.395f;
            }
            if (c == 1) {
                sub = 116.3f;
                div = 57.12f;
            }
            if (c == 0) {
                sub = 103.5f;
                div = 57.375f;
            }
            for (int h = 0; h < height; h++) {
              for (int w = 0; w < width; w++) {
                size_t idx = batch_pos * channels * width * height + c * width * height + h * width + w;
                blob_data[idx] = (blob_data[idx] - sub) / div;
              }
            }
          }
          for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
              size_t idx0 = batch_pos * channels * width * height + 0 * width * height + h * width + w;
              size_t idx2 = batch_pos * channels * width * height + 2 * width * height + h * width + w;
              float tmp = blob_data[idx0];
              blob_data[idx0] = blob_data[idx2];
              blob_data[idx2] = tmp;
            }
          }
        }
    } else if (blob->_layout == "NHWC") {
        size_t nielements = width * height * channels;
        if (blob->_precision == FP32) {
            for (size_t i = 0; i < nielements; i++) {
                blob_data[i] = (static_cast<float>(result_image.data[i]) - 128.f )/ 127.f;
            }
            if (blob->_colourFormat == "RGB") {
                for (size_t i = 0; i < nielements; i += channels) {
                    float tmp = blob_data[i];
                    blob_data[i] = blob_data[i + 2];
                    blob_data[i + 2] = tmp;
                }
            }
        } else if (blob->_precision == U8) {
            for (size_t i = 0; i < nielements; i += 3) {
                blob_data[i] = result_image.data[i + 2];
                blob_data[i+1] = result_image.data[i + 1];
                blob_data[i+2] = result_image.data[i];
            }
        }
    }

    return res;
}

std::map<std::string, cv::Size> convertToBlob(std::vector<std::string> names, int batch_pos, VBlob* blob, PreprocessingOptions preprocessingOptions) {
    if (blob->_data == nullptr) {
        THROW_USER_EXCEPTION(1) << "Blob was not allocated";
    }

    std::function<cv::Size(std::string, int, VBlob*, PreprocessingOptions)> add_func;

    switch (blob->_precision) {
    case FP32:
        add_func = &addToBlob<float>;
        break;
    case FP16:
    case Q78:
    case I16:
    case U16:
        add_func = &addToBlob<short>;
        break;
    default:
        add_func = &addToBlob<uint8_t>;
    }

    std::map<std::string, Size> res;
    for (size_t b = 0; b < names.size(); b++) {
        std::string name = names[b];
        Size orig_size = add_func(name, batch_pos + b, blob, preprocessingOptions);
        res.insert(std::pair<std::string, Size>(name, orig_size));
    }

    return res;
}

Size ImageDecoder::loadToBlob(std::string name, std::shared_ptr<VBlob> blob, PreprocessingOptions preprocessingOptions) {
    std::vector<std::string> names = { name };
    return loadToBlob(names, blob, preprocessingOptions).at(name);
}

std::map<std::string, cv::Size> ImageDecoder::loadToBlob(std::vector<std::string> names, std::shared_ptr<VBlob> blob, PreprocessingOptions preprocessingOptions) {
    return convertToBlob(names, 0, blob.get(), preprocessingOptions);
}

Size ImageDecoder::insertIntoBlob(std::string name, int batch_pos, std::shared_ptr<VBlob> blob, PreprocessingOptions preprocessingOptions) {
    return convertToBlob({ name }, batch_pos, blob.get(), preprocessingOptions).at(name);
}
