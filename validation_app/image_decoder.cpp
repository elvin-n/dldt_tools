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
cv::Size addToBlob(std::string name, int batch_pos, VBlob *blob, const std::vector<VPreprocessingStep> &preprocessingOptions) {
  VShape blobSize = blob->_shape;
  int width = blob->_layout == "NCHW" ? static_cast<int>(blobSize[3]) : static_cast<int>(blobSize[2]);
  int height = blob->_layout == "NCHW" ? static_cast<int>(blobSize[2]) : static_cast<int>(blobSize[1]);
  int channels = blob->_layout == "NCHW" ? static_cast<int>(blobSize[1]) : static_cast<int>(blobSize[3]);
  T *blob_data = static_cast<T *>(blob->_data);
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

  for (const auto &p : preprocessingOptions) {
    if (p.type_ == "resize") {
      Mat resized_image;
      cv::resize(orig_image, resized_image, Size(p.size_, p.size_), 0, 0, INTER_AREA);
      orig_image = resized_image;
    } else if (p.type_ == "crop") {
      size_t cx = 0;
      size_t cy = 0;
      if (p.use_pillow_) {
        cx = orig_image.cols / 2 - width / 2;
        cy = orig_image.rows / 2 - height / 2;
      }
      cv::Rect cropRect(cx, cy, width, height);
      Mat croped_image = orig_image(cropRect);
      orig_image = croped_image;
    } else if (p.type_ == "rgb_to_bgr") {
      // assuming that we have 1CHW, where C should be eq to 3
      if (channels!= 3) {
        throw std::string("Got preprocessing step 'rgb_to_bgr' but input assumes number of channel not eq to 3");
      }
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          if (orig_image.elemSize1() == 1) {
            unsigned char tmp = orig_image.at<cv::Vec3b>(h, w)[0];
            orig_image.at<cv::Vec3b>(h, w)[0] = orig_image.at<cv::Vec3b>(h, w)[2];
            orig_image.at<cv::Vec3b>(h, w)[2] = tmp;
          } else if (orig_image.elemSize1() == 4) {
            float tmp = orig_image.at<cv::Vec3f>(h, w)[0];
            orig_image.at<cv::Vec3f>(h, w)[0] = orig_image.at<cv::Vec3f>(h, w)[2];
            orig_image.at<cv::Vec3f>(h, w)[2] = tmp;
          }
        }
      }
    } else if (p.type_ == "normalization") {
      // create
      Mat tmpImg(orig_image.rows, orig_image.cols, CV_32FC3);
      unsigned char *ucdata = orig_image.data;
      float *ddata = reinterpret_cast<float *>(tmpImg.data);
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          for (int c = 0; c < channels; c++) {
            if (orig_image.elemSize1() == 1) {
              tmpImg.at<cv::Vec3f>(h, w)[c] = orig_image.at<cv::Vec3b>(h, w)[c];
            } else if (orig_image.elemSize1() == 4) {
              tmpImg.at<cv::Vec3f>(h, w)[c] = orig_image.at<cv::Vec3f>(h, w)[c];
            }
            if (p.mean_.size() == 1) {
              tmpImg.at<cv::Vec3f>(h, w)[c] -= p.mean_[0];
            } else if (p.mean_.size() == 3) {
              tmpImg.at<cv::Vec3f>(h, w)[c] -= p.mean_[c];
            }
            if (p.std_.size() == 1) {
              tmpImg.at<cv::Vec3f>(h, w)[c] /= p.std_[0];
            } else if (p.std_.size() == 3) {
              tmpImg.at<cv::Vec3f>(h, w)[c] /= p.std_[c];
            }
          }
        }
      }
      orig_image = tmpImg;
    }
  }

  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int pos = 0;
        if (blob->_layout == "NCHW") {
          pos = batch_pos * channels * width * height + c * width * height + h * width + w;
        } else if (blob->_layout == "NHWC") {
          pos = batch_pos * channels * width * height + h * width * channels + w * channels + c;
        } else {
          throw std::string("Unexpected destination layout during filling of the image");
        }
        blob_data[pos] =
          static_cast<T>(orig_image.at<cv::Vec3f>(h, w)[c]);
      }
    }
  }
  return res;
}

std::map<std::string, cv::Size> convertToBlob(std::vector<std::string> names, int batch_pos, VBlob* blob,
                                              const std::vector<VPreprocessingStep> &preprocessingOptions) {
  if (blob->_data == nullptr) {
    THROW_USER_EXCEPTION(1) << "Blob was not allocated";
  }

  std::function<cv::Size(std::string, int, VBlob *, const std::vector<VPreprocessingStep>&)> add_func;

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

Size ImageDecoder::insertIntoBlob(std::string name, int batch_pos, std::shared_ptr<VBlob> blob, const std::vector<VPreprocessingStep> &preprocessingOptions) {
  return convertToBlob({ name }, batch_pos, blob.get(), preprocessingOptions).at(name);
}
