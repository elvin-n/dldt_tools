// Copyright (C) 2020 The DLDT Tools Authors
// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <opencv2/opencv.hpp>

#include <vector>
#include <memory>
#include <string>
#include <iostream>

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"
#if defined(__ANDROID__) && (defined(__arm__) || defined(__aarch64__))
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#endif

/*
Fore delegate?
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

*/

#ifdef UNICODE
#include <tchar.h>
#endif

#include <chrono>

#ifndef UNICODE
#define tcout std::cout
#define _T(STR) STR
#else
#define tcout std::wcout
#endif

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

#ifndef UNICODE
int main(int argc, char *argv[]) {
#else
int wmain(int argc, wchar_t *argv[]) {
#endif
    try {
        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (argc != 4) {
            tcout << _T("Usage : ./tflite_hello_classification <path_to_model> <path_to_image> <device_name>") << std::endl;
            return EXIT_FAILURE;
        }

        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        const std::string device_name{argv[3]};

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Read TFLite IR (.tflite file) ------------
        /// NOTE: The current API requires that a FlatBufferModel instance be kept alive
        /// by the client as long as it is in use by any dependent Interpreter
        Time::time_point time1 = Time::now();
        std::unique_ptr<tflite::FlatBufferModel> model;
        model = tflite::FlatBufferModel::BuildFromFile(input_model.c_str());
        if (!model) {
          std::cerr << "Error while opening the tflite file." << std::endl;
          return EXIT_FAILURE;
        }

        Time::time_point time2 = Time::now();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Loading model to the device ------------------------------------------
        std::unique_ptr<tflite::Interpreter> interpreter;
        tflite::ops::builtin::BuiltinOpResolver resolver;

        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
          std::cerr << "Failed to construct interpreter." << std::endl;
          return EXIT_FAILURE;
        }
        // interpreter->UseNNAPI(s->old_accel);
        // interpreter->SetAllowFp16PrecisionForFp32(true);

        // forcedly set number of threads as 1 for a while
        interpreter->SetNumThreads(1);

        // there is offloading part
        TfLiteDelegate* delegate = nullptr;
        if (device_name == "GPU") {
/*        #if defined(__ANDROID__)
          TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
          gpu_opts.inference_preference =
              TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
          gpu_opts.inference_priority1 =
              s->allow_fp16 ? TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY
                            : TFLITE_GPU_INFERENCE_PRIORITY_MAX_PRECISION;
          auto delegate = evaluation::CreateGPUDelegate(&gpu_opts);
        #else
          auto delegate = evaluation::CreateGPUDelegate();
        #endif

          if (!delegate) {
            std::cerr << "GPU acceleration is unsupported on this platform.";
            return -1;
          } else {
            delegates.emplace("GPU", std::move(delegate));
          }
*/
        } else if (device_name == "DSP") {
#if defined(__ANDROID__) && (defined(__arm__) || defined(__aarch64__))
          TfLiteHexagonInit();
          TfLiteHexagonDelegateOptions options({0});
          delegate = TfLiteHexagonDelegateCreate(&options);
          if (!delegate) {
            std::cerr << "Hexagon acceleration is unsupported on this platform.";
            TfLiteHexagonTearDown();
            return -1;
          } /*else {
            delegates.emplace("Hexagon", std::move(delegate));
          }*/
#endif
        } else if (device_name == "CPU") {
            // default execution unit, no additional actions are required
        } else {
           std::cerr << "The device name is not valid. Please select CPU/DSP/GPU." << std::endl;
           return -1;
        }

        if (delegate) {
          if (interpreter->ModifyGraphWithDelegate(delegate) !=
              kTfLiteOk) {
            std::cerr << "Failed to apply TFLite delegate." << std::endl;
            return -1;
          }
          std::cout << "Applied deligation successfully" << std::endl;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Prepare input --------------------------------------------------------
        Time::time_point time4 = Time::now();
        int input = interpreter->inputs()[0];
        const std::vector<int> inputs = interpreter->inputs();
        if (interpreter->AllocateTensors() != kTfLiteOk) {
          std::cerr << "Failed to allocate tensors!" << std::endl;
          return EXIT_FAILURE;
        }

        // get input dimension from the input tensor metadata
        // assuming one input only
        TfLiteIntArray* dims = interpreter->tensor(input)->dims;
        int wanted_height = dims->data[1];
        int wanted_width = dims->data[2];
        int wanted_channels = dims->data[3];
        cv::Mat image = cv::imread(input_image_path);
        cv::Mat resized_image(image);
        cv::resize(image, resized_image, cv::Size(wanted_width, wanted_height));

        switch (interpreter->tensor(input)->type) {
          case kTfLiteFloat32:
            {
              // Follow copying of image is not optimal since it happens in two passes
              float *tf = interpreter->typed_tensor<float>(input);
              size_t nielements = wanted_height * wanted_width * wanted_channels;
              for (size_t i = 0; i < nielements; i++) {
                  tf[i] = (static_cast<float>(resized_image.data[i]) - 127.5) / 127.5;
              }

              // reorder from BGR to RGB:
              // snpe-tensorflow-to-dlc has follow parameters --input_encoding "input" bgr --input_type "input" image
              // unfortunately they do not work
              // if they worked, I would not have such code here
              for (size_t i = 0; i < nielements; i += wanted_channels) {
                  float tmp = tf[i];
                  tf[i] = tf[i + 2];
                  tf[i + 2] = tmp;
              }
            }
            break;
          case kTfLiteUInt8:
            {
              uint8_t *tf = interpreter->typed_tensor<uint8_t>(input);
              // reorder from BGR to RGB:
              // snpe-tensorflow-to-dlc has follow parameters --input_encoding "input" bgr --input_type "input" image
              // unfortunately they do not work
              // if they worked, I would not have such code here
              size_t nielements = wanted_height * wanted_width * wanted_channels;
              for (size_t i = 0; i < nielements; i += wanted_channels) {
                  tf[i] = resized_image.data[i + 2];
                  tf[i+1] = resized_image.data[i + 1];
                  tf[i+2] = resized_image.data[i];
              }
            }
            break;
          default:
            std::cerr << "cannot handle model input type!" <<
            interpreter->tensor(input)->type << std::endl;
            return EXIT_FAILURE;
        }

        Time::time_point time7 = Time::now();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        if (interpreter->Invoke() != kTfLiteOk) {
            std::cerr << "Error while executing the network." << std::endl;
            return EXIT_FAILURE;
        }
        Time::time_point time8 = Time::now();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Process output ------------------------------------------------------
        int output = interpreter->outputs()[0];
        TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
        // assume output dims to be something like (1, 1, ... ,size)
        auto output_size = output_dims->data[output_dims->size - 1];
        switch (interpreter->tensor(output)->type) {
          case kTfLiteFloat32:
            {
              float* oData = interpreter->typed_output_tensor<float>(0);
              std::map<float, int> ordered;
              for (size_t j = 0; j < output_size; j++) {
                  ordered[oData[j]] = j;
              }

              int s = 0;
              std::cout << std::fixed << std::setprecision(5);
              for (auto it = ordered.crbegin(); it != ordered.crend() && s < 5; it++, s++) {
                  std::cout << it->second << "    " << it->first << std::endl;
              }
            }

            break;
          case kTfLiteUInt8:
            {
              uint8_t* oData = interpreter->typed_output_tensor<uint8_t>(0);
              std::map<float, int> ordered;
              for (size_t j = 0; j < output_size; j++) {
                  float value = oData[j] + (static_cast<float>(j) / 10000.f);
                  value = value / 256.f;
                  ordered[value] = j;
              }

              int s = 0;
              std::cout << std::fixed << std::setprecision(5);
              for (auto it = ordered.crbegin(); it != ordered.crend() && s < 5; it++, s++) {
                  std::cout << it->second << "    " << it->first << std::endl;
              }
            }
            break;
          default:
            std::cerr << "cannot handle network output type " <<
              interpreter->tensor(output)->type << std::endl;
            return EXIT_FAILURE;
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Print timing info ----------------------------------------------------
/*        Time::time_point time9 = Time::now();
        std::cout << "Read model: " << std::chrono::duration_cast<ns>(time2 - time1).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Load of the model to SNPE: " << std::chrono::duration_cast<ns>(time3 - time2).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Load of the picture: " << std::chrono::duration_cast<ns>(time4 - time3).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Resize of the picture: " << std::chrono::duration_cast<ns>(time5 - time4).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Copying data from U8 to float array for input: " << std::chrono::duration_cast<ns>(time6 - time5).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Create of the ITensor: " << std::chrono::duration_cast<ns>(time7 - time6).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Inference: " << std::chrono::duration_cast<ns>(time8 - time7).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Post processing of output results: " << std::chrono::duration_cast<ns>(time9- time8).count() * 0.000001 << " ms" << std::endl;
*/
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
