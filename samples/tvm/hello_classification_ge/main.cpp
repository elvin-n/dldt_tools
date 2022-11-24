// Copyright (C) 2020 group of contributors
// SPDX-License-Identifier: Apache-2.0
//


#include <opencv2/opencv.hpp>

#include <vector>
#include <memory>
#include <string>
#include <iostream>

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

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
            tcout << _T("Usage : ./tvm_hello_classification <path_to_model> <path_to_image> <device>") << std::endl;
            return EXIT_FAILURE;
        }

        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        const std::string device{argv[3]};

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Read Model ------------
        // load in the library
        Time::time_point time1 = Time::now();
        DLDevice ctx;
        if (device == "CPU") {
            ctx = {kDLCPU, 0};
        } else if (device == "Metal") {
            ctx = {kDLMetal, 0};
        } else if (device == "OpenCL") {
            ctx = {kDLOpenCL, 0};
        } else {
            tcout << _T("Expecting <device> to be eq to 'CPU' or 'Metal'") << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Try to load model" << std::endl;
        tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(input_model);
        std::cout << "loaded model" << std::endl;
        // create the graph runtime module
        tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
        tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
        tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
        tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
        tvm::runtime::PackedFunc run = gmod.GetFunction("run");
        std::cout << "Get all packed functions" << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Prepare input --------------------------------------------------------

        // TODO(amalyshe): To identify if we can take shapes from TVM functions
        Time::time_point time2 = Time::now();
        size_t width = 224, height = 224;
        size_t nielements = 3 * width * height;

        cv::Mat image = cv::imread(input_image_path);
        std::cout << "Image is read" << std::endl;
        cv::Mat resized_image(image);
        cv::resize(image, resized_image, cv::Size(width, height));

        // Use the C++ API
        DLDevice ctxCPU{kDLCPU, 0};
        tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 224, 224, 3}, DLDataType{kDLFloat, 32, 1}, ctxCPU);

        // normalization parameters are for ONNX Mobilenet v2
        for (int i = 0; i < nielements; i += 3) {
            static_cast<float*>(x->data)[i] = (resized_image.data[i+2] - 123.6f) / 58.395f;
            static_cast<float*>(x->data)[i+1] = (resized_image.data[i+1] - 116.3f) / 57.12f;
            static_cast<float*>(x->data)[i+2] = (resized_image.data[i] - 103.5f) / 57.375f;
        }

        // reorder from NHWC to NCHW
        float *tmpBuffer = (float *)malloc(nielements * 4);
        for (size_t sd = 0; sd < width * height; sd++) {
            for (size_t channel = 0; channel < 3; channel++) {
                size_t pidx = channel * width * height + sd;
                tmpBuffer[pidx] = static_cast<float*>(x->data)[sd * 3 + channel];
            }
        }

        for (size_t j = 0; j < nielements; j++) {
            static_cast<float*>(x->data)[j] = tmpBuffer[j];
        }

        free(tmpBuffer);
        Time::time_point time3 = Time::now();

        // set the right input
        set_input(0, x);
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Do inference --------------------------------------------------------
        std::cout << "Do inference" << std::endl;
        // warm up
        run();

        Time::time_point time4 = Time::now();
        run();
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
        Time::time_point time5 = Time::now();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Process output ------------------------------------------------------
        tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 1000}, DLDataType{kDLFloat, 32, 1}, ctxCPU);

        tvm::runtime::NDArray y2 = get_output(0);
        y.CopyFrom(y2);
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);

        // Looking for the Top5 values and print them in format
        // class_id     probability/value of the latest tensor in case of softmax absence
        const float* oData = static_cast<float*>(y->data);
        std::map<float, int> ordered;
        for (size_t j = 0; j < 1000; j++) {
            ordered[oData[j]] = j;
        }

        int s = 0;
        std::cout << std::fixed << std::setprecision(5);
        for (auto it = ordered.crbegin(); it != ordered.crend() && s < 5; it++, s++) {
            std::cout << it->second << "    " << it->first << std::endl;
        }
        Time::time_point time6 = Time::now();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Print timing info ----------------------------------------------------
        std::cout << "Load of the model: " << std::chrono::duration_cast<ns>(time2 - time1).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Load of the picture, preprocessing (OpenCV): " << std::chrono::duration_cast<ns>(time3 - time2).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Load of input data to model: " << std::chrono::duration_cast<ns>(time4 - time3).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Inference: " << std::chrono::duration_cast<ns>(time5 - time4).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Post processing of output results: " << std::chrono::duration_cast<ns>(time6- time5).count() * 0.000001 << " ms" << std::endl;
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
