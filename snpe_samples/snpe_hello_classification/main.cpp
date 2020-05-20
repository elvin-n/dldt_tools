// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <opencv2/opencv.hpp>

#include <DlContainer/IDlContainer.hpp>
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/UDLFunc.hpp"
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>

#include <vector>
#include <memory>
#include <string>
#include <iostream>

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
            tcout << _T("Usage : ./snpe_hello_classification <path_to_model> <path_to_image> <device_name>") << std::endl;
            return EXIT_FAILURE;
        }

        const std::string input_model{argv[1]};
        const std::string input_image_path{argv[2]};
        const std::string device_name{argv[3]};

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Read IR Generated by SNPE tools (.dl file) ------------
        Time::time_point time1 = Time::now();
        std::unique_ptr<zdl::DlContainer::IDlContainer> container;
        container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(input_model.c_str()));
        if (container == nullptr) {
           std::cerr << "Error while opening the container file." << std::endl;
           return EXIT_FAILURE;
        }
        Time::time_point time2 = Time::now();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Loading model to the device ------------------------------------------
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
        if (device_name == "GPU") {
            runtime = zdl::DlSystem::Runtime_t::GPU;
        } else if (device_name == "DSP") {
            runtime = zdl::DlSystem::Runtime_t::DSP;
        } else if (device_name == "CPU") {
            runtime = zdl::DlSystem::Runtime_t::CPU;
        } else {
           std::cerr << "The device is not valid. Set default CPU runtime." << std::endl;
           runtime = zdl::DlSystem::Runtime_t::CPU;
        }

        std::unique_ptr<zdl::SNPE::SNPE> snpe;
        zdl::DlSystem::RuntimeList runtimeList;

        zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

        if(runtimeList.empty()) {
            runtimeList.add(runtime);
        }

        snpe = snpeBuilder.setRuntimeProcessorOrder(runtimeList)
        // .setUseUserSuppliedBuffers(true) TODO (amalyshe) to clarify what this option means
           .build();

        Time::time_point time3 = Time::now();
        if (snpe == nullptr) {
           std::cerr << "Error during creation of SNPE object." << std::endl;
           return EXIT_FAILURE;
        }

        // TODO(amalyshe) add logger which would allow to review SNPE work through snpe-diagview

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Prepare input --------------------------------------------------------
        zdl::DlSystem::UserBufferMap inputMap, outputMap;
        zdl::DlSystem::TensorMap inputTensorMap, outputTensorMap;
        std::vector<std::unique_ptr<zdl::DlSystem::IUserBuffer>> snpeUserBackedInputBuffers, snpeUserBackedOutputBuffers;
        std::unordered_map<std::string, std::vector<uint8_t>> applicationOutputBuffers;

        zdl::DlSystem::TensorShape iTShape = snpe->getInputDimensions();
        // verification of input dimensions number - should be 4, NHWC
        if (iTShape.rank() != 4) {
            std::cerr << "Input layer expect data having " << iTShape.rank() << " dimensions shape, while we expect 4" << std::endl;
            return EXIT_FAILURE;
        }

        // Note that nput dimensions are embedded into DLC model during conversion,
        // but in some cases can be overridden via SNPEBuilder::setInputDimensions()
        // (see description in C++ API) at SNPE object creation/build time.
        //
        // FOR SSD:  Due to PriorBox layer folding in the model converter, input/network
        // resizing is not possible.

        const zdl::DlSystem::Dimension *shapes = iTShape.getDimensions();
        std::cout << "Input Shape (" << shapes[0] << "," << shapes[1] << "," << shapes[2] << "," << shapes[3] << ")" << std::endl;

        cv::Mat image = cv::imread(input_image_path);
        std::cout << "Image parameters: Channels=" << image.channels() << ", width=" << image.cols <<
            ", height=" << image.rows << std::endl;

        Time::time_point time4 = Time::now();
        cv::Mat resized_image(image);
        cv::resize(image, resized_image, cv::Size(shapes[2], shapes[1]));
        std::cout << "Resized image parameters: Channels=" << resized_image.channels()<< ", width=" << resized_image.cols <<
        ", height=" << resized_image.rows << std::endl;
        Time::time_point time5 = Time::now();
        std::unique_ptr<zdl::DlSystem::ITensor> inputTensor =
            zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(snpe->getInputDimensions());

        Time::time_point time6 = Time::now();
        zdl::DlSystem::ITensor *t = inputTensor.get();
        if (!inputTensor.get()) {
            std::cerr << "Could not create SNPE input tensor" << std::endl;
            return EXIT_FAILURE;
        }


        // Follow copying of image is not optimal since it happens in two passes
        float *tf = reinterpret_cast<float *>(&(*inputTensor->begin()));
        size_t nielements = shapes[1] * shapes[2] * shapes[3];
        for (size_t i = 0; i < nielements; i++) {
            tf[i] = static_cast<float>(resized_image.data[i]) / 255.f;
        }

        // reorder from BGR to RGB:
        // snpe-tensorflow-to-dlc has follow parameters --input_encoding "input" bgr --input_type "input" image
        // unfortunately they do not work
        // if they worked, I would not have such code here
        for (size_t i = 0; i < nielements; i += shapes[3]) {
            float tmp = tf[i];
            tf[i] = tf[i + 2];
            tf[i + 2] = tmp;
        }

        Time::time_point time7 = Time::now();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        bool execStatus = snpe->execute(t, outputTensorMap);
        Time::time_point time8 = Time::now();
        // Save the execution results only if successful
        if (!execStatus) {
            std::cerr << "Error while executing the network." << std::endl;
            return EXIT_FAILURE;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Process output ------------------------------------------------------
        zdl::DlSystem::StringList outNames = outputTensorMap.getTensorNames();
        if (outNames.size() != 1) {
            std::cerr << "Output String list has " << outNames.size() << " elements, while one is expected" << std::endl;
        }

        // Looking for the Top5 values and print them in format
        // class_id     probability/value of the latest tensor in case of softmax absence
        zdl::DlSystem::ITensor *outTensor = outputTensorMap.getTensor(outNames.at(0));
        const float* oData = reinterpret_cast<float*>(&(*outTensor->begin()));
        std::map<float, int> ordered;
        for (size_t j = 0; j < outTensor->getSize(); j++) {
            ordered[oData[j]] = j;
        }

        int s = 0;
        std::cout << std::fixed << std::setprecision(5);
        for (auto it = ordered.crbegin(); it != ordered.crend() && s < 5; it++, s++) {
            std::cout << it->second << "    " << it->first << std::endl;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Print timing info ----------------------------------------------------
        Time::time_point time9 = Time::now();
        std::cout << "Read model: " << std::chrono::duration_cast<ns>(time2 - time1).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Load of the model to SNPE: " << std::chrono::duration_cast<ns>(time3 - time2).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Load of the picture: " << std::chrono::duration_cast<ns>(time4 - time3).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Resize of the picture: " << std::chrono::duration_cast<ns>(time5 - time4).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Copying data from U8 to float array for input: " << std::chrono::duration_cast<ns>(time6 - time5).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Create of the ITensor: " << std::chrono::duration_cast<ns>(time7 - time6).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Inference: " << std::chrono::duration_cast<ns>(time8 - time7).count() * 0.000001 << " ms" << std::endl;
        std::cout << "Post processing of output results: " << std::chrono::duration_cast<ns>(time9- time8).count() * 0.000001 << " ms" << std::endl;
        // -----------------------------------------------------------------------------------------------------
    } catch (const std::exception & ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
