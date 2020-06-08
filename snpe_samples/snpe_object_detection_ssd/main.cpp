// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gflags/gflags.h>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <map>
#include <chrono>


#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include "object_detection_sample_ssd.h"

#include <DlContainer/IDlContainer.hpp>
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPE.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/UDLFunc.hpp"
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>

#include <opencv2/opencv.hpp>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;


bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }

    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    return true;
}

/**
* \brief The entry point for the Inference Engine object_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
int main(int argc, char *argv[]) {
    try {
        /** This sample covers certain topology and cannot be generalized for any object detection one **/

        // --------------------------- 1. Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        // -----------------------------------------------------------------------------------------------------

        // -----------------------------------------------------------------------------------------------------
        // --------------------------- 1. Read IR Generated by SNPE tools (.dl file) ------------
        Time::time_point time1 = Time::now();
        std::unique_ptr<zdl::DlContainer::IDlContainer> container;
        container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(FLAGS_m.c_str()));
        if (container == nullptr) {
            std::cerr << "Error while opening the container file." << std::endl;
            return EXIT_FAILURE;
        }
        Time::time_point time2 = Time::now();

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Loading model to the device ------------------------------------------
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
        if (FLAGS_d == "GPU") {
            runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
        } else if (FLAGS_d == "DSP") {
            runtime = zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
        } else if (FLAGS_d == "CPU") {
            runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
        } else {
            std::cerr << "The device is not valid. Set default CPU runtime." << std::endl;
            runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
        }

        std::unique_ptr<zdl::SNPE::SNPE> snpe;
        zdl::DlSystem::RuntimeList runtimeList;
        runtimeList.add(runtime);
        // add CPU fallback
        if (runtime != zdl::DlSystem::Runtime_t::CPU_FLOAT32) {
            runtimeList.add(zdl::DlSystem::Runtime_t::CPU_FLOAT32);
        }

        zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

        // amalyshe: this is standard names of the layers for models been crfeated using Tensor Flow object detection API
        // (at least i TF 1.xx versions)
        // let's stick to these names
        // if we do not enumerate these names explicitly, there will be only one explicit output responsible for classes
        // adding of second layer gives us three more buffers which will have boxes and scores
        zdl::DlSystem::StringList outputs;
        outputs.append("add");
        outputs.append("Postprocessor/BatchMultiClassNonMaxSuppression");

        snpe = snpeBuilder.setOutputLayers(outputs)
            .setCPUFallbackMode(true) //old (TO BE deprecated) API - still works !
            // .setRuntimeProcessor(runtime).setCPUFallbackMode(true) //old (TO BE deprecated) API - still works !
            .setRuntimeProcessorOrder(runtimeList) //new API - doesn't work properly?!
            // .setUseUserSuppliedBuffers(true) TODO (amalyshe) to clarify what this option means
            .build();

        Time::time_point time3 = Time::now();
        if (snpe == nullptr) {
            std::cerr << "Error during creation of SNPE object." << std::endl;
            return EXIT_FAILURE;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Prepare input --------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> images;
        parseInputFilesArguments(images);
        if (images.empty()) throw std::logic_error("No suitable images were found");

        if (images.size() > 1) {
            slog::info << "There are " << images.size() << " images pointed for the input, will process only the first" << slog::endl;
        }

        /** Collect images data ptrs **/
        // std::vector<std::shared_ptr<unsigned char>> imagesData, originalImagesData;
        unsigned char * originalImagesData;
        size_t imageWidths, imageHeights;

        zdl::DlSystem::TensorShape iTShape = snpe->getInputDimensions();
        // verification of input dimensions number - should be 4, NHWC
        if (iTShape.rank() != 4) {
            std::cerr << "Input layer expect data having " << iTShape.rank() << " dimensions shape, while we expect 4" << std::endl;
            return EXIT_FAILURE;
        }
        const zdl::DlSystem::Dimension *shapes = iTShape.getDimensions();

        cv::Mat image = cv::imread(images[0]);
        imageWidths = image.cols;
        imageHeights = image.rows;
        std::cout << "Image parameters: Channels=" << image.channels() << ", width=" << image.cols <<
            ", height=" << image.rows << std::endl;

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
            tf[i] = (static_cast<float>(resized_image.data[i]) - 128.f )/ 128.f;
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

        // --------------------------- 4. Do inference --------------------------------------------------------
        /* Running the request synchronously */
        zdl::DlSystem::TensorMap outputTensorMap;
        bool execStatus = snpe->execute(t, outputTensorMap);
        Time::time_point time8 = Time::now();
        // Save the execution results only if successful
        if (!execStatus) {
            std::cerr << "Error while executing the network." << std::endl;
            return EXIT_FAILURE;
        }

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Process output ------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        zdl::DlSystem::StringList outNames = outputTensorMap.getTensorNames();

        for (size_t i = 0; i < outNames.size(); i++) {
            std::cout << outNames.at(i) << " (";
            zdl::DlSystem::ITensor *outTensori = outputTensorMap.getTensor(outNames.at(i));
            zdl::DlSystem::TensorShape shapei = outTensori->getShape();
            for (size_t j = 0; j < shapei.rank(); j++) {
                std::cout << shapei[j] << ", ";
            }
            std::cout  << ")" << std::endl;
        }

        if (outNames.size() != 4) {
            std::cerr << "Output list has " << outNames.size() << " elements, while three are expected" << std::endl;
            return EXIT_FAILURE;
        }

        // as I mentioned above, the names of the layers are pretty hardcoded and we do not have flexibility to change
        // these names sop far
        std::string scoresName = "Postprocessor/BatchMultiClassNonMaxSuppression_scores";
        std::string classesName = "detection_classes:0";
        // std::string classesName = "Postprocessor/BatchMultiClassNonMaxSuppression_classes";
        std::string boxesName = "Postprocessor/BatchMultiClassNonMaxSuppression_boxes";

        // Looking for the Top5 values and print them in format
        // class_id     probability/value of the latest tensor in case of softmax absence
        zdl::DlSystem::ITensor *outTensorScores = outputTensorMap.getTensor(scoresName.c_str());
        zdl::DlSystem::ITensor *outTensorClasses = outputTensorMap.getTensor(classesName.c_str());
        zdl::DlSystem::ITensor *outTensorBoxes = outputTensorMap.getTensor(boxesName.c_str());
        zdl::DlSystem::TensorShape scoresShape = outTensorScores->getShape();
        if (scoresShape.rank() != 2) {
            std::cerr << "Scores should have two axis" << std::endl;
            return EXIT_FAILURE;
        }

        const float *oScores = reinterpret_cast<float *>(&(*outTensorScores->begin()));
        const float *oClasses = reinterpret_cast<float *>(&(*outTensorClasses->begin()));
        const float *oBoxes = reinterpret_cast<float *>(&(*outTensorBoxes->begin()));

        std::vector<int> boxes;
        std::vector<int> classes;

        for (size_t curProposal = 0; curProposal < scoresShape[1]; curProposal++) {
            float confidence = oScores[curProposal];
            float label = static_cast<int>(oClasses[curProposal]);
            // boxes have follow layout top, left, bottom, right
            // according to this link: https://www.tensorflow.org/lite/models/object_detection/overview
            auto ymin = static_cast<int>(oBoxes[4 * curProposal] * imageHeights);
            auto xmin = static_cast<int>(oBoxes[4 * curProposal + 1] * imageWidths);
            auto ymax = static_cast<int>(oBoxes[4 * curProposal + 2] * imageHeights);
            auto xmax = static_cast<int>(oBoxes[4 * curProposal + 3] * imageWidths);


            std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
                "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << 0;

            if (confidence > 0.5) {
                // Drawing only objects with >50% probability
                classes.push_back(label);
                boxes.push_back(xmin);
                boxes.push_back(ymin);
                boxes.push_back(xmax - xmin);
                boxes.push_back(ymax - ymin);
                std::cout << " WILL BE PRINTED!";
            }
            std::cout << std::endl;
        }

        addRectangles(image.data, imageHeights, imageWidths, boxes, classes,
                      BBOX_THICKNESS);
        const std::string image_path = "out.bmp";
        if (writeOutputBmp(image_path.c_str(), image.data, imageHeights, imageWidths)) {
            slog::info << "Image " + image_path + " created!" << slog::endl;
        } else {
            throw std::logic_error(std::string("Can't create a file: ") + image_path);
        }



        //const float *oData = reinterpret_cast<float *>(&(*outTensor->begin()));
/*


        const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
        MemoryBlob::CPtr moutput = as<MemoryBlob>(output_blob);
        if (!moutput) {
            throw std::logic_error("We expect output to be inherited from MemoryBlob, "
                                   "but by fact we were not able to cast output to MemoryBlob");
        }
        // locked memory holder should be alive all time while access to its buffer happens
        auto moutputHolder = moutput->rmap();
        const float *detection = moutputHolder.as<const PrecisionTrait<Precision::FP32>::value_type *>();

        std::vector<std::vector<int> > boxes(batchSize);
        std::vector<std::vector<int> > classes(batchSize);

        // Each detection has image_id that denotes processed image
        for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) {
            auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
            if (image_id < 0) {
                break;
            }

            float confidence = detection[curProposal * objectSize + 2];
            auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
            auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * imageWidths[image_id]);
            auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * imageHeights[image_id]);
            auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * imageWidths[image_id]);
            auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * imageHeights[image_id]);

            std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
                "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

            if (confidence > 0.5) {
                // Drawing only objects with >50% probability
                classes[image_id].push_back(label);
                boxes[image_id].push_back(xmin);
                boxes[image_id].push_back(ymin);
                boxes[image_id].push_back(xmax - xmin);
                boxes[image_id].push_back(ymax - ymin);
                std::cout << " WILL BE PRINTED!";
            }
            std::cout << std::endl;
        }

        for (size_t batch_id = 0; batch_id < batchSize; ++batch_id) {
            addRectangles(originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id], boxes[batch_id], classes[batch_id],
                          BBOX_THICKNESS);
            const std::string image_path = "out_" + std::to_string(batch_id) + ".bmp";
            if (writeOutputBmp(image_path, originalImagesData[batch_id].get(), imageHeights[batch_id], imageWidths[batch_id])) {
                slog::info << "Image " + image_path + " created!" << slog::endl;
            } else {
                throw std::logic_error(std::string("Can't create a file: ") + image_path);
            }
        }
*/
        // -----------------------------------------------------------------------------------------------------
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    slog::info << slog::endl << "This sample is an API example, for any performance measurements "
                                "please use the dedicated benchmark_app tool" << slog::endl;
    return 0;
}
