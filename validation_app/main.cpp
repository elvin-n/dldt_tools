// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief The entry point for Inference Engine validation application
 * @file validation_app/main.cpp
 */
#include <gflags/gflags.h>
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include <limits>
#include <iomanip>
#include <memory>
#ifndef _WIN32
#include <dlfcn.h>
#else
#include<Windows.h>
#endif

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "user_exception.hpp"
#include "ClassificationProcessor.hpp"
#include "SSDObjectDetectionProcessor.hpp"
#include "YOLOObjectDetectionProcessor.hpp"
#include "backend.hpp"
#include "ValidationConfig.h"

typedef Backend*(*createBackend)();

using namespace std;

/// @brief Message for help argument
static const char help_message[] = "Print a help message";

static const char config_message[] = "Config file describing validation workflow";

static const char backend_message[] = "Required. Defines a backend which will perform inference of the neural network";
static const char tf_message[] = "Required. Defines a target framework to run validation on";
/// @brief Message for images argument
static const char image_message[] = "Required. Folder with validation images. Path to a directory with validation images. For Classification models,"
                                    " the directory must contain folders named as labels with images inside or a .txt file with"
                                    " a list of images. For Object Detection models, the dataset must be in"
                                    " VOC format.";
/// @brief Message for plugin_path argument
static const char plugin_path_message[] = "Required. Path to an .xml file with a trained model, including model name and "
                                          "extension.";
/// @brief Message for model argument
static const char model_message[] = "Required. Path to an .xml file with a trained model";
/// @brief Message for plugin argument
static const char plugin_message[] = "Plugin name. For example, CPU. If this parameter is passed, "
                                     "the sample looks for a specified plugin only.";
/// @brief Message for assigning cnn calculation to device
static const char target_device_message[] = "Target device to infer on: CPU (default), GPU, FPGA, HDDL or MYRIAD."
                                            " The application looks for a suitable plugin for the specified device.";
/// @brief Message for label argument
static const char label_message[] = "Path to a file with labels for a model";
/// @brief Message for batch argumenttype
static const char batch_message[] = "Batch size value. If not specified, the batch size value is taken from IR";
/// @brief Message for dump argument
static const char dump_message[] = "Dump file names and inference results to a .csv file";
/// @brief Message for network type
static const char type_message[] = "Type of an inferred network (\"C\" by default)";
/// @brief Message for pp-type
static const char preprocessing_type[] = "Preprocessing type. Options: \"None\", \"Resize\", \"ResizeCrop\"";
/// @brief Message for pp-crop-size
static const char preprocessing_size[] = "Preprocessing size (used with ppType=\"ResizeCrop\")";
static const char preprocessing_width[] = "Preprocessing width (overrides -ppSize, used with ppType=\"ResizeCrop\")";
static const char preprocessing_height[] = "Preprocessing height (overrides -ppSize, used with ppType=\"ResizeCrop\")";

static const char obj_detection_annotations_message[] = "Required for Object Detection models. Path to a directory"
                                                        " containing an .xml file with annotations for images.";

static const char obj_detection_classes_message[] = "Required for Object Detection models. Path to a file containing"
                                                    " a list of classes";

static const char obj_detection_subdir_message[] = "Directory between the path to images (specified with -i) and image name (specified in the"
                                                   " .xml file). For VOC2007 dataset, use JPEGImages.";
static const char obj_detection_kind_message[] = "Type of an Object Detection model. Options: SSD";

/// @brief Message for GPU custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels."
                                           "Absolute path to an .xml file with the kernel descriptions.";

/// @brief Message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. "
                                                 "Absolute path to a shared library with the kernel implementations";

/// @brief Message for labels file
static const char labels_file_message[] = "Labels file path. The labels file contains names of the dataset classes";

static const char zero_background_message[] = "\"Zero is a background\" flag. Some networks are trained with a modified"
                                              " dataset where the class IDs "
                                              " are enumerated from 1, but 0 is an undefined \"background\" class"
                                              " (which is never detected)";

static const char plain_output_message[] = "Flag for plain output";


/// @brief Network type options and their descriptions
static const char* types_descriptions[][2] = {
    { "C", "classification" },
//    { "SS", "semantic segmentation" },    // Not supported yet
    { "OD", "object detection" },
    { nullptr, nullptr }
};


/// @brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);
/// @brief Define parameter for backend to use for inference of the model <br>
/// It is a required parameter
DEFINE_string(backend, "", backend_message);
/// @brief Define parameter for config <br>
/// It is a required parameter
DEFINE_string(config, "", config_message);

DEFINE_string(target_framework, "", tf_message);
/// @brief Define parameter for a path to images <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);
/// @brief Define parameter for a path to model file <br>
/// It is a required parameter
DEFINE_string(m, "", model_message);
/// @brief Define parameter for a plugin name <br>
/// It is a required parameter
DEFINE_string(p, "", plugin_message);
/// @brief Define parameter for a path to a file with labels <br>
/// Default is empty
DEFINE_string(OCl, "", label_message);
/// @brief Define parameter for a path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, "", plugin_path_message);
/// @brief Define parameter for a target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);
/// @brief Define parameter for batch size <br>
/// Default is 0 (which means that batch size is not specified)
DEFINE_int32(b, 0, batch_message);
/// @brief Define flag to dump results to a file <br>
DEFINE_bool(dump, false, dump_message);
/// @brief Define parameter for a network type parameter
DEFINE_string(t, "C", type_message);

/// @brief Define parameter for preprocessing type
DEFINE_string(ppType, "", preprocessing_type);

/// @brief Define parameter for preprocessing size
DEFINE_int32(ppSize, 0, preprocessing_size);
DEFINE_int32(ppWidth, 0, preprocessing_width);
DEFINE_int32(ppHeight, 0, preprocessing_height);

DEFINE_bool(Czb, false, zero_background_message);

DEFINE_string(ODa, "", obj_detection_annotations_message);

DEFINE_string(ODc, "", obj_detection_classes_message);

DEFINE_string(ODsubdir, "", obj_detection_subdir_message);

/// @brief Define parameter for a type of Object Detection network
DEFINE_string(ODkind, "SSD", obj_detection_kind_message);

/// @brief Define parameter for GPU kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Define parameter for a path to CPU library with user layers <br>
/// It is an optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Flag for printing plain text
DEFINE_bool(plain, false, plain_output_message);

DEFINE_string(lbl, "", labels_file_message);

/**
 * @brief This function shows a help message
 */
static void showUsage() {
    std::cout << std::endl;
    std::cout << "Usage: validation_app [OPTION]" << std::endl << std::endl;
    std::cout << "Available options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                        " << help_message << std::endl;
    std::cout << "    -backend                  " << backend_message << std::endl;
    std::cout << "    -t <type>                 " << type_message << std::endl;
    for (int i = 0; types_descriptions[i][0] != nullptr; i++) {
        std::cout << "      -t \"" << types_descriptions[i][0] << "\" for " << types_descriptions[i][1] << std::endl;
    }
    std::cout << "    -i <path>                 " << image_message << std::endl;
    std::cout << "    -m <path>                 " << model_message << std::endl;
    std::cout << "    -lbl <path>               " << labels_file_message << std::endl;
    std::cout << "    -l <absolute_path>        " << custom_cpu_library_message << std::endl;
    std::cout << "    -c <absolute_path>        " << custom_cldnn_message << std::endl;
    std::cout << "    -d <device>               " << target_device_message << std::endl;
    std::cout << "    -b N                      " << batch_message << std::endl;
    std::cout << "    -ppType <type>            " << preprocessing_type << std::endl;
    std::cout << "    -ppSize N                 " << preprocessing_size << std::endl;
    std::cout << "    -ppWidth W                " << preprocessing_width << std::endl;
    std::cout << "    -ppHeight H               " << preprocessing_height << std::endl;
    std::cout << "    --dump                    " << dump_message << std::endl;

    std::cout << std::endl;
    std::cout << "    Classification-specific options:" << std::endl;
    std::cout << "      -Czb true               " << zero_background_message << std::endl;

    std::cout << std::endl;
    std::cout << "    Object detection-specific options:" << std::endl;
    std::cout << "      -ODkind <kind>          " << obj_detection_kind_message << std::endl;
    std::cout << "      -ODa <path>             " << obj_detection_annotations_message << std::endl;
    std::cout << "      -ODc <file>             " << obj_detection_classes_message << std::endl;
    std::cout << "      -ODsubdir <name>        " << obj_detection_subdir_message << std::endl << std::endl;
}

enum NetworkType {
    Undefined = -1,
    Classification,
    ObjDetection
};

std::string strtolower(const std::string& s) {
    std::string res = s;
    std::transform(res.begin(), res.end(), res.begin(), ::tolower);
    return res;
}

/**
 * @brief The main function of Inference Engine sample application
 * @param argc - The number of arguments
 * @param argv - Arguments
 * @return 0 if all good
 */
int main(int argc, char *argv[]) {
    try {
      UserExceptions ee;
      std::string libName;
        // ---------------------------Parsing and validating input arguments--------------------------------------
        slog::info << "Parsing input parameters" << slog::endl;

        bool noOptions = argc == 1;

        gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
        if (FLAGS_h || noOptions) {
            showUsage();
            return 1;
        }
        if (!FLAGS_config.empty()) ee << UserException(1, "Config file is not specified (missing -config option)");
        YAML::Node config = YAML::LoadFile(FLAGS_config);
        ValidationConfig a(config);
        auto launcher = a.getLauncherByFramwork(FLAGS_target_framework, FLAGS_d);
        auto dataset = a.getDatasetsByFramwork(FLAGS_target_framework);
        if (!launcher) {
          THROW_USER_EXCEPTION(2) << "Cannot get launcher from config";
        }
        if (!dataset) {
          THROW_USER_EXCEPTION(2) << "Cannot get dataset from config";
        }
#ifdef __APPLE__
        libName = std::string("lib") + launcher->framework_ + "_backend.dylib";
#elif _WIN32
        libName = launcher->framework_ + "_backend.dll";
#else
        libName = std::string("lib") + launcher->framework_ + "_backend.so";
#endif


        // try to load backend
        Backend* backend = nullptr;
#ifndef _WIN32
        void *shared_object = dlopen(libName.c_str(), RTLD_LAZY);
        if (!shared_object) {
            THROW_USER_EXCEPTION(2) << "Cannot open '" << libName  << "' library" << dlerror();
        }
        // void* procAddr = nullptr;
        Backend* (*createBackend)() = (Backend * (*)())dlsym(shared_object, "createBackend");
        if (createBackend == nullptr)
            THROW_USER_EXCEPTION(2) << "dlSym cannot locate method 'createBackend': " << dlerror();
        backend = createBackend();
#else
        HMODULE h = LoadLibrary(libName.c_str());
        if (!h) {
            THROW_USER_EXCEPTION(2) << "Cannot open '" << libName << "' library";
        }
        Backend* (*createBackend)() = (Backend * (*)())GetProcAddress(h, "createBackend");
        if (createBackend == nullptr)
            THROW_USER_EXCEPTION(2) << "GetProcAddress cannot locate method 'createBackend' in " << libName;
        backend = createBackend();
#endif
        if (!backend) {
#ifndef _WIN32
            THROW_USER_EXCEPTION(2) << "Cannot create inference backend: " << dlerror();
#else
            THROW_USER_EXCEPTION(2) << "Cannot create inference backend";
#endif
        }


        // TODO(amalyshe) evaluate what we can do with extensions
        //if (!FLAGS_l.empty()) {
        //    // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        //    IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
        //    ie.AddExtension(extension_ptr, "CPU");
        //    slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        //}

        //if (!FLAGS_c.empty()) {
        //    // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        //    ie.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } }, "GPU");
        //    slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        //}

        CsvDumper dumper(FLAGS_dump);

        std::shared_ptr<Processor> processor;

        if (launcher->adapter_ == "classification") {
            processor = std::shared_ptr<Processor>(
                new ClassificationProcessor(backend, launcher, FLAGS_d, {}, FLAGS_i, FLAGS_b, dumper, dataset));
        } /*else if (launcher->adapter_ == "ObjDetection") {
          if (FLAGS_ODkind == "SSD") {
            // work around for object detection models from tensorflow
            std::vector<std::string> outputs;
            if (FLAGS_backend == "snpe_backend") {
              outputs.push_back("add");
              outputs.push_back("Postprocessor/BatchMultiClassNonMaxSuppression");
            }
            processor = std::shared_ptr<Processor>(
              new SSDObjectDetectionProcessor(backend, FLAGS_m, outputs, FLAGS_d, FLAGS_i, FLAGS_ODsubdir, FLAGS_b,
                                              0.5, dumper, FLAGS_ODa, FLAGS_ODc));
          } else if (FLAGS_ODkind == "YOLO") {
            processor = std::shared_ptr<Processor>(
              new YOLOObjectDetectionProcessor(backend, FLAGS_m, { }, FLAGS_d, FLAGS_i, FLAGS_ODsubdir, FLAGS_b,
                                               0.5, dumper, FLAGS_ODa, FLAGS_ODc));
          }
        } */ else {
            THROW_USER_EXCEPTION(2) <<  "Unknown network type specified" << FLAGS_ppType;
        }
        if (!processor.get()) {
            THROW_USER_EXCEPTION(2) <<  "Processor pointer is invalid" << FLAGS_ppType;
        }
        slog::info << (FLAGS_d.empty() ? "Plugin: " + FLAGS_p : "Device: " + FLAGS_d) << slog::endl;
        shared_ptr<Processor::InferenceMetrics> pIM = processor->Process(FLAGS_plain);
        processor->Report(*pIM.get());

        if (dumper.dumpEnabled()) {
            slog::info << "Dump file generated: " << dumper.getFilename() << slog::endl;
        }
    } catch (const UserExceptions& ex) {
        if (ex.list().size() == 1) {
            slog::err << "Input problem: " << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        } else {
            slog::err << "Input problems: \n" << ex.what() << slog::endl;
            showUsage();
            return ex.list().begin()->exitCode();
        }
    }
    return 0;
}
