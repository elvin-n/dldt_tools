// Copyright (C) 2020 group of contributors
// SPDX-License-Identifier: Apache-2.0
//


#include <opencv2/opencv.hpp>

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/vm/vm.h"
#include <tvm/runtime/container/adt.h>

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
            tcout << _T("Expecting <device> to be eq to 'CPU', 'OpenCL' or 'Metal'") << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << "Try to load model" << std::endl;
        // hardcoding for .so file so far
        std::string model_base = input_model.substr(0, input_model.length() - 3);
        std::string consts_path = model_base + ".const";
        std::string vm_exec_code_path = model_base + ".ro";
        tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(input_model);

        std::ifstream code(vm_exec_code_path, std::ios::binary);
        std::stringstream ss;
        ss << code.rdbuf();

        tvm::runtime::Module exec_mod_ = tvm::runtime::vm::Executable::Load(ss.str(), lib);
        if (exec_mod_.get() == nullptr) {
            std::cout << "Failed to load module" << std::endl;
            return -1;
        }
        const tvm::runtime::vm::Executable* tmp = exec_mod_.as<tvm::runtime::vm::Executable>();
        auto exec_ = tvm::runtime::GetObjectPtr<tvm::runtime::vm::Executable>(const_cast<tvm::runtime::vm::Executable*>(tmp));
        exec_->LoadLateBoundConstantsFromFile(consts_path);

        auto vm = tvm::runtime::make_object<tvm::runtime::vm::VirtualMachine>();
        vm->LoadExecutable(exec_);

        std::cout << "consts  loaded\n";

        // Initialize the VM for the specified device. If the device is not a CPU,
        // We'll need to add a CPU context to drive it.
        int arity;
        if (ctx.device_type == kDLCPU) {
        arity = 3;
        } else {
        arity = 6;
        }
        uint64_t alloc_type = uint64_t(tvm::runtime::vm::AllocatorType::kPooled);
        // If target device is not CPU, we should pass CPU context in any case
        uint64_t device_id = 0;
        std::vector<TVMValue> init_vals(arity);
        std::vector<int> codes(arity);
        tvm::runtime::TVMArgsSetter setter(init_vals.data(), codes.data());
        setter(0, (int)ctx.device_type);
        setter(1, device_id);
        setter(2, alloc_type);
        // Also initialize a CPU device context.
        if (ctx.device_type != kDLCPU) {
            setter(3, (int)kDLCPU);
            setter(4, device_id);
            setter(5, alloc_type);
        }
        tvm::runtime::TVMRetValue rv;
        // Call the packed func with the init arguments.
        vm->GetFunction("init", nullptr).CallPacked(tvm::runtime::TVMArgs(init_vals.data(), codes.data(), arity), &rv);

        std::cout << "Finished initialization of virtual machine" << std::endl;
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Prepare input --------------------------------------------------------
        Time::time_point time2 = Time::now();
        size_t width = 224, height = 224;
        size_t nielements = 3 * width * height;

        cv::Mat image = cv::imread(input_image_path);
        std::cout << "Image is read" << std::endl;
        cv::Mat resized_image(image);
        cv::resize(image, resized_image, cv::Size(width, height));

        // Use the C++ API
        DLDevice ctxCPU{kDLCPU, 0};
        tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, ctxCPU);

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

        // set the input
        {
            size_t num_total_args = 2;
            std::vector<TVMValue> tvm_values(num_total_args);
            std::vector<int> tvm_type_codes(num_total_args);
            ::tvm::runtime::TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
            const std::string func_name = "main";
            setter(0, func_name.c_str());
            setter(1, x);

            tvm::runtime::PackedFunc set_input = vm->GetFunction("set_input", nullptr);
            tvm::runtime::TVMRetValue rv;
            set_input.CallPacked(tvm::runtime::TVMArgs(tvm_values.data(), tvm_type_codes.data(), int(num_total_args)), &rv);
        }
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);

        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Do inference --------------------------------------------------------
        std::cout << "Do inference" << std::endl;
        // warm up
        tvm::runtime::PackedFunc run_func = vm->GetFunction("invoke", nullptr);
        run_func("main");
        // run();

        Time::time_point time4 = Time::now();
        tvm::runtime::ObjectRef out = run_func("main");
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
        Time::time_point time5 = Time::now();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Process output ------------------------------------------------------
        tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 1000}, DLDataType{kDLFloat, 32, 1}, ctxCPU);

        if (out.as<tvm::runtime::ADTObj>()) {
            auto adt = tvm::Downcast<tvm::runtime::ADT>(out);
            tvm::runtime::NDArray arr = tvm::Downcast<tvm::runtime::NDArray>(adt[0]);
            arr.CopyTo(y);
        } else {
            tvm::runtime::NDArray arr = tvm::Downcast<tvm::runtime::NDArray>(out);
            arr.CopyTo(y);
        }


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
