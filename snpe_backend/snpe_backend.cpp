// Copyright 2020 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include "snpe_backend.hpp"

#include <string.h>
#include <DlContainer/IDlContainer.hpp>
#include "DlSystem/RuntimeList.hpp"
#include "SNPE/SNPEBuilder.hpp"
#include "DlSystem/UDLFunc.hpp"
#include <DlSystem/ITensorFactory.hpp>
#include <SNPE/SNPEFactory.hpp>

bool SNPEBackend::loadModel(const std::string &model, const std::string &device,
                            const std::vector<std::string> &outputs,
                            const std::map<std::string, std::string> &config) {

    try {

        // --------------------------- 1. Read IR Generated by SNPE tools (.dl file) ------------
        std::unique_ptr<zdl::DlContainer::IDlContainer> container;
        container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(model.c_str()));
        if (container == nullptr) {
            std::cerr << "Error while opening the container file." << std::endl;
            return EXIT_FAILURE;
        }

        // --------------------------- 2. Loading model to the device ------------------------------------------
        zdl::DlSystem::Runtime_t runtime = zdl::DlSystem::Runtime_t::CPU;
        if (device == "GPU") {
            runtime = zdl::DlSystem::Runtime_t::GPU_FLOAT16;
        } else if (device == "DSP") {
            runtime = zdl::DlSystem::Runtime_t::DSP_FIXED8_TF;
        } else if (device == "CPU") {
            runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
        } else {
            std::cerr << "The device is not valid. Set default CPU runtime." << std::endl;
            runtime = zdl::DlSystem::Runtime_t::CPU_FLOAT32;
        }

        zdl::DlSystem::RuntimeList runtimeList;
        runtimeList.add(runtime);
        // add CPU fallback
        if (runtime != zdl::DlSystem::Runtime_t::CPU_FLOAT32) {
            runtimeList.add(zdl::DlSystem::Runtime_t::CPU_FLOAT32);
        }

        zdl::SNPE::SNPEBuilder snpeBuilder(container.get());

        zdl::DlSystem::StringList snpeOutputs;
        for (auto o : outputs) {
            snpeOutputs.append(o.c_str());
        }
        _snpe = snpeBuilder.setOutputLayers(snpeOutputs)
            .setRuntimeProcessor(runtime).setCPUFallbackMode(true) //old (TO BE deprecated) API - still works !
            // .setRuntimeProcessorOrder(runtimeList)
            // .setUseUserSuppliedBuffers(true) TODO (amalyshe) to clarify what this option means
            .setDebugMode(false)
            .setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::HIGH_PERFORMANCE)
            .setProfilingLevel(zdl::DlSystem::ProfilingLevel_t::OFF)
            .build();
        if (!_snpe) {
            return false;
        }

        zdl::DlSystem::StringList inputNames = _snpe->getInputTensorNames();
        for (size_t i = 0; i < inputNames.size(); i++) {
            std::string name = inputNames.at(i);
            auto bufferAttributesOpt = _snpe->getInputOutputBufferAttributes(name.c_str());
            if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for input tensor ") + name);
            // calculate the size of buffer required by the input tensor
            const zdl::DlSystem::TensorShape bufferShape = (*bufferAttributesOpt)->getDims();

            IOInfo info;
            info._precision = FP32;
            info._shape.resize(bufferShape.rank());
            for (size_t j = 0; j < bufferShape.rank(); j++) {
                info._shape[j] = bufferShape.getDimensions()[j];
            }
            _inputInfo[name] = info;

            std::shared_ptr<zdl::DlSystem::ITensor> inputTensor =
                zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(bufferShape);
            _inputTensorMap.add(name.c_str(), inputTensor.get());
            tensors.push_back(inputTensor);

            auto vblob = std::make_shared<VBlob>();
            vblob->_precision = FP32;
            vblob->_shape = info._shape;
            vblob->_data = reinterpret_cast<void *>(&(*inputTensor->begin()));
            vblob->_ownMemory = false;
            vblob->_layout = "NHWC";
            vblob->_colourFormat = "RGB";
            _blobs[name] = vblob;
        }

        zdl::DlSystem::StringList outputNames = _snpe->getOutputTensorNames();
        for (size_t o = 0; o < outputNames.size(); o++) {
            std::string name = outputNames.at(o);
            auto bufferAttributesOpt = _snpe->getInputOutputBufferAttributes(name.c_str());
            if (!bufferAttributesOpt) throw std::runtime_error(std::string("Error obtaining attributes for output tensor ") + name);
            // calculate the size of buffer required by the input tensor
            const zdl::DlSystem::TensorShape &bufferShape = (*bufferAttributesOpt)->getDims();

            IOInfo info;
            info._precision = FP32;
            info._shape.resize(bufferShape.rank());
            for (size_t j = 0; j < bufferShape.rank(); j++) {
                info._shape[j] = bufferShape.getDimensions()[j];
            }
            _outputInfo[name] = info;

            auto vblob = std::make_shared<VBlob>();
            vblob->_precision = FP32;
            vblob->_shape = info._shape;
            vblob->_data = (unsigned char *)malloc(product(vblob->_shape) * sizeof(float));
            _blobs[name] = vblob;
        }
    } catch (std::exception &ex) {
        return false;
    }

    return true;
}

void SNPEBackend::report(const InferenceMetrics &im) const {

}
void SNPEBackend::infer() {
    bool execStatus = _snpe->execute(_inputTensorMap, _outputTensorMap);

    zdl::DlSystem::StringList outNames = _outputTensorMap.getTensorNames();
    for (size_t i = 0; i < outNames.size(); i++) {
        auto vblob = _blobs[outNames.at(i)];
        zdl::DlSystem::ITensor *outTensor = _outputTensorMap.getTensor(outNames.at(i));
        if (vblob && outTensor) {
            const void *oData = reinterpret_cast<void *>(&(*outTensor->begin()));
            memcpy(vblob->_data, oData, product(vblob->_shape) * sizeof(float));
        }
    }
}

std::shared_ptr<VBlob> SNPEBackend::getBlob(const std::string &name) {
    return _blobs[name];
}


void SNPEBackend::release() {
    delete this;
}

VInputInfo SNPEBackend::getInputDataMap() const {
    return _inputInfo;
}

VOutputInfo SNPEBackend::getOutputDataMap() const {
    return _outputInfo;
}

Backend* createBackend() {
    return new SNPEBackend();
}

