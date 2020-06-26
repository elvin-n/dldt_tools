// Copyright 2020 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include "ie_backend.hpp"

bool IEBackend::loadModel(const std::string &model, const std::string &device,
                          const std::vector<std::string> &outputs,
                          const std::map<std::string, std::string>& config) {

    try {
        InferenceEngine::CNNNetwork network = _core.ReadNetwork(model, model.substr(0, model.size() - 4) + ".bin");
        InferenceEngine::InputsDataMap inputInfo = network.getInputsInfo();
        for (auto i : inputInfo) {
            i.second->setPrecision(InferenceEngine::Precision::FP32);
        }
        InferenceEngine::OutputsDataMap outInfo = network.getOutputsInfo();
        _executableNetwork = _core.LoadNetwork(network, device, config);
        inferRequest = _executableNetwork.CreateInferRequest();
        // go over inputs and outputs and create host memory blobs for them
        for (auto i : inputInfo) {
            auto dims = i.second->getTensorDesc().getDims();
            size_t size = product(dims) * sizeof(float);
            _blobs[i.first] = std::make_shared<VBlob>();
            _blobs[i.first]->_shape = dims;
            _blobs[i.first]->_data = (unsigned char *)malloc(size);
            auto ieblob = InferenceEngine::make_shared_blob<float>(i.second->getTensorDesc(),
                                                                   static_cast<float*>(_blobs[i.first]->_data), product(dims));
            inferRequest.SetBlob(i.first, ieblob);

            IOInfo info;
            switch(i.second->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                info._precision = FP32;
            default:
                info._precision = UNSPECIFIED;
            }

            info._shape = dims;
            _inputInfo[i.first] = info;
        }
        for (auto o : outInfo) {
            auto dims = o.second->getTensorDesc().getDims();
            size_t size = product(dims) * sizeof(float);
            _blobs[o.first] = std::make_shared<VBlob>();
            _blobs[o.first]->_shape = dims;
            _blobs[o.first]->_data = (unsigned char *)malloc(size);
            auto ieblob = InferenceEngine::make_shared_blob<float>(o.second->getTensorDesc(),
                                                                   static_cast<float *>(_blobs[o.first]->_data), product(dims));
            inferRequest.SetBlob(o.first, ieblob);

            IOInfo info;
            switch (o.second->getTensorDesc().getPrecision()) {
            case InferenceEngine::Precision::FP32:
                info._precision = FP32;
            default:
                info._precision = UNSPECIFIED;
            }

            info._shape = dims;
            _outputInfo[o.first] = info;


        }
    } catch (std::exception & ex) {
        return false;
    }

    return true;
}

void IEBackend::report(const InferenceMetrics &im) const {

}
bool IEBackend::infer() {
    try {
        inferRequest.Infer();
        return true;
    } catch (std::exception&) {
        return false;
    }
}

std::shared_ptr<VBlob> IEBackend::getBlob(const std::string &name) {
    return _blobs[name];
}


void IEBackend::release() {
    delete this;
}

VInputInfo IEBackend::getInputDataMap() const {
    return _inputInfo;
}

VOutputInfo IEBackend::getOutputDataMap() const {
    return _outputInfo;
}

Backend* createBackend() {
    return new IEBackend();
}

