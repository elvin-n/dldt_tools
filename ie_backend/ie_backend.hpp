// Copyright 2020 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include "backend.hpp"
#include <inference_engine.hpp>

extern "C" {
Backend* createBackend();
}

class IEBackend : public Backend {
public:
    virtual bool loadModel(const std::string &model, const std::string &device,
                           const std::vector<std::string> &outputs,
                           const std::map<std::string, std::string>& config)override;
    virtual std::shared_ptr<InferenceMetrics> process(bool streamOutput = false)override
    {
        return nullptr;
    }
    virtual void report(const InferenceMetrics &im) const override;
    virtual void infer()override;
    virtual void release()override;

    virtual std::shared_ptr<VBlob> getBlob(const std::string &name)override;

    virtual VInputInfo getInputDataMap() const override;
    virtual VOutputInfo getOutputDataMap() const override;

protected:
    InferenceEngine::Core _core;
    InferenceEngine::ExecutableNetwork _executableNetwork;
    InferenceEngine::InferRequest inferRequest;
    std::map<std::string, std::shared_ptr<VBlob> > _blobs;

    VInputInfo _inputInfo;
    VOutputInfo _outputInfo;
};
