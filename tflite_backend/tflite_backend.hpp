// Copyright 2020 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include "backend.hpp"

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/kernels/register.h"

extern "C" {
Backend* createBackend();
}

class TFLiteBackend : public Backend {
public:
    virtual bool loadModel(const VLauncher *launcher, const std::string &device,
                           const std::vector<std::string> &outputs,
                           const std::map<std::string, std::string>& config)override;
    virtual std::shared_ptr<InferenceMetrics> process(bool streamOutput = false)override
    {
        return nullptr;
    }
    virtual void report(const InferenceMetrics &im) const override;
    virtual bool infer()override;
    virtual void release()override;

    virtual std::shared_ptr<VBlob> getBlob(const std::string &name)override;

    virtual VInputInfo getInputDataMap() const override;
    virtual VOutputInfo getOutputDataMap() const override;

protected:
    std::map<std::string, std::shared_ptr<VBlob> > _blobs;

    VInputInfo _inputInfo;
    VOutputInfo _outputInfo;

    // this container will retain and release snpe tensors since TensorMap operate with raw pointers
    std::unique_ptr<tflite::FlatBufferModel> _model;
    std::unique_ptr<tflite::Interpreter> _interpreter;
};
