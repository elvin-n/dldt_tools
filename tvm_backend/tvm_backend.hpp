// Copyright 2021 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include "backend.hpp"
#include "tvm/runtime/module.h"

extern "C" {
#ifdef _WIN32
__declspec(dllexport) Backend* createBackend();
#else
Backend* createBackend();
#endif
}

class VLauncher;
class TVMBackend : public Backend {
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
  tvm::runtime::PackedFunc run_;
  tvm::runtime::PackedFunc setInput_;
  tvm::runtime::PackedFunc getInput_;
  tvm::runtime::PackedFunc getOutput_;
  tvm::runtime::Module gmod_;

  //DLDevice ctx_;
  DLContext ctx_;
  // TODO: which object retain TVM network not to be released?
  tvm::runtime::Module mod_factory_;
  tvm::runtime::NDArray x_, y_;

  std::map<std::string, std::shared_ptr<VBlob> > _blobs;

  VInputInfo _inputInfo;
  VOutputInfo _outputInfo;
};
