// Copyright 2021 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license
#include "tvm_backend.hpp"

#include <vector>
#include <memory>
#include <string>
#include <iostream>

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"
#include "ValidationConfig.h"

bool TVMBackend::loadModel(const VLauncher *launcher, const std::string &device,
                          const std::vector<std::string> &outputs,
                          const std::map<std::string, std::string>& config) {
  // Device changes Context
  // DLDeviceType target = kDLCPU;
  DLDeviceType target = kDLCPU;
  if (device == "CPU") {
    target = kDLCPU;
  } else if (device == "OpenCL") {
    target = kDLOpenCL;
  } else if (device == "Metal") {
    target = kDLMetal;
  }
  // ctx_ = DLDevice{target, 0};
  ctx_ = DLContext{target, 0};
  std::cout << "Entered TVMBackend::loadModel, call tvm::runtime::Module::LoadFromFile(" << launcher->model_ << ")" << std::endl;
  mod_factory_ = tvm::runtime::Module::LoadFromFile(launcher->model_);
  // create the graph runtime module
  gmod_ = mod_factory_.GetFunction("default")(ctx_);
  std::cout << "Call default Packed Func - DONE" << std::endl;

  tvm::runtime::PackedFunc get_num_inputs = gmod_.GetFunction("get_num_inputs");
  tvm::runtime::PackedFunc get_input = gmod_.GetFunction("get_input");
  tvm::runtime::PackedFunc get_num_outputs = gmod_.GetFunction("get_num_outputs");
  tvm::runtime::PackedFunc get_output = gmod_.GetFunction("get_output");
  setInput_ = gmod_.GetFunction("set_input");
  getInput_ = gmod_.GetFunction("get_input");
  getOutput_ = gmod_.GetFunction("get_output");
  run_ = gmod_.GetFunction("run");

  int ninputs = get_num_inputs();
  ninputs = 1;
  for (size_t i = 0; i < ninputs; i++) {
    std::string name = "i" + std::to_string(i);
    tvm::runtime::NDArray input = get_input(i);
    auto shape = input.Shape();
    if (shape.size() ==4) {
      auto arr_type = input.DataType();
      std::cout << "input: " << i << "(";
      for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i] << ",";
      }
      std::cout << "), type: " << arr_type << std::endl;

      IOInfo info;
      if (arr_type.is_float()) {
        info._precision = FP32;
      } else if (arr_type.is_uint() && arr_type.bits() == 8) {
        info._precision = U8;
      } else {
        return false;
      }

      info._shape.resize(shape.size());
      for (size_t j = 0; j < shape.size(); j++) {
          info._shape[j] = shape[j];
      }
      _inputInfo[name] = info;

      auto vblob = std::make_shared<VBlob>();
      vblob->_precision = info._precision;
      vblob->_shape = info._shape;

      //DLDevice ctx{kDLCPU, 0}; //kDLMetal
      DLContext ctx{kDLCPU, 0}; //kDLMetal
      x_ = tvm::runtime::NDArray::Empty(shape,
        DLDataType{kDLFloat, 32, 1}, ctx);
      //setInput(i, x_);
      vblob->_data = reinterpret_cast<void *>(x_->data);
      //vblob->_data = reinterpret_cast<void *>(input->data);
      vblob->_ownMemory = false;
      //size_t size = product(info._shape) * sizeof(float);
      //vblob->_ownMemory = true;
      //vblob->_data = (unsigned char *)malloc(size);

      if (launcher->inputs_.size() == 1) {
        vblob->_layout = launcher->inputs_[0].layout_;
      } else {
        vblob->_layout = "NCHW";
      }
      vblob->_colourFormat = "RGB";
      _blobs[name] = vblob;
    }
  }

  int noutputs = get_num_outputs();
  for (size_t i = 0; i < noutputs; i++) {
    std::string name = "o" + std::to_string(i);
    tvm::runtime::NDArray output = get_output(i);
    auto shape = output.Shape();
    auto arr_type = output.DataType();

    std::cout << "output: " << i << "(";
    for (size_t i = 0; i < shape.size(); i++) {
      std::cout << shape[i] << ",";
    }
    std::cout << ")" << std::endl;


    IOInfo info;
    if (arr_type.is_float()) {
      info._precision = FP32;
    } else if (arr_type.is_uint() && arr_type.bits() == 8) {
      info._precision = U8;
    } else {
      return false;
    }

    info._shape.resize(shape.size());
    for (size_t j = 0; j < shape.size(); j++) {
        info._shape[j] = shape[j];
    }
    _outputInfo[name] = info;

    auto vblob = std::make_shared<VBlob>();
    vblob->_precision = info._precision;
    vblob->_shape = info._shape;
    //DLDevice ctx{kDLCPU, 0}; //kDLMetal
    DLContext ctx{kDLCPU, 0}; //kDLMetal
    y_ = tvm::runtime::NDArray::Empty(shape,
      DLDataType{kDLFloat, 32, 1}, ctx);
    vblob->_data = reinterpret_cast<void *>(y_->data);
    //vblob->_data = reinterpret_cast<void *>(output->data);
    vblob->_ownMemory = false;
    _blobs[name] = vblob;
  }

  return true;
}

void TVMBackend::report(const InferenceMetrics &im) const {

}
bool TVMBackend::infer() {
    try {
      setInput_(0, x_);
      run_();
      tvm::runtime::NDArray output = getOutput_(0);
      output.CopyTo(y_);
      TVMSynchronize(ctx_.device_type, ctx_.device_id, nullptr);
        return true;
    } catch (std::exception&) {
      return false;
    }
}

std::shared_ptr<VBlob> TVMBackend::getBlob(const std::string &name) {
    return _blobs[name];
}


void TVMBackend::release() {
  delete this;
}

VInputInfo TVMBackend::getInputDataMap() const {
  return _inputInfo;
}

VOutputInfo TVMBackend::getOutputDataMap() const {
  return _outputInfo;
}

Backend* createBackend() {
  return new TVMBackend();
}

