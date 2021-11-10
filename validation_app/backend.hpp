// Copyright 2020 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include <samples/console_progress.hpp>
#include <memory>
#include <limits>
#include <string>
#include <map>
#include <vector>
#include <numeric>
#include "ValidationConfig.h"

// #include "inference_engine.hpp"

#pragma once

enum evPrecision : uint8_t {
    UNSPECIFIED = 255, /**< Unspecified value. Used by default */
    MIXED = 0,         /**< Mixed value. Can be received from network. No applicable for tensors */
    FP32 = 10,         /**< 32bit floating point value */
    FP16 = 11,         /**< 16bit floating point value, 5 bit for exponent, 10 bit for mantisa */
    BF16 = 12,         /**< 16bit floating point value, 8 bit for exponent, 7 bit for mantisa*/
    Q78 = 20,          /**< 16bit specific signed fixed point precision */
    I16 = 30,          /**< 16bit signed integer value */
    U8 = 40,           /**< 8bit unsigned integer value */
    I8 = 50,           /**< 8bit signed integer value */
    U16 = 60,          /**< 16bit unsigned integer value */
    I32 = 70,          /**< 32bit signed integer value */
    I64 = 72,          /**< 64bit signed integer value */
    U64 = 73,          /**< 64bit unsigned integer value */
    BIN = 71,          /**< 1bit integer value */
    BOOL8 = 41,         /**< 8bit bool type */
    CUSTOM = 80        /**< custom precision has it's own name and size of elements */
};

typedef std::vector<size_t> VShape;
struct IOInfo {
    VShape _shape;
    evPrecision _precision = UNSPECIFIED;
};

typedef std::map<std::string, IOInfo> VInputInfo;
typedef std::map<std::string, IOInfo> VOutputInfo;

inline size_t product(const VShape &dims) noexcept {
    if (dims.empty()) return 0;
    return std::accumulate(std::begin(dims), std::end(dims), (size_t)1, std::multiplies<size_t>());
}

struct VBlob {
    VShape _shape;
    void* _data = nullptr;
    evPrecision _precision = FP32;
    bool _ownMemory = true;
    std::string _layout = "NCHW"; // NCHW/NHWC
    std::string _colourFormat = "BGR"; // BGR /RGB

    ~VBlob() {
        if (_ownMemory) {
            free(_data);
        }
    }
};

class Backend {
public:
    struct InferenceMetrics {
        int nRuns = 0;
        double minDuration = std::numeric_limits<double>::max();
        double maxDuration = 0;
        double totalTime = 0;

        virtual ~InferenceMetrics() { }  // Type has to be polymorphic
    };

    /**
     * TODO(amalyshe) add more extended error handling mechanism instead of just bool
     */
    virtual bool loadModel(const VLauncher* launcher, const std::string &device,
                           const std::vector<std::string>& outputs,
                           const std::map<std::string, std::string> &config) = 0;
    /**
     * TODO(amalyshe) need to remove export of shared_ptr between interfaces
     */
    virtual std::shared_ptr<InferenceMetrics> process(bool streamOutput = false) = 0;
    virtual void report(const InferenceMetrics &im) const = 0;
    virtual bool infer() = 0;

    virtual std::shared_ptr<VBlob> getBlob(const std::string &name) = 0;

    /**
     * deteltes current object. required to avoid collisions between different C++ libraries if ever
     */
    virtual void release() = 0;

    virtual VInputInfo getInputDataMap() const = 0;
    virtual VOutputInfo getOutputDataMap() const = 0;

    virtual ~Backend() { }
};
