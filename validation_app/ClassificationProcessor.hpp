// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <limits>
#include <string>
#include <memory>

#include "classification_set_generator.hpp"
#include "Processor.hpp"

using namespace std;

class ClassificationProcessor : public Processor {
    const int TOP_COUNT = 5;

    struct ClassificationInferenceMetrics : public InferenceMetrics {
    public:
        int top1Result = 0;
        int topCountResult = 0;
        int total = 0;
    };

protected:
    bool zeroBackground;
public:
    ClassificationProcessor(Backend *backend,
                            const VLauncher * launcher,
                            const std::vector<std::string> &outputs,
                            const std::string &flags_i,
                            int flags_b,
                            CsvDumper& dumper,
                            const VDataset *dataset);

    std::shared_ptr<InferenceMetrics> Process(bool stream_output);
    virtual void Report(const InferenceMetrics& im);
    virtual ~ClassificationProcessor() { }
};
