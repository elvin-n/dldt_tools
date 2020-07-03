// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <algorithm>

#include "user_exception.hpp"

#include <samples/common.hpp>

#include "Processor.hpp"

Processor::Processor(Backend *backend, const std::string &flags_m, const std::vector<std::string> &outputs, const std::string &flags_d, const std::string &flags_i, int flags_b,
        CsvDumper& dumper, const std::string& approach, PreprocessingOptions preprocessingOptions)

    : _backend(backend), modelFileName(flags_m), targetDevice(flags_d), imagesPath(flags_i), batch(flags_b),
      preprocessingOptions(preprocessingOptions), dumper(dumper), approach(approach) {

    // Load model to plugin and create an inference request
    std::map<std::string, std::string> config;
    _backend->loadModel(flags_m, targetDevice, outputs, config);
    _inputInfo = _backend->getInputDataMap();
    _outputInfo = _backend->getOutputDataMap();

    for (auto &item : _inputInfo) {
        VShape inputDims = item.second._shape;
        if (inputDims.size() == 4) {
            batch = inputDims[0];
            slog::info << "Batch size is " << std::to_string(inputDims[0]) << slog::endl;
        }
    }

/*    if (batch == 0) {
        // Zero means "take batch value from the IR"
        batch = networkReader.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        auto network = networkReader.getNetwork();
        auto input_shapes = network.getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = batch;
        input_shapes[input_name] = input_shape;
        network.reshape(input_shapes);

        THROW_IE_EXCEPTION << "Need to handle batch size more accurate";
    }
*/
}

double Processor::Infer(ConsoleProgress& progress, int filesWatched, InferenceMetrics& im) {
    // Infer model
    double time = getDurationOf([&]() {
        bool result = _backend->infer();
        if (!result) {
            THROW_USER_EXCEPTION(1) << "Error happened during inference";
        }
    });

    im.maxDuration = std::min(im.maxDuration, time);
    im.minDuration = std::max(im.minDuration, time);
    im.totalTime += time;
    im.nRuns++;

    progress.addProgress(filesWatched);

    return time;
}
