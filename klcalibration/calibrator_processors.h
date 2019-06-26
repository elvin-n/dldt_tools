// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include "inference_engine.hpp"
#include "ClassificationProcessor.hpp"
#include "data_stats.h"
#include <map>
#include <memory>

/**
 * Calibrator class representing unified stages for calibration of any kind of networks
*/
class Int8Calibrator {
public:
    /**
     * Initializes state to collect accuracy of FP32 network and collect statistic
     * of activations. The statistic of activations is stored in _statData and has all max/min for all
     * layers and for all pictures
     * The inference of all pictures and real collect of the statistic happen  during call of
     * Processor::Process()
     */
    void collectFP32Statistic();

    void collectFP32ActivationsKL();
    InferenceEngine::NetworkStatsMap optimizeActivationKL();

protected:
    /**
     * This function should be called from final callibrator after and each Infer for each picture
     * It calculates by layer accuracy drop and as well it also collect activation values statistic
     */
    void collectCalibrationStatistic(size_t pics);

    bool _collectStatistic = true;
    bool _collectFP32ActivationsKLHist = false;

    InferencePlugin _pluginI8C;
    std::string _modelFileNameI8C;
    InferenceEngine::CNNNetReader networkReaderC;
    InferenceEngine::InferRequest _inferRequestI8C;
    int _cBatch = 0;

    size_t _nPictures = 0;

private:
    /**
     * helper function for getting statistic for input layers. For getting statistic for them, we are
     * adding scalshift just after the input with scale == 1 and shift == 0
     */
    CNNLayerPtr addScaleShiftBeforeLayer(std::string name, InferenceEngine::CNNLayer::Ptr beforeLayer,
                                         size_t port, std::vector<float> scale);

    std::map<std::string, std::string> _inputsFromLayers;
    AggregatedDataStats _statData;
    std::map<std::string, std::vector<TensorHistogram > > _agregatedHist;
};

/**
 * This class represents the only one generalized metric which will be used for comparison of
 * accuracy drop
 */
struct CalibrationMetrics : public ClassificationProcessor::InferenceMetrics {
public:
    float AccuracyResult = 0;
};

/**
 * Сalibration class for classification networks.
 * Responsible for proper post processing of results and calculate of Top1 metric which is used as
 * universal metric for accuracy and particiapted in verification of accuracy drop
 */
class ClassificationCalibrator : public ClassificationProcessor, public Int8Calibrator {
public:
    ClassificationCalibrator(int nPictures, const std::string &flags_m, const std::string &flags_d,
                             const std::string &flags_i, int flags_b,
                              InferenceEngine::InferencePlugin plugin, CsvDumper &dumper, const std::string &flags_l,
                              PreprocessingOptions preprocessingOptions, bool zeroBackground);

    shared_ptr<InferenceMetrics> Process(bool stream_output = false) override;
};

