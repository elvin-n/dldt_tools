// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "calibrator_processors.h"
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <iomanip>
#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <list>
#include <limits>
#include "details/ie_cnn_network_tools.h"
#include "details/caseless.hpp"
#include "user_exception.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

using InferenceEngine::details::InferenceEngineException;

CNNLayerPtr Int8Calibrator::addScaleShiftBeforeLayer(std::string name, CNNLayer::Ptr beforeLayer, size_t port, std::vector<float> scale) {
    if (beforeLayer->insData.size() < port) {
        THROW_USER_EXCEPTION(2) << "cannot find appropraite port for addScaleShiftBeforeLayer";
    }

    DataPtr pData = beforeLayer->insData[port].lock();
    LayerParams params;
    params.name = name;
    params.precision = Precision::FP32;
    params.type = "ScaleShift";
    CNNLayerPtr lptr = std::make_shared<ScaleShiftLayer>(params);
    ScaleShiftLayer *pScaleShift = dynamic_cast<ScaleShiftLayer *>(lptr.get());

    SizeVector wdims({ pData->dims[2] });

    if (scale.size() == 1) {
        scale.resize(wdims[0]);
        for (size_t i = 1; i < wdims[0]; i++) {
            scale[i] = scale[0];
        }
    }

    if (scale.size() != pData->dims[2]) {
        THROW_IE_EXCEPTION << "Failed to add scaleshift before " << beforeLayer->name << " due to scales and layer output dims incossitency";
    }

    Blob::Ptr weights = nullptr;
    weights = make_shared_blob<float>(Precision::FP32, Layout::C, wdims);
    weights->allocate();
    float *buffer = weights->buffer().as<float *>();
    if (buffer == nullptr) {
        THROW_IE_EXCEPTION << "Could not allocate weights buffer";
    }
    for (size_t i = 0; i < pData->dims[2]; i++) {
        buffer[i] = scale[i];
    }
    pScaleShift->_weights = weights;


    SizeVector bdims({ pData->dims[2] });
    Blob::Ptr biases = nullptr;
    biases = make_shared_blob<float>(Precision::FP32, Layout::C, bdims);
    biases->allocate();
    buffer = biases->buffer().as<float *>();
    for (size_t i = 0; i < pData->dims[2]; i++) {
        buffer[i] = 0.f;
    }
    pScaleShift->_biases = biases;

    Data *edge2 = new Data(*pData.get());
    DataPtr newEdge(edge2);
    lptr->insData.push_back(pData);
    lptr->outData.push_back(newEdge);
    newEdge->name = /*"EdgeAfter_" +*/ params.name;
    newEdge->creatorLayer = lptr;
    newEdge->inputTo.clear();
    newEdge->inputTo[beforeLayer->name] = beforeLayer;

    pData->inputTo.erase(beforeLayer->name);
    pData->inputTo[params.name] = lptr;

    for (size_t i = 0; i < beforeLayer->insData.size(); i++) {
        DataPtr d = beforeLayer->insData[i].lock();
        if (d == pData) {
            beforeLayer->insData[i] = newEdge;
            break;
        }
    }
    return lptr;
}


void Int8Calibrator::collectFP32Statistic() {
    _collectStatistic = true;
    _collectFP32ActivationsKLHist = false;

    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());
    if (_cBatch == 0) {
        // Zero means "take batch value from the IR"
        _cBatch = networkReaderC.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        auto input_shapes = networkReaderC.getNetwork().getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = _cBatch;
        input_shapes[input_name] = input_shape;
        networkReaderC.getNetwork().reshape(input_shapes);
    }

    auto network = networkReaderC.getNetwork();


    std::vector<CNNLayerPtr> layersAfterInputs;

    std::string hackPrefix = "scaleshifted_input:";

    for (auto &&layer : network) {
        if (layer->insData.size() > 0) {
            std::string inName = layer->input()->getName();
            for (auto &&input : network.getInputsInfo()) {
                if (inName == input.first) {
                    layersAfterInputs.push_back(layer);
                    _inputsFromLayers[hackPrefix + layer->name] = inName;
                }
            }
        }
    }

    for (auto &&layer : layersAfterInputs) {
        std::string firstInputName = hackPrefix + layer->name;
        auto scaleShiftLayer = addScaleShiftBeforeLayer(firstInputName, layer, 0, { 1.f });
        ((ICNNNetwork&)network).addLayer(scaleShiftLayer);
    }


    // 1. add all layers as output one
    for (auto &&layer : network) {
        std::string layerType = network.getLayerByName(layer->name.c_str())->type;
        if (layerType != "Const") {
            if (/*layerType != "Split" &&*/layerType != "Input") {
                network.addOutput(layer->name);
            }
            _statData.registerLayer(layer->name);
        }
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();
}

void Int8Calibrator::collectFP32ActivationsKL() {
    _collectStatistic = false;
	_collectFP32ActivationsKLHist = true;

    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());
    if (_cBatch != 0) {
        auto input_shapes = networkReaderC.getNetwork().getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = _cBatch;
        input_shapes[input_name] = input_shape;
        networkReaderC.getNetwork().reshape(input_shapes);
    }

    auto network = networkReaderC.getNetwork();

    std::set<CNNLayerPtr> toCalibrate;
    // defines layers which should be calibrated, put them into the container
    std::vector<CNNLayerPtr> ordered = InferenceEngine::details::CNNNetSortTopologically(networkReaderC.getNetwork());
    for (auto l : ordered) {
        if (CaselessEq<std::string>()(l->type, "convolution")
            || CaselessEq<std::string>()(l->type, "eltwise")
            || CaselessEq<std::string>()(l->type, "pooling")
            || CaselessEq<std::string>()(l->type, "relu")) { // relu is required only for eltwise case, should be removed
            toCalibrate.insert(l);
        }
    }

    // 1. add layers before calibrated one as output layers
    for (auto layer : toCalibrate) {
        for (size_t j = 0; j < layer->insData.size(); j++) {
            auto lOutput = network.getLayerByName(layer->insData[j].lock()->getCreatorLayer().lock()->name.c_str());
            if (lOutput) {
                network.addOutput(lOutput->name);
				// identify if layer produce data only to DW convolution:
				bool nextDWConvolution = false;
				size_t channels = _statData.getNumberChannels(lOutput->name);
				std::vector<TensorHistogram> th;
/*                    if (CaselessEq<std::string>()(layer->type, "convolution")) {
					ConvolutionLayer* conv = dynamic_cast<ConvolutionLayer*>(layer.get());
					if (conv->_group == channels &&
						lOutput->outData.size() == 1 &&
						lOutput->outData[0]->inputTo.size() == 1 ) {
						nextDWConvolution = true;
					}
				}
*/

				if (nextDWConvolution == false) {
					float minVal = std::numeric_limits<float>::max();
					float maxVal = std::numeric_limits<float>::min();

					for (size_t i = 0; i < channels; i++) {
						float minTmp, maxTmp;
						_statData.getDataMinMax(lOutput->name, i, minTmp, maxTmp, 100.f);
						if (minVal > minTmp) {
							minVal = minTmp;
						}
						if (maxVal < maxTmp) {
							maxVal = maxTmp;
						}
					}
					TensorHistogram ts(minVal, maxVal);
					th.push_back(ts);
				} else {
					for (size_t i = 0; i < channels; i++) {
						float minTmp, maxTmp;
						_statData.getDataMinMax(lOutput->name, i, minTmp, maxTmp, 100.f);
						TensorHistogram ts(minTmp, maxTmp);
						th.push_back(ts);
					}
				}
				_agregatedHist[lOutput->name] = th;

            }
        }
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();
}

void Int8Calibrator::collectCalibrationStatistic(size_t pics) {
	// http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
	// https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    if (_collectFP32ActivationsKLHist) {
        auto it = _agregatedHist.begin();
        while (it != _agregatedHist.end()) {
            Blob::Ptr expected = _inferRequestI8C.GetBlob(it->first);
            float* fexpected = expected->buffer().as<float*>();
            if (it->second.size() == 1) {
                it->second[0].addValues(fexpected, expected->size());
            } else {
                size_t channels = it->second.size();
                Blob::Ptr expected = _inferRequestI8C.GetBlob(it->first);
                size_t plainSize = expected->size() / channels;
                float* fexpected = expected->buffer().as<float*>();
                for (size_t i = 0; i < channels; i++) {
                    it->second[i].addValues(fexpected, plainSize);
                    fexpected += plainSize;
                }
            }
            it++;
        }
    }

    if (_collectStatistic) {
        for (auto l : _statData.registeredLayers()) {
            auto outBlob = _inferRequestI8C.GetBlob(l);

            std::string outName = l;
            if (_inputsFromLayers.find(l) != _inputsFromLayers.end()) {
                outName = _inputsFromLayers[l];
            }

            size_t N, C;
            if (outBlob->dims().size() == 4 && outBlob->layout() == Layout::NCHW) {
                // TODO(amalyshe) cahnge to using of tensor desc
                N = pics;
                C = outBlob->dims()[2];
            } else if (outBlob->dims().size() == 2 && outBlob->layout() == Layout::NC) {
                N = pics;
                C = outBlob->dims()[0];
            } else {
                continue;
            }

            // Counting min/max outputs per channel
            for (size_t n = 0; n < N; n++) {
                if (outBlob->dims().size() == 4) {
                    size_t _HW = outBlob->dims()[0] * outBlob->dims()[1];
                    for (size_t c = 0; c < C; c++) {
                        if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                            float *ptr = &outBlob->buffer().as<float *>()[(n * C + c) * _HW];
                            _statData.addTensorStatistics(outName, c, ptr, _HW);
                        } else if (outBlob->getTensorDesc().getPrecision() == Precision::U8) {
                            uint8_t *ptr = &outBlob->buffer().as<uint8_t *>()[(n * C + c) * _HW];
                            _statData.addTensorStatistics(outName, c, ptr, _HW);
                        } else {
                            throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                        }
                    }
                } else if (outBlob->dims().size() == 2) {
                    if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                        float *ptr = &outBlob->buffer().as<float *>()[n * C];
                        _statData.addTensorStatistics(outName, 0, ptr, C);
                    } else if (outBlob->getTensorDesc().getPrecision() == Precision::U8) {
                        uint8_t *ptr = &outBlob->buffer().as<uint8_t *>()[n * C];
                        _statData.addTensorStatistics(outName, 0, ptr, C);
                    } else {
                        throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                    }
                }
            }
        }
    }
}

InferenceEngine::NetworkStatsMap Int8Calibrator::optimizeActivationKL() {
    InferenceEngine::NetworkStatsMap netNodesStats;
    auto it = _agregatedHist.begin();
    while (it != _agregatedHist.end()) {
        NetworkNodeStatsPtr nodeStats;
        size_t channels = _statData.getNumberChannels(it->first);
        nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats(channels));
        netNodesStats[it->first] = nodeStats;

        for (size_t i = 0; i < it->second.size(); i++) {
//            std::cout << "Optimizing output from " << it->first << std::endl;
            TensorHistogramHelper helper(it->second[i]);
            std::pair<float, float> effective;
            helper.minimizeKlDivergence(effective);
//            std::cout << "KL optimization found min = " << effective.first << " max = " << effective.second <<std::endl;
//            std::cout << "Absolut min = " << it->second[0].minValue() << " max = " << it->second[0].maxValue() <<std::endl;

            if (it->second.size() == 1) {
                for (size_t c = 0; c < channels; c++) {
                    nodeStats->_minOutputs[c] = effective.first;
                    nodeStats->_maxOutputs[c] = effective.second;
                }
            } else {
                nodeStats->_minOutputs[i] = effective.first;
                nodeStats->_maxOutputs[i] = effective.second;
            }
        }

        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::min();

        for (size_t c = 0; c < channels; c++) {
            if (minVal > nodeStats->_minOutputs[c]) {
                minVal = nodeStats->_minOutputs[c];
            }
            if (maxVal < nodeStats->_maxOutputs[c]) {
                maxVal = nodeStats->_maxOutputs[c];
            }
        }
        for (size_t c = 0; c < channels; c++) {
            if (nodeStats->_minOutputs[c] == 0.f &&
                nodeStats->_maxOutputs[c] == 0.f) {
                nodeStats->_minOutputs[c] = minVal;
                nodeStats->_maxOutputs[c] = maxVal;
            }
        }

        it++;
    }

    it = _agregatedHist.begin();
    while (it != _agregatedHist.end()) {
        auto pool = networkReaderC.getNetwork().getLayerByName(it->first.c_str());
        if (CaselessEq<std::string>()(pool->type, "pooling")) {
            auto itStat = netNodesStats.find(it->first);
            if (itStat != netNodesStats.end()) {
                auto itStat2 = netNodesStats.find(pool->insData[0].lock()->name);
                if (itStat2 == netNodesStats.end()) {
                    // case of conv->relu->pooling->conv
                    netNodesStats[pool->insData[0].lock()->name] = itStat->second;
                } else {
                    // a case of average pooling having stat from both sides - should use only previosu one
                    netNodesStats[it->first] = netNodesStats[pool->insData[0].lock()->name];
                }

            }
        }
        it++;
    }

    it = _agregatedHist.begin();
    while (it != _agregatedHist.end()) {
        auto concat = networkReaderC.getNetwork().getLayerByName(it->first.c_str());
        if (CaselessEq<std::string>()(concat->type, "concat")) {
            auto itStat = netNodesStats.find(it->first);
            if (itStat != netNodesStats.end()) {
                // split statistics to the input channels
                size_t curOutChannel = 0;
                for (auto i : concat->insData) {
                    // define number of channels, copy this number of channels from concat outputs
                    auto iLayer = i.lock()->getCreatorLayer().lock();

                    NetworkNodeStatsPtr nodeStats;
                    size_t channels = _statData.getNumberChannels(iLayer->name);
                    nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats(channels));
                    netNodesStats[iLayer->name] = nodeStats;
                    for (size_t c = 0; c < channels; c++) {
                        nodeStats->_minOutputs[c] = itStat->second->_minOutputs[c + curOutChannel];
                        nodeStats->_maxOutputs[c] = itStat->second->_maxOutputs[c + curOutChannel];
                    }
                    curOutChannel += channels;
                }
            }
        }
        it++;
    }


    return netNodesStats;
}

//--------------------------------------------------------------------------------------------------

ClassificationCalibrator::ClassificationCalibrator(int nPictures, const std::string &flags_m,
                                                   const std::string &flags_d, const std::string &flags_i,
                                                   int flags_b, InferenceEngine::InferencePlugin plugin,
                                                   CsvDumper &dumper, const std::string &flags_l,
                                                     PreprocessingOptions preprocessingOptions, bool zeroBackground) :
    ClassificationProcessor(flags_m, flags_d, flags_i, flags_b,
                            plugin, dumper, flags_l,
                            preprocessingOptions, zeroBackground) {
    _modelFileNameI8C = modelFileName;
    _pluginI8C = plugin;
    _nPictures = nPictures;
    _cBatch = flags_b;
}

shared_ptr<Processor::InferenceMetrics> ClassificationCalibrator::Process(bool stream_output) {
    inferRequest = _inferRequestI8C;
    int top1Result = 0, total = 0;

    ClassificationSetGenerator generator;

    try {
        generator.readLabels(labelFileName);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        slog::warn << "Can't read labels file " << labelFileName << slog::endl;
        slog::warn << "Error: " << ex.what() << slog::endl;
    }
    auto validationMap = generator.getValidationMap(imagesPath);

    if (validationMap.empty()) {
        THROW_IE_EXCEPTION << "The validation dataset in " << imagesPath << "is empty. Check the dataset file or folder and the labels file";
    }

    ImageDecoder decoder;

    // ----------------------------Do inference-------------------------------------------------------------
    std::vector<int> expected(batch);
    std::vector<std::string> files(batch);

    if (!_nPictures) {
        _nPictures = validationMap.size();
    }


    ConsoleProgress progress(_nPictures, stream_output);

    CalibrationMetrics im;

    std::string firstInputName = this->inputInfo.begin()->first;
    std::string firstOutputName = this->outInfo.begin()->first;
    auto firstInputBlob = inferRequest.GetBlob(firstInputName);
    auto firstOutputBlob = inferRequest.GetBlob(firstOutputName);

    size_t ipics = 0;
    auto iter = validationMap.begin();
    while (iter != validationMap.end() && ipics < _nPictures) {
        size_t b = 0;
        int filesWatched = 0;
        for (; b < batch && iter != validationMap.end() && ipics + b < _nPictures ; b++, iter++, filesWatched++) {
            expected[b] = iter->first;
            try {
                decoder.insertIntoBlob(iter->second, b, *firstInputBlob, preprocessingOptions);
                files[b] = iter->second;
            } catch (const InferenceEngineException &iex) {
                slog::warn << "Can't read file " << iter->second << slog::endl;
                slog::warn << "Error: " << iex.what() << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
        }
        ipics += batch;

        Infer(progress, filesWatched, im);
        collectCalibrationStatistic(b);

        std::vector<unsigned> results;
        InferenceEngine::TopResults(1, *firstOutputBlob, results);
        for (size_t i = 0; i < b; i++) {
            int expc = expected[i];
            if (zeroBackground) expc++;
            bool top1Scored = (static_cast<int>(results[i]) == expc);
            if (top1Scored) top1Result++;
            total++;
        }
    }
    progress.finish();

    if (total == 0) {
        throw std::logic_error("total can't be equal to zero");
    }

    im.AccuracyResult = static_cast<float>(top1Result) / static_cast<float>(total);

    return std::shared_ptr<Processor::InferenceMetrics>(new CalibrationMetrics(im));
}

