// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include "user_exception.hpp"

#include "ClassificationProcessor.hpp"
#include "Processor.hpp"

ClassificationProcessor::ClassificationProcessor(Backend *backend, const std::string &flags_m, const std::vector<std::string> &outputs,
                                                 const std::string &flags_d, const std::string &flags_i, int flags_b,
        CsvDumper& dumper, const std::string& flags_l,
        PreprocessingOptions preprocessingOptions, bool zeroBackground)
    : Processor(backend, flags_m, outputs, flags_d, flags_i, flags_b, dumper, "Classification network", preprocessingOptions), zeroBackground(zeroBackground) {

    // Change path to labels file if necessary
    if (flags_l.empty()) {
        labelFileName = fileNameNoExt(modelFileName) + ".labels";
    } else {
        labelFileName = flags_l;
    }
}

ClassificationProcessor::ClassificationProcessor(Backend *backend, const std::string &flags_m, const std::vector<std::string> &outputs,
                                                 const std::string &flags_d, const std::string &flags_i, int flags_b,
                                                 CsvDumper &dumper, const std::string &flags_l, bool zeroBackground)
    : ClassificationProcessor(backend, flags_m, outputs, flags_d, flags_i, flags_b, dumper, flags_l,
            PreprocessingOptions(false, ResizeCropPolicy::ResizeThenCrop, 256, 256), zeroBackground) {
}

inline void TopResults(unsigned int n, shared_ptr<VBlob> input, std::vector<unsigned> &output) {
    VShape dims = input->_shape;
    size_t input_rank = dims.size();
    if (!input_rank || !dims[0])
        THROW_USER_EXCEPTION(1) << "Input blob has incorrect dimensions!";
    size_t batchSize = dims[0];
    size_t blobSize = product(input->_shape);
    std::vector<unsigned> indexes(blobSize / batchSize);

    n = static_cast<unsigned>(std::min<size_t>((size_t)n, blobSize));

    output.resize(n * batchSize);

    for (size_t i = 0; i < batchSize; i++) {
        size_t offset = i *(blobSize / batchSize);
        float *batchData = static_cast<float*>(input->_data);
        batchData += offset;

        std::iota(std::begin(indexes), std::end(indexes), 0);
        std::partial_sort(std::begin(indexes), std::begin(indexes) + n, std::end(indexes),
                          [&batchData](unsigned l, unsigned r) {
                              return batchData[l] > batchData[r];
                          });
        for (unsigned j = 0; j < n; j++) {
            output.at(i * n + j) = indexes.at(j);
        }
    }
}


std::shared_ptr<Processor::InferenceMetrics> ClassificationProcessor::Process(bool stream_output) {
     slog::info << "Collecting labels" << slog::endl;
     ClassificationSetGenerator generator;

     auto validationMap = generator.getValidationMap(imagesPath);
     ImageDecoder decoder;

     // ----------------------------Do inference-------------------------------------------------------------
     slog::info << "Starting inference" << slog::endl;

     std::vector<int> expected(batch);
     std::vector<std::string> files(batch);

     ConsoleProgress progress(validationMap.size(), stream_output);

     ClassificationInferenceMetrics im;

     std::string firstInputName = this->_inputInfo.begin()->first;
     std::string firstOutputName = this->_outputInfo.begin()->first;
     auto firstInputBlob = _backend->getBlob(firstInputName);
     auto firstOutputBlob = _backend->getBlob(firstOutputName);


     auto iter = validationMap.begin();
     while (iter != validationMap.end()) {
         size_t b = 0;
         int filesWatched = 0;
         for (; b < batch && iter != validationMap.end(); b++, iter++, filesWatched++) {
             expected[b] = iter->first;
             try {
                 decoder.insertIntoBlob(iter->second, b, firstInputBlob, preprocessingOptions);
                 files[b] = iter->second;
             } catch (const std::exception& iex) {
                 slog::warn << "Can't read file " << iter->second << slog::endl;
                 slog::warn << "Error: " << iex.what() << slog::endl;
                 // Could be some non-image file in directory
                 b--;
                 continue;
             }
         }
         Infer(progress, filesWatched, im);

         std::vector<unsigned> results;
         auto firstOutputData = static_cast<float*>(firstOutputBlob->_data);
         TopResults(TOP_COUNT, firstOutputBlob, results);

         for (size_t i = 0; i < b; i++) {
             int expc = expected[i];
             if (zeroBackground) expc++;

             bool top1Scored = (static_cast<int>(results[0 + TOP_COUNT * i]) == expc);
             dumper << "\"" + files[i] + "\"" << top1Scored;
             if (top1Scored) im.top1Result++;
             for (int j = 0; j < TOP_COUNT; j++) {
                 unsigned classId = results[j + TOP_COUNT * i];
                 if (static_cast<int>(classId) == expc) {
                     im.topCountResult++;
                 }
                 dumper << classId << firstOutputData[classId + i * (product(firstOutputBlob->_shape) / batch)];
             }
             dumper.endLine();
             im.total++;
         }
     }
     progress.finish();

     return std::shared_ptr<Processor::InferenceMetrics>(new ClassificationInferenceMetrics(im));
}

void ClassificationProcessor::Report(const Processor::InferenceMetrics& im) {
    Processor::Report(im);
    if (im.nRuns > 0) {
        const ClassificationInferenceMetrics& cim = dynamic_cast<const ClassificationInferenceMetrics&>(im);

        cout << "Top1 accuracy: " << OUTPUT_FLOATING(100.0 * cim.top1Result / cim.total) << "% (" << cim.top1Result << " of "
                << cim.total << " images were detected correctly, top class is correct)" << "\n";
        cout << "Top5 accuracy: " << OUTPUT_FLOATING(100.0 * cim.topCountResult / cim.total) << "% (" << cim.topCountResult << " of "
            << cim.total << " images were detected correctly, top five classes contain required class)" << "\n";
    }
}

