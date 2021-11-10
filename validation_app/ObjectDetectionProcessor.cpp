// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <map>
#include <list>
#include <algorithm>
#include <memory>
#include <utility>

#include "ObjectDetectionProcessor.hpp"
#include "Processor.hpp"
#include "user_exception.hpp"

#include <samples/common.hpp>
#include <samples/slog.hpp>

ObjectDetectionProcessor::ObjectDetectionProcessor(Backend *backend,
                                                   const VLauncher *launcher,
                                                   const std::vector<std::string> &outputs,
                                                   const std::string &flags_i,
                                                   const std::string &subdir,
                                                   int flags_b,
                                                   double threshold,
                                                   CsvDumper &dumper,
                                                   const std::string &flags_a,
                                                   const std::string &classes_list_file,
                                                   const VDataset *dataset,
                                                   bool scaleProposalToInputSize)
  : Processor(backend, launcher, outputs, launcher->device_, flags_i, flags_b, dumper, dataset->preprocSteps_),
              annotationsPath(flags_a), subdir(subdir), threshold(threshold), scaleProposalToInputSize(scaleProposalToInputSize) {
    // To support faster-rcnn having several inputs we need to identify input dedicated for image correctly
    for (auto &item : _inputInfo) {
        if (item.second._shape.size() == 4) {
            inputDims = item.second._shape;
            picInputName = item.first;
        } else if (item.second._shape.size() == 2) {
            auto inputScale = _backend->getBlob(item.first);
            float *sdata = static_cast<float *>(inputScale->_data);
            sdata[0] = 600.f;
            sdata[1] = 1024.f;
            sdata[2] = 1.f;
        }
    }

    std::ifstream clf(classes_list_file);
    if (!clf) {
        throw UserException(1) <<  "Classes list file \"" << classes_list_file << "\" not found or inaccessible";
    }

    while (!clf.eof()) {
        std::string line;
        std::getline(clf, line, '\n');

        if (line != "") {
            // look for the latest space symbol
            size_t spos = line.find_last_of(" ");
            std::string className;
            int classId;
            className = line.substr(0, spos);
            std::string strId = line.substr(spos, line.length() - spos);
            istringstream streamId(strId);
            streamId >> classId;
            classes.insert(std::pair<std::string, int>(className, classId));
        }
    }
}

shared_ptr<Processor::InferenceMetrics> ObjectDetectionProcessor::Process(bool stream_output) {
    // Parsing PASCAL VOC2012 format
    VOCAnnotationParser vocAnnParser;
    slog::info << "Collecting VOC annotations from " << annotationsPath << slog::endl;
    VOCAnnotationCollector annCollector(annotationsPath);
    slog::info << annCollector.annotations().size() << " annotations collected" << slog::endl;

    if (annCollector.annotations().size() == 0) {
        ObjectDetectionInferenceMetrics emptyIM(this->threshold);

        return std::shared_ptr<InferenceMetrics>(new ObjectDetectionInferenceMetrics(emptyIM));
    }

    // Getting desired results from annotations
    std::map<std::string, ImageDescription> desiredForFiles;

    for (auto& ann : annCollector.annotations()) {
        std::list<DetectedObject> dobList;
        for (auto& obj : ann.objects) {
            DetectedObject dob(classes[obj.name], static_cast<float>(obj.bndbox.xmin),
                static_cast<float>(obj.bndbox.ymin), static_cast<float>(obj.bndbox.xmax),
                static_cast<float>(obj.bndbox.ymax), 1.0f, obj.difficult != 0);
            dobList.push_back(dob);
        }
        ImageDescription id(dobList);
        desiredForFiles.insert(std::pair<std::string, ImageDescription>(ann.folder + "/" + (!subdir.empty() ? subdir + "/" : "") + ann.filename, id));
    }
    // -----------------------------------------------------------------------------------------------------

    // ----------------------------Do inference-------------------------------------------------------------
    slog::info << "Starting inference" << slog::endl;

    std::vector<VOCAnnotation> expected(batch);

    ConsoleProgress progress(annCollector.annotations().size(), stream_output);

    ObjectDetectionInferenceMetrics im(threshold);

    vector<VOCAnnotation>::const_iterator iter = annCollector.annotations().begin();

    std::map<std::string, ImageDescription> scaledDesiredForFiles;

    auto firstInputBlob = _backend->getBlob(picInputName);

    ImageDecoder decoder;
    while (iter != annCollector.annotations().end()) {
        std::vector<std::string> files;
        size_t b = 0;

        int filesWatched = 0;
        for (; b < batch && iter != annCollector.annotations().end(); b++, iter++, filesWatched++) {
            expected[b] = *iter;
            string filename = iter->folder + "/" + (!subdir.empty() ? subdir + "/" : "") + iter->filename;
            try {
                Size orig_size = decoder.insertIntoBlob(std::string(imagesPath) + "/" + filename, b, firstInputBlob, preprocessingOptions);
                float scale_x, scale_y;

                scale_x = 1.0f / iter->size.width;  // orig_size.width;
                scale_y = 1.0f / iter->size.height;  // orig_size.height;

                if (scaleProposalToInputSize) {
                    // looking for the channel axis
                    if (firstInputBlob->_shape[1] == 3) {
                        scale_x *= firstInputBlob->_shape[3];
                        scale_y *= firstInputBlob->_shape[2];
                    } else if (firstInputBlob->_shape[3] == 3) {
                        scale_x *= firstInputBlob->_shape[2];
                        scale_y *= firstInputBlob->_shape[1];
                    }
                }

                // Scaling the desired result (taken from the annotation) to the network size
                scaledDesiredForFiles.insert(std::pair<std::string, ImageDescription>(filename, desiredForFiles.at(filename).scale(scale_x, scale_y)));

                files.push_back(filename);
            } catch (const std::exception& iex) {
                slog::warn << "Can't read file " << this->imagesPath + "/" + filename << slog::endl;
                slog::warn << "Error: " << iex.what() << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
        }

        if (files.size() == batch) {
            // Infer model
            Infer(progress, filesWatched, im);

            // Processing the inference result
            std::map<std::string, std::list<DetectedObject>> detectedObjects = processResult(files);

            for (auto f : detectedObjects) {
                for (auto o : f.second) {
                    dumper << f.first << o.objectType << o.ymin << o.xmin << o.ymax << o.xmax << o.prob;
                    dumper.endLine();
                }
            }

            // Calculating similarity
            //
            for (size_t b = 0; b < files.size(); b++) {
                ImageDescription result(detectedObjects[files[b]]);
                im.apc.consumeImage(result, scaledDesiredForFiles.at(files[b]));
            }
        }
    }
    progress.finish();

    // -----------------------------------------------------------------------------------------------------

    // ---------------------------Postprocess output blobs--------------------------------------------------
    slog::info << "Processing output blobs" << slog::endl;

    return std::shared_ptr<InferenceMetrics>(new ObjectDetectionInferenceMetrics(im));
}

void ObjectDetectionProcessor::Report(const Processor::InferenceMetrics& im) {
    const ObjectDetectionInferenceMetrics& odim = dynamic_cast<const ObjectDetectionInferenceMetrics&>(im);
    Processor::Report(im);
    if (im.nRuns > 0) {
        std::map<int, double> appc = odim.apc.calculateAveragePrecisionPerClass();

        std::cout << "Average precision per class table: " << std::endl << std::endl;
        std::cout << "Class\tAP" << std::endl;

        double mAP = 0;
        for (auto i : appc) {
            std::cout << std::fixed << std::setprecision(3) << i.first << "\t" << i.second << std::endl;
            mAP += i.second;
        }
        mAP /= appc.size();
        std::cout << std::endl << std::fixed << std::setprecision(4) << "Mean Average Precision (mAP): " << mAP << std::endl;
    }
}
