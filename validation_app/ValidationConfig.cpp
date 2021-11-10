// Copyright 2021 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#include "ValidationConfig.h"
#include <iostream>
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

static std::vector<float> convertStrParams(const std::string& params) {
  std::vector<float> ret;
  if (params.find(",") == std::string::npos) {
    float i = std::stof(params);
    ret.push_back(i);
  } else {
    // look for coma as delimiter, expecting three values
    int posb1 = params.find("(");
    int posb2 = params.find(")");
    int pos = params.find(",");
    int pos2 = params.find(",", pos+1);
    ret.push_back(std::stof(params.substr(posb1 + 1, pos - posb1 - 1)));
    ret.push_back(std::stof(params.substr(pos + 1, pos2 - pos - 1)));
    ret.push_back(std::stof(params.substr(pos2 + 1, posb2 - pos2 - 1)));
  }
  return ret;
}

ValidationConfig::ValidationConfig(YAML::Node* config) {
  // going over all models in the yaml file
  auto models = (*config)["models"];
  if (models) {
    for (std::size_t i = 0; i < models.size(); i++) {
      Model model;
      model.name_ = models[i]["name"].as<std::string>();

      auto ylaunchers = models[i]["launchers"];
      for (size_t j = 0; j < ylaunchers.size(); j++) {
        VLauncher l;
        l.framework_ = ylaunchers[j]["framework"].as<std::string>();
        l.adapter_ = ylaunchers[j]["adapter"].as<std::string>();
        if (ylaunchers[j]["device"]) {
          l.device_ = ylaunchers[j]["device"].as<std::string>();
        }

        auto inputs = ylaunchers[j]["inputs"];
        for (size_t k = 0; k < inputs.size(); k++) {
          VInput inp;
          inp.name_ = inputs[k]["name"].as<std::string>();
          inp.layout_ = inputs[k]["layout"].IsDefined() ? inputs[k]["layout"].as<std::string>() : "";
          l.inputs_.push_back(inp);
        }

        auto modelName = ylaunchers[j]["model"];
        if (modelName.IsDefined()) {
          l.model_ = modelName.as<std::string>();
          if (l.framework_ == "tvm") {
#ifdef __APPLE__
            l.model_ = l.model_ + ".dylib";
#elif _WIN32
            l.model_ = l.model_ + ".dll";
#else
            l.model_ = l.model_ + ".so";
#endif
          }
        } else {
          if (l.framework_ == "tvm") {
  #ifdef __APPLE__
            l.model_ = model.name_ + ".dylib";
  #elif _WIN32
            l.model_ = model.name_ + ".dll";
  #else
            l.model_ = model.name_ + ".so";
  #endif
          } else if (l.framework_ == "onnx_runtime") {
            l.model_ = model.name_ + ".onnx";
          } else if (l.framework_ == "dlsdk") {
            l.model_ = model.name_ + ".xml";
          } else if (l.framework_ == "caffe") {
            l.model_ = model.name_ + ".caffenet";
          } else if (l.framework_ == "tf") {
            l.model_ = model.name_ + ".pb";
          } else if (l.framework_ == "snpe") {
            l.model_ = model.name_ + ".dlc";
          } else {
            throw std::string("cannot create name of the model for this framework");
          }
        }
        // TODO(amalyshe): make reading of inputs
        model.launchers_.push_back(l);
      }

      auto ydatasets = models[i]["datasets"];
      for (size_t j = 0; j < ydatasets.size(); j++) {
        VDataset d;
        d.name_ = ydatasets[j]["name"].as<std::string>();
        if (ydatasets[j]["reader"]) {
          d.reader_ = ydatasets[j]["reader"].as<std::string>();
        }
        auto ypreprocessing = ydatasets[j]["preprocessing"];
        for (size_t k = 0; k < ypreprocessing.size(); k++) {
          VPreprocessingStep p;
          p.type_ = ypreprocessing[k]["type"].as<std::string>();
          // read size:
          if (ypreprocessing[k]["size"]) {
            p.size_ = ypreprocessing[k]["size"].as<int>();
          }
          if (ypreprocessing[k]["interpolation"]) {
            p.interpolation_ = ypreprocessing[k]["interpolation"].as<std::string>();
          }
          if (ypreprocessing[k]["aspect_ratio_scale"]) {
            p.aspect_ratio_scale_ = ypreprocessing[k]["aspect_ratio_scale"].as<std::string>();
          }
          if (ypreprocessing[k]["use_pillow"]) {
            p.use_pillow_ = ypreprocessing[k]["use_pillow"].as<bool>();
          }
          if (ypreprocessing[k]["std"]) {
            p.std_ = convertStrParams(ypreprocessing[k]["std"].as<std::string>());
          }
          if (ypreprocessing[k]["mean"]) {
            p.mean_ = convertStrParams(ypreprocessing[k]["mean"].as<std::string>());
          }
          d.preprocSteps_.push_back(p);
        }
        model.datasets_.push_back(d);
      }
      models_.push_back(model);
    }
  }
}

const VLauncher* ValidationConfig::getLauncherByFramwork(const std::string &framework, const std::string & device) const {
  // look for the requested framework
  for (const auto &m : models_) {
    for (const auto &l : m.launchers_) {
      if (l.framework_ == framework &&
          (l.device_ == device || l.device_ == "")) {
        return &l;
      }
    }
  }
  return nullptr;
}
const VDataset* ValidationConfig::getDatasetsByFramwork(const std::string &framework) const {
  // look for the requested framework
  for (const auto &m : models_) {
    for (const auto &l : m.launchers_) {
      if (l.framework_ == framework) {
        return &m.datasets_[0];
      }
    }
  }
  return nullptr;
}
