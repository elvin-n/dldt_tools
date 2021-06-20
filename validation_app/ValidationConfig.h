// Copyright 2021 the dldt tools authors. All rights reserved.
// Use of this source code is governed by a BSD-style license

#pragma once

#include <string>
#include <vector>
#include <map>

#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

struct VInput {
  std::string name_;
  std::string type_;
  std::string layout_;
  std::vector<int> shape_;
};

struct VLauncher {
  std::string framework_;
  std::string device_;
  std::string adapter_;
  std::vector<VInput> inputs_;

  std::string model_;
};

struct VPreprocessingStep {
  std::string type_;
  int size_;
  std::string aspect_ratio_scale_;
  bool use_pillow_;
  std::string interpolation_;
  std::vector <float> std_;
  std::vector<float> mean_;
};

struct VDataset {
  std::string name_;
  std::string reader_;
  std::vector<VPreprocessingStep> preprocSteps_;
};

struct Model {
  std::string name_;
  std::vector<VLauncher> launchers_;
  std::vector<VDataset> datasets_;
};

class ValidationConfig {
public:
  ValidationConfig(YAML::Node& config);
  const VLauncher*  getLauncherByFramwork(const std::string &framework, const std::string &device) const;
  const VDataset* getDatasetsByFramwork(const std::string &framework) const;
private:
  std::vector<Model> models_;
};

