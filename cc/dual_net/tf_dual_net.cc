// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cc/dual_net/tf_dual_net.h"

#include <algorithm>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "cc/constants.h"
#include "cc/file/path.h"
#include "cc/logging.h"
#include "tensorflow/c/c_api.h"
#include "wtf/macros.h"

namespace minigo {
namespace {

class TfDualNet : public Model {
 public:
  TfDualNet(const std::string& graph_path,
            const FeatureDescriptor& feature_desc,
            const std::string& model_bytes);
  ~TfDualNet() override;

  void RunMany(const std::vector<const ModelInput*>& inputs,
               std::vector<ModelOutput*>* outputs,
               std::string* model_name) override;

 private:
  void Reserve(int capacity);

  TF_Graph* graph_ = nullptr;
  TF_Session* session_ = nullptr;
  TF_Status* status_ = nullptr;

  TF_Output input_op_;
  TF_Output policy_op_;
  TF_Output value_op_;

  std::vector<TF_Tensor*> inputs_;
  const std::string graph_path_;
  int batch_capacity_ = 0;
  TF_DataType input_type_ = TF_FLOAT;
};

TfDualNet::TfDualNet(const std::string& graph_path,
                     const FeatureDescriptor& feature_desc,
                     const std::string& model_bytes)
    : Model(std::string(file::Stem(file::Basename(graph_path))), feature_desc),
      graph_path_(graph_path) {
  status_ = TF_NewStatus();

  TF_SessionOptions* session_opts = TF_NewSessionOptions();
  // Set GPU options if needed.
  // uint8_t config[10] = {0x32, 0x01, 0x20}; // Very minimal config for allow_growth=true
  // TF_SetConfig(session_opts, config, 3, status_);

  const char* tags[] = {"serve"};
  graph_ = TF_NewGraph();
  session_ = TF_LoadSessionFromSavedModel(session_opts, nullptr, graph_path.c_str(), tags, 1, graph_, nullptr, status_);
  TF_DeleteSessionOptions(session_opts);

  if (TF_GetCode(status_) != TF_OK) {
    MG_LOG(INFO) << "Failed to load SavedModel from " << graph_path << ": " << TF_Message(status_);
    MG_LOG(INFO) << "Falling back to GraphDef import.";

    if (graph_) TF_DeleteGraph(graph_);
    graph_ = TF_NewGraph();

    TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
    TF_Buffer* graph_def_buf = TF_NewBuffer();
    graph_def_buf->data = const_cast<char*>(model_bytes.data());
    graph_def_buf->length = model_bytes.size();
    graph_def_buf->data_deallocator = nullptr;

    TF_GraphImportGraphDef(graph_, graph_def_buf, import_opts, status_);
    TF_DeleteImportGraphDefOptions(import_opts);
    TF_DeleteBuffer(graph_def_buf);

    if (TF_GetCode(status_) != TF_OK) {
      MG_LOG(FATAL) << "Failed to import GraphDef: " << TF_Message(status_);
    }

    TF_SessionOptions* session_opts2 = TF_NewSessionOptions();
    session_ = TF_NewSession(graph_, session_opts2, status_);
    TF_DeleteSessionOptions(session_opts2);
    if (TF_GetCode(status_) != TF_OK) {
      MG_LOG(FATAL) << "Failed to create session: " << TF_Message(status_);
    }
  }

  input_op_ = {TF_GraphOperationByName(graph_, "pos_tensor"), 0};
  policy_op_ = {TF_GraphOperationByName(graph_, "policy_output"), 0};
  value_op_ = {TF_GraphOperationByName(graph_, "value_output"), 0};

  if (input_op_.oper == nullptr) {
      // Try serving default names
      input_op_ = {TF_GraphOperationByName(graph_, "serving_default_pos_tensor"), 0};
      policy_op_ = {TF_GraphOperationByName(graph_, "serving_default_policy_output"), 0};
      value_op_ = {TF_GraphOperationByName(graph_, "serving_default_value_output"), 0};
  }

  if (input_op_.oper == nullptr) MG_LOG(FATAL) << "Could not find input node 'pos_tensor'";
  if (policy_op_.oper == nullptr) MG_LOG(FATAL) << "Could not find output node 'policy_output'";
  if (value_op_.oper == nullptr) MG_LOG(FATAL) << "Could not find output node 'value_output'";

  input_type_ = TF_OperationOutputType(input_op_);
  MG_LOG(INFO) << "Model " << graph_path_ << " has input type " << input_type_;
}

TfDualNet::~TfDualNet() {
  for (auto* t : inputs_) if (t) TF_DeleteTensor(t);

  if (session_) {
    TF_CloseSession(session_, status_);
    TF_DeleteSession(session_, status_);
  }
  if (graph_) TF_DeleteGraph(graph_);
  if (status_) TF_DeleteStatus(status_);
}

static void DeallocateTensor(void* data, size_t len, void* arg) {
  free(data);
}

void TfDualNet::RunMany(const std::vector<const ModelInput*>& inputs,
                        std::vector<ModelOutput*>* outputs,
                        std::string* model_name) {
  Reserve(inputs.size());

  WTF_SCOPE("TfDualNet::Run: inputs, capacity", size_t, int)
  (inputs.size(), batch_capacity_);
  MG_CHECK(inputs.size() == outputs->size());

  auto shape = feature_descriptor().GetInputShape(batch_capacity_);
  if (input_type_ == TF_FLOAT) {
    WTF_SCOPE("Features::SetFloat: inputs", int)(inputs.size());
    Tensor<float> features(shape, static_cast<float*>(TF_TensorData(inputs_[0])));
    feature_descriptor().set_floats(inputs, &features);
  } else {
    WTF_SCOPE("Features::SetBool: inputs", size_t)(inputs.size());
    Tensor<uint8_t> features(shape, static_cast<uint8_t*>(TF_TensorData(inputs_[0])));
    feature_descriptor().set_bytes(inputs, &features);
  }

  TF_Output inputs_ops[] = {input_op_};
  TF_Output outputs_ops[] = {policy_op_, value_op_};
  TF_Tensor* run_outputs[2] = {nullptr, nullptr};

  TF_SessionRun(session_, nullptr, inputs_ops, inputs_.data(), 1, outputs_ops, run_outputs, 2, nullptr, 0, nullptr, status_);

  if (TF_GetCode(status_) != TF_OK) {
    MG_LOG(FATAL) << "Failed to run session: " << TF_Message(status_);
  }

  Tensor<float> policy({batch_capacity_, kNumMoves}, static_cast<float*>(TF_TensorData(run_outputs[0])));
  Tensor<float> value({batch_capacity_}, static_cast<float*>(TF_TensorData(run_outputs[1])));

  {
    WTF_SCOPE("Model::GetOutputs: outputs", size_t)(outputs->size());
    Model::GetOutputs(inputs, policy, value, absl::MakeSpan(*outputs));
  }

  TF_DeleteTensor(run_outputs[0]);
  TF_DeleteTensor(run_outputs[1]);

  if (model_name != nullptr) {
    *model_name = graph_path_;
  }
}

void TfDualNet::Reserve(int capacity) {
  MG_CHECK(capacity > 0);
  if (capacity == batch_capacity_) {
    return;
  }

  for (auto* t : inputs_) if (t) TF_DeleteTensor(t);
  inputs_.clear();

  auto shape = feature_descriptor().GetInputShape(capacity);
  int64_t dims[] = {shape[0], shape[1], shape[2], shape[3]};
  size_t nbytes = TF_DataTypeSize(input_type_) * shape[0] * shape[1] * shape[2] * shape[3];
  void* data = malloc(nbytes);
  inputs_.push_back(TF_NewTensor(input_type_, dims, 4, data, nbytes, &DeallocateTensor, nullptr));

  batch_capacity_ = capacity;
}

}  // namespace

TfDualNetFactory::TfDualNetFactory(absl::string_view device) {
  place_on_gpu_ = device.empty() || device == "gpu";
}

std::unique_ptr<Model> TfDualNetFactory::NewModel(const ModelDefinition& def) {
  MG_CHECK(def.metadata.Get<std::string>("engine") == "tf");

  auto feature_desc =
      FeatureDescriptor::Create(def.metadata.Get<std::string>("input_features"),
                                def.metadata.Get<std::string>("input_layout"));

  return absl::make_unique<TfDualNet>(def.path, feature_desc, def.model_bytes);
}

}  // namespace minigo
