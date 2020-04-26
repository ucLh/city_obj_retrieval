#ifndef TF_WRAPPER_EMBEDDING_TENSORFLOW_WRAPPER_CORE_H
#define TF_WRAPPER_EMBEDDING_TENSORFLOW_WRAPPER_CORE_H

#include "tensorflow_base.h"
#include "tensorflow_auxiliary.h"
#include "common/common_ops.h"

#include <string>
#include <vector>

class TensorFlowWrapperCore {
public:
  enum INPUT_TYPE { DT_FLOAT, DT_UINT8 };

  TensorFlowWrapperCore() = default;
  virtual ~TensorFlowWrapperCore();

  TensorFlowWrapperCore(const TensorFlowWrapperCore &) = delete;
  TensorFlowWrapperCore(TensorFlowWrapperCore &&that);

  virtual bool load(const std::string &filename,
                    const std::string &input_node_name);

  virtual inline std::string inference(const std::vector<cv::Mat> &imgs);

  virtual inline bool is_loaded() const { return is_loaded_; }

  virtual void clear_session();

  virtual inline std::string get_name() const { return name_; }
  virtual void set_name(const std::string &name);

  std::string get_path() const;

  bool get_aggressive_optimization_gpu_enabled() const;
  void set_aggressive_optimization_gpu_enabled(bool enabled);

  bool get_allow_soft_placement() const;
  void set_allow_soft_placement(bool allow_soft_placement);

  bool get_cpu_only() const;
  void set_cpu_only(bool cpu_only);

  bool get_aggressive_optimization_cpu_enabled() const;
  ///
  /// \brief setAgressiveOptimizationCPUEnabled JIT optimizations for CPU. Only
  /// for CPU Only mode.
  ///
  void set_aggressive_optimization_cpu_enabled(bool enabled);

  int get_gpu_number() const;
  // If -1 may use all visible GPUs. Otherwise that GPU number that was set.
  // Override with default device in the model
  void set_gpu_number(int value);

  // Uses set_gpu_number but checks whether the setting was successful
  bool set_gpu_number_preferred(int value);

  bool set_input_output(std::vector<std::string> in_nodes,
                        std::vector<std::string> out_nodes);

  double get_gpu_memory_fraction() const;
  void set_gpu_memory_fraction(double gpu_memory_fraction);

  std::string get_visible_devices() const;
  void set_visible_devices(const std::string &visible_devices);

  bool get_allow_growth() const;
  void set_allow_growth(bool allow_growth);

protected:
  //    virtual void clear_model() = 0;
  //    virtual void clear_data() = 0;

  void get_input_node_name_from_graph_if_possible(
      const std::string &input_node_name);

  tensorflow::Status status_;

  /// values for covert image before processing
  short input_height_ = 256;
  short input_width_ = 256;
  short input_depth_ = 3;

  std::vector<float> mean_ = {0, 0, 0};
  bool convert_to_float_ = false;
  ///_______________________________________

  /// values for inference
  std::vector<std::string> input_node_names_;
  std::vector<std::string> output_node_names_;

  ///_______________________________________

  std::vector<tensorflow::Tensor> output_tensors_;

  void parse_name(const std::string &filename);
  tensorflow::SessionOptions configure_session();
  void configure_graph();

  /// \brief getTensorFromGraph Method for extracting tensors from graph. For
  /// usage, model must be loaded and Session must be active.
  /// \param tensor_name Name in the Graph
  /// \return Empty Tensor if failed, otherwise extructed Tensor
  tensorflow::Tensor get_tensor_from_graph(const std::string &tensor_name);

  bool is_loaded_ = false;
  bool agres_optim_enabled_ = false;

  /// Mostly that nedeed for XLA, because it's not possible to enable XLA for
  /// CPU on session level But is possible manually. Works only if cpu only
  /// mode.
  bool agres_optim_cpu_enabled_ = false;

  bool allow_soft_placement_ = true;
  bool cpu_only_ = false;

  std::string name_ = "UnknownModel";
  std::string path_ = "";
  std::string visible_devices_ = "";

  tensorflow::GraphDef graph_def_;
  tensorflow::Session *session_ = nullptr;

  int gpu_number_ = -1;
  double gpu_memory_fraction_ = 0.;
  bool allow_growth_ = true;
};

#endif // TF_WRAPPER_EMBEDDING_TENSORFLOW_WRAPPER_CORE_H
