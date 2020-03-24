#ifndef TF_WRAPPER_EMBEDDING_TENSORFLOW_WRAPPER_CORE_H
#define TF_WRAPPER_EMBEDDING_TENSORFLOW_WRAPPER_CORE_H

#include "tensorflow_base.h"
#include "tensorflow_auxiliary.h"
#include "wrapper_legacy.h"
#include "common/common_ops.h"

#include <string>
#include <vector>

class TensorflowWrapperCore {
public:
  enum INPUT_TYPE { DT_FLOAT, DT_UINT8 };

  TensorflowWrapperCore() = default;
  virtual ~TensorflowWrapperCore();

  TensorflowWrapperCore(const TensorflowWrapperCore &) = delete;
  TensorflowWrapperCore(TensorflowWrapperCore &&that);

  virtual bool load(const std::string &filename,
                    const std::string &inputNodeName);

  virtual inline std::string inference(const std::vector<cv::Mat> &imgs);

  virtual inline bool is_loaded() const { return _is_loaded; }

  virtual void clear_session();

  virtual inline std::string get_name() const { return _name; }
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

  tensorflow::Status _status;

  /// values for covert image before processing
  short _input_height = 256;
  short _input_width = 256;
  short _input_depth = 3;

  std::vector<float> _mean = {0, 0, 0};
  bool _convert_to_float = false;
  ///_______________________________________

  /// values for inference
  std::vector<std::string> _input_node_names;
  std::vector<std::string> _output_node_names;

  ///_______________________________________

  std::vector<tensorflow::Tensor> _output_tensors;

  void parse_name(const std::string &filename);
  tensorflow::SessionOptions configure_session();
  void configure_graph();

  ///
  /// \brief getTensorFromGraph Method for extracting tensors from graph. For
  /// usage, model must be loaded and Session must be active. \param tensor_name
  /// Name in the Graph \return Empty Tensor if failed, otherwise extructed
  /// Tensor
  ///
  tensorflow::Tensor get_tensor_from_graph(const std::string &tensor_name);

  using ConvertFunctionType =
      decltype(&(wrapper_legacy::convert_mat_to_tensor<tensorflow::DT_FLOAT>));

  ConvertFunctionType get_convert_function(INPUT_TYPE type) {
    if (type == INPUT_TYPE::DT_FLOAT) {
      return wrapper_legacy::convert_mat_to_tensor<tensorflow::DT_FLOAT>;
    }
    /// Actually we don't need support for int operations because we don't have
    /// strong hardware limits.
    //        else if (type == INPUT_TYPE::DT_UINT8) {
    //            return tf_aux::convertMatToTensor<tensorflow::DT_UINT8>;
    //        }
    else
      throw std::invalid_argument("not implemented");
  }

  bool _is_loaded = false;
  bool _agres_optim_enabled = false;

  /// Mostly that nedeed for XLA, because it's not possible to enable XLA for
  /// CPU on session level But is possible manually. Works only if cpu only
  /// mode.
  bool _agres_optim_cpu_enabled = false;

  bool _allow_soft_placement = true;
  bool _cpu_only = false;

  std::string _name = "UnknownModel";
  std::string _path = "";
  std::string _visible_devices = "";

  tensorflow::GraphDef _graph_def;
  tensorflow::Session *_session = nullptr;

  int _gpu_number = -1;
  double _gpu_memory_fraction = 0.;
  bool _allow_growth = false;
};

#endif // TF_WRAPPER_EMBEDDING_TENSORFLOW_WRAPPER_CORE_H
