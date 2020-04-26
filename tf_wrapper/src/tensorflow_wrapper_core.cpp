#include "tf_wrapper/tensorflow_wrapper_core.h"

TensorFlowWrapperCore::TensorFlowWrapperCore(TensorFlowWrapperCore &&that) {
  session_ = that.session_;
  is_loaded_ = that.is_loaded_;

  name_ = std::move(that.name_);
  path_ = std::move(that.path_);
  graph_def_ = std::move(that.graph_def_);
}

TensorFlowWrapperCore::~TensorFlowWrapperCore() {
  clear_session();
  auto a = session_->Close();
  common_ops::delete_safe(session_);
}

// TODO Disable graph optimization. We assume that graph already optimized.
tensorflow::SessionOptions TensorFlowWrapperCore::configure_session() {
  using namespace tensorflow;

  SessionOptions opts;
  opts.config.set_allow_soft_placement(allow_soft_placement_);

  GPUOptions *gpu_options = new GPUOptions;
#ifdef TFDEBUG
  // opts.config.set_log_device_placement(true);
#endif
  if (cpu_only_) {
    auto device_map = opts.config.mutable_device_count();
    if (device_map) {
      tf_aux::debug_output("Warning", "Disabling GPU!!!");
      (*device_map)["GPU"] = 0;
    }
  } else {
    if (!visible_devices_.empty()) {
      gpu_options->set_visible_device_list(visible_devices_);
    }
    gpu_options->set_per_process_gpu_memory_fraction(gpu_memory_fraction_);
    gpu_options->set_allow_growth(allow_growth_);
  }

  GraphOptions *graph_opts = new GraphOptions;
  /// TODO: Needs tests, maybe not all options is ok
  OptimizerOptions *optim_opts = new OptimizerOptions;
  // OptimizerOptions_GlobalJitLevel_ON_2 turn on compilation, with higher
  // values being more aggressive.  Higher values may reduce opportunities for
  // parallelism and may use more memory.  (At present, there is no distinction,
  // but this is expected to change.)

  // TODO think about jit
  //    optim_opts->set_global_jit_level( (agres_optim_enabled_ ?
  //    OptimizerOptions_GlobalJitLevel_ON_2
  //                                                    :
  //                                                    OptimizerOptions_GlobalJitLevel_OFF)
  //                                                    );
  optim_opts->set_do_common_subexpression_elimination(
      agres_optim_enabled_ ? true : false);
  optim_opts->set_do_constant_folding(agres_optim_enabled_ ? true : false);
  optim_opts->set_do_function_inlining(agres_optim_enabled_ ? true : false);
  //
  graph_opts->set_allocated_optimizer_options(optim_opts);
  //
  opts.config.set_allocated_graph_options(graph_opts);
  opts.config.set_allocated_gpu_options(gpu_options);

  return opts;
}

// TODO Think about graph configuration
void TensorFlowWrapperCore::configure_graph() {
  using namespace tensorflow;
  if (cpu_only_ && agres_optim_cpu_enabled_)
    graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:XLA_CPU:0",
                            &graph_def_);
  if (!cpu_only_ && gpu_number_ >= 0) {
    //        graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:GPU:"
    //        + std::to_string(gpu_number_), &graph_def_);
    graph::SetDefaultDevice("/device:GPU:" + std::to_string(gpu_number_),
                            &graph_def_);
  }
}

bool TensorFlowWrapperCore::load(const std::string &filename,
                                 const std::string &input_node_name) {
  using namespace tensorflow;

  // Configuration for session
  SessionOptions opts = configure_session();
  if (session_) {
    session_->Close();
    common_ops::delete_safe(session_);
  }
  // Blame Tensorflow developers for NewSession mem leak.
  // It may appear on some versions.
  status_ = NewSession(opts, &session_);
  if (!status_.ok()) {
    tf_aux::debug_output("tf error: ", status_.ToString());

    return is_loaded_ = false;
  }
  status_ = ReadBinaryProto(Env::Default(), filename, &graph_def_);
  if (!status_.ok()) {
    tf_aux::debug_output("tf error: ", status_.ToString());

    return is_loaded_ = false;
  }
  configure_graph();
  status_ = session_->Create(graph_def_);
  if (!status_.ok()) {
    tf_aux::debug_output("tf error: ", status_.ToString());
    return is_loaded_ = false;
  } else {
    tf_aux::debug_output("WRAPPER_STATUS", "Graph successfully loaded!");
  }
  parse_name(filename);

  get_input_node_name_from_graph_if_possible(input_node_name);

  path_ = filename;
  return is_loaded_ = true;
}

std::string TensorFlowWrapperCore::inference(const std::vector<cv::Mat> &imgs) {
  using namespace tensorflow;

  Tensor input;
  tf_aux::convert_mat_to_tensor_v2<tensorflow::DT_FLOAT>(imgs, input);
  std::vector in_tensor_shape = tf_aux::get_tensor_shape(input);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {input_node_names_[0], input}};
  std::cout << input_node_names_[0] << std::endl;
  status_ = session_->Run(inputs, output_node_names_, {}, &output_tensors_);

  std::cerr << "NETWORK_STATUS: " << status_ << std::endl;
  return status_.ToString();
}

// std::string TensorFlowWrapperCore
void TensorFlowWrapperCore::set_name(const std::string &name) { name_ = name; }

void TensorFlowWrapperCore::clear_session() { output_tensors_.clear(); }

void TensorFlowWrapperCore::parse_name(const std::string &filename) {
  auto last_slash = filename.rfind("/");
  if (last_slash == std::string::npos) {
    last_slash = 0;
  }

  auto last_dot = filename.rfind(".");
  if (last_dot == std::string::npos) {
    name_ = "UnknownModel";
    return;
  }

  if (last_slash > last_dot) {
    name_ = "UnknownModel";
    return;
  }

  name_ = filename.substr(last_slash + 1, (last_dot - last_slash) - 1);
}

void TensorFlowWrapperCore::get_input_node_name_from_graph_if_possible(
    const std::string &input_node_name) {
  using namespace tensorflow;

  const Tensor &names_tensor = get_tensor_from_graph(input_node_name);
  if (names_tensor.NumElements() == 1) {
    const auto &names_mapped = names_tensor.tensor<std::string, 1>();
    //#ifdef TFDEBUG
    std::cerr << "Input node name:\n------------------" << std::endl;
    //#endif
    input_node_names_[0] = names_mapped(0);
    //#ifdef TFDEBUG
    std::cerr << names_mapped(0) << std::endl;
    //#endif

    //#ifdef TFDEBUG
    std::cerr << "------------------\nInput node name loaded" << std::endl;
    //#endif
  }
}

tensorflow::Tensor
TensorFlowWrapperCore::get_tensor_from_graph(const std::string &tensor_name) {
  using namespace tensorflow;

  if (tensor_name.empty()) {
    return Tensor();
  }

  if (!is_loaded_) {
    return Tensor();
  }

  tensorflow::Status status;
  std::vector<tensorflow::Tensor> tensors;

  status = session_->Run({}, {tensor_name}, {}, &tensors);

  tf_aux::debug_output("Sucessfully run graph! Status is: ", status.ToString());

  if (!status.ok()) {
    return Tensor();
  }

  return tensors[0];
}

bool TensorFlowWrapperCore::get_allow_growth() const { return allow_growth_; }

void TensorFlowWrapperCore::set_allow_growth(bool allow_growth) {
  allow_growth_ = allow_growth;
}

std::string TensorFlowWrapperCore::get_visible_devices() const {
  return visible_devices_;
}

void TensorFlowWrapperCore::set_visible_devices(
    const std::string &visible_devices) {
  visible_devices_ = visible_devices;
}

double TensorFlowWrapperCore::get_gpu_memory_fraction() const {
  return gpu_memory_fraction_;
}

void TensorFlowWrapperCore::set_gpu_memory_fraction(
    double gpu_memory_fraction) {
  if (gpu_memory_fraction > 1.0) {
    gpu_memory_fraction = 1.0;
  }
  if (gpu_memory_fraction < 0.0) {
    gpu_memory_fraction = 0.1;
  }

  gpu_memory_fraction_ = gpu_memory_fraction;
}

int TensorFlowWrapperCore::get_gpu_number() const { return gpu_number_; }

void TensorFlowWrapperCore::set_gpu_number(int value) { gpu_number_ = value; }

bool TensorFlowWrapperCore::set_gpu_number_preferred(int value) {
  set_gpu_number(value);
  const int gpu_num_value = get_gpu_number();
  if (gpu_num_value != value) {
    std::cerr << "GPU number was not set" << std::endl;
    return false;
  }

  return true;
}

bool TensorFlowWrapperCore::set_input_output(
    std::vector<std::string> in_nodes, std::vector<std::string> out_nodes) {
  input_node_names_ = std::move(in_nodes);
  output_node_names_ = std::move(out_nodes);
  return true;
}

bool TensorFlowWrapperCore::get_aggressive_optimization_cpu_enabled() const {
  return agres_optim_cpu_enabled_;
}

void TensorFlowWrapperCore::set_aggressive_optimization_cpu_enabled(
    bool enabled) {
  agres_optim_cpu_enabled_ = enabled;
}

bool TensorFlowWrapperCore::get_cpu_only() const { return cpu_only_; }

void TensorFlowWrapperCore::set_cpu_only(bool cpu_only) {
  cpu_only_ = cpu_only;
}

bool TensorFlowWrapperCore::get_allow_soft_placement() const {
  return allow_soft_placement_;
}

void TensorFlowWrapperCore::set_allow_soft_placement(
    bool allow_soft_placement) {
  allow_soft_placement_ = allow_soft_placement;
}

bool TensorFlowWrapperCore::get_aggressive_optimization_gpu_enabled() const {
  return agres_optim_enabled_;
}

void TensorFlowWrapperCore::set_aggressive_optimization_gpu_enabled(
    bool enabled) {
  agres_optim_enabled_ = enabled;
}

std::string TensorFlowWrapperCore::get_path() const { return path_; }
