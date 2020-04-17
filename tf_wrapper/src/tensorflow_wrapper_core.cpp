#include "tf_wrapper/tensorflow_wrapper_core.h"

TensorFlowWrapperCore::TensorFlowWrapperCore(TensorFlowWrapperCore &&that) {
  _session = that._session;
  _is_loaded = that._is_loaded;

  _name = std::move(that._name);
  _path = std::move(that._path);
  _graph_def = std::move(that._graph_def);
}

TensorFlowWrapperCore::~TensorFlowWrapperCore() {
  clear_session();
  auto a = _session->Close();
  common_ops::delete_safe(_session);
}

// TODO Disable graph optimization. We assume that graph already optimized.
tensorflow::SessionOptions TensorFlowWrapperCore::configure_session() {
  using namespace tensorflow;

  SessionOptions opts;
  opts.config.set_allow_soft_placement(_allow_soft_placement);

  GPUOptions *gpu_options = new GPUOptions;
#ifdef TFDEBUG
  // opts.config.set_log_device_placement(true);
#endif
  if (_cpu_only) {
    auto device_map = opts.config.mutable_device_count();
    if (device_map) {
      tf_aux::debug_output("Warning", "Disabling GPU!!!");
      (*device_map)["GPU"] = 0;
    }
  } else {
    if (!_visible_devices.empty()) {
      gpu_options->set_visible_device_list(_visible_devices);
    }
    gpu_options->set_per_process_gpu_memory_fraction(_gpu_memory_fraction);
    gpu_options->set_allow_growth(_allow_growth);
  }

  GraphOptions *graph_opts = new GraphOptions;
  /// TODO: Needs tests, maybe not all options is ok
  OptimizerOptions *optim_opts = new OptimizerOptions;
  // OptimizerOptions_GlobalJitLevel_ON_2 turn on compilation, with higher
  // values being more aggressive.  Higher values may reduce opportunities for
  // parallelism and may use more memory.  (At present, there is no distinction,
  // but this is expected to change.)

  // TODO think about jit
  //    optim_opts->set_global_jit_level( (_agres_optim_enabled ?
  //    OptimizerOptions_GlobalJitLevel_ON_2
  //                                                    :
  //                                                    OptimizerOptions_GlobalJitLevel_OFF)
  //                                                    );
  optim_opts->set_do_common_subexpression_elimination(
      _agres_optim_enabled ? true : false);
  optim_opts->set_do_constant_folding(_agres_optim_enabled ? true : false);
  optim_opts->set_do_function_inlining(_agres_optim_enabled ? true : false);
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
  if (_cpu_only && _agres_optim_cpu_enabled)
    graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:XLA_CPU:0",
                            &_graph_def);
  if (!_cpu_only && _gpu_number >= 0) {
    //        graph::SetDefaultDevice("/job:localhost/replica:0/task:0/device:GPU:"
    //        + std::to_string(_gpu_number), &_graph_def);
    graph::SetDefaultDevice("/device:GPU:" + std::to_string(_gpu_number),
                            &_graph_def);
  }
}

bool TensorFlowWrapperCore::load(const std::string &filename,
                                 const std::string &input_node_name) {
  using namespace tensorflow;

  // Configuration for session
  SessionOptions opts = configure_session();
  if (_session) {
    _session->Close();
    common_ops::delete_safe(_session);
  }
  // Blame Tensorflow developers for NewSession mem leak.
  // It may appear on some versions.
  _status = NewSession(opts, &_session);
  if (!_status.ok()) {
    tf_aux::debug_output("tf error: ", _status.ToString());

    return _is_loaded = false;
  }
  _status = ReadBinaryProto(Env::Default(), filename, &_graph_def);
  if (!_status.ok()) {
    tf_aux::debug_output("tf error: ", _status.ToString());

    return _is_loaded = false;
  }
  configure_graph();
  _status = _session->Create(_graph_def);
  if (!_status.ok()) {
    tf_aux::debug_output("tf error: ", _status.ToString());
    return _is_loaded = false;
  } else {
    tf_aux::debug_output("WRAPPER_STATUS", "Graph successfully loaded!");
  }
  parse_name(filename);

  get_input_node_name_from_graph_if_possible(input_node_name);

  _path = filename;
  return _is_loaded = true;
}

std::string TensorFlowWrapperCore::inference(const std::vector<cv::Mat> &imgs) {
  using namespace tensorflow;

  Tensor input = get_convert_function(INPUT_TYPE::DT_FLOAT)(
      imgs, _input_height, _input_width, _input_depth, _convert_to_float,
      _mean);

  std::vector in_tensor_shape = tf_aux::get_tensor_shape(input);
  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {_input_node_names[0], input}};
  std::cout << _input_node_names[0] << std::endl;
  _status = _session->Run(inputs, _output_node_names, {}, &_output_tensors);
  std::cerr << "NETWORK_STATUS: " << _status << std::endl;
  return _status.ToString();
}

// std::string TensorFlowWrapperCore
void TensorFlowWrapperCore::set_name(const std::string &name) { _name = name; }

void TensorFlowWrapperCore::clear_session() { _output_tensors.clear(); }

void TensorFlowWrapperCore::parse_name(const std::string &filename) {
  auto last_slash = filename.rfind("/");
  if (last_slash == std::string::npos) {
    last_slash = 0;
  }

  auto last_dot = filename.rfind(".");
  if (last_dot == std::string::npos) {
    _name = "UnknownModel";
    return;
  }

  if (last_slash > last_dot) {
    _name = "UnknownModel";
    return;
  }

  _name = filename.substr(last_slash + 1, (last_dot - last_slash) - 1);
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
    _input_node_names[0] = names_mapped(0);
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

  if (!_is_loaded) {
    return Tensor();
  }

  tensorflow::Status status;
  std::vector<tensorflow::Tensor> tensors;

  status = _session->Run({}, {tensor_name}, {}, &tensors);

  tf_aux::debug_output("Sucessfully run graph! Status is: ", status.ToString());

  if (!status.ok()) {
    return Tensor();
  }

  return tensors[0];
}

bool TensorFlowWrapperCore::get_allow_growth() const { return _allow_growth; }

void TensorFlowWrapperCore::set_allow_growth(bool allow_growth) {
  _allow_growth = allow_growth;
}

std::string TensorFlowWrapperCore::get_visible_devices() const {
  return _visible_devices;
}

void TensorFlowWrapperCore::set_visible_devices(
    const std::string &visible_devices) {
  _visible_devices = visible_devices;
}

double TensorFlowWrapperCore::get_gpu_memory_fraction() const {
  return _gpu_memory_fraction;
}

void TensorFlowWrapperCore::set_gpu_memory_fraction(
    double gpu_memory_fraction) {
  if (gpu_memory_fraction > 1.0) {
    gpu_memory_fraction = 1.0;
  }
  if (gpu_memory_fraction < 0.0) {
    gpu_memory_fraction = 0.1;
  }

  _gpu_memory_fraction = gpu_memory_fraction;
}

int TensorFlowWrapperCore::get_gpu_number() const { return _gpu_number; }

void TensorFlowWrapperCore::set_gpu_number(int value) { _gpu_number = value; }

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
  _input_node_names = std::move(in_nodes);
  _output_node_names = std::move(out_nodes);
  return true;
}

bool TensorFlowWrapperCore::get_aggressive_optimization_cpu_enabled() const {
  return _agres_optim_cpu_enabled;
}

void TensorFlowWrapperCore::set_aggressive_optimization_cpu_enabled(
    bool enabled) {
  _agres_optim_cpu_enabled = enabled;
}

bool TensorFlowWrapperCore::get_cpu_only() const { return _cpu_only; }

void TensorFlowWrapperCore::set_cpu_only(bool cpu_only) {
  _cpu_only = cpu_only;
}

bool TensorFlowWrapperCore::get_allow_soft_placement() const {
  return _allow_soft_placement;
}

void TensorFlowWrapperCore::set_allow_soft_placement(
    bool allow_soft_placement) {
  _allow_soft_placement = allow_soft_placement;
}

bool TensorFlowWrapperCore::get_aggressive_optimization_gpu_enabled() const {
  return _agres_optim_enabled;
}

void TensorFlowWrapperCore::set_aggressive_optimization_gpu_enabled(
    bool enabled) {
  _agres_optim_enabled = enabled;
}

std::string TensorFlowWrapperCore::get_path() const { return _path; }
