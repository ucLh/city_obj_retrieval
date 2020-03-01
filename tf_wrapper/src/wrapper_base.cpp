#include "tf_wrapper/wrapper_base.h"
#include "tf_wrapper/tensorflow_embeddings.h"
#include "tf_wrapper/wrapper_interfaces.h"
#include <utility>

WrapperBase::WrapperBase() {
  db_handler = std::make_unique<DataHandling>();
  inference_handler = std::make_unique<TensorFlowEmbeddingsInterface>();

  topN = 1;
}

bool WrapperBase::prepare_for_inference() {
  if (!db_handler->load_config()) {
    std::cerr << "Can't load config!" << std::endl;
    return false;
  }

  _input_nodes = {db_handler->get_config_input_node()};
  _output_nodes = {db_handler->get_config_output_node()};

  std::cout << "Config was loaded" << std::endl;
  db_handler->load_database();
  std::cout << "Database was loaded" << std::endl;
  list_of_imgs =
      fs_img::list_imgs(db_handler->get_config_imgs_path()); // TODO rewrite it
  _check_for_updates();
  if (!list_of_imgs.empty())
    _add_updates();
  else
    std::cout << "No new images found" << std::endl;

  return true;
}

std::vector<WrapperBase::distance>
WrapperBase::inference_and_matching(std::string img_path) {
  std::vector<float> embedding;

  topN = db_handler->get_config_top_n();
  cv::Mat img = fs_img::read_img(img_path, db_handler->get_config_input_size());

  if (!inference_handler->is_loaded())
    inference_handler->load(db_handler->get_config_pb_path(), _input_nodes[0]);

  inference_handler->set_input_output(_input_nodes, _output_nodes);
  inference_handler->inference({img});

  embedding = inference_handler->get_output_embeddings()[0];

  _matching(db_handler->get_data_vec_base(), embedding);
  inference_handler->clear_session();

  return distances;
}

bool WrapperBase::_add_updates() {
  std::cout << "Adding updates to database..." << std::endl;
  cv::Mat img; // TODO rethink this logic..
  if (!inference_handler->is_loaded())
    inference_handler->load(db_handler->get_config_pb_path(),
                            db_handler->get_config_input_node());
  inference_handler->set_input_output(_input_nodes, _output_nodes);
  std::vector<float> out_embedding; // TODO remember about batch
  DataHandling::data_vec_entry new_data;
  for (const auto &img_path : list_of_imgs) {
    img = fs_img::read_img(img_path, db_handler->get_config_input_size());
    inference_handler->inference({img}); // TODO remember about batch
    new_data.embedding =
        inference_handler->get_output_embeddings()[0]; // TODO BATCH
    inference_handler->clear_session();
    new_data.filepath = img_path;
    db_handler->add_element_to_data_vec_base(new_data);
    db_handler->add_json_entry(new_data);
  }
  return true;
}

bool WrapperBase::_check_for_updates() {
  for (const auto &entry : db_handler->get_data_vec_base()) {
    for (auto img_path = list_of_imgs.begin();
         img_path != list_of_imgs.end();) {
      if (*img_path == entry.filepath) {
        list_of_imgs.erase(img_path);
      } else {
        img_path++;
      }
    }
  }
  if (!list_of_imgs.empty()) {
    for (const auto &entry : list_of_imgs) {
      std::cout << "Found new data " << entry << std::endl;
    }
  }

  return true;
}
bool sort_by_dist(const WrapperBase::distance &a,
                  const WrapperBase::distance &b) {
  return (a.dist < b.dist);
}

bool WrapperBase::_matching(
    const std::vector<DBInterface::data_vec_entry> &base,
    std::vector<float> &target) {
  distances.clear();
  WrapperBase::distance distance;

  if (base.empty() or target.empty())
    return false;

  for (auto &it : base) {
    distance.dist =
        EmbeddingMatching::calc_distance_euclid(it.embedding, target);
    distance.path = it.filepath;
    distances.push_back(distance);
  }
  std::sort(distances.begin(), distances.end(), sort_by_dist);
  if (topN > distances.size())
    topN = distances.size();
  distances.erase(distances.begin() + topN, distances.end());

  return true;
}

float EmbeddingMatching::calc_distance_euclid(std::vector<float> base,
                                              std::vector<float> target) {
  float sum = 0;
  auto target_it = target.begin();
  for (auto base_it = base.begin(); base_it != base.end();
       ++base_it, ++target_it) {
    sum += pow((*base_it - *target_it), 2);
  }
  return sqrt(sum);
}

float EmbeddingMatching::calc_distance_cosine(std::vector<float> base,
                                              std::vector<float> target) {
  float numerator = 0, denominator_base = 0, denominator_target = 0;
  auto target_it = target.begin();
  for (auto base_it = base.begin(); base_it != base.end();
       ++base_it, ++target_it) {
    numerator += *base_it * *target_it;
    denominator_base += *base_it * *base_it;
    denominator_target += *target_it * *target_it;
  }
  float expr = numerator / (sqrt(denominator_base) * sqrt(denominator_target));
  return 1 - expr;
}
