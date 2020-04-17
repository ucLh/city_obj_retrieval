#include "tf_wrapper/embeddings_wrapper.h"
#include "tf_wrapper/inference_handlers.h"
#include "tf_wrapper/tensorflow_embeddings.h"
#include <utility>

EmbeddingsWrapper::EmbeddingsWrapper() {
  db_handler = std::make_unique<DataHandling>();
  inference_handler = std::make_unique<EmbeddingsInferenceHandler>();
  topN = 1;
}

bool EmbeddingsWrapper::load_config(std::string config_path) {
  db_handler->set_config_path(std::move(config_path));
  if (!db_handler->load_config()) {
    std::cerr << "Can't load config!" << std::endl;
    return false;
  }
  inference_handler->set_input_output({db_handler->get_config_input_node()},
                                      {db_handler->get_config_output_node()});
  inference_handler->load(db_handler->get_config_pb_path(),
                          db_handler->get_config_input_node());
  _is_configured = true;
  std::cout << "Config was loaded" << std::endl;

  return true;
}

bool EmbeddingsWrapper::prepare_for_inference(std::string config_path) {
  if (!load_config(std::move(config_path))) {
    return false;
  }

  if (!db_handler->load_database()) {
    std::cerr << "Can't load database" << std::endl;
    return false;
  }
  std::cout << "Database was loaded" << std::endl;

  list_of_imgs = fs_img::list_imgs(db_handler->get_config_imgs_path());
  _check_for_updates();
  if (!list_of_imgs.empty()) {
    _add_updates();
  } else
    std::cout << "No new images found" << std::endl;

  return true;
}

std::vector<EmbeddingsWrapper::distance>
EmbeddingsWrapper::inference_and_matching(std::string img_path) {
  if (!_is_configured) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    exit(1); // TODO: rethink it
  }
  std::vector<float> embedding;

  topN = db_handler->get_config_top_n();
  cv::Mat img = fs_img::read_img(img_path);

  inference_handler->inference({img});

  embedding = inference_handler->get_output_embeddings()[0];

  _matching(db_handler->get_data_vec_base(), embedding);
  inference_handler->clear_data();

  return distances;
}

bool EmbeddingsWrapper::_add_updates() {
  std::cout << "Adding updates to database..." << std::endl;
  cv::Mat img;
  if (!_is_configured) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    return false;
  }
  std::vector<float> out_embedding;
  DataHandling::data_vec_entry new_data;
  for (size_t i = 0; i < list_of_imgs.size(); ++i) {
    std::cout << i << " of " << list_of_imgs.size() << "\r" << std::flush;
    img = fs_img::read_img(list_of_imgs[i]);
    inference_handler->inference({img});
    new_data.embedding = inference_handler->get_output_embeddings()[0];
    inference_handler->clear_data();
    new_data.filepath = list_of_imgs[i];
    db_handler->add_element_to_data_vec_base(new_data);
    db_handler->add_json_entry(new_data);
  }
  return true;
}

bool EmbeddingsWrapper::_check_for_updates() {
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
bool sort_by_dist(const EmbeddingsWrapper::distance &a,
                  const EmbeddingsWrapper::distance &b) {
  return (a.dist < b.dist);
}

bool EmbeddingsWrapper::_matching(
    const std::vector<IDataBase::data_vec_entry> &base,
    std::vector<float> &target) {
  distances.clear();
  EmbeddingsWrapper::distance distance;

  if (base.empty() or target.empty()) {
    return false;
  }

  for (auto &it : base) {
    distance.dist =
        EmbeddingMatching::calc_distance_euclid(it.embedding, target);
    distance.path = it.filepath;
    distances.push_back(distance);
  }
  std::sort(distances.begin(), distances.end(), sort_by_dist);
  if (topN > distances.size()) {
    topN = distances.size();
  }
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
