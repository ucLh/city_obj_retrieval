#include "tf_wrapper/embeddings_wrapper.h"
#include "tf_wrapper/inference_handlers.h"
#include "tf_wrapper/tensorflow_embeddings.h"
#include <utility>

EmbeddingsWrapper::EmbeddingsWrapper() {
  db_handler_ = std::make_unique<DataHandling>();
  inference_handler_ = std::make_unique<EmbeddingsInferenceHandler>();
  topN = 1;
}

bool EmbeddingsWrapper::load_config(std::string config_path) {
  db_handler_->set_config_path(std::move(config_path));
  if (!db_handler_->load_config()) {
    std::cerr << "Can't load config!" << std::endl;
    return false;
  }
  inference_handler_->set_input_output({db_handler_->get_config_input_node()},
                                      {db_handler_->get_config_output_node()});
  inference_handler_->load(db_handler_->get_config_pb_path(),
                          db_handler_->get_config_input_node());
  is_configured_ = true;
  std::cout << "Config was loaded" << std::endl;

  return true;
}

bool EmbeddingsWrapper::prepare_for_inference(std::string config_path) {
  if (!load_config(std::move(config_path))) {
    return false;
  }

  if (!db_handler_->load_database()) {
    std::cerr << "Can't load database" << std::endl;
    return false;
  }
  std::cout << "Database was loaded" << std::endl;

  list_of_imgs_ = fs_img::list_imgs(db_handler_->get_config_imgs_path());
  check_for_updates();
  if (!list_of_imgs_.empty()) {
    add_updates();
  } else
    std::cout << "No new images found" << std::endl;

  return true;
}

std::vector<EmbeddingsWrapper::distance>
EmbeddingsWrapper::inference_and_matching(const std::string &img_path) {
  cv::Mat img = fs_img::read_img(img_path);
  return inference_and_matching(img);
}

std::vector<EmbeddingsWrapper::distance>
EmbeddingsWrapper::inference_and_matching(cv::Mat img) {
  if (!is_configured_) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    exit(1); // TODO: rethink it
  }
  std::vector<float> embedding;

  topN = db_handler_->get_config_top_n();

  inference_handler_->inference({std::move(img)});

  embedding = inference_handler_->get_output_embeddings()[0];

  matching(db_handler_->get_data_vec_base(), embedding);
  inference_handler_->clear_data();

  return distances_;
}

bool EmbeddingsWrapper::add_updates() {
  std::cout << "Adding updates to database..." << std::endl;
  cv::Mat img;
  if (!is_configured_) {
    std::cerr << "You need to configure wrapper first!" << std::endl;
    return false;
  }
  std::vector<float> out_embedding;
  DataHandling::data_vec_entry new_data;
  for (size_t i = 0; i < list_of_imgs_.size(); ++i) {
    std::cout << "Wrapper Info: " << i << " of " << list_of_imgs_.size()
              << " was processed"
              << "\r" << std::flush;
    img = fs_img::read_img(list_of_imgs_[i]);
    inference_handler_->inference({img});
    new_data.embedding = inference_handler_->get_output_embeddings()[0];
    inference_handler_->clear_data();
    new_data.filepath = list_of_imgs_[i];
    db_handler_->add_element_to_data_vec_base(new_data);
    db_handler_->add_json_entry(new_data);
  }
  return true;
}

bool EmbeddingsWrapper::check_for_updates() {
  for (const auto &entry : db_handler_->get_data_vec_base()) {
    for (auto img_path = list_of_imgs_.begin();
         img_path != list_of_imgs_.end();) {
      if (*img_path == entry.filepath) {
        list_of_imgs_.erase(img_path);
      } else {
        img_path++;
      }
    }
  }
  if (!list_of_imgs_.empty()) {
    for (const auto &entry : list_of_imgs_) {
      std::cout << "Found new data " << entry << std::endl;
    }
  }

  return true;
}
bool sort_by_dist(const EmbeddingsWrapper::distance &a,
                  const EmbeddingsWrapper::distance &b) {
  return (a.dist < b.dist);
}

bool EmbeddingsWrapper::matching(
    const std::vector<IDataBase::data_vec_entry> &base,
    std::vector<float> &target) {
  distances_.clear();
  EmbeddingsWrapper::distance distance;

  if (base.empty() or target.empty()) {
    return false;
  }

  for (auto &it : base) {
    distance.dist =
        EmbeddingMatching::calc_distance_euclid(it.embedding, target);
    distance.path = it.filepath;
    distances_.push_back(distance);
  }
  std::sort(distances_.begin(), distances_.end(), sort_by_dist);
  if (topN > distances_.size()) {
    topN = distances_.size();
  }
  distances_.erase(distances_.begin() + topN, distances_.end());

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
