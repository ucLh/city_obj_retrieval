#include "metrics_base.h"
#include "tf_wrapper/common/common_ops.h"
#include <iostream>
#include <set>

std::vector<EmbeddingsWrapper::distance>
MetricsBase::inference_and_matching(std::string img_path) {
  return EmbeddingsWrapper::inference_and_matching(img_path);
}

float MetricsBase::get_metrics(std::string &queries_path, int top_N_classes) {
  float metrics;
  std::vector<std::string> test_imgs_paths = fs_img::list_imgs(queries_path);
  testimg_entry test_img;
  std::vector<EmbeddingsWrapper::distance> matched_images_list;

  std::cout << "Start prepearing for inference" << std::endl;
  prepare_for_inference("config.json");
  std::string series_path = db_handler->get_config_imgs_path();
  std::cout << "Preparaing for inference was finished" << std::endl;
  std::cout << "Finding TOP " << top_N_classes << " among "
            << this->db_handler->get_config_top_n() << std::endl;
  float val_correct = 0.f;

  int i = 0;
  for (const auto &test_img_path : test_imgs_paths) {
    test_img.img_path = test_img_path;
    test_img.img_class =
        common_ops::extract_class(test_img_path, series_path, queries_path);
    test_img.img = fs_img::read_img(test_img_path);

    matched_images_list = inference_and_matching(test_img.img_path);
    auto proposed_classes =
        choose_classes(matched_images_list, test_img, top_N_classes,
                       queries_path, series_path);
    if (test_img.is_correct) {
      ++val_correct;
    } else {
      db_handler->add_error_entry(test_img.img_class, test_img.img_path,
                                  proposed_classes[0]);
    }
    test_img.img_classes_proposed = proposed_classes;
    test_img.distance = matched_images_list[0].dist;
    ++i;
    std::cout << i << " of " << test_imgs_paths.size() << "\r" << std::flush;
  }

  metrics = val_correct / test_imgs_paths.size() * 100.f;
  std::cout << "Accuracy is : " << metrics << "%" << std::endl;
  std::cout << "Got " << val_correct << " out of " << test_imgs_paths.size()
            << " right" << std::endl;

  return metrics;
}

bool MetricsBase::prepare_for_inference(std::string config_path) {
  return EmbeddingsWrapper::prepare_for_inference(config_path);
}

std::vector<std::string> MetricsBase::choose_classes(
    const std::vector<EmbeddingsWrapper::distance> &matched_images_list,
    testimg_entry &test_img, unsigned int top_N_classes,
    std::string &queries_path, std::string &series_path) {

  std::set<std::string> top_classes_set;
  std::string test_class;

  for (const auto &res_it : matched_images_list) {
    test_class =
        common_ops::extract_class(res_it.path, series_path, queries_path);
    top_classes_set.insert(test_class);

    if (top_classes_set.size() >= top_N_classes)
      break;
  }

  test_img.is_correct = top_classes_set.count(test_img.img_class) != 0;

  std::vector<std::string> top_classes_vec(top_classes_set.begin(),
                                           top_classes_set.end());

  return top_classes_vec;
}
