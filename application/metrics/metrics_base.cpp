#include "metrics_base.h"
#include "tf_wrapper/common/common_ops.h"
#include <iostream>
#include <set>

std::vector<WrapperBase::distance>
MetricsBase::inference_and_matching(std::string img_path) {
  return WrapperBase::inference_and_matching(img_path);
}

bool IsCorrect(MetricsBase::testimg_entry &entry) { return entry.is_correct; }

float MetricsBase::getMetrics(std::string &testimg_path, int top_N_classes) {
  std::vector<std::string> test_imgs_paths = fs_img::list_imgs(testimg_path);
  testimg_entry test_img;
  std::vector<WrapperBase::distance> test_distance;
  std::string test_class;

  std::cout << "Start prepearing for inference" << std::endl;
  prepare_for_inference();
  std::cout << "Preparaing for inference was finished" << std::endl;
  std::cout << "Finding TOP " << top_N_classes << " among "
            << this->db_handler->get_config_top_n() << std::endl;

  for (const auto &test_img_path : test_imgs_paths) {
    test_img.img_path = test_img_path;
    test_img.img_class = common_ops::extract_class(test_img_path);
    test_img.img =
        fs_img::read_img(test_img_path, db_handler->get_config_input_size());

    testimg_vector.emplace_back(test_img);
  }

  for (auto it = testimg_vector.begin(); it != testimg_vector.end(); ++it) {
    test_distance = inference_and_matching(it->img_path);
    auto proposed_classes = choose_classes(test_distance, it, top_N_classes);
    if (!it->is_correct) {
      db_handler->add_error_entry(it->img_class, it->img_path, test_class);
    }

    // it->is_correct = test_class == it->img_class; //So much simplified so
    // wow.
    it->img_classes_proposed = proposed_classes;
    it->distance = test_distance[0].dist;
    std::cout << it - testimg_vector.begin() + 1 << " of "
              << testimg_vector.size() << "\r" << std::flush;
  }

  float val_correct =
      std::count_if(testimg_vector.begin(), testimg_vector.end(), IsCorrect);

  float metrics = val_correct / testimg_vector.size() * 100.f;
  std::cout << "Accuracy is : " << metrics << "%" << std::endl;
  std::cout << "Got " << val_correct << " out of " << testimg_vector.size()
            << " right" << std::endl;

  return metrics;
}

bool MetricsBase::prepare_for_inference() {
  return WrapperBase::prepare_for_inference();
}

std::vector<std::string> MetricsBase::choose_classes(
    const std::vector<WrapperBase::distance> &matched_images_list,
    std::vector<testimg_entry>::iterator &it, unsigned int top_N_classes) {
  std::set<std::string> top_classes_set;
  std::string test_class;

  for (const auto &res_it : matched_images_list) {
    test_class = common_ops::extract_class(res_it.path);
    top_classes_set.insert(test_class);

    if (top_classes_set.size() >= top_N_classes) {
      break;
    }
  }

  it->is_correct = top_classes_set.count(it->img_class) != 0;

  return std::vector<std::string>(top_classes_set.begin(),
                                  top_classes_set.end());
}
