#include "metrics_base.h"
#include "tf_wrapper/common/common_ops.h"
#include "tf_wrapper/segmentation_wrapper.h"
#include <iostream>
#include <set>

//std::vector<EmbeddingsWrapper::distance>
//MetricsBase::inference_and_matching(std::string img_path) {
//  return EmbeddingsWrapper::inference_and_matching(img_path);
//}

float MetricsBase::get_metrics(std::string &queries_path, int top_N_classes,
                               bool use_segmentation) {
  queries_path_ = queries_path;

  std::vector<std::string> test_imgs_paths = fs_img::list_imgs(queries_path);
  testimg_entry test_img;
  std::vector<EmbeddingsWrapper::distance> matched_images_list;
  std::vector<cv::Mat> seg_images;

  if (use_segmentation) {
    std::cout << "Running image segmentation" << std::endl;
    SegmentationWrapper seg_wrapper;
    seg_wrapper.prepare_for_inference("seg_config.json");
    seg_wrapper.process_images(test_imgs_paths);
    seg_images = seg_wrapper.get_masked({8, 11, 13});
  }

  std::cout << "Start prepearing for inference" << std::endl;
  prepare_for_inference("embed_config.json");
  std::string series_path = db_handler_->get_config_imgs_path();
  std::cout << "Preparaing for inference was finished" << std::endl;
  std::cout << "Finding TOP " << top_N_classes << " among "
            << db_handler_->get_config_top_n() << std::endl;

  float val_correct = 0.f, metrics;
  int i = 0;
  double ap_sum = 0.f;
  for (const auto &test_img_path : test_imgs_paths) {
    test_img.img_path = test_img_path;
    test_img.img_class =
        common_ops::extract_class(test_img_path, series_path, queries_path);

    if (use_segmentation) {
      test_img.img = std::move(seg_images[i]);
    } else {
      test_img.img = fs_img::read_img(test_img_path);
    }

    matched_images_list = inference_and_matching(test_img.img_path);
    auto proposed_classes =
        choose_classes(matched_images_list, test_img, top_N_classes,
                       queries_path, series_path);
    ap_sum +=
        calculate_average_precision(matched_images_list, test_img.img_class);
    if (test_img.is_correct) {
      ++val_correct;
    } else {
      db_handler_->add_error_entry(test_img.img_class, test_img.img_path,
                                   proposed_classes[0]);
    }
    test_img.img_classes_proposed = proposed_classes;
    test_img.distance = matched_images_list[0].dist;
    ++i;
    std::cout << "Wrapper Info: " << i << " of " << test_imgs_paths.size()
              << " was processed"
              << "\r" << std::flush;
  }
  std::cout << std::endl;

  metrics = val_correct / test_imgs_paths.size() * 100.f;
  double mean_ap = ap_sum / test_imgs_paths.size();
  std::cout << "Accuracy is : " << metrics << "%" << std::endl;
  std::cout << "Mean average precision is : " << mean_ap << std::endl;
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

double MetricsBase::calculate_average_precision(
    const std::vector<EmbeddingsWrapper::distance> &matched_images_list,
    const std::string &target_class_name) {
  double sum = 0;
  int appearance_count = 0;
  auto series_path = db_handler_->get_config_imgs_path();
  for (int i = 0; i < matched_images_list.size(); ++i) {
    auto class_name = common_ops::extract_class(matched_images_list[i].path,
                                                series_path, queries_path_);
    if (target_class_name == class_name) {
      ++appearance_count;
      sum += (double)appearance_count / (i + 1);
    }
  }
  if (appearance_count == 0) {
    return 0;
  }
  return sum / appearance_count;
}