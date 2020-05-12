#include "metrics_base.h"
#include "gtest/gtest.h"
#include <algorithm>

class MetricsBaseTester : public MetricsBase {
public:
  std::vector<std::string> choose_classes_wrapper(
      const std::vector<EmbeddingsWrapper::distance> &matched_images_list,
      testimg_entry &test_img, unsigned int top_N_classes,
      std::string &queries_path, std::string &series_path) {

    return MetricsBase::choose_classes(matched_images_list, test_img,
                                       top_N_classes, queries_path,
                                       series_path);
  }

  static bool check_correctness(MetricsBase::testimg_entry &entry) {
    return entry.is_correct;
  }

  float calculate_ap(
      const std::vector<EmbeddingsWrapper::distance> &matched_images_list,
      const std::string &target_class_name) {
    return calculate_average_precision(matched_images_list, target_class_name);
  }

  bool set_queries_path(const std::string &queries_path) {
    queries_path_ = queries_path;
    return true;
  }

  bool set_series_path(const std::string &series_path) {
    db_handler_->set_config_imgs_path(series_path);
    return true;
  }
};

TEST(_choose_classes, _chooses_topN_unique_classes) {
  MetricsBaseTester wrapper;

  std::vector<EmbeddingsWrapper::distance> image_list = {
      {0.1, "series/build1/img/a.jpg"}, {0.2, "series/build2/img/a.jpg"},
      {0.2, "series/build1/img/a.jpg"}, {0.2, "series/build3/img/a.jpg"},
      {0.2, "series/build2/img/a.jpg"}, {0.2, "series/build4/img/a.jpg"},
      {0.2, "series/build5/img/a.jpg"}, {0.2, "series/build6/img/a.jpg"},
      {0.2, "series/build7/img/a.jpg"},
  };

  MetricsBase::testimg_entry img_entry1, img_entry2;
  img_entry1.img_class = "build6";
  img_entry2.img_class = "build2";
  std::vector<MetricsBase::testimg_entry> img_entry_vec = {img_entry1,
                                                           img_entry2};
  std::string series_path = "series";
  std::string queries_path = "queries";

  for (auto it : img_entry_vec) {
    auto proposed_classes = wrapper.choose_classes_wrapper(
        image_list, it, 5, series_path, queries_path);
  }

  float val_correct = std::count_if(img_entry_vec.begin(), img_entry_vec.end(),
                                    MetricsBaseTester::check_correctness);

  auto res_vec = wrapper.choose_classes_wrapper(image_list, img_entry_vec[0], 5,
                                                series_path, queries_path);
  ASSERT_FALSE(img_entry_vec[0].is_correct);
  ASSERT_TRUE(img_entry_vec[1].is_correct);
  ASSERT_EQ(val_correct, 1.0);
  ASSERT_TRUE(std::find(res_vec.begin(), res_vec.end(), "build1") !=
              res_vec.end());
  ASSERT_TRUE(std::find(res_vec.begin(), res_vec.end(), "build2") !=
              res_vec.end());
  ASSERT_TRUE(std::find(res_vec.begin(), res_vec.end(), "build3") !=
              res_vec.end());
  ASSERT_TRUE(std::find(res_vec.begin(), res_vec.end(), "build4") !=
              res_vec.end());
  ASSERT_TRUE(std::find(res_vec.begin(), res_vec.end(), "build5") !=
              res_vec.end());
  ASSERT_FALSE(std::find(res_vec.begin(), res_vec.end(), "build6") !=
               res_vec.end());
  ASSERT_FALSE(std::find(res_vec.begin(), res_vec.end(), "build7") !=
               res_vec.end());
};

TEST(_calculate_ap, _calculates_ap_correctly) {
  MetricsBaseTester wrapper;

  std::vector<EmbeddingsWrapper::distance> image_list = {
      {0.1, "series/build1/img/a.jpg"}, {0.2, "series/build2/img/a.jpg"},
      {0.2, "series/build1/img/a.jpg"}, {0.2, "series/build3/img/a.jpg"},
      {0.2, "series/build2/img/a.jpg"}, {0.2, "series/build1/img/a.jpg"},
      {0.2, "series/build5/img/a.jpg"}, {0.2, "series/build6/img/a.jpg"},
      {0.2, "series/build7/img/a.jpg"},
  };
  wrapper.set_series_path("series/");
  wrapper.set_queries_path("/queries");

  float ap = wrapper.calculate_ap(image_list, "build1");
  ASSERT_FLOAT_EQ(ap, (1.0 + 2.0 / 3.0 + 1.0 / 2.0) / 3.0);

  ap = wrapper.calculate_ap(image_list, "build2");
  ASSERT_FLOAT_EQ(ap, (1.0 / 2.0 + 2.0 / 5.0) / 2.0);

  ap = wrapper.calculate_ap(image_list, "build3");
  ASSERT_FLOAT_EQ(ap, 1.0 / 4.0);

  ap = wrapper.calculate_ap(image_list, "build4");
  ASSERT_FLOAT_EQ(ap, 0.0);
};