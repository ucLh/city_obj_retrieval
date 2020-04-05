#include "metrics_base.h"
#include "gtest/gtest.h"
#include <algorithm>

class MetricsBaseTester : public MetricsBase {
public:
  std::vector<std::string> choose_classes_wrapper(
      const std::vector<EmbeddingsBase::distance> &matched_images_list,
      testimg_entry &test_img, unsigned int top_N_classes = 2) {
    return MetricsBase::choose_classes(matched_images_list, test_img,
                                       top_N_classes);
  }

  static bool check_correctness(MetricsBase::testimg_entry &entry) {
    return entry.is_correct;
  }
};

TEST(_choose_classes, _chooses_topN_unique_classes) {
  MetricsBaseTester wrapper;

  std::vector<EmbeddingsBase::distance> image_list = {
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

  for (auto it : img_entry_vec) {
    auto proposed_classes = wrapper.choose_classes_wrapper(image_list, it, 5);
  }

  float val_correct = std::count_if(img_entry_vec.begin(), img_entry_vec.end(),
                                    MetricsBaseTester::check_correctness);
  
  auto res_vec =
      wrapper.choose_classes_wrapper(image_list, img_entry_vec[0], 5);
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
