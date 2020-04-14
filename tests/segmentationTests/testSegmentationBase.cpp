#include "tf_wrapper/common/fs_handling.h"
#include "tf_wrapper/segmentation_base.h"
#include "gtest/gtest.h"

#include <vector>

void load_images(std::vector<cv::Mat> &images, const std::string &dir_path) {
  auto image_paths = fs_img::list_imgs(dir_path);
  for (const auto &path : image_paths) {
    images.emplace_back(fs_img::read_img(path));
  }
}

void check_equality(const std::vector<cv::Mat> &results,
                    const std::vector<cv::Mat> &preprocessed) {
  bool are_equal;
  for (unsigned long i = 0; i < results.size(); ++i) {
    are_equal = cv::countNonZero(preprocessed[i] != results[i]) == 0;
    ASSERT_TRUE(are_equal);
  }
}

TEST(segmentation_modes, preserves_order) {
  std::vector<cv::Mat> indexed_images, colored_images, masked_images;
  load_images(indexed_images, "segmented_images/indexed");
  load_images(colored_images, "segmented_images/colored");
  load_images(masked_images, "segmented_images/masked");

  // Sanity check
  ASSERT_TRUE(cv::countNonZero(indexed_images[0] != indexed_images[0]) == 0);

  SegmentationWrapperBase seg_wrapper;
  seg_wrapper.prepare_for_inference("seg_config.json");
  seg_wrapper.process_images();
  auto indexed_results = seg_wrapper.get_indexed(true);
  auto colored_results = seg_wrapper.get_colored(true);
  auto masked_results = seg_wrapper.get_masked(true, {8, 11, 13});
  colored_results = seg_wrapper.get_colored(true);
  colored_results = seg_wrapper.get_colored(true);
  indexed_results = seg_wrapper.get_indexed(true);
  indexed_results = seg_wrapper.get_indexed(true);
  masked_results = seg_wrapper.get_masked(true, {8, 11, 13});
  masked_results = seg_wrapper.get_masked(true, {8, 11, 13});

  ASSERT_EQ(indexed_results.size(), indexed_images.size());
  ASSERT_EQ(colored_results.size(), colored_images.size());
  ASSERT_EQ(masked_results.size(), masked_images.size());
  check_equality(indexed_results, indexed_images);
  check_equality(colored_results, colored_images);
  check_equality(masked_results, masked_images);
}

TEST(process_images, inferences_input) {
  SegmentationWrapperBase seg_wrapper;
  seg_wrapper.prepare_for_inference("seg_config.json");
  seg_wrapper.process_images();
  int size = seg_wrapper.get_indexed(true).size();
  ASSERT_EQ(size, 4);
}

TEST(process_images_w_paths, inferences_input) {
  auto image_paths = fs_img::list_imgs("helsinki_2_apteeki");

  SegmentationWrapperBase seg_wrapper;
  seg_wrapper.prepare_for_inference("seg_config.json");
  seg_wrapper.process_images(image_paths);
  int size = seg_wrapper.get_indexed(true).size();
  ASSERT_EQ(size, 4);
}

TEST(process_images_w_images, inferences_input) {
  std::vector<cv::Mat> images;
  load_images(images, "helsinki_2_apteeki");

  SegmentationWrapperBase seg_wrapper;
  seg_wrapper.prepare_for_inference("seg_config.json");
  seg_wrapper.process_images(images);
  int size = seg_wrapper.get_indexed(true).size();
  ASSERT_EQ(size, 4);
}

TEST(process_images, handles_multiple_calls) {
  std::vector<cv::Mat> colored_images1, colored_images2, images1, images2;
  load_images(colored_images1, "segmented_images/colored");
  load_images(colored_images2, "segmented_images/helsinki_3");
  load_images(images1, "helsinki_2_apteeki");
  load_images(images2, "helsinki_3_usadba");

  SegmentationWrapperBase seg_wrapper;
  seg_wrapper.prepare_for_inference("seg_config.json");
  
  seg_wrapper.process_images(images1);
  auto colored_results1 = seg_wrapper.get_colored(true);
  colored_results1 = seg_wrapper.get_colored(true);
  ASSERT_EQ(colored_results1.size(), colored_images1.size());
  check_equality(colored_results1, colored_images1);

  seg_wrapper.process_images(images2);
  auto colored_results2 = seg_wrapper.get_colored(true);
  colored_results2 = seg_wrapper.get_colored(true);
  ASSERT_EQ(colored_results2.size(), colored_images2.size());
  check_equality(colored_results2, colored_images2);
}