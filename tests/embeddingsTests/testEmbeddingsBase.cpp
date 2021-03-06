#include <utility>

#include "tf_wrapper/common/common_ops.h"
#include "tf_wrapper/embeddings_wrapper.h"
#include "gtest/gtest.h"

class EmbeddingsWrapperTester : public EmbeddingsWrapper {
public:
  auto matching(std::vector<DataHandling::data_vec_entry> &base,
                std::vector<float> &target) {
    return EmbeddingsWrapper::matching(base, target);
  }

  auto add_updates() { return EmbeddingsWrapper::add_updates(); }

  auto check_for_updates() { return EmbeddingsWrapper::check_for_updates(); }

  auto get_distances() { return distances_; }

  auto set_data_vec_base(std::vector<DataHandling::data_vec_entry> &vec_base) {
    db_handler_->set_data_vec_base(vec_base);
  }

  auto get_data_vec_base() { return db_handler_->get_data_vec_base(); }

  auto set_list_of_imgs(std::vector<std::string> &list_of_imgs) {
    this->list_of_imgs_ = list_of_imgs;
  }

  auto get_list_of_imgs() { return list_of_imgs_; }

  auto load_config(const std::string &config_path) {
    return EmbeddingsWrapper::load_config(config_path);
  }
};

TEST(_matching, _matching_testMatchingCorrectnes_Test) {

  EmbeddingsWrapperTester wrapper;

  wrapper.topN = 3;

  std::vector<float> test_target;

  test_target = {0.7, 0.7, 0.7};

  std::vector<DataHandling::data_vec_entry> test_base;

  DataHandling::data_vec_entry test_entry_farthest;
  DataHandling::data_vec_entry test_entry_closest;
  DataHandling::data_vec_entry test_entry_middle;

  test_entry_farthest.embedding = {0.05, 0.05, 0.05};
  test_entry_farthest.filepath = "/testpath/series/farthest_res/randimg.jpg";

  test_entry_closest.embedding = {0.6, 0.6, 0.6};
  test_entry_closest.filepath = "/testpath/series/closest_res/randimg.jpg";

  test_entry_middle.embedding = {0.3, 0.3, 0.3};
  test_entry_middle.filepath = "/testpath/series/middle_res/randimg.jpg";

  test_base.emplace_back(test_entry_farthest);
  test_base.emplace_back(test_entry_closest);
  test_base.emplace_back(test_entry_middle);

  wrapper.matching(test_base, test_target);

  ASSERT_EQ(wrapper.get_distances()[0].path,
            "/testpath/series/closest_res/randimg.jpg");
  ASSERT_EQ(wrapper.get_distances()[1].path,
            "/testpath/series/middle_res/randimg.jpg");
  ASSERT_EQ(wrapper.get_distances()[2].path,
            "/testpath/series/farthest_res/randimg.jpg");

  //    common_ops::delete_safe(wrapper);
}

TEST(_check_for_updates, _check_no_changes) {
  std::vector<DataHandling::data_vec_entry> test_base;

  DataHandling::data_vec_entry test_entry_farthest;
  DataHandling::data_vec_entry test_entry_closest;
  DataHandling::data_vec_entry test_entry_middle;

  test_entry_farthest.embedding = {0.05, 0.05, 0.05};
  test_entry_farthest.filepath = "/testpath/series/farthest_res/randimg.jpg";

  test_entry_closest.embedding = {0.6, 0.6, 0.6};
  test_entry_closest.filepath = "/testpath/series/closest_res/randimg.jpg";

  test_entry_middle.embedding = {0.3, 0.3, 0.3};
  test_entry_middle.filepath = "/testpath/series/middle_res/randimg.jpg";

  test_base.emplace_back(test_entry_farthest);
  test_base.emplace_back(test_entry_closest);
  test_base.emplace_back(test_entry_middle);

  std::vector<std::string> test_list_of_imgs;
  test_list_of_imgs.emplace_back(test_entry_farthest.filepath);
  test_list_of_imgs.emplace_back(test_entry_closest.filepath);
  test_list_of_imgs.emplace_back(test_entry_middle.filepath);

  EmbeddingsWrapperTester wrapper;
  wrapper.set_data_vec_base(test_base);
  wrapper.set_list_of_imgs(test_list_of_imgs);
  wrapper.check_for_updates();

  ASSERT_TRUE(wrapper.get_list_of_imgs().empty());
}

TEST(_check_for_updates, _check_some_changes) {
  std::vector<DataHandling::data_vec_entry> test_base;

  DataHandling::data_vec_entry test_entry_closest;
  DataHandling::data_vec_entry test_entry_middle;

  test_entry_closest.embedding = {0.6, 0.6, 0.6};
  test_entry_closest.filepath = "/testpath/series/closest_res/randimg.jpg";

  test_entry_middle.embedding = {0.3, 0.3, 0.3};
  test_entry_middle.filepath = "/testpath/series/middle_res/randimg.jpg";

  test_base.emplace_back(test_entry_closest);
  test_base.emplace_back(test_entry_middle);

  std::vector<std::string> test_list_of_imgs;
  test_list_of_imgs.emplace_back("/testpath/series/farthest_res/randimg.jpg");
  test_list_of_imgs.emplace_back(test_entry_closest.filepath);
  test_list_of_imgs.emplace_back(test_entry_middle.filepath);

  EmbeddingsWrapperTester wrapper;
  wrapper.set_data_vec_base(test_base);
  wrapper.set_list_of_imgs(test_list_of_imgs);
  wrapper.check_for_updates();
  ASSERT_FALSE(wrapper.get_list_of_imgs().empty());
}

TEST(_check_for_updates, _remembers_images_that_are_not_present_anymore) {
  std::vector<DataHandling::data_vec_entry> test_base;

  DataHandling::data_vec_entry test_entry_farthest;
  DataHandling::data_vec_entry test_entry_closest;
  DataHandling::data_vec_entry test_entry_middle;

  test_entry_farthest.embedding = {0.05, 0.05, 0.05};
  test_entry_farthest.filepath = "/testpath/series/farthest_res/randimg.jpg";

  test_entry_closest.embedding = {0.6, 0.6, 0.6};
  test_entry_closest.filepath = "/testpath/series/closest_res/randimg.jpg";

  test_entry_middle.embedding = {0.3, 0.3, 0.3};
  test_entry_middle.filepath = "/testpath/series/middle_res/randimg.jpg";

  test_base.emplace_back(test_entry_farthest);
  test_base.emplace_back(test_entry_closest);
  test_base.emplace_back(test_entry_middle);

  std::vector<std::string> test_list_of_imgs;
  test_list_of_imgs.emplace_back(test_entry_farthest.filepath);
  test_list_of_imgs.emplace_back(test_entry_closest.filepath);

  EmbeddingsWrapperTester wrapper;
  wrapper.set_data_vec_base(test_base);
  wrapper.set_list_of_imgs(test_list_of_imgs);
  wrapper.check_for_updates();

  ASSERT_TRUE(wrapper.get_list_of_imgs().empty());
}

TEST(_add_updates, adds_new_images) {
  EmbeddingsWrapperTester wrapper;

  std::vector<DataHandling::data_vec_entry> test_base;

  DataHandling::data_vec_entry test_entry_farthest;

  test_entry_farthest.embedding = {0.05, 0.05, 0.05};
  test_entry_farthest.filepath = "/testpath/series/farthest_res/randimg.jpg";

  test_base.emplace_back(test_entry_farthest);

  std::vector<std::string> test_list_of_imgs;
  test_list_of_imgs.emplace_back("./Lenna.jpg");

  wrapper.set_data_vec_base(test_base);
  wrapper.load_config("./config.json");
  wrapper.set_list_of_imgs(test_list_of_imgs);

  wrapper.add_updates();

  auto new_vec_base = wrapper.get_data_vec_base();

  ASSERT_EQ(new_vec_base[1].filepath, "./Lenna.jpg");
}

TEST(calc_distance_cosine, test_codine_distance_calculation) {
  std::vector<float> base{1.f, 2.f, 5.f};
  std::vector<float> target{5.f, 2.f, 1.f};

  float dist = EmbeddingMatching::calc_distance_cosine(base, target);
  ASSERT_EQ(dist, 16.f / 30.f);
}
