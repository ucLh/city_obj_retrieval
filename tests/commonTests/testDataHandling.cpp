#include "tf_wrapper/common/common_ops.h"
#include "gtest/gtest.h"

TEST(load_config, load_config_unexisted_Test) {
  DataHandling data_handler;
  data_handler.config_path = "unexisted_config.json";
  ASSERT_FALSE(data_handler.load_config());
}

TEST(load_config, load_config_load_Test) {
  DataHandling data_handler;
  data_handler.config_path = "sample_config.json";
  data_handler.load_config();
  auto config = data_handler.config;

  ASSERT_EQ(config.input_size, cv::Size(256, 256));
  ASSERT_EQ(config.datafile_path, "this/is/test/path/testdatafile.txt");
  ASSERT_EQ(config.imgs_path, "this/is/test/path/test_img_path/");
  ASSERT_EQ(config.pb_path, "this/is/test/path/testpb.pb");
  ASSERT_EQ(config.input_node, "test_input_node:0");
  ASSERT_EQ(config.output_node, "test_output_node:0");
}

TEST(load_database, load_database_load_Test) {
  DataHandling data_handler;
  data_handler.config_path = "sample_config.json";
  data_handler.load_config();
  data_handler.load_database();

  ASSERT_EQ(data_handler.data_vec_base[0].filepath,
            "/this/is/test/path/testimg.jpg");
  ASSERT_EQ(data_handler.data_vec_base[0].embedding[0], -0.12846693396568299);
}

TEST(add_json_entry, add_json_entry_add_Test) {
  DataHandling data_handler;

  auto new_entry = DataHandling::data_vec_entry();
  new_entry.filepath = "this/is/test/filepath/testimg1.jpg";
  new_entry.embedding = {0.0010000000474974514};

  data_handler.config_path = "sample_config.json";
  data_handler.load_config();
  data_handler.load_database();
  data_handler.add_json_entry(new_entry);

  data_handler.load_database();

  ASSERT_EQ(data_handler.data_vec_base[1].filepath,
            "this/is/test/filepath/testimg1.jpg");
  ASSERT_EQ(data_handler.data_vec_base[1].embedding[0], 0.0010000000474974514);
}

TEST(read_img, read_img_read_Test) {
  auto size = cv::Size_<int>(256, 256);
  cv::Mat test_img = fs_img::read_img("Lenna.jpg");

  ASSERT_FALSE(test_img.empty());
  //    Comment these lines out for now. New checkpoints have resize op inside
  //    them ASSERT_EQ(test_img.rows, 256); ASSERT_EQ(test_img.cols, 256);
}

TEST(list_imgs, list_imgs_lis_Test) {
  std::vector<std::string> list_of_images = fs_img::list_imgs(".");

  ASSERT_EQ(list_of_images[0], "./Lenna.jpg");
}