#include "tf_wrapper/common/common_ops.h"
#include "gtest/gtest.h"

TEST(PATH, testClassExtraction) {
  using namespace common_ops;
  ASSERT_ANY_THROW(extract_class(""));
  ASSERT_EQ(extract_class("/some/test/path/is_here/series/category_name/images/"
                          "this_is_img__name.jpg"),
            "category_name");
  ASSERT_EQ(extract_class("/some/test/path/is_here/series/category_name/images/"
                          "this_is_img__name.jpg"),
            "category_name");
  ASSERT_EQ(extract_class("/some/test/path/is_here/series/category_name-name/"
                          "images/this_is_img__name.jpg"),
            "category_name-name");
  ASSERT_EQ(
      extract_class("/some/test/path/is_here/series/category_name-name.skip/"
                    "images/this_is_img__name.jpg"),
      "category_name-name.skip");
  ASSERT_EQ(
      extract_class("/some/test/path/is_here/series/category_name__DEVICE_NAME/"
                    "images/this_is_img__name.jpg"),
      "category_name");
  ASSERT_EQ(
      extract_class(
          "/some/test/path/is_here/series/category_name__DEVICE_NAME.trash/"
          "images/this_is_img__name.jpg"),
      "category_name");
  ASSERT_EQ(
      extract_class("/some/test/path/is_here/series/category_name__DEVICE_NAME/"
                    "images/this_is_img__name.jpg"),
      "category_name");
  ASSERT_EQ(extract_class(
                "/some/test/path/is_here/series/"
                "category_name.skip__DEVICE_NAME/images/this_is_img__name.jpg"),
            "category_name.skip");

  ASSERT_EQ(extract_class("/some/test/path/is_here/queries/category_name/"
                          "images/this_is_img__name.jpg"),
            "category_name");
  ASSERT_EQ(extract_class("/some/test/path/is_here/queries/category_name/"
                          "images/this_is_img__name.jpg"),
            "category_name");
  ASSERT_EQ(extract_class("/some/test/path/is_here/queries/category_name-name/"
                          "images/this_is_img__name.jpg"),
            "category_name-name");
  ASSERT_EQ(
      extract_class("/some/test/path/is_here/queries/"
                    "category_name__DEVICE_NAME/images/this_is_img__name.jpg"),
      "category_name");
  ASSERT_EQ(
      extract_class(
          "/some/test/path/is_here/queries/category_name__DEVICE_NAME.trash/"
          "images/this_is_img__name.jpg"),
      "category_name");
  ASSERT_EQ(
      extract_class("/some/test/path/is_here/queries/"
                    "category_name__DEVICE_NAME/images/this_is_img__name.jpg"),
      "category_name");
  ASSERT_EQ(extract_class(
                "/some/test/path/is_here/queries/"
                "category_name.skip__DEVICE_NAME/images/this_is_img__name.jpg"),
            "category_name.skip");
  ASSERT_EQ(
      extract_class("/some/test/path/is_here/queries/category_name-name.skip/"
                    "images/this_is_img__name.jpg"),
      "category_name-name.skip");
}