#include "tf_wrapper/common/common_ops.h"
#include "tf_wrapper/embeddings_wrapper.h"
#include "gtest/gtest.h"

TEST(EXECUTION, EXECUTION_Acceptance_Test) {
  std::string inFileName = "queries/helsinki_1_andante/IMG_7813_4.jpg";
  std::string gt_class = common_ops::extract_class(inFileName);
  EmbeddingsWrapper tf_wrapper;
  tf_wrapper.prepare_for_inference("config.json");
  tf_wrapper.topN = 1;

  std::vector<EmbeddingsWrapper::distance> results =
      tf_wrapper.inference_and_matching(inFileName);
  std::string predicted_class = common_ops::extract_class(results[0].path);
  ASSERT_EQ(predicted_class, gt_class);
  //    common_ops::delete_safe(tf_wrapper);
}
