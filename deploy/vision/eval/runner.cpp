// Evaluation adapter for the already installed, pinned yolo_ros core.
// Preprocessing, inference and postprocessing all belong to that framework.
#include <yolo_ros/yolo/model_factory.hpp>
#include <opencv2/opencv.hpp>
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: vision-eval MODEL TASK PROVIDER CACHE INPUT_TSV OUTPUT_JSONL\n";
    return 2;
  }
  try {
    yolo_ros::yolo::utils::YoloParams params{};
    params.model_path = argv[1];
    params.model_type = argv[2];
    params.provider = argv[3];
    params.device = params.provider == "cpu" ? "cpu" : "tensorrt:0";
    params.trt_engine_cache_path = argv[4];
    params.trt_fp16_enable = false;
    params.trt_engine_cache_enable = true;
    params.cuda_graph_enable = false;
    params.threshold = .5f;
    params.iou = .45f;
    params.max_det = 100;
    params.n_threads = 2;
    params.max_fps = 0; // Offline sequential processing, not a timing benchmark.
    params.img_width = params.img_height = 640;
    params.input_color = "rgb";
    auto model = yolo_ros::yolo::create_model(params);
    if (model->active_provider() != params.provider) {
      throw std::runtime_error("Requested execution provider was not selected");
    }
    std::ifstream inputs(argv[5]);
    std::ofstream output(argv[6], std::ios::out | std::ios::trunc);
    if (!inputs || !output) throw std::runtime_error("Cannot open input/output");
    output << std::setprecision(10);
    std::string path;
    long image_id;
    int count = 0;
    while (inputs >> image_id >> path) {
      auto image = cv::imread(path, cv::IMREAD_COLOR);
      if (image.empty()) throw std::runtime_error("Cannot read image: " + path);
      const auto started = std::chrono::steady_clock::now();
      auto detections = model->detect(image);
      // Same publication cap as DetectionPlugin::process_camera/process_batch.
      if (detections.size() > 100) detections.resize(100);
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - started).count();
      output << "{\"image_id\":" << image_id << ",\"elapsed_ms\":" << elapsed
             << ",\"predictions\":[";
      for (size_t i = 0; i < detections.size(); ++i) {
        const auto &d = detections[i];
        if (i) output << ',';
        output << "{\"class_id\":" << d.class_id << ",\"score\":" << d.score
               << ",\"bbox\":[" << d.bbox.center.position.x - d.bbox.size.x / 2
               << ',' << d.bbox.center.position.y - d.bbox.size.y / 2 << ','
               << d.bbox.size.x << ',' << d.bbox.size.y << ']';
        if (params.model_type == "pose") {
          std::array<double, 51> points{};
          for (const auto &point : d.keypoints.data) {
            if (point.id < 1 || point.id > 17) throw std::runtime_error("Invalid keypoint ID");
            const size_t offset = (point.id - 1) * 3;
            points[offset] = point.point.x;
            points[offset + 1] = point.point.y;
            points[offset + 2] = point.score;
          }
          output << ",\"keypoints\":[";
          for (size_t k = 0; k < points.size(); ++k) {
            if (k) output << ',';
            output << points[k];
          }
          output << ']';
        }
        output << '}';
      }
      output << "]}\n";
      output.flush();
      if (!output) throw std::runtime_error("Cannot persist output");
      if (++count % 100 == 0) std::cout << "completed=" << count << std::endl;
    }
    if (!inputs.eof()) throw std::runtime_error("Malformed input list");
    std::cout << "completed=" << count << " provider=" << model->active_provider() << std::endl;
    return count > 0 ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
