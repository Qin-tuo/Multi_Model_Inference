// Native rosbag2 playback; admission uses actual publisher/reader matches.
#include <rosbag2_transport/player.hpp>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

class MatchedPlayer : public rosbag2_transport::Player {
public:
  using rosbag2_transport::Player::Player;
  bool readers_matched() {
    const auto publishers = get_publishers();
    if (publishers.empty()) return false;
    for (const auto &entry : publishers)
      if (entry.second->get_subscription_count() == 0) return false;
    return true;
  }
};

int main(int argc, char **argv) {
  if (argc != 2) return 2;
  const char *domain = std::getenv("ROS_DOMAIN_ID");
  if (!domain || std::string(domain) == "42") {
    std::cerr << "An explicit isolated ROS domain is required\n";
    return 2;
  }
  rclcpp::init(argc, argv);
  rosbag2_storage::StorageOptions storage;
  storage.uri = argv[1]; storage.storage_id = "mcap";
  rosbag2_transport::PlayOptions options;
  options.rate = 1.; options.read_ahead_queue_size = 8;
  options.start_paused = true; options.disable_keyboard_controls = true;
  options.clock_publish_on_topic_publish = true;
  options.clock_publish_frequency = 50.;
  for (const auto &topic : {
      "/camera/camera/color/image_raw",
      "/camera/camera/aligned_depth_to_color/image_raw",
      "/camera/camera/aligned_depth_to_color/camera_info"}) {
    options.topic_qos_profile_overrides.emplace(topic, rclcpp::QoS(10).reliable());
  }
  auto player = std::make_shared<MatchedPlayer>(storage, options);
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(player);
  std::thread spin([&executor] { executor.spin(); });
  int status = 0;
  try {
    if (!player->play() || !player->wait_for_playback_to_start(std::chrono::seconds(10)))
      throw std::runtime_error("Native player did not start paused");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (true) {
      if (player->readers_matched()) break;
      if (!rclcpp::ok() || std::chrono::steady_clock::now() > deadline)
        throw std::runtime_error("Actual replay publishers have no matched readers");
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    std::cout << "Matched readers for all recorded topics; resuming native player" << std::endl;
    player->resume();
    if (!player->wait_for_playback_to_finish(std::chrono::seconds(90)))
      throw std::runtime_error("Native playback deadline exceeded");
  } catch (const std::exception &error) {
    std::cerr << error.what() << std::endl;
    status = 1;
  }
  player->stop(); executor.cancel(); spin.join(); player.reset(); rclcpp::shutdown();
  return status;
}
