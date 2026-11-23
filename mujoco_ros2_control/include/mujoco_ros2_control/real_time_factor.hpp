#pragma once

#include <optional>

namespace mujoco_ros2_control
{

/**
 * @brief Measures sim seconds per wall second over consecutive windows of wall time.
 *
 * Dependency-free so it can be unit tested without a simulation.
 */
class RealTimeFactorWindow
{
public:
  explicit RealTimeFactorWindow(double window) : window_(window)
  {
  }

  /**
   * @brief Add a (wall, sim) sample, both in seconds.
   * @return The factor over the window that this sample closes, otherwise nothing.
   */
  std::optional<double> update(double wall, double sim)
  {
    // A sim time that jumps back is a reset: the window must not span it.
    if (!started_ || sim < start_sim_)
    {
      start(wall, sim);
      return std::nullopt;
    }
    const double elapsed = wall - start_wall_;
    if (elapsed < window_)
    {
      return std::nullopt;
    }
    const double factor = (sim - start_sim_) / elapsed;
    start(wall, sim);
    return factor;
  }

  /// Forget the open window. Call it while paused, or the pause reads as a slow sim.
  void reset()
  {
    started_ = false;
  }

private:
  void start(double wall, double sim)
  {
    start_wall_ = wall;
    start_sim_ = sim;
    started_ = true;
  }

  double window_;
  double start_wall_ = 0.0;
  double start_sim_ = 0.0;
  bool started_ = false;
};

}  // namespace mujoco_ros2_control
