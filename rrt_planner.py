import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
import time

class Node:
    """RRT Node in (x, y, theta) configuration space."""
    def __init__(self, x, y, theta, parent=None):
        self.x = x
        self.y = y
        self.theta = self.normalize_angle(theta)
        self.parent = parent
        self.cost = 0.0
        self.path_to_parent = []

    def __repr__(self):
        return f"Node({self.x:.2f}, {self.y:.2f}, {self.theta:.2f})"

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to be within [-pi, pi]"""
        return (angle + math.pi) % (2 * math.pi) - math.pi

class RRTPlanner:
    """
    Rapidly-exploring Random Tree Planner for a rectangular robot.
    Configuration: (x, y, theta)
    """
    def __init__(self, start, goal, bounds, obstacles, robot_dims, params):
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
        self.bounds = bounds
        self.obstacles = obstacles
        self.robot_L, self.robot_W = robot_dims

        self.max_iter = params.get('max_iter', 5000)
        self.step_len = params.get('step_len', 0.5)
        self.goal_sample_rate = params.get('goal_sample_rate', 0.1)
        self.goal_dist_threshold = params.get('goal_dist_threshold', 1.0)
        self.angular_weight = params.get('angular_weight', 0.5)
        self.animation_update_rate = params.get('animation_update_rate', 100)

        self.tree = [self.start]
        self.path = None

    def _get_robot_corners(self, x, y, theta):
        L, W = self.robot_L, self.robot_W
        local_corners = np.array([
            [L/2, W/2], [-L/2, W/2], [-L/2, -W/2], [L/2, -W/2]
        ])
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        world_corners = (R @ local_corners.T).T + np.array([x, y])
        return world_corners

    def _is_inside_rect_obstacle(self, px, py, ox, oy, ow, oh):
        return (ox <= px <= ox + ow and oy <= py <= oy + oh)

    def _is_point_inside_robot(self, px, py, rx, ry, rtheta):
        L, W = self.robot_L, self.robot_W
        p_translated_x = px - rx
        p_translated_y = py - ry

        cos_theta = np.cos(-rtheta)
        sin_theta = np.sin(-rtheta)

        p_local_x = p_translated_x * cos_theta - p_translated_y * sin_theta
        p_local_y = p_translated_x * sin_theta + p_translated_y * cos_theta

        return (abs(p_local_x) <= L/2) and (abs(p_local_y) <= W/2)

    def _is_collision_free(self, x, y, theta):
        xmin, xmax, ymin, ymax = self.bounds
        if not (xmin < x < xmax and ymin < y < ymax):
            return False

        robot_corners = self._get_robot_corners(x, y, theta)

        for ox, oy, ow, oh in self.obstacles:
            for rx, ry in robot_corners:
                if self._is_inside_rect_obstacle(rx, ry, ox, oy, ow, oh):
                    return False

            obstacle_corners = [
                (ox, oy), (ox + ow, oy), (ox + ow, oy + oh), (ox, oy + oh)
            ]
            for px, py in obstacle_corners:
                if self._is_point_inside_robot(px, py, x, y, theta):
                    return False
        return True

    def _distance_metric(self, node1, node2):
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        d_theta = Node.normalize_angle(node1.theta - node2.theta)
        return np.sqrt(dx**2 + dy**2 + self.angular_weight * d_theta**2)

    def _get_nearest_node(self, q_rand):
        distances = [self._distance_metric(node, q_rand) for node in self.tree]
        nearest_index = np.argmin(distances)
        return self.tree[nearest_index]

    def _steer(self, q_near, q_rand):
        dx = q_rand.x - q_near.x
        dy = q_rand.y - q_near.y
        dist_xy = np.sqrt(dx**2 + dy**2)
        d_theta = Node.normalize_angle(q_rand.theta - q_near.theta)

        if dist_xy > self.step_len:
            scale = self.step_len / dist_xy
            new_x = q_near.x + dx * scale
            new_y = q_near.y + dy * scale
            new_theta = q_near.theta + d_theta * scale
        else:
            new_x = q_rand.x
            new_y = q_rand.y
            new_theta = q_rand.theta

        new_theta = Node.normalize_angle(new_theta)

        num_steps = 10
        path_to_new = []
        is_safe = True

        d_theta_new = Node.normalize_angle(new_theta - q_near.theta)

        for i in range(1, num_steps + 1):
            t = i / num_steps
            px = q_near.x + (new_x - q_near.x) * t
            py = q_near.y + (new_y - q_near.y) * t
            ptheta = q_near.theta + d_theta_new * t
            ptheta = Node.normalize_angle(ptheta)

            if not self._is_collision_free(px, py, ptheta):
                is_safe = False
                break

            path_to_new.append((px, py, ptheta))

        if is_safe:
            return Node(new_x, new_y, new_theta), path_to_new
        else:
            return None, []

    def _get_random_sample(self):
        if random.random() < self.goal_sample_rate:
            return self.goal

        xmin, xmax, ymin, ymax = self.bounds

        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        theta = random.uniform(-np.pi, np.pi)

        return Node(x, y, theta)

    def _extract_path(self, last_node):
        path = []
        current = last_node
        while current is not None:
            path.append((current.x, current.y, current.theta))
            if current.path_to_parent:
                path.extend(reversed(current.path_to_parent))
            current = current.parent

        path.reverse()
        return path

# --- Visualization Functions ---
def plot_static_environment(ax, planner, title="RRT Planning..."):
    """Draws the static parts of the environment."""
    ax.clear() # Clear previous plot
    xmin, xmax, ymin, ymax = planner.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)

    # Obstacles
    for ox, oy, ow, oh in planner.obstacles:
        rect = patches.Rectangle((ox, oy), ow, oh, facecolor='gray', edgecolor='black')
        ax.add_patch(rect)

    # Start and Goal
    ax.plot(planner.start.x, planner.start.y, 'o', color='blue', markersize=10, label='Start')
    ax.plot(planner.goal.x, planner.goal.y, '*', color='red', markersize=15, label='Goal')
    ax.legend()

def plot_final_path(ax, planner, robot_display_indices):
    """Draws the final path and robot footprints on the axes."""
    if not planner.path:
        return

    path_x = [p[0] for p in planner.path]
    path_y = [p[1] for p in planner.path]
    ax.plot(path_x, path_y, '-r', linewidth=2.0, label='Final Path')

    for i in robot_display_indices:
        if i < len(planner.path):
            x, y, theta = planner.path[i]
            corners = planner._get_robot_corners(x, y, theta)

            robot_footprint = patches.Polygon(corners, closed=True,
                                              fill=False, edgecolor='red',
                                              linewidth=1.0, alpha=0.8,
                                              linestyle='--')
            ax.add_patch(robot_footprint)

            ax.plot([x, x + planner.robot_L/2 * np.cos(theta)],
                    [y, y + planner.robot_L/2 * np.sin(theta)], 'r-')

    handles, labels = ax.get_legend_handles_labels()
    if 'Final Path' not in labels:
        handles.append(plt.Line2D([0], [0], color='r', linewidth=2, label='Final Path'))
        labels.append('Final Path')
    ax.legend(handles=handles, labels=labels)

def run_planning_animation(planner, scenario_name=""):
    """
    Runs the main RRT planning loop with live animation.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.ion()

    plot_static_environment(ax, planner, title=f"RRT Planning ({scenario_name})")
    plt.show()

    start_time = time.time()

    path_found = False
    for i in range(planner.max_iter):
        q_rand = planner._get_random_sample()
        q_nearest = planner._get_nearest_node(q_rand)

        q_new, path_to_new = planner._steer(q_nearest, q_rand)

        if q_new:
            q_new.parent = q_nearest
            q_new.cost = q_nearest.cost + np.linalg.norm([q_new.x - q_nearest.x, q_new.y - q_nearest.y])
            q_new.path_to_parent = path_to_new
            planner.tree.append(q_new)

            ax.plot([q_new.x, q_nearest.x], [q_new.y, q_nearest.y], '-k', linewidth=0.5, alpha=0.3)

            if planner._distance_metric(q_new, planner.goal) < planner.goal_dist_threshold:
                final_node, final_path = planner._steer(q_new, planner.goal)
                if final_node:
                    final_node.parent = q_new
                    final_node.path_to_parent = final_path
                    planner.path = planner._extract_path(final_node)

                    path_found = True
                    print(f"Path found in {i+1} iterations!")
                    break

        if (i + 1) % planner.animation_update_rate == 0:
            ax.set_title(f"RRT ({scenario_name}) - Iteration {i+1}/{planner.max_iter}")
            plt.pause(0.001)

    end_time = time.time()
    print(f"Planning took {end_time - start_time:.2f} seconds.")

    plt.ioff()

    if path_found:
        title = f"RRT Result ({scenario_name}) - Path Found ({i+1} iterations)"
        print("Path found! Drawing final path.")
        robot_display_indices = [0]
        path_len = len(planner.path)
        robot_display_indices.extend([
            int(path_len * 0.33),
            int(path_len * 0.66),
            path_len - 1
        ])
        robot_display_indices = sorted(list(set(robot_display_indices)))

        # Redraw static environment for a clean final plot
        plot_static_environment(ax, planner, title=title)
        plot_final_path(ax, planner, robot_display_indices)
    else:
        title = f"RRT Result ({scenario_name}) - Path Not Found"
        print("RRT failed to find a path.")
        ax.set_title(title)

    plt.show()

def run_narrow_passage_scenario():
    """SCENARIO 1: The U-shaped narrow passage."""
    print("Running Scenario 1: Narrow Passage")

    MAP_BOUNDS = (0, 10, 0, 10)
    ROBOT_DIMS = (1.0, 0.5)
    OBSTACLES = [
        (2.5, 3, 5, 0.5),
        (2.5, 3, 0.5, 4),
        (7.0, 3, 0.5, 4),
    ]
    RRT_PARAMS = {
        'max_iter': 10000,
        'step_len': 0.3,
        'goal_sample_rate': 0.20,
        'goal_dist_threshold': 0.5,
        'angular_weight': 0.5,
        'animation_update_rate': 100
    }
    START_POSE = (5.0, 1.5, np.radians(90))
    GOAL_POSE = (5.0, 5.0, np.radians(90))

    planner = RRTPlanner(
        start=START_POSE,
        goal=GOAL_POSE,
        bounds=MAP_BOUNDS,
        obstacles=OBSTACLES,
        robot_dims=ROBOT_DIMS,
        params=RRT_PARAMS
    )

    run_planning_animation(planner, scenario_name="Narrow Passage")

def run_scattered_scenario():
    """SCENARIO 2: Scattered obstacles."""
    print("Running Scenario 2: Scattered Obstacles")

    MAP_BOUNDS = (0, 10, 0, 10)
    ROBOT_DIMS = (1.0, 0.5)
    OBSTACLES = [
        (1, 1, 1, 1),
        (3, 2, 1.5, 1),
        (2, 4, 1, 2),
        (5, 5, 1, 1),
        (7, 3, 1, 3),
        (6, 7, 2, 1),
        (8, 8, 1, 1.5),
    ]
    RRT_PARAMS = {
        'max_iter': 5000,
        'step_len': 0.4,
        'goal_sample_rate': 0.10,
        'goal_dist_threshold': 0.5,
        'angular_weight': 0.5,
        'animation_update_rate': 100
    }
    START_POSE = (1.0, 9.0, np.radians(-90))
    GOAL_POSE = (9.0, 1.0, np.radians(-90))

    planner = RRTPlanner(
        start=START_POSE,
        goal=GOAL_POSE,
        bounds=MAP_BOUNDS,
        obstacles=OBSTACLES,
        robot_dims=ROBOT_DIMS,
        params=RRT_PARAMS
    )

    run_planning_animation(planner, scenario_name="Scattered")


if __name__ == '__main__':
    # Run scenario 1
    run_narrow_passage_scenario()

    # Run scenario 2 (Comment out the line above and uncomment the line below)
    # run_scattered_scenario()