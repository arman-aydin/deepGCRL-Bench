"""Example evaluating goal-conditioned policy rollouts in OGBench environments."""

import ogbench


def evaluate_policy_tasks(env_name: str = 'humanoidmaze-large-navigate-v0', num_tasks: int = 5):
    """Evaluate an agent across standardized goal tasks."""
    env, _, _ = ogbench.make_env_and_datasets(env_name)

    for task_id in range(1, num_tasks + 1):
        obs, info = env.reset(options=dict(task_id=task_id, render_goal=False))
        goal = info['goal']
        done = False
        step = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

        success = info.get('success', False)
        print(f"Task ID {task_id}: {'Success' if success else 'Failed'} in {step} steps.")


if __name__ == '__main__':
    evaluate_policy_tasks()
