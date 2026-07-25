import ogbench

# Make an environment and datasets (they will be automatically downloaded)
dataset_name = 'humanoidmaze-large-navigate-v0'
env, train_dataset, val_dataset = ogbench.make_env_and_datasets(dataset_name)

# Train your offline goal-conditioned RL agent on the dataset
# ...

# Evaluate the agent
for task_id in [1, 2, 3, 4, 5]:
    # Reset  the environment and set the evaluation task
    ob, info = env.reset(
        options=dict(
            task_id=task_id, # Set the evaluation task. Each environment provides five
                             # evaluation goals, and `task_id` must be in [1, 5]
            render_goal=True, # Set to `True` to get a rendered goal image (optional)
        )
    )

    goal = info['goal'] # Get the goal observation to pass to the agent
    goal_rendered = info['goal_rendered']   # Get the rendered goal image (optional)

    done = False
    while not done:
        action = env.action_space.sample()  # Replace this with your agent's action
        ob, reward, terminated, truncated, info = env.step(action)  # Gymnasium-style step
        # If the agent reaches the goal, `terminated` will be `True`. If the episode length
        # exceeds the maximum length without reaching the goal, `truncated` will be `True`.
        # `reward` is 1 if the agent reaches the goal and 0 otherwise.
        done = terminated or truncated
        frame = env.render()    # Render the current frame (optional)

    success = info['success']   # Whether the agent reached the goal (0 or 1)
                                # `terminated` also indicates this