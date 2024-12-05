import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
from student_submissions.s2210xxx.policy2210xxx import Policy2210xxx
from student_submissions.s2210xxx.policy2350030 import Policy2350030
import time

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 100

if __name__ == "__main__":
    # # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test GreedyPolicy
    # gd_policy = GreedyPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = gd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1
    # print("GreedyPolicy done")
    # Reset the environment
    # observation, info = env.reset(seed=42)

    # # Test RandomPolicy
    # rd_policy = RandomPolicy()
    # ep = 0
    # while ep < NUM_EPISODES:
    #     action = rd_policy.get_action(observation, info)
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     if terminated or truncated:
    #         observation, info = env.reset(seed=ep)
    #         print(info)
    #         ep += 1
    # print("RandomPolicy done")
    # Uncomment the following code to test your policy
    # Reset the environment
    observation, info = env.reset(seed=42)
    print(info)
    print(len(observation["stocks"]))
    print(
        "================================================================================"
    )
    print(len(observation["products"]))
    # print(observation["stocks"])
    while len(observation["products"]) <= 10:
        observation, info = env.reset(seed=42)

    policy2210xxx = Policy2350030()
    for _ in range(NUM_EPISODES):
        print(
            "================================================================================"
        )
        action = policy2210xxx.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(reward)
        print(terminated)
        print(truncated)
        print(info)

        if terminated or truncated:
            observation, info = env.reset()
time.sleep(3)
env.close()
