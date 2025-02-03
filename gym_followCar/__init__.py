from gymnasium.envs.registration import register

register(
  id="followCar-v0",
  entry_point="gym_followCar.followCarEnvironment:followCar"
)

