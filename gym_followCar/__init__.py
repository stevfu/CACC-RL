from gymnasium.envs.registration import register

register(
  id="followCar-v1",
  entry_point="gym_followCar.followCarEnvironment-v1:followCar_v1"
)

register(
  id="followCar-v0",
  entry_point="gym_followCar.followCarEnvironment:followCar"
)


