import matplotlib.pyplot as plt
from env import BattlecodeConfig, BattlecodeEnv

for _ in range(100):
    env = BattlecodeEnv(BattlecodeConfig())
    env.reset()
