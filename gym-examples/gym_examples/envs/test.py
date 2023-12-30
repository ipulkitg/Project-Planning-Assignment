import gymnasium as gym
from gymnasium.spaces.utils import flatten_space
from gymnasium import spaces
from gymnasium.spaces import Dict,Discrete,Box


# space = {
#     "position": spaces.Discrete(2),
#     "velocity": spaces.Box(0, 1, shape=(2, 2))
#          }

space = Dict({"position": Discrete(2), "velocity": Box(0, 1, shape=(2, 2))})

print(flatten_space(space))

space = Dict(
            {
                # "agent": spaces.Dict(agentDict),
                # "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "factory": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "recharge": spaces.Box(0, size - 1, shape=(2,), dtype=int)
                "bot_1_distances":  Dict(
                    {
                        "factory_raw":  Discrete(9),
                        "factory_inter":  Discrete(9),
                        "factory_final":  Discrete(9),
                        "factory_delivery":  Discrete(9),
                        "recharge":  Discrete(9)
                    }
                ),
                "bot_2_distances":  Dict({
                        "factory_raw":  Discrete(9),
                        "factory_inter":  Discrete(9),
                        "factory_final":  Discrete(9),
                        "factory_delivery":  Discrete(9),
                        "recharge":  Discrete(9)
                }),
                "bot_3_distances":  Dict({
                        "factory_raw":  Discrete(9),
                        "factory_inter":  Discrete(9),
                        "factory_final":  Discrete(9),
                        "factory_delivery":  Discrete(9),
                        "recharge":  Discrete(9)
                }),
                "bot_charges":  Dict({
                        "bot_1":  Discrete(101),
                        "bot_2":  Discrete(101),
                        "bot_3":  Discrete(101)
                }),
                "factory_raw":  Dict({
                        "factory_raw":  Discrete(1),
                        "factory_inter":  Discrete(1),
                        "factory_final":  Discrete(1),
                        "factory_delivery":  Discrete(1)
                }),
                "factory_proc_buff":  Dict({
                        "factory_raw":  Discrete(101),
                        "factory_inter":  Discrete(101),
                        "factory_final":  Discrete(101)
                })
            }
        )


print(flatten_space(space))
