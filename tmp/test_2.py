import gym


class InfiniteLake:
    def __init__(self):
        self.__env_cnt = 1
        self.__list_env = [gym.make('FrozenLake-v0')]
        self.__state_dim = self.get_state_dim().n
        self.__action_dim = self.get_action_dim().n

    def step(self, a):
        ns, r, d, info = self.__list_env[self.__env_cnt - 1].step(a)
        if self.__s == ns:
            self.__list_env.append(gym.make('FrozenLake-v0'))
            ns = self.__list_env[self.__env_cnt].reset()
            self.__env_cnt += 1
        ns += (self.__env_cnt - 1) * self.__state_dim

        d = False
        return ns, r, d, info

    def reset(self):
        self.__s = self.__list_env[0].reset()
        return self.__s

    def get_state_dim(self):
        return self.__list_env[0].observation_space

    def get_action_dim(self):
        return self.__list_env[0].action_space

    def render(self):
        self.__list_env[self.__env_cnt - 1].render()

    def __str__(self):
        return "this is a discrete env, use action_dim.n to get the size, same with state_dim"


if __name__ == '__main__':
    env = InfiniteLake()
    s = env.reset()
    for _ in range(100000):
        # env.render()

        a = env.get_action_dim().sample()
        ns, r, d, info = env.step(a)
        # print(s, a, r, ns, d)
        s = ns
        if d:
            env.render()
            break
    print(s, a, r, ns, d)
