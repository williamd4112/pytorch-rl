import time
import threading

from tensorpack.utils.fs import mkdir_p
from tensorpack.utils.stats import StatCounter
from tensorpack.RL.envbase import RLEnvironment, DiscreteActionSpace

from unity3d_env import Unity3DEnvironment

__all__ = ['GymEnv']
_ENV_LOCK = threading.Lock()

class Unity3DPlayer(RLEnvironment):
    ACTION_TABLE = [(2.0, 0.0), # Forward
                    (-2.0, 0.0), # Backward
                    (1.0, 0.5), # Forward-Right
                    (-1.0, 0.5), # Backward-Right
                    (1.0, -0.5), # Forward-Left
                    (1.0, -0.5) ] # Backward-Left 

    def __init__(self, connection, dumpdir=None, viz=False, auto_restart=True):
        if connection != None:
            with _ENV_LOCK:
                self.gymenv = Unity3DEnvironment(server_address=connection)
            self.use_dir = dumpdir

            self.reset_stat()
            self.rwd_counter = StatCounter()
            self.restart_episode()
            self.auto_restart = auto_restart
            self.viz = viz

    def restart_episode(self):
        self.rwd_counter.reset()
        self._ob = self.gymenv.reset()

    def finish_episode(self):
        self.stats['score'].append(self.rwd_counter.sum)

    def current_state(self):
        if self.viz:
            self.gymenv.render()
            time.sleep(self.viz)
        return self._ob

    def action(self, act):
        env_act = self.ACTION_TABLE[act]
        self._ob, r, isOver, info = self.gymenv.step(env_act)
        self.rwd_counter.feed(r)
        if isOver:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return r, isOver

    def get_action_space(self):
        return DiscreteActionSpace(len(self.ACTION_TABLE))

    def close(self):
        self.gymenv.close()

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1])
    p = Unity3DPlayer(connection=('127.0.0.1', port))
    p.restart_episode()
    for i in range(100):
        r, done = p.action(0)
        print(r)
    p.close()

