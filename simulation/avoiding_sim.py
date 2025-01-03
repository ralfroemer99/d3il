import logging
import multiprocessing as mp
import os
import random
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

import numpy as np
import torch
import wandb
import matplotlib
import matplotlib.pyplot as plt

from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Avoiding_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_trajectories: int = 30,
            action_space: str = 'vel',
            obstacles: list = None,
            with_constraints: bool = False
    ):
        super().__init__(seed, device, render, n_cores)

        self.n_trajectories = n_trajectories

        # -------- Make constraint list: Obstacles, halfspace constraints, dynamic constraints
        self.action_space = action_space
        self.obstacles = obstacles
        self.with_constraints = with_constraints
        self.constraints = []
        if self.obstacles is not None:      # Obstacle constraints
            for obs in self.obstacles:
                x_center, y_center, radius = obs
                if self.action_space == 'pos':
                    self.constraints.append(['sphere_outside', [0, 1], [x_center, y_center], radius])
                elif self.action_space == 'pos_vel':
                    self.constraints.append(['sphere_outside', [2, 3], [x_center, y_center], radius])
        if self.action_space == 'pos_vel':
            self.constraints.extend([['deriv', [2, 0]], ['deriv', [3, 1]]])
        # --------

    def eval_agent(self, agent, n_trajectories, mode_encoding, successes, robot_c_pos, pid, cpu_set, ax=None):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = ObstacleAvoidanceEnv(render=self.render, obstacles=self.obstacles)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for i in range(n_trajectories):

            agent.reset()

            print(f'core {pid}, Rollout {i}')

            obs = env.reset()

            pred_action = env.robot_state()
            fixed_z = pred_action[2:]
            done = False

            c_pos = [env.robot.current_c_pos]

            while not done:

                obs = np.concatenate((pred_action[:2], obs))

                # -------- Arguments for the projector --------
                extra_args = {}
                if self.action_space == 'pos':
                    extra_args['plot_info'] = {'ax': ax, 'dims': [0, 1]}
                elif self.action_space == 'pos_vel':
                    extra_args['plot_info'] = {'ax': ax, 'dims': [2, 3]}
                elif self.action_space == 'vel':
                    extra_args['plot_info'] = {'ax': ax, 'integrate_vel': True}

                if self.with_constraints: extra_args['constraint_info'] = {'constraints': self.constraints, 'dt': 1, 'skip_initial': True}
                # extra_args = {'constraints': self.constraints, 'dt': 1, 'skip_initial': True, 'plot_info': plot_info} if self.with_constraints else {}
                agent_output = agent.predict(obs, extra_args=extra_args)
                if self.action_space == 'vel':
                    pred_vel = agent_output
                elif self.action_space == 'pos':
                    pred_vel = agent_output - obs[:2]
                elif self.action_space == 'pos_vel':
                    pred_vel = agent_output[:, :2]
                else:
                    raise ValueError('Invalid action space')
                # --------------------------------------------

                pred_action = pred_vel[0] + obs[:2]

                pred_action = np.concatenate((pred_action, fixed_z, [0, 1, 0, 0]), axis=0)

                obs, reward, done, info = env.step(pred_action)

                c_pos.append(env.robot.current_c_pos)

            c_pos = torch.tensor(np.array(c_pos))[:, :2]
            robot_c_pos[pid * n_trajectories + i, :c_pos.shape[0], :] = c_pos

            mode_encoding[pid * n_trajectories + i, :] = torch.tensor(info[0])
            successes[pid * n_trajectories + i] = torch.tensor(info[1])

    ################################
    # we use multi-process for the simulation
    # n_trajectories: rollout policy for n times
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        robot_c_pos = torch.zeros([self.n_trajectories, 251, 2]).share_memory_()

        mode_encoding = torch.zeros([self.n_trajectories, 9]).share_memory_()
        successes = torch.zeros(self.n_trajectories).share_memory_()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))

        # start = self.seed * 20
        # end = start + 20
        #
        # cpu_set = cpu_set[start:end]
        print("there are cpus: ", num_cpu)

        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent": agent,
                        "n_trajectories": self.n_trajectories // self.n_cores,
                        "mode_encoding": mode_encoding,
                        "successes": successes,
                        "robot_c_pos": robot_c_pos,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i + 1])
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, self.n_trajectories, mode_encoding, successes, robot_c_pos, 0, set([0]), ax)
            
        np.save(f"{self.working_dir}/robot_c_pos.npy", robot_c_pos.numpy())
        success_rate = torch.mean(successes).item()

        # calculate entropy
        data = mode_encoding[successes == 1].numpy()
        data_decimal = data.dot(1 << np.arange(data.shape[-1]))
        _, counts = np.unique(data_decimal, return_counts=True)
        mode_dist = counts / np.sum(counts)
        entropy = - np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))

        wandb.log({'score': (success_rate * 0.8 + entropy * 0.2)})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy': entropy})

        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')

        # Plot trajectories
        for i in range(self.n_trajectories):
            traj_length = torch.sum(robot_c_pos[i, :, 0] != 0)
            ax.plot(robot_c_pos[i, :traj_length, 0], robot_c_pos[i, :traj_length, 1])
        ax.set_xlim([0.2, 0.8])
        ax.set_ylim([-0.3, 0.4])
        centers = [[0.5, -0.1], [0.425, 0.08], [0.575, 0.08], [0.35, 0.26], [0.5, 0.26], [0.65, 0.26]]
        radii = [0.03, 0.025, 0.025, 0.025, 0.025, 0.025]
        for i, center in enumerate(centers):
            ax.add_patch(matplotlib.patches.Circle(center, radii[i], color='r'))
        ax.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)
        for obs in self.obstacles:
            x_center, y_center, radius = obs
            ax.add_patch(matplotlib.patches.Circle([x_center, y_center], radius, color='b'))
        plt.savefig(f"{self.working_dir}/trajectories.png")

        return successes, entropy