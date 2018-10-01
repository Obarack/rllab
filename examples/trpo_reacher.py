from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.gym_env import GymEnv

from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

#env = normalize(SwimmerEnv())
env = normalize(GymEnv("Reacher-v1"))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(64, 64)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=2500,
    max_path_length=100,
    n_itr=100,
    discount=0.995,
    step_size=0.01,
)
algo.train_mf()





''' def train_mf(self):
        self.start_worker()
        self.init_opt()
        logz.configure_output_dir("/home/hendawy/Desktop/2DOF_Robotic_Arm_withSphereObstacle/Rr",1003)
        for itr in range(self.current_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                paths = self.sampler.obtain_samples(itr)
                samples_data,analysis_data = self.sampler.process_samples(itr, paths)
                self.log_diagnostics(paths)
                optimization_data=self.optimize_policy(itr, samples_data)
                logz.log_tabular('Iteration', analysis_data["Iteration"])
                # In terms of true environment reward of your rolled out trajectory using the MPC controller
                logz.log_tabular('AverageDiscountedReturn',analysis_data["AverageDiscountedReturn"])
                logz.log_tabular('AverageReturns', analysis_data["AverageReturn"])
                logz.log_tabular('violation_cost', np.mean(samples_data["violation_cost"]))
                logz.log_tabular('boundary_violation_cost', np.mean(samples_data["boundary_violation_cost"]))
                logz.log_tabular('success_rate', samples_data["success_rate"])
                logz.log_tabular('successful_AverageReturn', np.mean(samples_data["successful_AverageReturn"]))
                logz.log_tabular('ExplainedVariance', analysis_data["ExplainedVariance"])
                logz.log_tabular('NumTrajs', analysis_data["NumTrajs"])
                logz.log_tabular('Entropy', analysis_data["Entropy"])
                logz.log_tabular('Perplexity', analysis_data["Perplexity"])
                logz.log_tabular('StdReturn', analysis_data["StdReturn"])
                logz.log_tabular('MaxReturn', analysis_data["MaxReturn"])
                logz.log_tabular('MinReturn', analysis_data["MinReturn"])
                logz.log_tabular('LossBefore', optimization_data["LossBefore"])
                logz.log_tabular('LossAfter', optimization_data["LossAfter"])
                logz.log_tabular('MeanKLBefore', optimization_data["MeanKLBefore"])
                logz.log_tabular('MeanKL', optimization_data["MeanKL"])
                logz.log_tabular('dLoss', optimization_data["dLoss"])
                logz.dump_tabular()
                logger.log("saving snapshot...")
                params = self.get_itr_snapshot(itr, samples_data)
                self.current_itr = itr + 1
                params["algo"] = self
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot:
                    self.update_plot()
                    if self.pause_for_plot:
                        input("Plotting evaluation run: Press Enter to "
                                  "continue...")

        self.shutdown_worker()














def rollout(env, agent, max_path_length=np.inf, animated=True, speedup=1,
            always_return_paths=True):
    violation_cost=0
    boundary_violation_cost=0
    observations = []
    actions = []
    rewards = []
    succ_rewards=[]
    succ_rate=0
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        #print(next_o,next_o.shape)
        if not (constrain_fn_simple(next_o)):
            violation_cost+=1
            rewards[-1]=rewards[-1]-20
            print("*******************************Actual Violation******************************************************")
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        succ_rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return
    succ_rate=1

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        violation_cost=violation_cost,
        boundary_violation_cost=boundary_violation_cost,
        succ_return=succ_rewards,
        succ_rate=succ_rate,
    )
        '''


