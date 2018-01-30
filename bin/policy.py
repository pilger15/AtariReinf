import gym
import torch.optim as optim
from torch.distributions import Categorical
from utils import *
from models import *
import config

"""
Do not touch these files if not necessary!
Change most parameters in main.py or config.py
"""
def main(args=Args()):
    if args.p_arch == 'CNN':
        p_model = CNN(args=args).cuda()
    elif args.p_arch == 'FC':
        p_model = FC(args=args).cuda()
    else:
        print("ERROR: unknown architecture in policy.py | (%s)" % args.p_arch)
        return

    p_memory = PolicyMemory(args.gamma, normalize_reward=False)
    state = State(args.state_length)
    optimizer = optim.Adam(p_model.parameters(), lr=args.p_learning_rate, weight_decay=1e-5)
    baseline = Baseline(100//args.batch_size)  # we compute the baseline for 100 rounds

    #resume
    if args.start_episode:
        # TODO implement resume
        p_model.load_state_dict(load_checkpoint(args.saving_prefix, args.game, args.p_arch, args.start_episode)['model_dict'])

    episode_number = args.start_episode
    env = gym.make(args.game)
    observation = env.reset()
    tmr = startTimer()
    while episode_number < args.max_episodes or not args.max_episodes:
        s = state.get_state(observation)
        probs = p_model.policy(Variable(s, volatile=True))
        cat_dist = Categorical(probs)
        sampled_action = cat_dist.sample().data

        observation, reward, done, info = env.step(args.action_dict[sampled_action[0]])
        terminal = True if reward != 0 else False
        p_memory.remember(s, sampled_action, reward, terminal, 0)                                                                           #save the zero to make it consistent with actor critic

        if terminal and p_memory.n_rewards == args.batch_size:
            observation_batch, action_batch, reward_batch, _ = p_memory.sample_batch()
            baseline.append(reward_batch.data)
            action_prob = p_model.policy(observation_batch)
            action_prob_taken = torch.gather(action_prob, dim=1, index=action_batch)
            log_prob_action_taken = torch.log(action_prob_taken)
            loss = - torch.sum(log_prob_action_taken * (reward_batch - baseline.get()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            observation = env.reset()
            state.clear()
            episode_number += 1
            print("Episode %d" % episode_number)

            if episode_number % args.save_interval == 0:
                time, tmr = resetTimer(tmr)
                save_checkpoint(args.saving_prefix, args.game, 'policy', p_model, args.p_learning_rate, episode_number, time, args)

        if reward > 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print(reward)


if __name__ == '__main__':
    main()

