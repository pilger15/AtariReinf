import gym
import torch.optim as optim
from torch.distributions import Categorical
from utils import *
from models import *
from config import *



def main(args=Args()):
    print('Starting actor critic algorithm...')
    if args.p_arch == 'CNN':
        p_model = CNN(args=args).cuda()
    elif args.p_arch == 'FC':
        p_model = FC(args=args).cuda()
    else:
        print("ERROR: unknown architecture in policy.py | (%s)" % args.p_arch)
        return

    if args.q_arch == 'CNN':
        q_model = CNN(args=args).cuda()
    elif args.q_arch == 'FC':
        q_model = FC(args=args).cuda()
    else:
        print("ERROR: unknown architecture in policy.py | (%s)" % args.p_arch)
        return

    q_memory = DQNMemory(replay_capacity=args.replay_capacity)
    p_memory = PolicyMemory(gamma=args.gamma, save_q_baseline=True, normalize_reward=False)

    q_optim = optim.RMSprop(q_model.parameters(), lr=args.q_learning_rate)
    p_optim = optim.RMSprop(p_model.parameters(), lr=args.p_learning_rate)
    state = State(args.state_length)
    criterion = nn.SmoothL1Loss(size_average=True)

    # if args.start_episode:
    #     # TODO implement resume
    #     q_model.load_state_dict(load_checkpoint(args.saving_prefix, args.game, args.q_arch, args.start_episode)['model_dict'])
    #     p_model.load_state_dict(load_checkpoint(args.saving_prefix, args.game, args.p_arch, args.start_episode)['model_dict'])

    episode_number = args.start_episode
    env = gym.make(args.game)
    observation = env.reset()
    tmr = startTimer()
    while episode_number < args.max_episodes or not args.max_episodes:
        s = state.get_state(observation)
        probs = p_model.policy(Variable(s, volatile=True))
        cat_dist = Categorical(probs)
        sampled_action = cat_dist.sample().data
        est_state_val = torch.max(q_model.action_values(Variable(s, volatile=True)), dim=-1)[0].data

        observation, reward, done, info = env.step(args.action_dict[sampled_action[0]])  # Get Reward for action
        terminal = True if reward != 0 else False
        p_memory.remember(s, sampled_action, reward, terminal, est_state_val)
        q_memory.remember(s, sampled_action, reward, terminal)

        if q_memory.memory_length == q_memory.replay_capacity:
            q_model.train()
            t_1, t_2, reward_batch, sampled_action_batch, is_terminal = q_memory.sample_mini_batch(args.mini_batch_size)
            next_state_val, _ = torch.max(q_model.action_values(t_2), dim=-1)
            action_val = torch.gather(q_model.action_values(t_1), dim=1, index=sampled_action_batch)
            est_action_val = reward_batch + args.gamma * (~ is_terminal).float() * next_state_val.data
            q_optim.zero_grad()
            q_loss = criterion(action_val, Variable(est_action_val).detach())
            q_loss.backward()
            q_optim.step()

        if terminal and p_memory.n_rewards % args.batch_size == 0:
            p_model.train()
            observation_batch, action_batch, reward_batch, est_state_val_batch = p_memory.sample_batch()
            action_prob = p_model.policy(observation_batch)
            action_prob_taken = torch.gather(action_prob, dim=1, index=action_batch)
            log_prob_action_taken = torch.log(action_prob_taken)
            p_loss = - torch.sum(log_prob_action_taken * (reward_batch - est_state_val_batch))
            p_optim.zero_grad()
            p_loss.backward()
            p_optim.step()

        if done:
            # if memory.memory_length == memory.replay_capacity:
            #     print('%f p_loss' %f ' q_loss  (p_loss.data[0], q_loss.data[0]))
            observation = env.reset()
            state.clear()
            episode_number += 1
            print("Episode %d" % episode_number)

            if episode_number % args.save_interval == 0:
                time, tmr = resetTimer(tmr)
                save_checkpoint(args.saving_prefix, args.game, 'policy', p_model, args.p_learning_rate, episode_number, time, args)         # TODO: save args seperatly loading
                save_checkpoint(args.saving_prefix, args.game, 'dqn', q_model, args.q_learning_rate, episode_number, time, args)

        if int(reward) == 1:  # Pong has either +1 or -1 reward exactly when game ends.
            print('Won!')

if __name__ == '__main__':
    main()


