import gym
import torch.optim as optim
from utils import *
from models import *
import torch
from config import *


"""
Do not touch these files if not necessary!
Change most parameters in main.py or config.py
"""
def main(args=Args()):
    print('Starting dqn algorithm...')
    if args.q_arch == 'CNN':
        q_model = CNN(args=args).cuda()
    elif args.q_arch == 'FC':
        q_model = FC(args=args).cuda()
    else:
        print("ERROR: unknown architecture in policy.py | (%s)" % args.p_arch)
        return

    q_memory = DQNMemory(replay_capacity=args.replay_capacity)
    optimizer = optim.RMSprop(q_model.parameters(), lr=args.q_learning_rate)
    criterion = nn.SmoothL1Loss(size_average=True)
    state = State(args.state_length)

    # if args.start_episode:
    #     # TODO implement resume
    #     q_model.load_state_dict(load_checkpoint(args.saving_prefix, args.game, args.q_model_type, args.start_episode)['model_dict'])

    env = gym.make(args.game)
    observation = env.reset()
    tmr = startTimer()
    episode_number = args.start_episode
    while episode_number < args.max_episodes or not args.max_episodes:
        q_model.eval()
        s = state.get_state(observation)
        sampled_action = torch.LongTensor(np.random.randint(0, args.n_action, size=1)).cuda()\
            if np.random.uniform() < epsilon_decay(episode_number) \
            else torch.max(q_model.action_values(Variable(s, volatile=True)), dim=-1)[1].data
        observation, reward, done, info = env.step(args.action_dict[sampled_action[0]])  # Get Reward for action
        terminal = True if reward != 0 else False
        q_memory.remember(s, sampled_action, reward, terminal)  # append memory to replay

        if q_memory.memory_length == q_memory.replay_capacity:  # check if replay memory is filled up
            t_1, t_2, reward_batch, sampled_action_batch, is_terminal = q_memory.sample_mini_batch(args.mini_batch_size)
            next_state_val, _ = torch.max(q_model.action_values(t_2), dim=-1)
            action_val = torch.gather(q_model.action_values(t_1), dim=1, index=sampled_action_batch)
            est_action_val = reward_batch + args.gamma * (~ is_terminal).float() * next_state_val.data
            optimizer.zero_grad()
            loss = criterion(action_val, Variable(est_action_val).detach())
            loss.backward()
            optimizer.step()
        # time, tmr = resetTimer(tmr)

        if done:
            # if memory.memory_length == memory.replay_capacity:
            #     print('%f loss' % (loss.data[0]))
            observation = env.reset()
            state.clear()
            episode_number += 1
            print("Episode %d" % episode_number)

            if episode_number % args.save_interval == 0:
                time, tmr = resetTimer(tmr)
                save_checkpoint(args.saving_prefix, args.game, 'dqn', q_model, args.q_learning_rate, episode_number, time, args)

        if reward > 0:  # Pong has either +1 or -1 reward exactly when game ends.
            print('Won!')


if __name__ == '__main__':
    main()

