import tensorflow
import numpy as numpy
import gym


for i in range(MAX_EPISODES):

	s= env.reset()   # initialize a random process N for action exploration
	ep_reward = 0  # receive initial observation state s.1

	for j in range(MAX_EP_STEPS):

		a = actor.choose_action(s) # select an action according to the current policy and exploration noise
		a = np.clip(np.random.normal(a,var),-2,2) # add randomness to action selection for exploration
		s_, r, done = env.step(a) # execute action a.t and observe reward r.t and overse new state s.t+1
		R.store_trasition(s,a,r,s_)  # store transition (s.tm a.t, r.t, s.t+1) in R

		# sample a random minibatch of N transitions(s.i, a.i, r.i, s.i+1) from R
		if M.pointer > MEMORY_CAPACITY:
			var = max([var*.9999, VAR_MIN])  # decay the action randomness
			this_R = R.sample(BATCH_SIZE)
			this_s = this_R[:, "STATE_DIM"]
			this_a = this_R[:,STATE_DIM: STATE_DIM + ACTION_DIM]
			this_r = this_R[:,-STATE_DIM-1: -STATE_DIM]
			this_s_ = this_R[:, -STATE_DIM:]

			# update critic by minimizing the loss
			critic.learn(this_s,this_a,this_r,this_s_)
			# update the actor policy using the samples policy gradient
			actor.learn(this.s)

		# update the target networks
		s=s_
		ep_reward += r

