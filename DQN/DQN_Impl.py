# Initialize replay memory D to capacity N 
# Initialize action-value funtion Q with random weights
def main():
	env = Maze()
	RL = DeepQNetwork(env.n_actions, env.n_features,
		learning_rate = 0.0.1,
		reward_decay = 0.9,
		e_greedy = 0.9,
		replace_target_iter = 200,
		memory_size = 2000)
	
	step = 0
	for episode in range(300):
		# initial observation // initialise sequence s.1 = {x.1} and preprocessed sequenced theta.1 = theta(s.1)
		observation = env.reset()
		# for t = 1, T do
		while True:

			env.render()
			#with probability E select a random action a.t # otherwise select a.t = max Q.start(theta(s.t),a; theta)
			action = RL.choose_action(observation)

			#execute action a.t in emulator and observe reward r.t and image x.t+1

			# store transition (theta.t, a.t, r.t, theta.t+1) in replay memory D 
			RL.store_transition(observation, action , reward, observation_)

			# sample random minibatch of transitions (theta.j, a.j , r.j, theta.j+1) from D
			if (step > 200)  and (step % 5 == 0):
				RL.learn()

			observation = observation_

			if done:
				break
			step += 1


def learn(self):
	#check to replace target parameters

	# set y.j = r.j  if theta.j+1 is terminal theta.j+1
	if self.learn_step_counter % self.replace_target_iter == 0:
		self.sess.run(self.replace_target_op)
		print('target parameters replaced')
	
	if self.memory_counter > self.memory_size:
		sample_index = np.random.choice(self.memory_size, size = self.batch.size)
	else:
		sample_index = np.random.choice(self.memory_counter, size = self.batch.size)
	batch_memory = self.memory[sample_index, :]

	q_next, q_eval = self.sess.run(
		[self.q_next, self.q_eval],
		feed_dict = {
			self.s_: batch_memory[:, -self.n_features:],  # fixed parameters
			self.s: batch_memory[:,:self.n_features],   # newest parameters
		})

	# change q_target = q_eval.copy()

	batch_index = np.arange(self.batch_size, dtype.int32)
	eval_act_index = batch_memory[:,self.n_features].astype(int)
	reward = batch_memory[:, self.n_features+1]

	# set y.j = r.j+r.max Q(theta.j+1, a.skim; theta) for non-terminals theta.j+1
	q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

