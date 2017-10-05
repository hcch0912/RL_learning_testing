
#Asynchronous advantage actor critic , for each actor leaner thread

# global counter T thread counter t

def work
# assume global shared parameter vector theta and theta v and counter T = 0
	global GLOBAL_RUNNING_R, GLOBAL_EP
	#initialize thread step counter t <- 1
	total_step = 1
	buffer_s, buffer_a, buffer_r =[],[],[]
	# run unitl counter is larger than the max counter value
	while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP
		# reset gradients: d.theta <- 0 and d.theta v <- 0
		# synchronize thread specific parameters theta skim = theta and theta v skim = theta v 
		# t.start = t
		#get state s.t
		s.self.env.reset()

		# repeat until terminal s.t or t - t.start == t.max
		while True:
			if self.name == 'W_0':
				self.env.render()
			# perform a.t according to policy pi(a.t| s.t; theta skim)
			a = self.AC.choose_actions(s)
			# receive reware r.t 
			s_, r, done, info = self.env.step(a)
			if done: r = -5
			ep_r += r
			buffer_s.append(s)
			buffer_a.append(a)
			buffer_r.append(r)

			# new state s.t+1
			s = s_
			# t <- t+1
			total_step += 1
			# T <- T+1
			# GLOBAL_EP += 1  ???

			# if s.t is terminal R = 0
			if done:
				if len(GLOBAL_RUNNING_R) == 0:
					GLOBAL_RUNNING_R.append(ep_r)

			# else for all non-terminal s.t,  R = V(s.t, theta v skim)
				else:
					GLOBAL_RUNNING_R.append(0.99*GLOBAL_RUNNING_R[-1] + 0.01*ep_r)
				GLOBAL_EP +=1 ###??
				breaks