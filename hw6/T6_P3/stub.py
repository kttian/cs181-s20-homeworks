# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import matplotlib.pyplot as plt

from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.iter = 0

		#hyperparameters
		self.alpha = 0.1 # learning rate
		self.gamma = 0.5 # discount rate
		self.epsilon = 0.001

		# We initialize our Q-value grid that has an entry for each action and state.
		# (action, rel_x, rel_y)
		self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.iter = 0

	def hyper(self, alpha, gamma, epsilon):
		self.alpha = alpha
		self.gamma = gamma
		self.epsilon = epsilon

	def print_hyper(self):
		print("alpha: ", self.alpha, 
			", gamma: ", self.gamma, 
			", epsilon: ", self.epsilon)



	def discretize_state(self, state):
		'''
		Discretize the position space to produce binned features.
		rel_x = the binned relative horizontal distance between the monkey and the tree
		rel_y = the binned relative vertical distance between the monkey and the tree        
		'''
		rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
		rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
		return (rel_x, rel_y) 

	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''

		# TODO (currently monkey just jumps around randomly)
		# 1. Discretize 'state' to get your transformed 'current state' features.
		# 2. Perform the Q-Learning update using 'current state' and the 'last state'.
		# 3. Choose the next action using an epsilon-greedy policy.
		self.iter += 1
		if(self.last_state == None or state == None):
			self.last_action = 0
			self.last_state = state
			return self.last_action

		# discretize state?
		r = self.last_reward #state["score"]
		a = int(self.last_action)
		s = self.last_state
		sp = state
		current_state = self.discretize_state(s)
		(x,y) = self.discretize_state(s)
		(xp,yp) = self.discretize_state(state)


		# perform q-learning update step

		#for ap in range(2):
		#print(self.Q)
		#print("Qp: ", r, Qp)
		#print("a, x, y: ", a,x,y)
		#print("current q: ", self.Q[a][x][y])
		#print(self.Q[a][current_state] + alpha * (r + gamma * Qp - self.Q[a][current_state]))
		alpha = self.alpha
		gamma = self.gamma
		decay = 0.95
		epsilon = self.epsilon * (decay ** self.iter)

		Qp = max(self.Q[0][xp][yp],self.Q[1][xp][yp])
		self.Q[a][x][y] = self.Q[a][x][y] + alpha * (r + gamma * Qp - self.Q[a][x][y])
		
		# action step
		rng = npr.rand()
		if rng < epsilon:
			new_action = int(npr.rand() < 0.5) # what threshold to use for random action
		else:
			new_action = int(self.Q[1][xp][yp] > self.Q[0][xp][yp])
			#new_action = np.argmax(self.Q[:][x][y])

		new_state = state
		self.last_action = new_action
		self.last_state  = new_state
		#print("action: ", new_action)
		#print("state: ", new_state)
		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''

		self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			pass
		
		# Save score history.
		hist.append(swing.score)

		# Reset the state of the learner.
		learner.reset()
	pg.quit()
	return


alphas = [0.5, 0.1, 0.01, 0.001]
gammas = [0.9, 0.8, 0.7, 0.6, 0.5]
epsilons = [0.1, 0.01, 0.001]

na = len(alphas)
ng = len(gammas)
ne = len(epsilons)

histories = []
for i in range(na):
	histories.append([])
	for j in range(ng):
		histories[i].append([])
		for k in range(ne):
			histories[i][j].append([])


def tune_hyper():
	#alphas = [0.001]
	#gammas = [0.8]
	#epsilons = [0.001]
	
	results = np.zeros(shape=(na,ng,ne))
	agent = Learner()

	
	nr = 1

	for i in range(na):
		for j in range(ng):
			for k in range(ne):
				for l in range(nr):
					a = alphas[i]
					g = gammas[j]
					e = epsilons[k]

					agent.hyper(a,g,e)
					#agent.print_hyper()
					hist = []
					run_games(agent, hist, 100, 100)
					#print("history: ", hist)
					results[i][j][k] += max(hist)
					print(a,g,e,max(hist))
					#print(hist)
					histories[i][j][k] = hist
					#plt.plot(hist)
					#t = "a=" + str(a) + "g=" + str(g) + "e=" + str(e)
					#plt.title(t)
					#plt.savefig(t + ".png")
				results[i][j][k] = int(results[i][j][k]/nr)
	print(results)
	print(np.unravel_index(np.argmax(results),results.shape))

def plot_hist():
	# make plot for best gamma
	
	for i in range(na):
		for k in range(ne):
			for j in range(ng):
				hist = histories[i][j][k]
				plt.plot(hist)
			a = alphas[i]
			g = gammas[j]
			e = epsilons[k]
			t = "a=" + str(a) + "e=" + str(e)
			plt.title(t )
			plt.savefig(t + ".png")
			plt.show()

def tune_decay():
	alphas = [0.001, 0.01, 0.1]
	gammas = [0.8]
	epsilons = [0.5, 0.1, 0.01, 0.001]
	#epsilons = [0.1]
	for i in range(len(alphas)):
		for j in range(len(gammas)):
			for k in range(len(epsilons)):
				hist = []
				agent = Learner()
				agent.hyper(alpha=alphas[i],gamma=gammas[j],epsilon=epsilons[k])
				run_games(agent, hist, 100, 100)
				print("history: ", max(hist))
				print(hist)
				plt.plot(hist)
				t = "epsilon_decay"+str(epsilons[k])
				plt.title(t)
				plt.savefig(t + ".png")
				plt.show()

if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. You can update t_len to be smaller to run it faster.
	agent.hyper(alpha=0.001,gamma=0.8,epsilon=0.1)
	run_games(agent, hist, 100, 100)
	agent.print_hyper()
	print("history: ", hist)

	tune_decay()

	# Save history. 
	np.save('hist',np.array(hist))

	#tune_hyper()
	#plot_hist()


