# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey


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

		# We initialize our Q-value grid that has an entry for each action and state.
		# (action, rel_x, rel_y)
		self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None

	def discretize_state(self, state):
		'''
		Discretize the position space to produce binned features.
		rel_x = the binned relative horizontal distance between the monkey and the tree
		rel_y = the binned relative vertical distance between the monkey and the tree        
		'''

		rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
		rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
		print("discretize ", rel_x, rel_y)
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
		alpha = 0.01 # learning rate
		gamma = 0.90 # discount rate
		epsilon = 0.1

		#for ap in range(2):
		print("q size: ", self.Q.shape)
		print(self.Q)
		Qp = max(self.Q[0][xp][yp],self.Q[1][xp][yp])
		print("Qp: ", r, Qp)
		print("a, x, y: ", a,x,y)
		print("current state: ", current_state)
		print("current q: ", self.Q[a][x][y])
		print(self.Q[a][current_state] + alpha * (r + gamma * Qp - self.Q[a][current_state]))
		self.Q[a][current_state] = self.Q[a][current_state] + alpha * (r + gamma * Qp - self.Q[a][current_state])
		print("q size after: ", self.Q.shape)
		print(self.Q)
		
		# action step
		rng = npr.rand()
		if rng < epsilon:
			new_action = int(npr.rand() < 0.5) # what threshold to use for random action
		else:
			new_action = self.Q[1][current_state] > self.Q[0][current_state]
			#new_action = np.argmax(self.Q[:][x][y])

		new_state = state
		self.last_action = new_action
		self.last_state  = new_state
		print("action: ", new_action)
		print("state: ", new_state)
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


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. You can update t_len to be smaller to run it faster.
	run_games(agent, hist, 100, 100)
	print("history: ", hist)

	# Save history. 
	np.save('hist',np.array(hist))


