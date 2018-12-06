# Reinforcement Learning
by Harish Kumar and Michael Bass

Power Point Presentation - https://github.com/CourseReps/ECEN689-Fall2018/blob/master/Students/SuperMB/ReinforcementLearning.Finished.pptx








## ECEN 689 - Applied Information Science - Tutorial

### Harish Kumar, Michael Bass

Abstract
We summarize and describe the material discussed in our tutorial on Reinforcement Learning. We start with the distinction
between reinforcement learning and other topics in reinforcement learning, describe the reinforcement learning problem, and then
discuss how it can be solved.

### I. REINFORCEMENTLEARNING

In a reinforcement learning problem, we train an agent to take optimal decisions in an environment to maximize a cumulative
reward. The agent is oftentimes a software program that has control over resources, hardware or even other agents.
Of the familiar sub-divisions in machine learning, supervised learning involves learning to optimally classify or regress when
given the ground truth, and unsupervised learning involves identifying patterns, groups and clusters within data.
Reinforcement learning is a significantly different paradigm where the agent is not explicitly told what the right behavior
is. Instead, the agent observes the consequences of its behavior in the form of a reward that is often delayed with regards to
the original action.

### II. PROBLEMFORMULATION

In a reinforcement learning problem, an Actor or an Agent takes different decisions in an environment, with the goal of
maximizing the total of the rewards it obtains from the environment in response to its decisions.
Reinforcement Learning problems are formulated mathematically as Markov Decision Processes. We use the example of
training a computer program to play Mario to illustrate a model Markov Decision Process.

- A Statesis a particular ”position” or ”status” of the environment as well as the agent. In the case of the Mario example,
    a state would be a particular set of positions for Mario, the various terrain obstacles, enemies and pits. Equivalently, we
    can propose that the whole image that we see on the screen is a particular state of the environment.
- An actionais a decision that the agent takes to move to a new state. In our example, choices to press different buttons
    in the game are all different actions that are available to us. The set of available actions may depend on the state.
    Actions taken by the agent may modify the environment and these are described by state transitions and their associate
    probabilities.
- The rewardrdescribes our objective during the whole process. The agent is expected to maximize the total cumulative
    reward over a long time horizon. The reward is a random variable that depends on the originating state, the action taken,
    and the terminal state.
We must note here that some actions may have small immediate reward, but may take us to a state from which much higher
rewards are possible in the future.
Thus, a Markov Decision Process is described by
- The set of statesS.
- The set of actionsA.
- The state transition probabilitiesP(st,at,st+1).
- The immediate reward distribution,R(st,at,st+1)

### III. POLICY

Given an MDP, it can be shown that there exists a methodology to observe the current statestselect actions such that the
expected total cumulative reward is maximized. Such a methodology of choosing actions based on the present state is called
a policy.
Thus, the goal of reinforcement learning is to learn a probability density functionπ(st,at)that describes which actions to
take in what states so that we can maximize the expected total cumulative reward over a long time horizon. It can also be
shown that if we have an infinite time-horizon, this policy is time-independent, i.e. the optimal action depends only on the
state and not at what time-step we reached that state.

### IV. Q-LEARNING

There are many different algorithms to learn an optimal policy on a Markov Decision Process, we describe Q-Learning as
an example.
Here, we calculate a set of values called the Q-Values,Q(st,at)that describe the long-term utility of taking an actionat
from the statest. The policy we should use after learning the Q-Values is to choose the action with the highest Q-Value at
each state. We also use a parameterγto discount future rewards and express the cumulative reward as

<img src="https://github.com/CourseReps/ECEN689-Fall2018/raw/master/Students/SuperMB/Images/Equation1.png/">
 
We use an iterative process and an update equation to arrive at the final Q-Values.

<img src="https://github.com/CourseReps/ECEN689-Fall2018/raw/master/Students/SuperMB/Images/Equation2.png/">

1) Start with an initial states 0 and a set of initial Q-Values. The choice of the initial Q-Values will decide whether our
model is exploratory or greedy during the learning process.
2) Observe all the Q-Values in the present statestand take the actionatwith the highest Q-Value.
3) Use the obtained rewardrtto update the Q-Values forQ(st,at).
4) Repeat 2 and 3 until the Q-Values converge, or for a fixed number of iterations.

### V. FUNCTIONAPPROXIMATION

We note here that for complex reinforcement learning problems, the state-action space is too large to enumerate all possible
elements of the setS×A. In fact, it is very probable that we only encounter a tiny fraction of all possible state-action
combinations during the learning process. In order to extend the knowledge that our agent has gained over the training phase
to unseen states encountered in the decision phase, we use a function approximator to learn the correspondence between
state-action pairs and their Q-Values.
An example of a function approximator that can learn the relation between states, actions and Q-Values is a neural network.
For instance, a neural network can take a statestas input and predict the Q-Value for every actionat. Such a network would
often be able to transfer what it learned over already visited states to states that have not been visited.
Neural networks can also learn to take the right action in a state, and this approach is knowns as policy gradient descent.

### VI. DEEPREINFORCEMENTLEARNING

In recent years, deep neural networks have helped make significant strides in reinforcement learning. Such networks have
learned to perform relatively complex tasks such as playing Atari games, robot navigation, etc.


