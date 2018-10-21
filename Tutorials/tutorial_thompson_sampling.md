# Thompson Sampling
Presented by Mason Rumuly and Siddharth Ajit the 9th of October 2018

## Summary of Contents
This tutorial covers the following topics:
1. Introduction to Multi-Arm Bandits
2. Exploration vs. Exploitation
3. The Thompson Sampling Algorithm
4. Thompson Sampling Performance
5. Extensions to Thompson Sampling
6. Further Reading

## Multi-Arm Bandits

Multi-Arm Bandits (MAB), which derive their name from a tongue-in-cheek description of Slot Machines, are perhaps the simplest incarnation of Reinforcement Learning. A MAB problem presents a number of actions which can be taken each round; these are the 'arms,' which are 'pulled' or acted on. Each arm creates a probabilistic reward when pulled. The reward probability is independent of the other arms and of all other actions. An arm must be pulled each round. The goal is to maximize the sum of rewards over a number of rounds, or equivalently to find and exploit the arm with the best payout probability.

## Exploration and Exploitation
Exploration : Gain or learn more information about the rewards by pulling all the arms; gather more information to improve future decisions.
Exploitation : Maximize the reward or minimize the regret by pulling the arms with greatest expected value; Make the best decision given the existing information.

In the context of MAB, the user must take action (play an arm) to learn about it. In short, sub optimal actions should be chosen to maximize long term benefits.

This can be better explained by a real life scenario
![image](https://user-images.githubusercontent.com/35848569/47262851-c50f8600-d4b8-11e8-9c49-13e98c4caf7e.png)

Visit the new restaurant to learn more information about it. You may like the new place and hopefully visit it more often or regret your decision.This is an example of exploration. Go to the usual favorite restaurant and “exploit” the utility or value associated with the decision but you may lose out on finding a better restaurant. This action represents exploitation.

Thompson sampling does not have a seperate exploration and exploitation phase.However,the distribution with the highest likelihood of payout gets chosen the most assuming a finite number of sampling rounds.




## Thompson Sampling Algorithm

Thompson Sampling uses a Bayesian probabilistic approach to learning and policy.
The first step is to construct a Beta distribution for each arm. The Beta distribution is a conjugate prior to the Bernoulli trial and its support coincides with the range of the unknown parameter. This makes it the natural representation of the uncertainty (or, equivalently, the knowledge) about each arm. The beta distribution takes two shape parameters, ```alpha``` and ```beta```. Initialize these to values ```alpha_0 = 1``` and ```beta_0 = 1```. This is a total memory requirement of two counts per arm.
Now, for each round:

- Sample the Beta distributed random variable associated with each arm.
- Pull the arm associated with the greatest sampled value
- Record the outome for the associated arm by incrementing the ```alpha``` parameter of its Beta distribution on success or the ```beta``` parameter on failure

A Python3 implementation of this algorithm may be viewed in the associated [demo notebook](../Students/mason-rumuly/tutorial/thompson_sampling_demo.ipynb).

## Performance

The Thompson Sampling algorithm naturally creates unbiased estimates of the parameter for each arm which are asymtotially exact. It also converges on the optimal policy, that is as the number of rounds approaches infinity the proportion of pulls to the optimal arm go to one.

It executes these in an optimal fashion, finding and exploiting the best arm as fast as possible, with ```O(log(rounds))``` pulls to non-optimal arms (see the Further Reading for proofs). It also avoids the problem of choosing exploration or exploitation, naturally trading off appropriately between the two in a probabilistic manner as it acquires information about the bandit.

## Extensions

There are many extentions and variations on implementation of Thompson Sampling.
First, for the Beta distribution, there are three natural initial values for ```(alpha_0, beta_0)```
- Haldane ```(0,0)```: While theoretically interesting, the Haldane prior is practically useless, as an arm which returns a failure on the first pull is 'killed,' that is it has a 0 probability of ever being pulled again.
- Bayesian ```(1,1)```: This prior assumes that any value of the Bernoulli prior is equally likely, and performs well with most formulations of the Multi-Arm Bandit; thus, it is the one used in the associated demonstration.
- Jeffrey ```(0.5,0.5)```: This prior is interesting as a 'non-informative' prior, that is it in some sense makes fewer assumptions than the Bayesian prior. It also has some interesting symmetries, and performs identically to the Bayesian prior for our simple formulation of the bandit. However, this prior loses some of the performance guarantees for some bandit formulations, so use with caution.

There are also multiple ways to keep track of the information acquired. As each resolves perfectly to every other, it resolves to a matter of convenience which to use:
- **Count of Successes, Count of Failures** These counts have the simplest and most direct update and corresponds 1:1 with the parameters of the Beta distribution
- **Count of Pulls, Count of Successes** or Failures; Since the number of pulls on an arm is the sum of successes and failures, this is identical to the original formulation
- **Count of Pulls, Sample Mean** This version corresponds most directly to the intuition about the information gleaned over time, where the number of pulls is inversely proportional to the uncertainty and the sample mean is an unbiased estimate of the Bernoulli parameter of the associated arm. It also only has one value which may go to infinity, but requires slightly more computation to update the other value.

Finally, there are all manner of changes used to accommodate alternate formulations of the bandit problem:
- Use Dirichlet distribution instead of Beta distribution for multiple payout amounts
- Use Gaussian distribution instead of Beta distribution for continuously-valued payout
- Forgetful version for Non-stationary bandits
- Contextual information for Contextual bandits
- General Reinforcement Learning adaptations
- and many more

## Further Reading

We have included some references to act as a springboard for personal study and which were used in assembling this tutorial.

- Original Thompson Sampling Paper: [Thompson, William R. "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples". Biometrika, 25(3–4):285–294, 1933.](https://www.dropbox.com/s/yhn9prnr5bz0156/1933-thompson.pdf)
- Best Performance on Bandits: [Lai, T.L., & Robbins, H. “Asymptotically Efficient Adaptive Allocation Rules”. Advances in applied Mathematics, 6, pp 4-22, 1985.](http://www.rci.rutgers.edu/~mnk/papers/Lai_robbins85.pdf)
- Proof of log(n) Regret for Thompson Sampling[Sampling:Kaufmann, E., Korda, N., & Munos, R. “Thompson Sampling: an Asymptotically Optimal Finite-Time Analysis”. Proceedings of the 23rd international conference on Algorithmic Learning Theory (ALT’12) , pp. 199-213. 2012.](https://doi.org/10.1007/978-3-642-34106-9_18)
- Analysis of Extension to Real-Reward Bandits:[Junya Honda, Akimichi Takemura.“Optimality of Thompson Sampling for Gaussian Bandits Depends on Priors”. Proceedings of the Seventeenth International Conference on Artificial Intelligence and Statistics, PMLR 33:375-383, 2014.](http://proceedings.mlr.press/v33/honda14.pdf)
- Wikipedia on [Multi-Arm Bandits](https://en.wikipedia.org/wiki/Bandit_problem)
- Wikipedia on [Thompson Sampling](https://en.wikipedia.org/wiki/Thompson_sampling)
