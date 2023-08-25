# AlphaZero-Wuziqi
Reinforcement Learning agent implementation of AlphaZero and AlphaGo Zero to play variations of Wuziqi (Gomoku).

Based on the papers:
[Mastering the game of Go Without human knowledge](https://www.nature.com/articles/nature24270/)

[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)

### Game basics:
Wuziqi (五子棋) is a board game (typically) played on a $15$ by $15$ board where the aim is to connect 5 of your stones in a row. Its other names are Gomoku/Gobang in other languages.

This game, like tic-tac-toe, is an example of an [m,n,k-game](https://en.wikipedia.org/wiki/M,n,k-game). Tic-tac-toe for example, is 3,3,3, while traditional Wuziqi is 15,15,5.

I used to play this game and slight variations with my grandfather as a kid. One such variation was to connect 4 in a row to win rather than 5, that we called Siziqi (四子棋).

In the interest of computation and speed, I have condensed the board size to $6$ by $6$ for Siziqi, and $7$ by $7$ for Wuziqi for our models to train on.

### Reinforcement Learning Agent and Training
As outlined in the papers, Alpha(Go) Zero uses reinforcement learning, based on Monte Carlo Tree Search combined with a policy-value neural network to guide decisions and moves.
While the rules of Go, Chess, and Wuziqi are all different, the algorithm is generalizable to manny different games.

However, since Wuziqi has much simpler rules and conditions, I simplified the neural network architecture that was outlined in the AlphaGo Zero paper.

The neural network learns on previous self-play data, of which is augmented through rotations and mirrored flips of the board state, which is using the fact that Wuziqi, like Go, is invariant under rotation and flips, a technique also used in the training of AlphaGo Zero to play Go.

As mentioned, the RL agent employs Monte Carlo Tree Search with a policy-value neural network. It plays against itself by playing against a basic instance of Monte Carlo Tree Search, with no neural network, with a larger number of playouts (1000, 2000, 3000, etc.) compared to the RL agent's number of playouts (400).



### Results:
Siziqi model managed to beat basic MCTS 10-0:
- with 1000 playouts by 550 plays.
- with 2000 playouts by 650 plays.
- with 3000 playouts by 750 plays.


### Files:
- Siziqi models: folder contains the models used when training to play Siziqi (6,6,4-game).
- **MonteCarloTreeSearch.py**: Implementation of the Monte Carlo Tree Search agent/player applied to the games used in the Alpha(Go) Zero papers
- **MonteCarloTreeSearchBasic.py**: A basic implementation of Monte Carlo Tree Search to serve as the opponent to the RL agent.
- **game.py**: Implementation of the rules and board for general m,n,k-games.
- **playtest.py**: Playtest the trained models at the game.
- **policy_value_network.py**: Implementation of the neural network used by the RL agent.
- **train.py**: Implementation of the training pipeline for the agent.