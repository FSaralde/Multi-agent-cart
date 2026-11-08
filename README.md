# 🤖🤖 Two Robots, One Broom

Two AI robots learn to balance a broom on a cart **together**.

## The idea (like you're 5)

A little cart rolls left and right, and a broom stands on top. If the broom
falls over, the game ends.

Two robots share the cart, and each one has its **own brain**:

- 🤖 **Lefty** can only push the cart ⬅️ (or rest).
- 🤖 **Righty** can only push the cart ➡️ (or rest).

Neither robot can balance the broom alone, and if they both push at once the
pushes cancel out. They can't talk to each other, so each one has to learn what
the other usually does. Every moment the broom stays up, **both** get a sticker
⭐, so they win or lose together.

A perfect game is **500 steps**. The game stops itself there.

## How they learn

| Trick | What it means |
|---|---|
| 🧠 Brains | Each robot has a small neural network that guesses "how many stickers will I get if I push, or if I rest?" |
| 🎲 Curiosity | At first they press random buttons to try things out, then less and less. |
| 📓 Diary | Everything that happens gets written down, and they study random pages later. |
| 🪞 Twin brains | A slow copy of each brain gives steady answers while the real brain changes. |

The grown-up name is **Independent Q-Learning (IQL)**, a multi-agent version of
**DQN** (Deep Q-Network), built with [TorchRL](https://github.com/pytorch/rl).

## Results

After about 150,000 steps (~3 minutes on a Mac CPU), the team scores
**500 / 500** on every test game.

```
 5,000 steps: 173   (bumping into each other)
30,000 steps: 500   (first perfect score)
40,000 steps: 125   (got mixed up for a while)
110,000+ steps: 500 every check
```

## Setup

Uses Python 3.13 (Python 3.15 has no PyTorch yet).

```
python3.13 -m venv .venv
.venv/bin/pip install torch torchrl tensordict "gymnasium[classic-control]" pygame
```

## Run it

Train the team (~3 minutes):

```
.venv/bin/python ghghgpy
```

Watch the trained team play (blue dot = Lefty pushing, red dot = Righty pushing):

```
.venv/bin/python ghghgpy --watch
```

> Tip: type commands one at a time. In zsh, a `#` note after a command
> gets passed to the program unless you run `setopt interactive_comments`.

## Change the settings

All the knobs are at the top of `ghghgpy` under **KNOBS**:

| Knob | What it does |
|---|---|
| `TOTAL_STEPS` | How long to train |
| `LEARNING_RATE` | How big each "aha!" adjustment is |
| `CURIOUS_START` / `CURIOUS_END` | How random the robots are at the start and end |
| `CURIOUS_FADE_STEPS` | How fast they calm down |
| `GAMMA` | How much they care about future stickers |
| `MAX_STEPS` | Length of a perfect game (500) |

## Files

| File | What it is |
|---|---|
| `ghghgpy` | The script: the game, both brains, training and watching |
| `team_brains.pt` | The trained brains of Lefty and Righty |
| `cartpole_brain.pt` | The brain from the earlier single-robot version |

## What about the traditional way?

You don't need AI to balance a broom. Engineers use **control theory**: one rule
built from physics, with no training at all.

```
score = a × lean + b × tipping speed + c × cart position + d × cart speed
if score > 0: push right, else: push left
```

The 4 numbers are picked by hand (**PID**) or calculated from the physics
(**LQR**). That one brain works from the very first try.

AI is useful when nobody can write the rule down, like two robots that have to
work out teamwork without talking.
