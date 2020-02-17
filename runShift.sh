#!/bin/bash
#WARNING: For replicating study put the json files
# into dict/mediumC and dict/smallC respectively.

#Extended Experiments, where we to turn the wide component off and on again

###MEDIUM Map with Shift###
#WDQN 3 feat
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-3feat-a --fixRandomSeed 1
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-3feat-b --fixRandomSeed 2
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-3feat-c --fixRandomSeed 3
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-3feat-d --fixRandomSeed 4
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-3feat-e --fixRandomSeed 5

#WDQN 2 feat
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-2feat-a --fixRandomSeed 1
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-2feat-b --fixRandomSeed 2
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-2feat-c --fixRandomSeed 3
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-2feat-d --fixRandomSeed 4
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-2feat-e --fixRandomSeed 5

#WDQN 1 feat
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-1feat-a --fixRandomSeed 1
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-1feat-b --fixRandomSeed 2
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-1feat-c --fixRandomSeed 3
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-1feat-d --fixRandomSeed 4
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l mediumClassic --path SWDQN-mediumC-1feat-e --fixRandomSeed 5


###SMALL MAP with Shift###
#WDQN 3 feat
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-3feat-a --fixRandomSeed 1
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-3feat-b --fixRandomSeed 2
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-3feat-c --fixRandomSeed 3
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-3feat-d --fixRandomSeed 4
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-3feat-e --fixRandomSeed 5

#WDQN 2 feat
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-2feat-a --fixRandomSeed 1
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-2feat-b --fixRandomSeed 2
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-2feat-c --fixRandomSeed 3
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-2feat-d --fixRandomSeed 4
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-2feat-e --fixRandomSeed 5

#WDQN 1 feat
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-1feat-a --fixRandomSeed 1
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-1feat-b --fixRandomSeed 2
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-1feat-c --fixRandomSeed 3
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-1feat-d --fixRandomSeed 4
python3 pacman.py -p PacmanPWDQN -n 11000 -x 11000 -g DirectionalGhost -l smallClassic --path SWDQN-smallC-1feat-e --fixRandomSeed 5
