# Wide & Deep Reinforcement Learning Extended for Grid-Based Games


Wide and Deep Reinforcement Learning (WDRL) implementation in Pac-man using our Wide Deep Q-networks (WDQN). 
This is an extension of the original paper that you can download [here](http://www.scitepress.org/PublicationsDetail.aspx?ID=0bLGtol9A6g=&t=1).

In this new version,
we developed the idea to try to turn the wide component off and on again. This creates replays for both a pure DQN and a
WDQN and thus forces the deep component to work independently (namely when the
wide component is switched off). 

For a complete explanation of our new version, you can access [here](http://www.scitepress.org/PublicationsDetail.aspx?ID=0bLGtol9A6g=&t=1) our chapter of this [book](https://www.springer.com/gp/book/9783030374938).

## WARNING!
This repository just includes the new files necessary for replicating the results of our chapter to the Springer book about Agents and Artificial Intelligence. This repository was created for being used in combination with the past one. For accessing our past repository and for a more detailed explanation of the code, you can click [here](https://github.com/JuanMMontoya/WDRL).

## Citation
Please cite our research if it was useful for you:

```
@incollection{
author="Montoya, Juan M.
and Doell, Christoph
and Borgelt, Christian",
editor="van den Herik, Jaap
and Rocha, Ana Paula
and Steels, Luc",
title="Wide and Deep Reinforcement Learning Extended for Grid-Based Action Games",
booktitle="Agents and Artificial Intelligence",
year="2019",
publisher="Springer International Publishing",
address="Cham",
pages="224--245",
isbn="978-3-030-37494-5"
}


```
