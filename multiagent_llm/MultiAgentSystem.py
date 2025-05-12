from abc import ABC

from agent import Agent


class MultiAgentSystem(Agent, ABC):
    """
    An Interface for Multi-agent Systems.
    Must implement the method "act".
    """
    def __init__(self, agents: dict[str, Agent]):
        self.__agents = agents

    def add_agent(self, agent, name: str=None):
        if name is None:
            self.__agents[str(len(self.__agents))] = agent
        else:
            self.__agents[name] = agent
