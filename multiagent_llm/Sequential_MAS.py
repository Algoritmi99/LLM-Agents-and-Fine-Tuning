from overrides import override

from .MultiAgentSystem import *


class SequentialMAS(MultiAgentSystem):
    """
    A sequential Multi-agent System (Both Ring and Linear Structure)
    """
    def __init__(self, agents: dict[str, Agent], k=1):
        """
        :param agents: A dictionary of agents. Keys must be numerical. The actions happen in order of the numerical keys.
        :param k: The number of times the sequence is to be repeated. Default is 1, which corresponds to a linear structure.
        """
        super().__init__(agents)
        for i in self.__agents.keys():
            i.isnumeric(), "Agents dictionary must have ordered numerical zero based keys for sequential MAS."
        self.__k = k

    @override
    def act(self, state: str) -> str:
        out = state
        keys = [int(i) for i in self.__agents.keys()]
        keys.sort()
        for _ in range(self.__k):
            for agent_id in keys:
                out = self.__agents[str(agent_id)].act(out)
        return out

    @override
    def add_agent(self, agent, name: str=None):
        if name is None:
            super().add_agent(agent, str(len(self.__agents)))
            return
        assert name is not None and name.isnumeric() and name not in self.__agents, "Name must be numeric and unique."
        self.__agents[str(name)] = agent