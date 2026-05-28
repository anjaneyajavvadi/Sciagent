import app.env_setup
from app.graph.graph import build_graph
from app.utils.logger import logger
from langgraph.checkpoint.memory import MemorySaver

_agent=None

def get_agent():
    global _agent
    if _agent is None:
        logger.info("Initializing agent")
        checkpointer=MemorySaver()
        _agent=build_graph(checkpointer=checkpointer)
        logger.info("Agent ready")

    return _agent