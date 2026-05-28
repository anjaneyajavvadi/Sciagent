import app.env_setup
from app.graph.graph import build_graph
from app.utils.logger import logger


_agent=None

def get_agent():
    global _agent
    if _agent is None:
        logger.info("Initializing agent")
        _agent=build_graph()
        logger.info("Agent ready")

    return _agent