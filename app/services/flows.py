import difflib
import json
import logging
from typing import Any, Dict

from app.core.config import FLOW_FILE
from app.utils.common import normalize_text

logger = logging.getLogger("swarai.flows")

try:
    with open(FLOW_FILE, "r", encoding="utf-8") as f:
        conversation_flow: Dict[str, Any] = json.load(f)
except Exception:
    logger.exception("Failed to load conversation flow, using empty flow.")
    conversation_flow = {}


class FlowManager:
    def __init__(self, flow: Dict[str, Any]):
        self.flow = flow or {}

    def get_node(self, node_id: str):
        return self.flow.get(node_id)

    def start_flow_for_user(self, session: dict):
        session.update({"in_flow": True, "node_id": "start", "context": {}, "change_target": None})
        return self.flow.get("start")

    def restart_flow(self, session: dict):
        session.update({"in_flow": False, "node_id": None, "context": {}, "change_target": None})
        return None

    def start_from_node(self, session: dict, node_id: str):
        """
        Start the flow from a specific node instead of 'start'.
        Used when PDF already provided partial requirements.
        """
        session.update(
            {
                "in_flow": True,
                "node_id": node_id,
                "change_target": None,
            }
        )
        return self.flow.get(node_id)

    def find_best_option(self, options: list, user_text: str, cutoff: float = 0.7):
        if not options:
            return None
        norm_input = normalize_text(user_text)
        labels = [normalize_text(opt.get("label", "")) for opt in options]
        for opt, lab in zip(options, labels):
            if lab == norm_input:
                return opt
            if norm_input in lab or lab in norm_input:
                return opt
        best = difflib.get_close_matches(norm_input, labels, n=1, cutoff=cutoff)
        if best:
            idx = labels.index(best[0])
            return options[idx]
        tokens = norm_input.split()
        for t in tokens:
            for opt, lab in zip(options, labels):
                if t in lab:
                    return opt
        return None


flow_manager = FlowManager(conversation_flow)
