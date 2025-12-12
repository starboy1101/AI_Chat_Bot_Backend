import json
import logging
import difflib
import os
from typing import Dict, Any
from config import FLOW_FILE
from utils import normalize_text

logger = logging.getLogger("swarai.flows")

# load flow file
_flow_path = FLOW_FILE if os.path.exists(FLOW_FILE) else os.path.join(os.path.dirname(__file__), FLOW_FILE)
try:
    with open(_flow_path, "r", encoding="utf-8") as f:
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

    def find_best_option(self, options: list, user_text: str, cutoff: float = 0.7):
        if not options:
            return None
        norm_input = normalize_text(user_text)
        labels = [normalize_text(opt.get("label", "")) for opt in options]
        # exact match
        for opt, lab in zip(options, labels):
            if lab == norm_input:
                return opt
            # token contains
            if norm_input in lab or lab in norm_input:
                return opt
        # fuzzy via difflib
        best = difflib.get_close_matches(norm_input, labels, n=1, cutoff=cutoff)
        if best:
            idx = labels.index(best[0])
            return options[idx]
        # try token-based matching
        tokens = norm_input.split()
        for t in tokens:
            for opt, lab in zip(options, labels):
                if t in lab:
                    return opt
        return None

flow_manager = FlowManager(conversation_flow)
