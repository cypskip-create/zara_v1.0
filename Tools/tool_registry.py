# tools/tool_registry.py
# Zara - Tool Registry
# Central registry where Zara discovers and accesses all tools
# This is how Zara knows what tools exist and when to use them

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger("zara.registry")


class ToolRegistry:
    """
    Central registry for all Zara tools.

    How it works:
        1. All tools register themselves here
        2. Zara queries the registry to find the right tool
        3. Zara calls tool.execute() with the right input
        4. Registry returns the result back to Zara

    Adding a new tool:
        1. Create the tool class in tools/your_tool.py
        2. Import it here
        3. Register it in _register_all_tools()
        That is all Zara needs to start using it.
    """

    def __init__(self):
        self._tools = {}
        self._register_all_tools()
        logger.info("Tool registry initialized with " + str(len(self._tools)) + " tools")

    def _register_all_tools(self):
        """Register all available tools. Add new tools here."""
        from .threat_detector import ThreatDetector
        from .fraud_detector import FraudDetector
        from .security_auditor import SecurityAuditor
        from .incident_responder import IncidentResponder
        from .compliance_checker import ComplianceChecker
        from .vulnerability_scanner import VulnerabilityScanner

        tools = [
            ThreatDetector(),
            FraudDetector(),
            SecurityAuditor(),
            IncidentResponder(),
            ComplianceChecker(),
            VulnerabilityScanner(),
        ]

        for tool in tools:
            self._tools[tool.name] = tool
            logger.info("Registered tool: " + tool.name)

    def get_tool(self, tool_name: str):
        """Get a specific tool by name."""
        tool = self._tools.get(tool_name)
        if not tool:
            logger.warning("Tool not found: " + tool_name)
        return tool

    def list_tools(self) -> List[Dict]:
        """List all available tools with their descriptions."""
        return [tool.info() for tool in self._tools.values()]

    def list_by_category(self, category: str) -> List[Dict]:
        """List tools filtered by category."""
        return [
            tool.info()
            for tool in self._tools.values()
            if tool.category == category
        ]

    def run_tool(self, tool_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a specific tool by name.
        This is the main method Zara uses to execute tools.

        Args:
            tool_name: name of the tool to run
            input_data: dict with tool-specific inputs

        Returns:
            tool result dict with success, result, error, time_ms
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "success": False,
                "error": "Tool not found: " + tool_name,
                "available_tools": list(self._tools.keys()),
            }
        return tool.execute(input_data)

    def detect_intent(self, user_input: str) -> Optional[str]:
        """
        Detect which tool Zara should use based on user input.
        This is how Zara decides which tool to call automatically.
        """
        text = user_input.lower()

        intent_map = [
            {
                "tool": "threat_detector",
                "keywords": [
                    "threat", "attack", "suspicious", "malicious",
                    "phishing", "ransomware", "malware", "hack",
                    "breach", "compromise", "infected",
                ],
            },
            {
                "tool": "fraud_detector",
                "keywords": [
                    "fraud", "transaction", "suspicious payment",
                    "sim swap", "account takeover", "stolen",
                    "unauthorized", "scam", "fake transfer",
                ],
            },
            {
                "tool": "security_auditor",
                "keywords": [
                    "audit", "review code", "check code", "secure code",
                    "vulnerability in code", "code review", "api key",
                    "hardcoded", "credentials in code",
                ],
            },
            {
                "tool": "incident_responder",
                "keywords": [
                    "incident", "response plan", "what do i do", "playbook",
                    "under attack", "been hacked", "ransomware hit",
                    "data breach happened", "what steps", "how to respond",
                    "we got hacked", "we have been breached",
                ],
            },
            {
                "tool": "compliance_checker",
                "keywords": [
                    "compliance", "ndpr", "popia", "data protection",
                    "regulation", "cbn", "cbk", "legal requirement",
                    "regulatory", "framework", "gdpr africa",
                ],
            },
            {
                "tool": "vulnerability_scanner",
                "keywords": [
                    "scan", "vulnerability", "weakness", "secure",
                    "how secure", "security check", "assess", "posture",
                    "mfa", "firewall", "encryption", "backup",
                ],
            },
        ]

        scores = {}
        for intent in intent_map:
            score = 0
            for keyword in intent["keywords"]:
                if keyword in text:
                    score += 1
            if score > 0:
                scores[intent["tool"]] = score

        if not scores:
            return None

        return max(scores, key=scores.get)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())