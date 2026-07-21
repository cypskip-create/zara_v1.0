# tools/fraud_detector.py
# Zara - Fraud Detector Tool
# Real-time transaction and identity fraud detection

from typing import Any, Dict, Optional
from .base_tool import BaseTool
from datetime import datetime


RISK_RULES = [
    {
        "name": "unusual_hour",
        "description": "Transaction at unusual hour (midnight to 5am)",
        "weight": 20,
        "check": lambda t: 0 <= t.get("hour", 12) <= 5,
    },
    {
        "name": "large_amount",
        "description": "Amount significantly above average",
        "weight": 25,
        "check": lambda t: t.get("amount", 0) > t.get("average_amount", 1000) * 5,
    },
    {
        "name": "new_device",
        "description": "Transaction from unrecognized device",
        "weight": 30,
        "check": lambda t: t.get("new_device", False),
    },
    {
        "name": "new_recipient",
        "description": "First time sending to this recipient",
        "weight": 15,
        "check": lambda t: t.get("new_recipient", False),
    },
    {
        "name": "rapid_transactions",
        "description": "Multiple transactions in short time",
        "weight": 35,
        "check": lambda t: t.get("transactions_last_hour", 0) > 5,
    },
    {
        "name": "foreign_location",
        "description": "Transaction from unusual geographic location",
        "weight": 40,
        "check": lambda t: t.get("location_mismatch", False),
    },
    {
        "name": "sim_recently_changed",
        "description": "SIM card changed in last 24 hours",
        "weight": 50,
        "check": lambda t: t.get("sim_changed_recently", False),
    },
    {
        "name": "account_recently_created",
        "description": "Account created less than 7 days ago",
        "weight": 20,
        "check": lambda t: t.get("account_age_days", 999) < 7,
    },
    {
        "name": "multiple_failed_auth",
        "description": "Multiple failed authentication attempts",
        "weight": 35,
        "check": lambda t: t.get("failed_auth_count", 0) >= 3,
    },
    {
        "name": "round_amount",
        "description": "Suspiciously round amount (common in fraud)",
        "weight": 10,
        "check": lambda t: t.get("amount", 1) % 1000 == 0 and t.get("amount", 0) > 0,
    },
]

RISK_LEVELS = {
    "LOW": (0, 30),
    "MEDIUM": (31, 60),
    "HIGH": (61, 80),
    "CRITICAL": (81, 100),
}


class FraudDetector(BaseTool):

    def __init__(self):
        super().__init__()
        self.name = "fraud_detector"
        self.description = "Real-time fraud detection for African mobile money and banking transactions. Detects SIM swap fraud, account takeover, and suspicious transaction patterns."
        self.category = "security"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        transaction = input_data.get("transaction", {})

        if not transaction:
            return {
                "success": False,
                "error": "No transaction data provided",
            }

        triggered_rules = []
        total_score = 0

        for rule in RISK_RULES:
            try:
                if rule["check"](transaction):
                    triggered_rules.append({
                        "rule": rule["name"],
                        "description": rule["description"],
                        "risk_weight": rule["weight"],
                    })
                    total_score += rule["weight"]
            except Exception:
                continue

        risk_score = min(100, total_score)

        risk_level = "LOW"
        for level, (low, high) in RISK_LEVELS.items():
            if low <= risk_score <= high:
                risk_level = level
                break

        action = self._get_action(risk_level)
        african_context = self._get_african_context(transaction, triggered_rules)

        return {
            "success": True,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "action": action,
            "triggered_rules": triggered_rules,
            "rules_triggered_count": len(triggered_rules),
            "transaction_id": transaction.get("id", "unknown"),
            "amount": transaction.get("amount"),
            "currency": transaction.get("currency", "unknown"),
            "african_context": african_context,
            "recommendation": self._get_recommendation(risk_level, triggered_rules),
        }

    def _get_action(self, risk_level: str) -> str:
        actions = {
            "LOW": "ALLOW",
            "MEDIUM": "FLAG_FOR_REVIEW",
            "HIGH": "REQUIRE_ADDITIONAL_VERIFICATION",
            "CRITICAL": "BLOCK_AND_ALERT",
        }
        return actions.get(risk_level, "FLAG_FOR_REVIEW")

    def _get_african_context(self, transaction: Dict, rules: list) -> str:
        rule_names = [r["rule"] for r in rules]

        if "sim_recently_changed" in rule_names:
            return "SIM swap fraud is the most common mobile money attack in Africa. A recently changed SIM combined with a transaction is a critical indicator."

        if "unusual_hour" in rule_names and "large_amount" in rule_names:
            return "Late night large transactions are a common pattern in African mobile money fraud where attackers wait until victims are asleep."

        if "location_mismatch" in rule_names:
            return "Geographic anomalies are significant in African mobile money fraud as most legitimate transactions occur within the user home region."

        return "Transaction analyzed against African mobile money fraud patterns."

    def _get_recommendation(self, risk_level: str, rules: list) -> str:
        if risk_level == "CRITICAL":
            return "Block transaction immediately. Send alert to account owner via alternative contact. Require in-person or video verification before any further transactions."
        elif risk_level == "HIGH":
            return "Pause transaction. Send OTP to registered email (not SMS in case of SIM swap). Require confirmation before proceeding."
        elif risk_level == "MEDIUM":
            return "Allow transaction but flag for manual review within 2 hours. Monitor account for further unusual activity."
        else:
            return "Transaction appears normal. Allow to proceed."