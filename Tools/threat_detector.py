# tools/threat_detector.py
# Zara - Threat Detector Tool
# Analyzes text, logs, or events for cybersecurity threats

from typing import Any, Dict, List
from .base_tool import BaseTool
import re


THREAT_PATTERNS = {
    "sim_swap": {
        "keywords": [
            "sim swap", "sim change", "sim replacement",
            "port out", "number transfer", "new sim",
        ],
        "severity": "CRITICAL",
        "description": "SIM swap fraud attempt detected",
        "recommendation": "Immediately freeze account and verify customer identity through secondary channel",
    },
    "phishing": {
        "keywords": [
            "click here", "verify your account", "suspended",
            "unusual activity", "confirm your details", "act now",
            "limited time", "your account will be", "reset password",
        ],
        "severity": "HIGH",
        "description": "Phishing attempt detected",
        "recommendation": "Do not click links. Report to security team. Verify through official channels only.",
    },
    "credential_stuffing": {
        "keywords": [
            "multiple failed login", "login attempts", "brute force",
            "invalid password", "account locked", "too many attempts",
        ],
        "severity": "HIGH",
        "description": "Credential stuffing or brute force attack detected",
        "recommendation": "Block IP, enforce MFA, notify account owner immediately",
    },
    "social_engineering": {
        "keywords": [
            "urgent", "emergency", "immediately", "wire transfer",
            "ceo request", "management request", "confidential transfer",
            "bypass approval", "dont tell anyone",
        ],
        "severity": "HIGH",
        "description": "Social engineering attempt detected",
        "recommendation": "Verify request through known contact. Never bypass approval processes under pressure.",
    },
    "malware_indicator": {
        "keywords": [
            "powershell", "cmd.exe", "regsvr32", "wscript",
            "mshta", "certutil", "bitsadmin", "rundll32",
        ],
        "severity": "CRITICAL",
        "description": "Malware execution indicator detected",
        "recommendation": "Isolate system immediately. Do not restart. Engage incident response team.",
    },
    "data_exfiltration": {
        "keywords": [
            "bulk download", "large file transfer", "unusual data volume",
            "after hours access", "database dump", "export all records",
        ],
        "severity": "HIGH",
        "description": "Potential data exfiltration detected",
        "recommendation": "Block transfer, preserve logs, investigate user activity immediately",
    },
    "ransomware": {
        "keywords": [
            "your files are encrypted", "bitcoin payment", "decrypt",
            "pay ransom", "files locked", "ransomware", ".locked",
        ],
        "severity": "CRITICAL",
        "description": "Ransomware activity detected",
        "recommendation": "Isolate all systems immediately. Do not pay. Activate incident response plan.",
    },
    "mobile_money_fraud": {
        "keywords": [
            "mpesa fraud", "mtn momo fraud", "airtel money fraud",
            "mobile money scam", "fake mpesa", "wrong number sent",
            "reverse transaction", "agent fraud",
        ],
        "severity": "HIGH",
        "description": "Mobile money fraud pattern detected",
        "recommendation": "Freeze transaction, verify through official operator channels, report to financial institution",
    },
}

SEVERITY_SCORES = {
    "CRITICAL": 10,
    "HIGH": 7,
    "MEDIUM": 4,
    "LOW": 2,
    "INFO": 1,
}


class ThreatDetector(BaseTool):

    def __init__(self):
        super().__init__()
        self.name = "threat_detector"
        self.description = "Analyzes text, logs, or events for African cybersecurity threats including SIM swap, phishing, mobile money fraud, and malware indicators"
        self.category = "security"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        text = input_data.get("text", "")
        context = input_data.get("context", "general")

        if not text:
            return {
                "success": False,
                "error": "No text provided for analysis",
                "threats": [],
            }

        text_lower = text.lower()
        detected_threats = []
        highest_severity = "INFO"
        highest_score = 0

        for threat_type, pattern in THREAT_PATTERNS.items():
            matched_keywords = []
            for keyword in pattern["keywords"]:
                if keyword.lower() in text_lower:
                    matched_keywords.append(keyword)

            if matched_keywords:
                score = SEVERITY_SCORES.get(pattern["severity"], 1)
                if score > highest_score:
                    highest_score = score
                    highest_severity = pattern["severity"]

                detected_threats.append({
                    "threat_type": threat_type,
                    "severity": pattern["severity"],
                    "description": pattern["description"],
                    "matched_keywords": matched_keywords,
                    "recommendation": pattern["recommendation"],
                })

        detected_threats.sort(
            key=lambda x: SEVERITY_SCORES.get(x["severity"], 0),
            reverse=True
        )

        return {
            "success": True,
            "threat_count": len(detected_threats),
            "overall_severity": highest_severity if detected_threats else "CLEAN",
            "is_threat": len(detected_threats) > 0,
            "threats": detected_threats,
            "context": context,
            "analysis_summary": self._generate_summary(detected_threats, text),
        }

    def _generate_summary(self, threats: List[Dict], text: str) -> str:
        if not threats:
            return "No threats detected. Text appears clean."

        critical = [t for t in threats if t["severity"] == "CRITICAL"]
        high = [t for t in threats if t["severity"] == "HIGH"]

        summary = "THREAT DETECTED. "
        if critical:
            summary += str(len(critical)) + " CRITICAL threat(s): "
            summary += ", ".join(t["threat_type"] for t in critical) + ". "
        if high:
            summary += str(len(high)) + " HIGH threat(s): "
            summary += ", ".join(t["threat_type"] for t in high) + ". "

        top_threat = threats[0]
        summary += "Priority action: " + top_threat["recommendation"]

        return summary