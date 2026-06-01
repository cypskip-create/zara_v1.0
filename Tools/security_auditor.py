# tools/security_auditor.py
# Zara by Nexara - Security Auditor Tool
# Scans code and configurations for security vulnerabilities

from typing import Any, Dict, List
from base_tool import BaseTool
import re


VULNERABILITY_PATTERNS = [
    {
        "id": "SEC-001",
        "name": "Hardcoded API Key",
        "severity": "CRITICAL",
        "pattern": r"(api_key|apikey|api-key)\s*=\s*['\"][a-zA-Z0-9_\-]{20,}['\"]",
        "description": "API key hardcoded in source code",
        "fix": "Move API keys to environment variables. Use os.environ.get('API_KEY') or a secrets manager.",
    },
    {
        "id": "SEC-002",
        "name": "Hardcoded Password",
        "severity": "CRITICAL",
        "pattern": r"(password|passwd|pwd)\s*=\s*['\"][^'\"]{4,}['\"]",
        "description": "Password hardcoded in source code",
        "fix": "Never hardcode passwords. Use environment variables or a secrets manager like HashiCorp Vault.",
    },
    {
        "id": "SEC-003",
        "name": "Hardcoded Secret Key",
        "severity": "CRITICAL",
        "pattern": r"(secret|secret_key|secretkey)\s*=\s*['\"][^'\"]{8,}['\"]",
        "description": "Secret key hardcoded in source code",
        "fix": "Use environment variables: SECRET_KEY = os.environ.get('SECRET_KEY')",
    },
    {
        "id": "SEC-004",
        "name": "SQL Injection Risk",
        "severity": "HIGH",
        "pattern": r"(execute|query)\s*\([^)]*\+[^)]*\)",
        "description": "String concatenation in SQL query - SQL injection risk",
        "fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
    },
    {
        "id": "SEC-005",
        "name": "No Input Validation",
        "severity": "HIGH",
        "pattern": r"request\.(get|post|form|args)\[['\"][^'\"]+['\"]\]",
        "description": "User input used without validation",
        "fix": "Validate and sanitize all user inputs before processing",
    },
    {
        "id": "SEC-006",
        "name": "Insecure HTTP",
        "severity": "MEDIUM",
        "pattern": r"http://(?!localhost|127\.0\.0\.1)",
        "description": "Insecure HTTP connection to external service",
        "fix": "Use HTTPS for all external connections. Enforce SSL certificate verification.",
    },
    {
        "id": "SEC-007",
        "name": "Debug Mode Enabled",
        "severity": "HIGH",
        "pattern": r"DEBUG\s*=\s*True",
        "description": "Debug mode enabled - exposes sensitive information in production",
        "fix": "Set DEBUG = False in production. Use environment variable: DEBUG = os.environ.get('DEBUG', 'False') == 'True'",
    },
    {
        "id": "SEC-008",
        "name": "Weak Cryptography",
        "severity": "HIGH",
        "pattern": r"(md5|sha1)\s*\(",
        "description": "Weak hashing algorithm used",
        "fix": "Use bcrypt for passwords or SHA-256/SHA-512 for data integrity",
    },
    {
        "id": "SEC-009",
        "name": "SSL Verification Disabled",
        "severity": "CRITICAL",
        "pattern": r"verify\s*=\s*False",
        "description": "SSL certificate verification disabled - vulnerable to MITM attacks",
        "fix": "Never disable SSL verification. Remove verify=False from all requests calls.",
    },
    {
        "id": "SEC-010",
        "name": "Sensitive Data in Logs",
        "severity": "MEDIUM",
        "pattern": r"(print|log)\s*\([^)]*\b(password|token|secret|key|pin)\b",
        "description": "Sensitive data potentially logged",
        "fix": "Never log passwords, tokens, or secrets. Mask sensitive fields before logging.",
    },
    {
        "id": "SEC-011",
        "name": "Exposed M-Pesa Credentials",
        "severity": "CRITICAL",
        "pattern": r"(consumer_key|consumer_secret|mpesa_passkey)\s*=\s*['\"][^'\"]{10,}['\"]",
        "description": "M-Pesa API credentials hardcoded",
        "fix": "Store M-Pesa credentials in environment variables. Never commit them to GitHub.",
    },
    {
        "id": "SEC-012",
        "name": "Exposed Paystack Keys",
        "severity": "CRITICAL",
        "pattern": r"(sk_live_|pk_live_)[a-zA-Z0-9]{20,}",
        "description": "Paystack live API keys exposed in code",
        "fix": "Immediately rotate these keys on the Paystack dashboard. Store in environment variables.",
    },
    {
        "id": "SEC-013",
        "name": "No Rate Limiting",
        "severity": "MEDIUM",
        "pattern": r"@app\.route\([^)]+\)\s*\ndef\s+\w+\s*\(",
        "description": "API endpoint without rate limiting",
        "fix": "Add rate limiting to all API endpoints using Flask-Limiter or similar",
    },
    {
        "id": "SEC-014",
        "name": "Unvalidated Redirect",
        "severity": "HIGH",
        "pattern": r"redirect\s*\(\s*request\.",
        "description": "Redirect using unvalidated user input - open redirect vulnerability",
        "fix": "Validate redirect URLs against a whitelist of allowed domains",
    },
    {
        "id": "SEC-015",
        "name": "Callback URL Not Verified",
        "severity": "HIGH",
        "pattern": r"callback_url\s*=\s*request\.",
        "description": "Payment callback URL from user input not verified",
        "fix": "Always verify callback URLs against pre-registered URLs. Never use user-supplied callback URLs.",
    },
]


class SecurityAuditor(BaseTool):

    def __init__(self):
        super().__init__()
        self.name = "security_auditor"
        self.description = "Scans code for security vulnerabilities including exposed credentials, injection flaws, and African fintech specific issues like exposed M-Pesa and Paystack keys."
        self.category = "security"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        code = input_data.get("code", "")
        language = input_data.get("language", "python")
        context = input_data.get("context", "general")

        if not code:
            return {
                "success": False,
                "error": "No code provided for audit",
            }

        vulnerabilities = []
        lines = code.split("\n")

        for vuln in VULNERABILITY_PATTERNS:
            matches = []
            for line_num, line in enumerate(lines, 1):
                if re.search(vuln["pattern"], line, re.IGNORECASE):
                    matches.append({
                        "line_number": line_num,
                        "line_content": line.strip(),
                    })

            if matches:
                vulnerabilities.append({
                    "id": vuln["id"],
                    "name": vuln["name"],
                    "severity": vuln["severity"],
                    "description": vuln["description"],
                    "fix": vuln["fix"],
                    "occurrences": matches,
                    "occurrence_count": len(matches),
                })

        vulnerabilities.sort(
            key=lambda x: {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x["severity"], 0),
            reverse=True
        )

        critical_count = len([v for v in vulnerabilities if v["severity"] == "CRITICAL"])
        high_count = len([v for v in vulnerabilities if v["severity"] == "HIGH"])
        medium_count = len([v for v in vulnerabilities if v["severity"] == "MEDIUM"])

        security_score = self._calculate_score(vulnerabilities, len(lines))

        return {
            "success": True,
            "security_score": security_score,
            "security_grade": self._get_grade(security_score),
            "total_vulnerabilities": len(vulnerabilities),
            "critical_count": critical_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "vulnerabilities": vulnerabilities,
            "lines_scanned": len(lines),
            "summary": self._generate_summary(vulnerabilities, security_score),
            "pass_security_check": critical_count == 0 and high_count == 0,
        }

    def _calculate_score(self, vulnerabilities: List[Dict], total_lines: int) -> int:
        score = 100
        weights = {"CRITICAL": 25, "HIGH": 15, "MEDIUM": 8, "LOW": 3}
        for vuln in vulnerabilities:
            score -= weights.get(vuln["severity"], 0)
        return max(0, score)

    def _get_grade(self, score: int) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_summary(self, vulnerabilities: List[Dict], score: int) -> str:
        if not vulnerabilities:
            return "Code passed security audit. No vulnerabilities detected."

        critical = [v for v in vulnerabilities if v["severity"] == "CRITICAL"]
        if critical:
            names = ", ".join(v["name"] for v in critical[:3])
            return "CRITICAL vulnerabilities found: " + names + ". Fix these before deploying to production."

        return (
            str(len(vulnerabilities)) + " vulnerabilities found. "
            "Security score: " + str(score) + "/100. "
            "Review and fix all HIGH severity issues before launch."
        )