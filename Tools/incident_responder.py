# tools/incident_responder.py
# Zara - Incident Responder Tool
# Generates step-by-step incident response playbooks
# specific to African legal and regulatory requirements

from typing import Any, Dict, List
from .base_tool import BaseTool


PLAYBOOKS = {
    "ransomware": {
        "name": "Ransomware Attack",
        "severity": "CRITICAL",
        "immediate_steps": [
            "Isolate all infected systems from the network immediately by unplugging ethernet cables and disabling WiFi",
            "Do NOT restart infected machines as this may destroy forensic evidence",
            "Photograph or screenshot the ransom note before touching anything",
            "Identify which systems are infected vs clean using network logs",
            "Activate your incident response team and notify C-suite within 30 minutes",
            "Check if backup systems are accessible and verify last clean backup date",
            "Document everything with timestamps from this moment forward",
        ],
        "short_term_steps": [
            "Engage a cybersecurity forensics firm with African experience",
            "Determine the ransomware variant using ID Ransomware (id-ransomware.malwarehunterteam.com)",
            "Check if a free decryptor exists at NoMoreRansom (nomoreransom.org)",
            "Assess the full scope of encrypted systems and affected data",
            "Notify law enforcement in your country",
            "Do NOT pay ransom without consulting security and legal counsel",
        ],
        "regulatory_reporting": {
            "Nigeria": "Report to ngCERT at incident@cert.gov.ng within 24 hours. Notify CBN if financial data involved.",
            "Kenya": "Report to KE-CIRT at kecirt@ke.go.ke within 24 hours. Notify CBK if customer financial data is affected.",
            "South Africa": "Report to CSIRT-SA. Notify Information Regulator under POPIA within 72 hours if personal data affected.",
            "Ghana": "Report to CERT-GH. Notify Bank of Ghana if financial institution data is involved.",
        },
        "recovery_steps": [
            "Rebuild systems from clean backups only - never rebuild from infected systems",
            "Patch all vulnerabilities exploited in the attack before restoring systems",
            "Reset all passwords and revoke all active sessions",
            "Monitor restored systems intensively for 30 days post-recovery",
            "Conduct post-incident review to identify root cause",
            "Update security controls to prevent recurrence",
        ],
    },
    "data_breach": {
        "name": "Data Breach",
        "severity": "HIGH",
        "immediate_steps": [
            "Identify the source and scope of the breach immediately",
            "Stop the ongoing breach by closing the compromised access vector",
            "Preserve all logs and evidence before any remediation",
            "Assess what data was exposed - personal data, financial data, credentials",
            "Notify your Data Protection Officer or legal counsel immediately",
            "Document everything with timestamps",
        ],
        "short_term_steps": [
            "Determine the number of individuals affected",
            "Assess the risk level to affected individuals",
            "Prepare breach notification communications",
            "Notify affected individuals if risk to their rights is high",
            "Engage forensics to determine full extent of breach",
        ],
        "regulatory_reporting": {
            "Nigeria": "Notify NITDA within 72 hours under NDPR. Notify CBN if financial data of customers is involved.",
            "Kenya": "Notify the Office of the Data Protection Commissioner within 72 hours under the Data Protection Act.",
            "South Africa": "Notify the Information Regulator and affected data subjects under POPIA Section 22.",
            "Ghana": "Notify the Data Protection Commission under the Data Protection Act 2012.",
        },
        "recovery_steps": [
            "Patch the vulnerability that caused the breach",
            "Reset all affected user credentials",
            "Implement additional monitoring on affected systems",
            "Review and strengthen access controls",
            "Conduct privacy impact assessment",
            "Update data breach response procedures",
        ],
    },
    "sim_swap": {
        "name": "SIM Swap Fraud",
        "severity": "CRITICAL",
        "immediate_steps": [
            "Freeze the affected customer account immediately",
            "Contact the mobile network operator fraud team to reverse the SIM swap",
            "Alert the customer through alternative contact method (email, backup number)",
            "Block all pending transactions from the account",
            "Preserve transaction logs and SIM swap timing records as evidence",
            "Identify and reverse any fraudulent transactions within the reversal window",
        ],
        "short_term_steps": [
            "File a report with the mobile network operator formal complaint process",
            "Assist customer in filing police report for insurance and legal purposes",
            "Investigate whether an insider at the mobile operator was involved",
            "Review other accounts that share the same phone number",
            "Check if other accounts were accessed using compromised 2FA",
        ],
        "regulatory_reporting": {
            "Nigeria": "Report to CBN Consumer Protection Department. File with EFCC if amount exceeds threshold.",
            "Kenya": "Report to Communications Authority of Kenya and DCI Cybercrime Unit.",
            "South Africa": "Report to FSCA if financial institution. File with Hawks Cybercrime Unit.",
            "Ghana": "Report to Bank of Ghana and National Communications Authority.",
        },
        "recovery_steps": [
            "Help customer restore SIM to their control",
            "Reset all account credentials after SIM is secured",
            "Enable stronger authentication methods - move away from SMS 2FA",
            "Educate customer about SIM swap fraud prevention",
            "Review and improve SIM swap detection controls",
        ],
    },
    "phishing": {
        "name": "Phishing Attack",
        "severity": "HIGH",
        "immediate_steps": [
            "Do not click any links or download attachments in the suspicious message",
            "Report the phishing message to your security team immediately",
            "If credentials were entered, change passwords immediately from a clean device",
            "Enable MFA on all accounts if not already active",
            "Check for any unauthorized access or transactions since the phishing occurred",
            "Preserve the original phishing message as evidence",
        ],
        "short_term_steps": [
            "Report the phishing site to Google Safe Browsing and your national CERT",
            "Notify other employees if a targeted phishing campaign is suspected",
            "Scan the device used to click the link for malware",
            "Review access logs for any unauthorized activity",
            "Notify customers if a phishing campaign impersonates your brand",
        ],
        "regulatory_reporting": {
            "Nigeria": "Report phishing sites to ngCERT. If customers are targeted, notify CBN.",
            "Kenya": "Report to KE-CIRT. Notify CBK if bank customers are being targeted.",
            "South Africa": "Report to SABRIC (South African Banking Risk Information Centre) if banking related.",
            "Ghana": "Report to CERT-GH and Bank of Ghana if financial institution phishing.",
        },
        "recovery_steps": [
            "Take down phishing domains through your registrar and hosting provider",
            "Issue customer warning through official channels",
            "Implement email authentication (DMARC, DKIM, SPF) to prevent brand spoofing",
            "Conduct staff phishing simulation training",
            "Review and improve email security controls",
        ],
    },
    "account_takeover": {
        "name": "Account Takeover",
        "severity": "HIGH",
        "immediate_steps": [
            "Lock the compromised account immediately",
            "Terminate all active sessions across all devices",
            "Send account lockout notification to the legitimate owner",
            "Review and reverse any unauthorized transactions within reversal window",
            "Preserve access logs showing the takeover timeline",
            "Identify the attack vector - credential stuffing, phishing, or SIM swap",
        ],
        "short_term_steps": [
            "Contact the legitimate account owner through verified alternative contact",
            "Verify identity before restoring account access",
            "Review other accounts that share similar credentials",
            "Check if the attack is part of a larger credential stuffing campaign",
            "Analyze IP addresses and device fingerprints used in the takeover",
        ],
        "regulatory_reporting": {
            "Nigeria": "Report to ngCERT if part of a coordinated attack. Notify CBN for financial accounts.",
            "Kenya": "Report to KE-CIRT. Financial institutions must notify CBK of significant fraud incidents.",
            "South Africa": "Report to CSIRT-SA. FSCA notification required for regulated financial services.",
            "Ghana": "Report to CERT-GH. Notify Bank of Ghana for banking sector incidents.",
        },
        "recovery_steps": [
            "Reset credentials and security questions after verifying identity",
            "Enable stronger MFA - authenticator app not SMS",
            "Set up suspicious login alerts for the account",
            "Monitor the account intensively for 30 days",
            "Investigate credential leak source and notify affected users",
        ],
    },
}


class IncidentResponder(BaseTool):

    def __init__(self):
        super().__init__()
        self.name = "incident_responder"
        self.description = "Generates step-by-step incident response playbooks for African organizations. Includes country-specific regulatory reporting requirements for Nigeria, Kenya, South Africa, and Ghana."
        self.category = "security"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        incident_type = input_data.get("incident_type", "").lower().replace(" ", "_")
        country = input_data.get("country", "")
        details = input_data.get("details", "")

        if incident_type not in PLAYBOOKS:
            closest = self._find_closest_playbook(incident_type)
            if closest:
                incident_type = closest
            else:
                return {
                    "success": False,
                    "error": "Unknown incident type: " + incident_type,
                    "available_types": list(PLAYBOOKS.keys()),
                }

        playbook = PLAYBOOKS[incident_type]
        regulatory = playbook.get("regulatory_reporting", {})
        regulatory_ci = {k.lower(): v for k, v in regulatory.items()}
        country_reporting = regulatory_ci.get(
            country.lower(),
            "Check with your national CERT and relevant financial regulator for specific reporting requirements in your country.",
        )

        return {
            "success": True,
            "incident_type": playbook["name"],
            "severity": playbook["severity"],
            "immediate_steps": playbook["immediate_steps"],
            "short_term_steps": playbook["short_term_steps"],
            "recovery_steps": playbook["recovery_steps"],
            "regulatory_reporting": {
                "your_country": country,
                "requirements": country_reporting,
                "all_countries": regulatory,
            },
            "total_steps": (
                len(playbook["immediate_steps"]) +
                len(playbook["short_term_steps"]) +
                len(playbook["recovery_steps"])
            ),
            "details_noted": details,
        }

    def _find_closest_playbook(self, incident_type: str) -> str:
        mappings = {
            "ransomware": "ransomware",
            "ransom": "ransomware",
            "breach": "data_breach",
            "data_breach": "data_breach",
            "leak": "data_breach",
            "sim": "sim_swap",
            "sim_swap": "sim_swap",
            "phishing": "phishing",
            "email_fraud": "phishing",
            "takeover": "account_takeover",
            "account_takeover": "account_takeover",
            "account_compromise": "account_takeover",
        }
        for key, value in mappings.items():
            if key in incident_type:
                return value
        return ""