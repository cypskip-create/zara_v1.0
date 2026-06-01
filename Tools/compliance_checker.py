# tools/compliance_checker.py
# Zara by Nexara - Compliance Checker Tool
# African cybersecurity and data protection compliance

from typing import Any, Dict, List
from base_tool import BaseTool


COMPLIANCE_FRAMEWORKS = {
    "nigeria_ndpr": {
        "name": "Nigeria Data Protection Regulation (NDPR)",
        "country": "Nigeria",
        "authority": "NITDA",
        "requirements": [
            {
                "id": "NDPR-1",
                "requirement": "Data Protection Impact Assessment",
                "description": "Conduct DPIA before processing sensitive personal data",
                "mandatory": True,
            },
            {
                "id": "NDPR-2",
                "requirement": "Data Protection Officer",
                "description": "Appoint a DPO for organizations processing large volumes of personal data",
                "mandatory": True,
            },
            {
                "id": "NDPR-3",
                "requirement": "Breach Notification",
                "description": "Report data breaches to NITDA within 72 hours",
                "mandatory": True,
            },
            {
                "id": "NDPR-4",
                "requirement": "Data Subject Rights",
                "description": "Implement mechanisms for data subjects to access, correct, and delete their data",
                "mandatory": True,
            },
            {
                "id": "NDPR-5",
                "requirement": "Third Party Processor Contracts",
                "description": "Ensure data processing agreements with all vendors",
                "mandatory": True,
            },
            {
                "id": "NDPR-6",
                "requirement": "Annual Audit",
                "description": "Conduct annual data protection audit and submit to NITDA",
                "mandatory": True,
            },
        ],
        "penalties": "Up to 2% of annual gross revenue or NGN 10 million whichever is greater",
    },
    "kenya_dpa": {
        "name": "Kenya Data Protection Act 2019",
        "country": "Kenya",
        "authority": "Office of the Data Protection Commissioner",
        "requirements": [
            {
                "id": "KE-DPA-1",
                "requirement": "Data Controller Registration",
                "description": "Register as a data controller with the ODPC",
                "mandatory": True,
            },
            {
                "id": "KE-DPA-2",
                "requirement": "Lawful Basis for Processing",
                "description": "Establish and document lawful basis for all data processing",
                "mandatory": True,
            },
            {
                "id": "KE-DPA-3",
                "requirement": "Breach Notification",
                "description": "Notify ODPC and data subjects of breaches within 72 hours",
                "mandatory": True,
            },
            {
                "id": "KE-DPA-4",
                "requirement": "Data Minimization",
                "description": "Collect only data necessary for the specified purpose",
                "mandatory": True,
            },
            {
                "id": "KE-DPA-5",
                "requirement": "Cross-border Transfer Controls",
                "description": "Ensure adequate protection for data transferred outside Kenya",
                "mandatory": True,
            },
        ],
        "penalties": "Up to KES 3 million or imprisonment for individuals",
    },
    "south_africa_popia": {
        "name": "South Africa Protection of Personal Information Act (POPIA)",
        "country": "South Africa",
        "authority": "Information Regulator",
        "requirements": [
            {
                "id": "POPIA-1",
                "requirement": "Information Officer Registration",
                "description": "Register Information Officer with the Information Regulator",
                "mandatory": True,
            },
            {
                "id": "POPIA-2",
                "requirement": "Eight Processing Conditions",
                "description": "Comply with all 8 conditions for lawful processing",
                "mandatory": True,
            },
            {
                "id": "POPIA-3",
                "requirement": "Security Safeguards",
                "description": "Implement appropriate technical and organizational security measures",
                "mandatory": True,
            },
            {
                "id": "POPIA-4",
                "requirement": "Breach Notification",
                "description": "Notify Information Regulator and data subjects as soon as reasonably possible",
                "mandatory": True,
            },
            {
                "id": "POPIA-5",
                "requirement": "Direct Marketing Opt-out",
                "description": "Obtain consent for direct marketing and honor opt-out requests",
                "mandatory": True,
            },
        ],
        "penalties": "Up to ZAR 10 million or 10 years imprisonment",
    },
    "cbn_cybersecurity": {
        "name": "CBN Cybersecurity Framework for Financial Institutions",
        "country": "Nigeria",
        "authority": "Central Bank of Nigeria",
        "requirements": [
            {
                "id": "CBN-1",
                "requirement": "Cybersecurity Policy",
                "description": "Board-approved cybersecurity policy reviewed annually",
                "mandatory": True,
            },
            {
                "id": "CBN-2",
                "requirement": "CISO Appointment",
                "description": "Appoint a Chief Information Security Officer reporting to the board",
                "mandatory": True,
            },
            {
                "id": "CBN-3",
                "requirement": "Incident Reporting",
                "description": "Report significant cyber incidents to CBN within 24 hours",
                "mandatory": True,
            },
            {
                "id": "CBN-4",
                "requirement": "Penetration Testing",
                "description": "Conduct penetration testing at least annually",
                "mandatory": True,
            },
            {
                "id": "CBN-5",
                "requirement": "Business Continuity Plan",
                "description": "Maintain and test cybersecurity business continuity plan",
                "mandatory": True,
            },
            {
                "id": "CBN-6",
                "requirement": "Third Party Risk Management",
                "description": "Assess and monitor cybersecurity risks from third party vendors",
                "mandatory": True,
            },
        ],
        "penalties": "Fines and sanctions as determined by CBN. License revocation for serious violations.",
    },
    "cbk_cybersecurity": {
        "name": "CBK Cybersecurity Guidelines",
        "country": "Kenya",
        "authority": "Central Bank of Kenya",
        "requirements": [
            {
                "id": "CBK-1",
                "requirement": "Cybersecurity Framework",
                "description": "Implement a documented cybersecurity framework aligned to CBK guidelines",
                "mandatory": True,
            },
            {
                "id": "CBK-2",
                "requirement": "Security Incident Reporting",
                "description": "Report cyber incidents to CBK within 24 hours of detection",
                "mandatory": True,
            },
            {
                "id": "CBK-3",
                "requirement": "Customer Data Protection",
                "description": "Implement controls to protect customer financial and personal data",
                "mandatory": True,
            },
            {
                "id": "CBK-4",
                "requirement": "Mobile Banking Security",
                "description": "Specific security controls for mobile banking and mobile money services",
                "mandatory": True,
            },
        ],
        "penalties": "CBK sanctions including fines and license revocation",
    },
}


class ComplianceChecker(BaseTool):

    def __init__(self):
        super().__init__()
        self.name = "compliance_checker"
        self.description = "Checks compliance requirements for African cybersecurity and data protection regulations including Nigeria NDPR, Kenya DPA, South Africa POPIA, CBN and CBK frameworks."
        self.category = "compliance"

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        country = input_data.get("country", "").lower()
        organization_type = input_data.get("organization_type", "general")
        completed_controls = input_data.get("completed_controls", [])

        applicable_frameworks = self._get_applicable_frameworks(country, organization_type)

        if not applicable_frameworks:
            return {
                "success": False,
                "error": "No frameworks found for country: " + country,
                "available_countries": ["Nigeria", "Kenya", "South Africa", "Ghana"],
            }

        compliance_results = []
        for framework_key in applicable_frameworks:
            framework = COMPLIANCE_FRAMEWORKS[framework_key]
            result = self._check_framework(framework, completed_controls)
            compliance_results.append(result)

        overall_compliant = all(r["compliant"] for r in compliance_results)
        total_requirements = sum(r["total_requirements"] for r in compliance_results)
        total_met = sum(r["requirements_met"] for r in compliance_results)

        return {
            "success": True,
            "country": country,
            "organization_type": organization_type,
            "overall_compliant": overall_compliant,
            "compliance_score": int((total_met / max(total_requirements, 1)) * 100),
            "frameworks_checked": len(compliance_results),
            "framework_results": compliance_results,
            "priority_actions": self._get_priority_actions(compliance_results),
        }

    def _get_applicable_frameworks(self, country: str, org_type: str) -> List[str]:
        framework_map = {
            "nigeria": ["nigeria_ndpr"],
            "kenya": ["kenya_dpa"],
            "south africa": ["south_africa_popia"],
        }

        frameworks = framework_map.get(country.lower(), [])

        if "fintech" in org_type.lower() or "bank" in org_type.lower():
            if "nigeria" in country.lower():
                frameworks.append("cbn_cybersecurity")
            elif "kenya" in country.lower():
                frameworks.append("cbk_cybersecurity")

        return frameworks

    def _check_framework(self, framework: Dict, completed_controls: List[str]) -> Dict:
        requirements = framework["requirements"]
        met = []
        not_met = []

        for req in requirements:
            is_met = req["id"] in completed_controls
            if is_met:
                met.append(req)
            else:
                not_met.append(req)

        return {
            "framework_name": framework["name"],
            "authority": framework["authority"],
            "country": framework["country"],
            "compliant": len(not_met) == 0,
            "total_requirements": len(requirements),
            "requirements_met": len(met),
            "requirements_not_met": len(not_met),
            "gaps": [
                {
                    "id": r["id"],
                    "requirement": r["requirement"],
                    "description": r["description"],
                }
                for r in not_met
            ],
            "penalties": framework.get("penalties", "See framework documentation"),
        }

    def _get_priority_actions(self, results: List[Dict]) -> List[str]:
        actions = []
        for result in results:
            for gap in result.get("gaps", [])[:3]:
                actions.append(
                    result["framework_name"] + " - " +
                    gap["requirement"] + ": " +
                    gap["description"]
                )
        return actions[:5]