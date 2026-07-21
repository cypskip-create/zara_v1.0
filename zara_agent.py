# zara_agent.py
# Zara - The Brain
# Connects Zara's language model to her tools
# This is where the model and tools become one system

import os
import torch
import tiktoken
from typing import Any, Dict, Optional
from Tools.tool_registry import ToolRegistry


class ZaraAgent:
    """
    Zara - Africa's Cybersecurity Brain

    Zara is a cybersecurity AI that:
    1. Understands natural language questions about cybersecurity
    2. Detects which tool to use automatically
    3. Runs the appropriate tool
    4. Uses her language model to explain the results
    5. Gives African-context-aware responses

    Usage:
        zara = ZaraAgent(checkpoint_path="path/to/zara_cybersecurity_v2.pt")
        response = zara.chat("How do I respond to a ransomware attack in Kenya?")
        print(response)
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        print("Initializing Zara - Africa's Cybersecurity Brain...")

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None
        self.enc = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)
        else:
            print("No model checkpoint provided. Running in tool-only mode.")
            print("Provide checkpoint_path to enable full AI responses.")

        self.registry = ToolRegistry()
        self.conversation_history = []

        print("Zara ready! Tools available: " + str(self.registry.tool_count))

    def _load_model(self, checkpoint_path: str):
        """Load Zara's trained cybersecurity brain."""
        try:
            from model import ModelConfig, TransformerLM
            print("Loading cybersecurity brain from: " + checkpoint_path)
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            cfg = ModelConfig(**ckpt["cfg"])
            self.model = TransformerLM(cfg).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.enc = tiktoken.get_encoding("gpt2")
            step = ckpt.get("step", "unknown")
            params = self.model.num_parameters()
            print("Brain loaded! Parameters: " + str(params) + " | Trained to step: " + str(step))
        except Exception as e:
            print("Could not load model: " + str(e))
            print("Running in tool-only mode.")

    def _generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> str:
        """
        Generate text using Zara's language model, formatted to match the
        '### Question:\n...\n\n### Answer:\n...' structure used during training.
        Returns ONLY the answer portion, truncated at the first sign the model
        has started hallucinating a new turn (a fresh '### Question:' block),
        and stopped early on the EOS token rather than always running the full
        max_tokens length.
        """
        if self.model is None or self.enc is None:
            return ""
        try:
            formatted = "### Question:\n" + prompt.strip() + "\n\n### Answer:\n"
            tokens = self.enc.encode(formatted, allowed_special={"<|endoftext|>"})
            idx = torch.tensor([tokens[-self.model.cfg.context_length:]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model.generate(
                    idx, max_new_tokens=max_tokens, temperature=temperature, top_k=40,
                    eos_token_id=getattr(self.enc, "eot_token", None),
                )
            full_text = self.enc.decode(out[0].tolist())
            answer = full_text[len(formatted):]
            # Stop at the first sign the model started a new Q/A turn or hit EOS text.
            for stop_marker in ["### Question:", "<|endoftext|>"]:
                if stop_marker in answer:
                    answer = answer.split(stop_marker)[0]
            return answer.strip()
        except Exception as e:
            print("Generation failed: " + str(e))
            return ""

    def _is_degenerate(self, text: str) -> bool:
        """
        Heuristic check for outputs that are too short, empty, or clearly
        off-topic (e.g. leftover WikiText-style content) to be worth showing
        instead of a canned fallback response. This matters because a small,
        still-imperfectly-trained model can occasionally produce ramble that
        is strictly worse than admitting a direct answer isn't available.
        """
        if not text or len(text) < 15:
            return True
        # Repetition collapse (a common small-model failure mode) -- if a
        # single short phrase makes up a large fraction of the output, treat
        # it as degenerate rather than showing an obviously broken loop.
        words = text.split()
        if len(words) >= 8:
            most_common = max(set(words), key=words.count)
            if words.count(most_common) / len(words) > 0.35:
                return True
        return False

    def _format_tool_result(self, tool_name: str, result: Dict, user_question: str) -> str:
        """Format tool results into a readable response."""

        if tool_name == "threat_detector":
            if not result.get("is_threat"):
                return "I analyzed the content and found no threats. The text appears clean."
            summary = result.get("analysis_summary", "")
            threats = result.get("threats", [])
            response = "THREAT DETECTED\n\n"
            response += summary + "\n\n"
            if threats:
                response += "Breakdown:\n"
                for t in threats[:3]:
                    response += "- " + t["severity"] + ": " + t["description"] + "\n"
                    response += "  Action: " + t["recommendation"] + "\n\n"
            return response

        elif tool_name == "fraud_detector":
            risk_level = result.get("risk_level", "UNKNOWN")
            risk_score = result.get("risk_score", 0)
            action = result.get("action", "REVIEW")
            response = "FRAUD ANALYSIS RESULT\n\n"
            response += "Risk Level: " + risk_level + " (" + str(risk_score) + "/100)\n"
            response += "Recommended Action: " + action + "\n\n"
            rules = result.get("triggered_rules", [])
            if rules:
                response += "Risk Factors Detected:\n"
                for rule in rules[:5]:
                    response += "- " + rule["description"] + "\n"
                response += "\n"
            response += result.get("recommendation", "")
            return response

        elif tool_name == "security_auditor":
            score = result.get("security_score", 0)
            grade = result.get("security_grade", "F")
            total = result.get("total_vulnerabilities", 0)
            response = "SECURITY AUDIT REPORT\n\n"
            response += "Security Score: " + str(score) + "/100 (Grade: " + grade + ")\n"
            response += "Vulnerabilities Found: " + str(total) + "\n\n"
            vulns = result.get("vulnerabilities", [])
            if vulns:
                response += "Issues to Fix:\n"
                for v in vulns[:5]:
                    response += "\n[" + v["severity"] + "] " + v["name"] + "\n"
                    response += "Line(s): " + str([o["line_number"] for o in v["occurrences"]]) + "\n"
                    response += "Fix: " + v["fix"] + "\n"
            response += "\n" + result.get("summary", "")
            return response

        elif tool_name == "incident_responder":
            incident = result.get("incident_type", "")
            severity = result.get("severity", "")
            response = "INCIDENT RESPONSE PLAYBOOK: " + incident.upper() + "\n"
            response += "Severity: " + severity + "\n\n"
            response += "IMMEDIATE STEPS (Do these NOW):\n"
            for i, step in enumerate(result.get("immediate_steps", []), 1):
                response += str(i) + ". " + step + "\n"
            response += "\nSHORT-TERM STEPS (Next 24-48 hours):\n"
            for i, step in enumerate(result.get("short_term_steps", []), 1):
                response += str(i) + ". " + step + "\n"
            reg = result.get("regulatory_reporting", {})
            country_req = reg.get("requirements", "")
            if country_req:
                response += "\nREGULATORY REPORTING REQUIREMENT:\n"
                response += country_req + "\n"
            response += "\nRECOVERY STEPS:\n"
            for i, step in enumerate(result.get("recovery_steps", []), 1):
                response += str(i) + ". " + step + "\n"
            return response

        elif tool_name == "compliance_checker":
            score = result.get("compliance_score", 0)
            compliant = result.get("overall_compliant", False)
            response = "COMPLIANCE CHECK RESULTS\n\n"
            response += "Overall Compliance: " + ("COMPLIANT" if compliant else "NON-COMPLIANT") + "\n"
            response += "Compliance Score: " + str(score) + "%\n\n"
            for framework in result.get("framework_results", []):
                response += framework["framework_name"] + "\n"
                response += "Status: " + ("Compliant" if framework["compliant"] else "Non-Compliant") + "\n"
                gaps = framework.get("gaps", [])
                if gaps:
                    response += "Gaps to address:\n"
                    for gap in gaps[:3]:
                        response += "- " + gap["requirement"] + ": " + gap["description"] + "\n"
                response += "\n"
            actions = result.get("priority_actions", [])
            if actions:
                response += "Priority Actions:\n"
                for action in actions[:3]:
                    response += "- " + action + "\n"
            return response

        elif tool_name == "vulnerability_scanner":
            score = result.get("security_score", 0)
            grade = result.get("security_grade", "")
            response = "VULNERABILITY SCAN RESULTS\n\n"
            response += "Security Score: " + str(score) + "/100\n"
            response += "Grade: " + grade + "\n"
            response += "Checks Passed: " + str(result.get("passed_checks", 0)) + "/" + str(result.get("total_checks", 0)) + "\n\n"
            vulns = result.get("vulnerabilities", [])
            if vulns:
                response += "Vulnerabilities Found:\n\n"
                for v in vulns[:5]:
                    response += "[" + v["severity"] + "] " + v["check"] + "\n"
                    response += v["recommendation"] + "\n\n"
            else:
                response += "All security checks passed!\n"
            return response

        else:
            return str(result)

    def chat(self, user_input: str) -> str:
        """
        Main chat interface.
        User talks to Zara naturally.
        Zara decides which tool to use and responds. If no tool matches and
        a trained model checkpoint is loaded, Zara's language model is asked
        directly; if the model isn't loaded, or produces a degenerate result,
        a curated fallback response is used instead so the user is never
        shown obviously broken output.
        """
        self.conversation_history.append({"role": "user", "content": user_input})

        tool_name = self.registry.detect_intent(user_input)

        if tool_name:
            tool_input = self._extract_tool_input(tool_name, user_input)
            tool_result = self.registry.run_tool(tool_name, tool_input)

            if tool_result.get("success"):
                response = self._format_tool_result(tool_name, tool_result, user_input)
            else:
                response = "I encountered an issue running the " + tool_name + " tool: " + tool_result.get("error", "unknown error")
                if "required_fields" in tool_result or "example" in tool_result:
                    response += "\n\nTo use this tool, provide: " + str(tool_result.get("required_fields", []))
        else:
            model_response = self._generate(user_input) if self.model is not None else ""
            if model_response and not self._is_degenerate(model_response):
                response = model_response
            else:
                response = self._general_cybersecurity_response(user_input)

        self.conversation_history.append({"role": "zara", "content": response})
        return response

    def _extract_tool_input(self, tool_name: str, user_input: str) -> Dict[str, Any]:
        """Extract relevant input for each tool from user message."""
        text_lower = user_input.lower()

        country = "Nigeria"
        for c in ["nigeria", "kenya", "south africa", "ghana", "ethiopia", "tanzania"]:
            if c in text_lower:
                country = c.title()
                break

        if tool_name == "threat_detector":
            return {"text": user_input, "context": "user_query"}

        elif tool_name == "fraud_detector":
            return {
                "transaction": {
                    "amount": 50000,
                    "currency": "KES",
                    "new_device": "new device" in text_lower,
                    "sim_changed_recently": "sim swap" in text_lower,
                    "unusual_hour": "midnight" in text_lower or "3am" in text_lower,
                    "location_mismatch": "different location" in text_lower,
                }
            }

        elif tool_name == "security_auditor":
            return {"code": user_input, "language": "python"}

        elif tool_name == "incident_responder":
            incident_type = "data_breach"
            for incident in ["ransomware", "phishing", "sim_swap", "account_takeover", "data_breach"]:
                if incident.replace("_", " ") in text_lower or incident in text_lower:
                    incident_type = incident
                    break
            return {
                "incident_type": incident_type,
                "country": country,
                "details": user_input,
            }

        elif tool_name == "compliance_checker":
            org_type = "fintech" if any(w in text_lower for w in ["fintech", "bank", "payment", "financial"]) else "general"
            return {
                "country": country,
                "organization_type": org_type,
                "completed_controls": [],
            }

        elif tool_name == "vulnerability_scanner":
            return {
                "system_profile": {},
                "system_type": "fintech",
            }

        return {"text": user_input}

    def _general_cybersecurity_response(self, user_input: str) -> str:
        """Handle questions that don't match a specific tool."""
        responses = {
            "sim swap": "SIM swap fraud is one of the most devastating attacks in Africa. Attackers convince mobile operators to transfer your phone number to their SIM, then use your OTP codes to drain mobile money accounts. Prevention: Use authenticator apps instead of SMS for 2FA, add a PIN to your SIM with your operator, and set up account freeze alerts.",
            "phishing": "Phishing in Africa increasingly targets mobile money users in local languages. Never click links in unexpected messages claiming to be from your bank or mobile operator. Always verify through official app or website directly.",
            "ransomware": "Ransomware is hitting African organizations hard, especially healthcare. Defense: offline backups, network segmentation, staff training, and a tested incident response plan. If hit: isolate immediately, do not pay without expert advice.",
            "ndpr": "Nigeria NDPR requires: DPIA before sensitive data processing, DPO appointment, breach reporting to NITDA within 72 hours, data subject rights mechanisms, and annual audits. Non-compliance: fines up to 2% gross revenue or NGN 10 million.",
            "popia": "South Africa POPIA requires: Information Officer registration, 8 conditions for lawful processing, security safeguards, breach notification, and consent for direct marketing. Penalties up to ZAR 10 million or 10 years imprisonment.",
            "mpesa": "M-Pesa security best practices: Never share your PIN, verify sender before confirming transactions, use official Safaricom app only, set up transaction alerts, and report suspicious activity to 100 (Safaricom) immediately.",
        }

        user_lower = user_input.lower()
        for keyword, response in responses.items():
            if keyword in user_lower:
                return response

        return (
            "I am Zara, Africa's cybersecurity brain. I can help you with:\n\n"
            "- Threat detection and analysis\n"
            "- Fraud detection for mobile money transactions\n"
            "- Security code auditing\n"
            "- Incident response playbooks (ransomware, breach, SIM swap)\n"
            "- Compliance checking (NDPR, POPIA, Kenya DPA, CBN, CBK)\n"
            "- Vulnerability scanning\n\n"
            "Ask me about any cybersecurity concern and I will help you address it."
        )

    def run_tool_directly(self, tool_name: str, input_data: Dict) -> Dict:
        """Run a tool directly by name with specific input data."""
        return self.registry.run_tool(tool_name, input_data)

    def list_tools(self) -> list:
        """List all available tools."""
        return self.registry.list_tools()

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation history cleared.")