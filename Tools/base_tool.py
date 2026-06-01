# tools/base_tool.py
# Zara by Nexara - Base Tool Class
# Every tool Zara uses inherits from this

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("zara.tools")


class BaseTool(ABC):
    """
    Base class for all Zara tools.
    Every tool must implement:
        - name: unique tool identifier
        - description: what the tool does
        - run(): executes the tool
    """

    def __init__(self):
        self.name = "base_tool"
        self.description = "Base tool"
        self.category = "general"
        self.version = "1.0"

    @abstractmethod
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool.
        Every tool must implement this.

        Args:
            input_data: dict with tool-specific inputs

        Returns:
            dict with:
                success: bool
                result: the tool output
                error: error message if failed
                tool: tool name
                time_ms: execution time
        """
        pass

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper around run() that handles errors and timing.
        Zara calls this instead of run() directly.
        """
        start = time.time()
        try:
            logger.info("Running tool: " + self.name)
            result = self.run(input_data)
            elapsed = int((time.time() - start) * 1000)
            result["tool"] = self.name
            result["time_ms"] = elapsed
            return result
        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            logger.error("Tool " + self.name + " failed: " + str(e))
            return {
                "success": False,
                "error": str(e),
                "tool": self.name,
                "time_ms": elapsed,
                "result": None,
            }

    def info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
        }