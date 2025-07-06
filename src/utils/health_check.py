"""
System health monitoring utilities for the DBC application.

This module provides comprehensive health checks for all system components
including vector database, LLM services, and file processing capabilities.
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime
import psutil
import os

from src.utils.logger import get_logger
from src.utils.config import get_settings

logger = get_logger(__name__)


class HealthChecker:
    """
    Comprehensive health monitoring for DBC application components.
    
    Monitors system resources, dependencies, and service availability.
    """
    
    def __init__(self):
        """Initialize health checker."""
        self.settings = get_settings()
        self.checks = {}
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """
        Check system resource availability.
        
        Returns:
            Dict[str, Any]: System resource status
        """
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(os.getcwd())
            
            return {
                "status": "healthy",
                "memory": {
                    "total_gb": round(memory.total / (1024**3), 2),
                    "available_gb": round(memory.available / (1024**3), 2),
                    "percent_used": memory.percent
                },
                "disk": {
                    "total_gb": round(disk.total / (1024**3), 2),
                    "free_gb": round(disk.free / (1024**3), 2),
                    "percent_used": round((disk.used / disk.total) * 100, 2)
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def check_dependencies(self) -> Dict[str, Any]:
        """
        Check availability of critical dependencies.
        
        Returns:
            Dict[str, Any]: Dependency status
        """
        dependencies = {}
        
        # Check core dependencies
        try:
            import openai
            dependencies["openai"] = {"status": "available", "version": openai.__version__}
        except ImportError as e:
            dependencies["openai"] = {"status": "missing", "error": str(e)}
        
        try:
            import chromadb
            dependencies["chromadb"] = {"status": "available", "version": chromadb.__version__}
        except ImportError as e:
            dependencies["chromadb"] = {"status": "missing", "error": str(e)}
        
        try:
            import langchain_community
            dependencies["langchain_community"] = {"status": "available"}
        except ImportError as e:
            dependencies["langchain_community"] = {"status": "missing", "error": str(e)}
        
        # Check PDF processing
        try:
            from unstructured.partition.pdf import partition_pdf
            dependencies["pdf_processing"] = {"status": "available"}
        except ImportError as e:
            dependencies["pdf_processing"] = {"status": "missing", "error": str(e)}
        
        # Check file processing
        file_processors = ["pypdf", "ebooklib", "python-docx", "beautifulsoup4"]
        for processor in file_processors:
            try:
                __import__(processor.replace("-", "_"))
                dependencies[processor] = {"status": "available"}
            except ImportError as e:
                dependencies[processor] = {"status": "missing", "error": str(e)}
        
        return dependencies
    
    async def check_file_permissions(self) -> Dict[str, Any]:
        """
        Check file system permissions for data directories.
        
        Returns:
            Dict[str, Any]: File permission status
        """
        permissions = {}
        
        # Check data directories
        data_dirs = [
            "data/books",
            "data/chroma_db", 
            "data/users"
        ]
        
        for dir_path in data_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                
                # Test write permission
                test_file = os.path.join(dir_path, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                
                permissions[dir_path] = {
                    "status": "accessible",
                    "readable": os.access(dir_path, os.R_OK),
                    "writable": os.access(dir_path, os.W_OK)
                }
            except Exception as e:
                permissions[dir_path] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return permissions
    
    async def run_comprehensive_check(self) -> Dict[str, Any]:
        """
        Run all health checks and return comprehensive status.
        
        Returns:
            Dict[str, Any]: Complete health status
        """
        logger.info("Running comprehensive health check...")
        
        # Run all checks concurrently
        results = await asyncio.gather(
            self.check_system_resources(),
            self.check_dependencies(),
            self.check_file_permissions(),
            return_exceptions=True
        )
        
        system_resources, dependencies, file_permissions = results[:3]
        
        # Calculate overall health
        issues = []
        warnings = []
        
        # Check for critical issues
        if isinstance(dependencies, dict):
            critical_deps = ["openai", "chromadb", "pdf_processing"]
            for dep in critical_deps:
                if dep in dependencies and dependencies[dep]["status"] != "available":
                    issues.append(f"Critical dependency missing: {dep}")
        
        # Check resource constraints
        if isinstance(system_resources, dict) and system_resources.get("status") == "healthy":
            memory = system_resources.get("memory", {})
            if memory.get("percent_used", 0) > 90:
                warnings.append("High memory usage detected")
            
            disk = system_resources.get("disk", {})
            if disk.get("percent_used", 0) > 95:
                issues.append("Disk space critically low")
        
        overall_status = "healthy"
        if issues:
            overall_status = "unhealthy"
        elif warnings:
            overall_status = "warning"
        
        return {
            "overall_status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues,
            "warnings": warnings,
            "details": {
                "system_resources": system_resources,
                "dependencies": dependencies,
                "file_permissions": file_permissions
            },
            "recommendations": self._generate_recommendations(issues, warnings)
        }
    
    def _generate_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
        """
        Generate actionable recommendations based on detected issues.
        
        Args:
            issues: List of critical issues
            warnings: List of warnings
            
        Returns:
            List[str]: Actionable recommendations
        """
        recommendations = []
        
        for issue in issues:
            if "Critical dependency missing" in issue:
                if "pdf_processing" in issue:
                    recommendations.append("Install PDF dependencies: pip install 'unstructured[pdf]'")
                elif "openai" in issue:
                    recommendations.append("Install OpenAI: pip install openai")
                elif "chromadb" in issue:
                    recommendations.append("Install ChromaDB: pip install chromadb")
            elif "Disk space critically low" in issue:
                recommendations.append("Free up disk space or increase storage capacity")
        
        for warning in warnings:
            if "High memory usage" in warning:
                recommendations.append("Consider increasing available memory or optimizing memory usage")
        
        if not recommendations:
            recommendations.append("System is healthy - no action required")
        
        return recommendations


# Global health checker instance
_health_checker = None


def get_health_checker() -> HealthChecker:
    """
    Get the global health checker instance.
    
    Returns:
        HealthChecker: Health checker instance
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker