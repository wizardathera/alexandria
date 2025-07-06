#!/usr/bin/env python3
"""
Test script for Content Relationships Explorer Backend

This script tests the new relationship API endpoints to ensure they work
correctly and return valid graph data structures for frontend visualization.
"""

import asyncio
import aiohttp
import json
import time
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30


@dataclass
class TestResult:
    """Result of a test case."""
    test_name: str
    passed: bool
    message: str
    response_time: float
    response_data: Optional[Dict] = None


class RelationshipEndpointTester:
    """Comprehensive tester for relationship API endpoints."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.results: List[TestResult] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def test_health_check(self) -> TestResult:
        """Test that the API is running."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/api/enhanced/health") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return TestResult(
                        test_name="Health Check",
                        passed=True,
                        message=f"API is healthy: {data.get('status')}",
                        response_time=response_time,
                        response_data=data
                    )
                else:
                    return TestResult(
                        test_name="Health Check",
                        passed=False,
                        message=f"Health check failed with status {response.status}",
                        response_time=response_time
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="Health Check",
                passed=False,
                message=f"Health check error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def test_content_list(self) -> TestResult:
        """Test listing content items."""
        start_time = time.time()
        
        try:
            async with self.session.get(f"{self.base_url}/api/enhanced/content") as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    content_items = data.get("content", [])
                    
                    return TestResult(
                        test_name="Content List",
                        passed=True,
                        message=f"Found {len(content_items)} content items",
                        response_time=response_time,
                        response_data=data
                    )
                else:
                    return TestResult(
                        test_name="Content List",
                        passed=False,
                        message=f"Content list failed with status {response.status}",
                        response_time=response_time
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="Content List",
                passed=False,
                message=f"Content list error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def test_content_relationships(self, content_id: str) -> TestResult:
        """Test getting relationships for a specific content item."""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}/api/enhanced/content/{content_id}/relationships"
            async with self.session.get(url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    relationships = data.get("relationships", [])
                    
                    # Validate response structure
                    required_fields = ["content_id", "relationships", "total_found", "returned"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        return TestResult(
                            test_name=f"Content Relationships ({content_id})",
                            passed=False,
                            message=f"Missing required fields: {missing_fields}",
                            response_time=response_time,
                            response_data=data
                        )
                    
                    # Validate relationship structure
                    if relationships:
                        rel = relationships[0]
                        rel_required_fields = [
                            "related_content_id", "related_title", "relationship_type",
                            "strength", "confidence"
                        ]
                        rel_missing_fields = [field for field in rel_required_fields if field not in rel]
                        
                        if rel_missing_fields:
                            return TestResult(
                                test_name=f"Content Relationships ({content_id})",
                                passed=False,
                                message=f"Relationship missing fields: {rel_missing_fields}",
                                response_time=response_time,
                                response_data=data
                            )
                    
                    return TestResult(
                        test_name=f"Content Relationships ({content_id})",
                        passed=True,
                        message=f"Found {len(relationships)} relationships",
                        response_time=response_time,
                        response_data=data
                    )
                    
                elif response.status == 404:
                    return TestResult(
                        test_name=f"Content Relationships ({content_id})",
                        passed=True,  # 404 is expected for non-existent content
                        message="Content not found (expected for test)",
                        response_time=response_time
                    )
                else:
                    return TestResult(
                        test_name=f"Content Relationships ({content_id})",
                        passed=False,
                        message=f"Request failed with status {response.status}",
                        response_time=response_time
                    )
                    
        except Exception as e:
            return TestResult(
                test_name=f"Content Relationships ({content_id})",
                passed=False,
                message=f"Request error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def test_relationships_graph(self, limit: int = 100) -> TestResult:
        """Test getting graph data for relationships."""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}/api/enhanced/relationships"
            params = {"limit": limit, "min_strength": 0.1}
            
            async with self.session.get(url, params=params) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate graph structure
                    required_fields = ["nodes", "edges", "stats"]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        return TestResult(
                            test_name="Relationships Graph",
                            passed=False,
                            message=f"Missing required fields: {missing_fields}",
                            response_time=response_time,
                            response_data=data
                        )
                    
                    nodes = data.get("nodes", [])
                    edges = data.get("edges", [])
                    stats = data.get("stats", {})
                    
                    # Validate node structure
                    if nodes:
                        node = nodes[0]
                        node_required_fields = ["id", "title", "content_type", "module_type"]
                        node_missing_fields = [field for field in node_required_fields if field not in node]
                        
                        if node_missing_fields:
                            return TestResult(
                                test_name="Relationships Graph",
                                passed=False,
                                message=f"Node missing fields: {node_missing_fields}",
                                response_time=response_time,
                                response_data=data
                            )
                    
                    # Validate edge structure
                    if edges:
                        edge = edges[0]
                        edge_required_fields = ["source", "target", "relationship_type", "strength", "confidence"]
                        edge_missing_fields = [field for field in edge_required_fields if field not in edge]
                        
                        if edge_missing_fields:
                            return TestResult(
                                test_name="Relationships Graph",
                                passed=False,
                                message=f"Edge missing fields: {edge_missing_fields}",
                                response_time=response_time,
                                response_data=data
                            )
                    
                    return TestResult(
                        test_name="Relationships Graph",
                        passed=True,
                        message=f"Graph with {len(nodes)} nodes, {len(edges)} edges",
                        response_time=response_time,
                        response_data=data
                    )
                else:
                    return TestResult(
                        test_name="Relationships Graph",
                        passed=False,
                        message=f"Request failed with status {response.status}",
                        response_time=response_time
                    )
                    
        except Exception as e:
            return TestResult(
                test_name="Relationships Graph",
                passed=False,
                message=f"Request error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def test_relationship_discovery(self, content_id: str) -> TestResult:
        """Test AI-powered relationship discovery."""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}/api/enhanced/relationships/discover"
            params = {
                "content_id": content_id,
                "max_relationships": 10,
                "min_confidence": 0.5
            }
            
            async with self.session.post(url, params=params) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    
                    # Validate discovery response structure
                    required_fields = [
                        "content_id", "discovered_relationships", 
                        "total_candidates_analyzed", "relationships_found"
                    ]
                    missing_fields = [field for field in required_fields if field not in data]
                    
                    if missing_fields:
                        return TestResult(
                            test_name=f"Relationship Discovery ({content_id})",
                            passed=False,
                            message=f"Missing required fields: {missing_fields}",
                            response_time=response_time,
                            response_data=data
                        )
                    
                    discovered = data.get("discovered_relationships", [])
                    
                    return TestResult(
                        test_name=f"Relationship Discovery ({content_id})",
                        passed=True,
                        message=f"Discovered {len(discovered)} relationships",
                        response_time=response_time,
                        response_data=data
                    )
                    
                elif response.status == 404:
                    return TestResult(
                        test_name=f"Relationship Discovery ({content_id})",
                        passed=True,  # 404 is expected for non-existent content
                        message="Content not found (expected for test)",
                        response_time=response_time
                    )
                else:
                    return TestResult(
                        test_name=f"Relationship Discovery ({content_id})",
                        passed=False,
                        message=f"Request failed with status {response.status}",
                        response_time=response_time
                    )
                    
        except Exception as e:
            return TestResult(
                test_name=f"Relationship Discovery ({content_id})",
                passed=False,
                message=f"Request error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def test_performance_with_large_dataset(self, num_requests: int = 5) -> TestResult:
        """Test performance with multiple concurrent requests."""
        start_time = time.time()
        
        try:
            # Create multiple concurrent requests to relationships graph endpoint
            tasks = []
            for i in range(num_requests):
                task = self.session.get(f"{self.base_url}/api/enhanced/relationships?limit=100")
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            response_time = time.time() - start_time
            
            successful_requests = 0
            failed_requests = 0
            
            for response in responses:
                if isinstance(response, Exception):
                    failed_requests += 1
                elif hasattr(response, 'status') and response.status == 200:
                    successful_requests += 1
                    response.close()
                else:
                    failed_requests += 1
                    if hasattr(response, 'close'):
                        response.close()
            
            avg_response_time = response_time / num_requests
            passed = successful_requests >= num_requests * 0.8  # 80% success rate
            
            return TestResult(
                test_name=f"Performance Test ({num_requests} requests)",
                passed=passed,
                message=f"Success: {successful_requests}/{num_requests}, Avg time: {avg_response_time:.2f}s",
                response_time=response_time
            )
            
        except Exception as e:
            return TestResult(
                test_name=f"Performance Test ({num_requests} requests)",
                passed=False,
                message=f"Performance test error: {str(e)}",
                response_time=time.time() - start_time
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all relationship endpoint tests."""
        print("🧪 Starting Content Relationships Explorer Backend Tests...")
        print("=" * 60)
        
        # Test 1: Health check
        result = await self.test_health_check()
        self.results.append(result)
        self._print_result(result)
        
        # Test 2: Content list (to get actual content IDs)
        result = await self.test_content_list()
        self.results.append(result)
        self._print_result(result)
        
        content_ids = []
        if result.passed and result.response_data:
            content_items = result.response_data.get("content", [])
            content_ids = [item["content_id"] for item in content_items[:3]]  # Test with first 3
        
        # Test 3: Content relationships (use actual content ID if available)
        test_content_id = content_ids[0] if content_ids else "test-content-id"
        result = await self.test_content_relationships(test_content_id)
        self.results.append(result)
        self._print_result(result)
        
        # Test 4: Relationships graph
        result = await self.test_relationships_graph()
        self.results.append(result)
        self._print_result(result)
        
        # Test 5: Relationship discovery
        result = await self.test_relationship_discovery(test_content_id)
        self.results.append(result)
        self._print_result(result)
        
        # Test 6: Performance test
        result = await self.test_performance_with_large_dataset()
        self.results.append(result)
        self._print_result(result)
        
        return self.results
    
    def _print_result(self, result: TestResult):
        """Print a formatted test result."""
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"{status} {result.test_name}")
        print(f"    Time: {result.response_time:.3f}s")
        print(f"    Message: {result.message}")
        if not result.passed and result.response_data:
            print(f"    Data: {json.dumps(result.response_data, indent=2)[:200]}...")
        print()
    
    def print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("=" * 60)
        print(f"📊 TEST SUMMARY: {passed}/{total} tests passed")
        print(f"🕒 Total execution time: {sum(r.response_time for r in self.results):.2f}s")
        
        if passed == total:
            print("🎉 All tests passed! Relationship endpoints are working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            failed_tests = [r for r in self.results if not r.passed]
            print(f"Failed tests: {[r.test_name for r in failed_tests]}")
        
        # Performance analysis
        avg_response_time = sum(r.response_time for r in self.results) / len(self.results)
        max_response_time = max(r.response_time for r in self.results)
        
        print(f"🚀 Performance: Avg {avg_response_time:.3f}s, Max {max_response_time:.3f}s")
        
        if max_response_time > 2.0:
            print("⚠️  Some endpoints exceeded 2s response time target")
        else:
            print("✅ All endpoints met <2s response time requirement")


async def main():
    """Main test execution function."""
    try:
        async with RelationshipEndpointTester() as tester:
            await tester.run_all_tests()
            tester.print_summary()
            
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted")
    except Exception as e:
        print(f"❌ Test execution failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())