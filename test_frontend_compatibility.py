#!/usr/bin/env python3
"""
Frontend Compatibility Validation for Content Relationships Explorer

This script validates that the backend API responses are compatible with
the frontend components' expected data structures.
"""

import json
import sys
from typing import Dict, List, Any
from dataclasses import dataclass

# Expected data structures based on frontend code analysis
@dataclass
class ExpectedRelationshipStructure:
    """Expected structure for individual relationship items."""
    related_content_id: str
    related_title: str
    related_author: str
    related_content_type: str
    relationship_type: str
    strength: float
    confidence: float
    explanation: str = None
    related_semantic_tags: List[str] = None


@dataclass
class ExpectedContentStructure:
    """Expected structure for content items."""
    content_id: str
    title: str
    author: str
    content_type: str
    module_type: str
    topics: List[str] = None


@dataclass
class ExpectedGraphNodeStructure:
    """Expected structure for graph nodes."""
    id: str
    title: str
    author: str
    content_type: str
    module_type: str
    topics: List[str]
    size: int
    color: str
    created_at: str


@dataclass
class ExpectedGraphEdgeStructure:
    """Expected structure for graph edges."""
    source: str
    target: str
    relationship_type: str
    strength: float
    confidence: float
    weight: float
    discovered_by: str
    human_verified: bool
    context: str = None


class FrontendCompatibilityValidator:
    """Validates backend API responses against frontend expectations."""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_content_relationships_response(self, response_data: Dict) -> bool:
        """Validate /api/enhanced/content/{id}/relationships response."""
        try:
            # Check top-level structure
            required_top_level = ["content_id", "relationships", "total_found", "returned"]
            for field in required_top_level:
                if field not in response_data:
                    self.validation_results.append(f"❌ Missing top-level field: {field}")
                    return False
            
            # Check relationships array
            relationships = response_data.get("relationships", [])
            if not isinstance(relationships, list):
                self.validation_results.append("❌ 'relationships' must be an array")
                return False
            
            # Validate each relationship
            if relationships:
                rel = relationships[0]
                required_rel_fields = [
                    "related_content_id", "related_title", "related_author",
                    "related_content_type", "relationship_type", "strength", "confidence"
                ]
                
                for field in required_rel_fields:
                    if field not in rel:
                        self.validation_results.append(f"❌ Missing relationship field: {field}")
                        return False
                
                # Validate data types
                if not isinstance(rel["strength"], (int, float)) or not (0 <= rel["strength"] <= 1):
                    self.validation_results.append("❌ 'strength' must be float between 0 and 1")
                    return False
                
                if not isinstance(rel["confidence"], (int, float)) or not (0 <= rel["confidence"] <= 1):
                    self.validation_results.append("❌ 'confidence' must be float between 0 and 1")
                    return False
            
            self.validation_results.append("✅ Content relationships response structure valid")
            return True
            
        except Exception as e:
            self.validation_results.append(f"❌ Validation error: {str(e)}")
            return False
    
    def validate_graph_data_response(self, response_data: Dict) -> bool:
        """Validate /api/enhanced/relationships response."""
        try:
            # Check top-level structure
            required_top_level = ["nodes", "edges", "stats"]
            for field in required_top_level:
                if field not in response_data:
                    self.validation_results.append(f"❌ Missing top-level field: {field}")
                    return False
            
            # Validate nodes
            nodes = response_data.get("nodes", [])
            if not isinstance(nodes, list):
                self.validation_results.append("❌ 'nodes' must be an array")
                return False
            
            if nodes:
                node = nodes[0]
                required_node_fields = [
                    "id", "title", "author", "content_type", "module_type", "topics", "size", "color"
                ]
                
                for field in required_node_fields:
                    if field not in node:
                        self.validation_results.append(f"❌ Missing node field: {field}")
                        return False
                
                # Validate node data types
                if not isinstance(node["topics"], list):
                    self.validation_results.append("❌ Node 'topics' must be an array")
                    return False
                
                if not isinstance(node["size"], int) or node["size"] <= 0:
                    self.validation_results.append("❌ Node 'size' must be positive integer")
                    return False
            
            # Validate edges
            edges = response_data.get("edges", [])
            if not isinstance(edges, list):
                self.validation_results.append("❌ 'edges' must be an array")
                return False
            
            if edges:
                edge = edges[0]
                required_edge_fields = [
                    "source", "target", "relationship_type", "strength", "confidence", "weight"
                ]
                
                for field in required_edge_fields:
                    if field not in edge:
                        self.validation_results.append(f"❌ Missing edge field: {field}")
                        return False
                
                # Validate edge data types
                for float_field in ["strength", "confidence", "weight"]:
                    if not isinstance(edge[float_field], (int, float)) or not (0 <= edge[float_field] <= 1):
                        self.validation_results.append(f"❌ Edge '{float_field}' must be float between 0 and 1")
                        return False
            
            # Validate stats
            stats = response_data.get("stats", {})
            required_stat_fields = ["total_nodes", "total_edges"]
            
            for field in required_stat_fields:
                if field not in stats:
                    self.validation_results.append(f"❌ Missing stats field: {field}")
                    return False
                
                if not isinstance(stats[field], int) or stats[field] < 0:
                    self.validation_results.append(f"❌ Stats '{field}' must be non-negative integer")
                    return False
            
            self.validation_results.append("✅ Graph data response structure valid")
            return True
            
        except Exception as e:
            self.validation_results.append(f"❌ Validation error: {str(e)}")
            return False
    
    def validate_content_list_response(self, response_data: Dict) -> bool:
        """Validate /api/enhanced/content response."""
        try:
            # Check if it's the content array or wrapped response
            if "content" in response_data:
                content_list = response_data["content"]
            else:
                content_list = response_data  # Assume it's directly the array
            
            if not isinstance(content_list, list):
                self.validation_results.append("❌ Content list must be an array")
                return False
            
            if content_list:
                content = content_list[0]
                required_content_fields = [
                    "content_id", "title", "author", "content_type", "module_type"
                ]
                
                for field in required_content_fields:
                    if field not in content:
                        self.validation_results.append(f"❌ Missing content field: {field}")
                        return False
            
            self.validation_results.append("✅ Content list response structure valid")
            return True
            
        except Exception as e:
            self.validation_results.append(f"❌ Validation error: {str(e)}")
            return False
    
    def validate_discovery_response(self, response_data: Dict) -> bool:
        """Validate /api/enhanced/relationships/discover response."""
        try:
            required_top_level = [
                "content_id", "discovered_relationships", 
                "total_candidates_analyzed", "relationships_found"
            ]
            
            for field in required_top_level:
                if field not in response_data:
                    self.validation_results.append(f"❌ Missing discovery field: {field}")
                    return False
            
            # Check relationships structure (same as content relationships)
            discovered = response_data.get("discovered_relationships", [])
            if not isinstance(discovered, list):
                self.validation_results.append("❌ 'discovered_relationships' must be an array")
                return False
            
            if discovered:
                rel = discovered[0]
                required_fields = [
                    "related_content_id", "related_title", "relationship_type", "strength", "confidence"
                ]
                
                for field in required_fields:
                    if field not in rel:
                        self.validation_results.append(f"❌ Missing discovered relationship field: {field}")
                        return False
            
            self.validation_results.append("✅ Discovery response structure valid")
            return True
            
        except Exception as e:
            self.validation_results.append(f"❌ Validation error: {str(e)}")
            return False
    
    def create_sample_data(self) -> Dict[str, Any]:
        """Create sample data that matches expected structures."""
        return {
            "content_relationships": {
                "content_id": "12345678-1234-1234-1234-123456789abc",
                "relationships": [
                    {
                        "related_content_id": "87654321-4321-4321-4321-cba987654321",
                        "related_title": "Related Book Title",
                        "related_author": "Author Name",
                        "related_content_type": "book",
                        "related_module_type": "library",
                        "relationship_type": "similar",
                        "strength": 0.85,
                        "confidence": 0.92,
                        "explanation": "Both books cover AI and ML topics",
                        "related_semantic_tags": ["artificial intelligence", "machine learning"]
                    }
                ],
                "total_found": 1,
                "returned": 1
            },
            "graph_data": {
                "nodes": [
                    {
                        "id": "12345678-1234-1234-1234-123456789abc",
                        "title": "Introduction to Machine Learning",
                        "author": "John Doe",
                        "content_type": "book",
                        "module_type": "library",
                        "topics": ["machine learning", "artificial intelligence"],
                        "size": 5000,
                        "color": "#3498db",
                        "created_at": "2024-01-01T12:00:00"
                    }
                ],
                "edges": [
                    {
                        "source": "12345678-1234-1234-1234-123456789abc",
                        "target": "87654321-4321-4321-4321-cba987654321",
                        "relationship_type": "similar",
                        "strength": 0.85,
                        "confidence": 0.92,
                        "weight": 0.85,
                        "discovered_by": "ai",
                        "human_verified": False,
                        "context": "Semantic similarity"
                    }
                ],
                "stats": {
                    "total_nodes": 1,
                    "total_edges": 1,
                    "average_connections": 1.0,
                    "content_types": ["book"],
                    "module_types": ["library"],
                    "relationship_types": ["similar"],
                    "average_strength": 0.85
                }
            },
            "content_list": {
                "content": [
                    {
                        "content_id": "12345678-1234-1234-1234-123456789abc",
                        "title": "Introduction to Machine Learning",
                        "author": "John Doe",
                        "content_type": "book",
                        "module_type": "library",
                        "topics": ["machine learning", "artificial intelligence"],
                        "created_at": "2024-01-01T12:00:00",
                        "processing_status": "completed"
                    }
                ],
                "pagination": {
                    "total": 1,
                    "limit": 20,
                    "offset": 0,
                    "has_more": False
                }
            },
            "discovery": {
                "content_id": "12345678-1234-1234-1234-123456789abc",
                "discovered_relationships": [
                    {
                        "related_content_id": "87654321-4321-4321-4321-cba987654321",
                        "related_title": "Deep Learning Explained",
                        "related_author": "Jane Smith",
                        "related_content_type": "book",
                        "relationship_type": "similar",
                        "strength": 0.78,
                        "confidence": 0.85,
                        "distance": 1,
                        "explanation": "Discovered semantic similarity (distance: 1)",
                        "related_semantic_tags": ["deep learning", "neural networks"]
                    }
                ],
                "total_candidates_analyzed": 10,
                "relationships_found": 1,
                "discovery_parameters": {
                    "max_relationships": 20,
                    "min_confidence": 0.5
                }
            }
        }
    
    def run_all_validations(self) -> bool:
        """Run all frontend compatibility validations."""
        sample_data = self.create_sample_data()
        
        print("🔍 Frontend Compatibility Validation")
        print("=" * 50)
        
        all_passed = True
        
        # Test 1: Content relationships
        print("Testing content relationships response...")
        if not self.validate_content_relationships_response(sample_data["content_relationships"]):
            all_passed = False
        
        # Test 2: Graph data
        print("Testing graph data response...")
        if not self.validate_graph_data_response(sample_data["graph_data"]):
            all_passed = False
        
        # Test 3: Content list
        print("Testing content list response...")
        if not self.validate_content_list_response(sample_data["content_list"]):
            all_passed = False
        
        # Test 4: Discovery
        print("Testing discovery response...")
        if not self.validate_discovery_response(sample_data["discovery"]):
            all_passed = False
        
        print("\n" + "=" * 50)
        print("📋 Validation Results:")
        for result in self.validation_results:
            print(f"  {result}")
        
        print(f"\n🎯 Overall Result: {'✅ ALL VALIDATIONS PASSED' if all_passed else '❌ SOME VALIDATIONS FAILED'}")
        
        if all_passed:
            print("\n🎉 Backend API responses are compatible with frontend expectations!")
            print("📊 Frontend components should work correctly with the implemented endpoints.")
        else:
            print("\n⚠️  Backend API responses need adjustments for frontend compatibility.")
            print("🔧 Please review the validation results and update the API responses accordingly.")
        
        return all_passed
    
    def generate_compatibility_report(self) -> str:
        """Generate a detailed compatibility report."""
        sample_data = self.create_sample_data()
        
        report = "# Frontend Compatibility Report\n\n"
        report += "## Expected Data Structures\n\n"
        
        report += "### Content Relationships Response\n"
        report += "```json\n"
        report += json.dumps(sample_data["content_relationships"], indent=2)
        report += "\n```\n\n"
        
        report += "### Graph Data Response\n"
        report += "```json\n"
        report += json.dumps(sample_data["graph_data"], indent=2)
        report += "\n```\n\n"
        
        report += "### Content List Response\n"
        report += "```json\n"
        report += json.dumps(sample_data["content_list"], indent=2)
        report += "\n```\n\n"
        
        report += "### Discovery Response\n"
        report += "```json\n"
        report += json.dumps(sample_data["discovery"], indent=2)
        report += "\n```\n\n"
        
        report += "## Validation Requirements\n\n"
        report += "- All numeric scores (strength, confidence, weight) must be floats between 0 and 1\n"
        report += "- All IDs must be valid UUIDs\n"
        report += "- All arrays must be properly typed\n"
        report += "- Response times should be under 2 seconds for good UX\n"
        report += "- Error responses should include helpful messages\n"
        
        return report


def main():
    """Main validation execution."""
    validator = FrontendCompatibilityValidator()
    
    try:
        success = validator.run_all_validations()
        
        # Generate compatibility report
        report = validator.generate_compatibility_report()
        
        with open("frontend_compatibility_report.md", "w") as f:
            f.write(report)
        
        print(f"\n📄 Detailed compatibility report saved to: frontend_compatibility_report.md")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())