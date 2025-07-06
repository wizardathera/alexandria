# Frontend Compatibility Report

## Expected Data Structures

### Content Relationships Response
```json
{
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
      "related_semantic_tags": [
        "artificial intelligence",
        "machine learning"
      ]
    }
  ],
  "total_found": 1,
  "returned": 1
}
```

### Graph Data Response
```json
{
  "nodes": [
    {
      "id": "12345678-1234-1234-1234-123456789abc",
      "title": "Introduction to Machine Learning",
      "author": "John Doe",
      "content_type": "book",
      "module_type": "library",
      "topics": [
        "machine learning",
        "artificial intelligence"
      ],
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
      "human_verified": false,
      "context": "Semantic similarity"
    }
  ],
  "stats": {
    "total_nodes": 1,
    "total_edges": 1,
    "average_connections": 1.0,
    "content_types": [
      "book"
    ],
    "module_types": [
      "library"
    ],
    "relationship_types": [
      "similar"
    ],
    "average_strength": 0.85
  }
}
```

### Content List Response
```json
{
  "content": [
    {
      "content_id": "12345678-1234-1234-1234-123456789abc",
      "title": "Introduction to Machine Learning",
      "author": "John Doe",
      "content_type": "book",
      "module_type": "library",
      "topics": [
        "machine learning",
        "artificial intelligence"
      ],
      "created_at": "2024-01-01T12:00:00",
      "processing_status": "completed"
    }
  ],
  "pagination": {
    "total": 1,
    "limit": 20,
    "offset": 0,
    "has_more": false
  }
}
```

### Discovery Response
```json
{
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
      "related_semantic_tags": [
        "deep learning",
        "neural networks"
      ]
    }
  ],
  "total_candidates_analyzed": 10,
  "relationships_found": 1,
  "discovery_parameters": {
    "max_relationships": 20,
    "min_confidence": 0.5
  }
}
```

## Validation Requirements

- All numeric scores (strength, confidence, weight) must be floats between 0 and 1
- All IDs must be valid UUIDs
- All arrays must be properly typed
- Response times should be under 2 seconds for good UX
- Error responses should include helpful messages
