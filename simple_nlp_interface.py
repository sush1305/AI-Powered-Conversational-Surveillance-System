#!/usr/bin/env python3
"""
Simplified NLP Interface for AI Surveillance System
This version works without LLM dependencies and provides basic query functionality
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import json
import re

logger = logging.getLogger(__name__)

class SimpleSemanticSearch:
    """Dummy semantic search for compatibility"""
    def update_from_database(self):
        """Dummy method for compatibility"""
        pass
    
    def search(self, query, limit=10):
        """Dummy search method"""
        return []

class SimpleNLPInterface:
    """Simplified NLP interface that works without LLM dependencies"""
    
    def __init__(self):
        self.llm_available = False
        self.semantic_search = SimpleSemanticSearch()
        logger.info("Simple NLP interface initialized (LLM-free mode)")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query using simple pattern matching"""
        try:
            query_lower = query.lower().strip()
            
            # Parse time-related queries
            time_info = self._parse_time_query(query_lower)
            
            # Parse event type queries
            event_types = self._parse_event_types(query_lower)
            
            # Parse location queries
            location_info = self._parse_location(query_lower)
            
            # Generate response
            response = {
                'status': 'success',
                'query': query,
                'parsed_info': {
                    'event_types': event_types,
                    'time_range': time_info,
                    'location': location_info,
                    'keywords': self._extract_keywords(query_lower)
                },
                'message': self._generate_response_message(query_lower, event_types, time_info),
                'suggestions': self._get_suggestions(query_lower)
            }
            
            logger.info(f"Processed query successfully: {query[:50]}...")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'status': 'error',
                'message': f'Error processing query: {str(e)}',
                'suggestions': ['Try asking about recent events', 'Check the web dashboard for detailed analytics']
            }
    
    def _parse_time_query(self, query: str) -> Dict[str, Any]:
        """Parse time-related information from query"""
        now = datetime.now()
        
        # Time patterns
        if 'today' in query:
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = now
        elif 'yesterday' in query:
            yesterday = now - timedelta(days=1)
            start_time = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        elif 'last week' in query or 'past week' in query:
            start_time = now - timedelta(days=7)
            end_time = now
        elif 'last hour' in query or 'past hour' in query:
            start_time = now - timedelta(hours=1)
            end_time = now
        elif 'last 24 hours' in query:
            start_time = now - timedelta(hours=24)
            end_time = now
        else:
            # Default to last 24 hours
            start_time = now - timedelta(hours=24)
            end_time = now
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'description': self._describe_time_range(start_time, end_time)
        }
    
    def _parse_event_types(self, query: str) -> List[str]:
        """Parse event types from query"""
        event_types = []
        
        # Event type patterns
        event_patterns = {
            'fall': ['fall', 'falling', 'fell', 'fallen'],
            'violence': ['violence', 'violent', 'fight', 'fighting', 'attack'],
            'crash': ['crash', 'collision', 'accident', 'hit'],
            'robbery': ['robbery', 'theft', 'steal', 'burglar', 'break'],
            'suspicious_activity': ['suspicious', 'unusual', 'strange', 'weird', 'motion']
        }
        
        for event_type, patterns in event_patterns.items():
            if any(pattern in query for pattern in patterns):
                event_types.append(event_type)
        
        # If no specific types found, include all
        if not event_types:
            event_types = ['all']
        
        return event_types
    
    def _parse_location(self, query: str) -> str:
        """Parse location information from query"""
        location_patterns = [
            'entrance', 'door', 'gate', 'lobby', 'hallway', 'corridor',
            'room', 'office', 'kitchen', 'bedroom', 'bathroom',
            'outside', 'garden', 'parking', 'garage', 'basement'
        ]
        
        for location in location_patterns:
            if location in query:
                return location
        
        return 'any'
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'when', 'where', 'why', 'how', 'show', 'me', 'all', 'any'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to 10 keywords
    
    def _generate_response_message(self, query: str, event_types: List[str], time_info: Dict[str, Any]) -> str:
        """Generate a helpful response message"""
        if 'all' in event_types:
            event_desc = "all events"
        else:
            event_desc = ", ".join(event_types)
        
        return f"Searching for {event_desc} in the time range: {time_info['description']}. Check the web dashboard for detailed results and analytics."
    
    def _describe_time_range(self, start_time: datetime, end_time: datetime) -> str:
        """Create human-readable time range description"""
        now = datetime.now()
        
        if start_time.date() == now.date():
            return "today"
        elif start_time.date() == (now - timedelta(days=1)).date():
            return "yesterday"
        elif (now - start_time).days <= 7:
            return "past week"
        else:
            return f"{start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}"
    
    def _get_suggestions(self, query: str) -> List[str]:
        """Get helpful suggestions based on query"""
        suggestions = [
            "Visit the web dashboard at http://127.0.0.1:8000 for detailed analytics",
            "Check the surveillance_data/harmful_events/ folder for critical events",
            "View recorded videos in surveillance_data/videos/ folder"
        ]
        
        if 'fall' in query:
            suggestions.append("Falls are automatically detected and saved as high-priority events")
        elif 'violence' in query:
            suggestions.append("Violence detection uses advanced AI models for accurate identification")
        elif 'today' not in query and 'recent' not in query:
            suggestions.append("Try asking about 'today' or 'recent events' for latest activity")
        
        return suggestions
    
    def get_available_features(self) -> Dict[str, Any]:
        """Get information about available features"""
        return {
            'llm_available': False,
            'simple_queries': True,
            'time_parsing': True,
            'event_type_detection': True,
            'location_parsing': True,
            'web_dashboard': True,
            'file_storage': True,
            'message': 'Simple NLP interface active. For advanced conversational AI, the LLM model needs to be properly configured.'
        }

if __name__ == "__main__":
    # Test the simple NLP interface
    nlp = SimpleNLPInterface()
    
    test_queries = [
        "Show me all falls from today",
        "Any violence detected yesterday?",
        "What happened in the last hour?",
        "Suspicious activity near the entrance"
    ]
    
    print("Testing Simple NLP Interface:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = nlp.process_query(query)
        print(f"Response: {result['message']}")
        print(f"Event Types: {result['parsed_info']['event_types']}")
        print(f"Time Range: {result['parsed_info']['time_range']['description']}")
