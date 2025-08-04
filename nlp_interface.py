import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import BaseOutputParser
import re
from database import db_manager
from config import config

logger = logging.getLogger(__name__)

class QueryParser(BaseOutputParser):
    """Parse LLM output for structured queries"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM response into structured query parameters"""
        try:
            # Extract time range
            time_pattern = r"TIME_RANGE:\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)\s*to\s*(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)"
            time_match = re.search(time_pattern, text)
            
            # Extract event types
            event_pattern = r"EVENT_TYPES:\s*\[(.*?)\]"
            event_match = re.search(event_pattern, text)
            
            # Extract keywords
            keyword_pattern = r"KEYWORDS:\s*\[(.*?)\]"
            keyword_match = re.search(keyword_pattern, text)
            
            # Extract semantic query
            semantic_pattern = r"SEMANTIC_QUERY:\s*(.*?)(?:\n|$)"
            semantic_match = re.search(semantic_pattern, text)
            
            result = {
                'time_range': None,
                'event_types': [],
                'keywords': [],
                'semantic_query': None
            }
            
            if time_match:
                start_time = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S" if len(time_match.group(1)) > 10 else "%Y-%m-%d")
                end_time = datetime.strptime(time_match.group(2), "%Y-%m-%d %H:%M:%S" if len(time_match.group(2)) > 10 else "%Y-%m-%d")
                result['time_range'] = (start_time, end_time)
                
            if event_match:
                event_types = [t.strip().strip('"\'') for t in event_match.group(1).split(',')]
                result['event_types'] = [t for t in event_types if t]
                
            if keyword_match:
                keywords = [k.strip().strip('"\'') for k in keyword_match.group(1).split(',')]
                result['keywords'] = [k for k in keywords if k]
                
            if semantic_match:
                result['semantic_query'] = semantic_match.group(1).strip()
                
            return result
            
        except Exception as e:
            logger.error(f"Error parsing query: {e}")
            return {
                'time_range': None,
                'event_types': [],
                'keywords': [],
                'semantic_query': text  # Fallback to original text
            }

class SemanticSearch:
    """Handles semantic search using sentence transformers and FAISS"""
    
    def __init__(self):
        self.model = SentenceTransformer(config.ai.embedding_model)
        self.index = None
        self.event_ids = []
        self.embeddings_cache = {}
        self.index_path = os.path.join(config.ai.vector_db_path, "faiss_index")
        self.metadata_path = os.path.join(config.ai.vector_db_path, "metadata.json")
        
        # Create vector DB directory
        os.makedirs(config.ai.vector_db_path, exist_ok=True)
        
        # Load existing index
        self.load_index()
        
    def add_event_embedding(self, event_id: int, description: str):
        """Add event embedding to the search index"""
        try:
            # Generate embedding
            embedding = self.model.encode([description])[0]
            
            # Store in cache
            self.embeddings_cache[event_id] = {
                'embedding': embedding,
                'description': description
            }
            
            # Rebuild index
            self._rebuild_index()
            
        except Exception as e:
            logger.error(f"Error adding event embedding: {e}")
            
    def search_similar_events(self, query: str, k: int = 10) -> List[Dict]:
        """Search for similar events using semantic similarity"""
        if not self.index or len(self.event_ids) == 0:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query])[0]
            
            # Search in FAISS index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), k
            )
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.event_ids):
                    event_id = self.event_ids[idx]
                    results.append({
                        'event_id': event_id,
                        'similarity_score': float(score),
                        'description': self.embeddings_cache[event_id]['description']
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar events: {e}")
            return []
            
    def _rebuild_index(self):
        """Rebuild FAISS index from cached embeddings"""
        if not self.embeddings_cache:
            return
            
        try:
            # Prepare embeddings matrix
            embeddings = []
            event_ids = []
            
            for event_id, data in self.embeddings_cache.items():
                embeddings.append(data['embedding'])
                event_ids.append(event_id)
                
            embeddings_matrix = np.array(embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings_matrix.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings_matrix)
            
            # Add to index
            self.index.add(embeddings_matrix)
            self.event_ids = event_ids
            
            # Save index
            self.save_index()
            
        except Exception as e:
            logger.error(f"Error rebuilding index: {e}")
            
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        try:
            if self.index:
                faiss.write_index(self.index, self.index_path)
                
            # Save metadata
            metadata = {
                'event_ids': self.event_ids,
                'embeddings_cache': {
                    str(k): {
                        'embedding': v['embedding'].tolist(),
                        'description': v['description']
                    }
                    for k, v in self.embeddings_cache.items()
                }
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f)
                
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            
    def load_index(self):
        """Load FAISS index and metadata from disk"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load index
                self.index = faiss.read_index(self.index_path)
                
                # Load metadata
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    
                self.event_ids = metadata['event_ids']
                
                # Reconstruct embeddings cache
                for event_id_str, data in metadata['embeddings_cache'].items():
                    event_id = int(event_id_str)
                    self.embeddings_cache[event_id] = {
                        'embedding': np.array(data['embedding']),
                        'description': data['description']
                    }
                    
                logger.info(f"Loaded semantic search index with {len(self.event_ids)} events")
                
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            
    def update_from_database(self):
        """Update embeddings from database events"""
        try:
            # Get all events from database
            events = db_manager.get_events(limit=10000)
            
            for event in events:
                if event['id'] not in self.embeddings_cache:
                    self.add_event_embedding(event['id'], event['description'])
                    
        except Exception as e:
            logger.error(f"Error updating from database: {e}")

class NLPInterface:
    """Natural language interface for querying surveillance data"""
    
    def __init__(self):
        self.semantic_search = SemanticSearch()
        self.llm = None
        self.query_chain = None
        self.llm_available = False
        
        # Initialize LLM if model exists
        if os.path.exists(config.ai.llama_model_path):
            try:
                self._initialize_llm()
                self.llm_available = True
                logger.info("LLM interface fully initialized")
            except Exception as e:
                logger.warning(f"LLM initialization failed, continuing without LLM features: {e}")
                self.llm_available = False
        else:
            logger.warning(f"LLM model not found at {config.ai.llama_model_path}, continuing without LLM features")
            self.llm_available = False
            
    def _initialize_llm(self):
        """Initialize the LLM with simplified approach"""
        try:
            logger.info("Attempting to initialize LLM...")
            
            # Check if model file exists
            if not os.path.exists(config.ai.llama_model_path):
                raise FileNotFoundError(f"Model file not found: {config.ai.llama_model_path}")
            
            # Try very basic initialization
            self.llm = LlamaCpp(
                model_path=config.ai.llama_model_path,
                n_ctx=512,
                verbose=False
            )
            
            # Simple test
            test_response = self.llm.invoke("Hello")
            logger.info("LLM initialized and tested successfully")
            
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            # Re-raise to be caught by __init__
            raise e
            
    def _create_query_chain(self):
        """Create query chain"""
        try:
            # Create prompt template
            prompt_template = """
You are a surveillance system query assistant. Parse the user's natural language query about surveillance events and extract structured information.

User Query: {query}

Extract the following information and format your response exactly as shown:

TIME_RANGE: YYYY-MM-DD HH:MM:SS to YYYY-MM-DD HH:MM:SS (if time mentioned)
EVENT_TYPES: ["fall", "weapon_detected", "vehicle_crash", "suspicious_activity"] (if event types mentioned)
KEYWORDS: ["keyword1", "keyword2"] (important keywords from query)
SEMANTIC_QUERY: simplified description for semantic search

Examples:
- "What happened at 2 PM yesterday?" → TIME_RANGE: 2024-01-01 14:00:00 to 2024-01-01 14:59:59
- "Show me all falls from last week" → EVENT_TYPES: ["fall"], TIME_RANGE: 2024-01-01 00:00:00 to 2024-01-07 23:59:59
- "Any suspicious activity near the entrance?" → EVENT_TYPES: ["suspicious_activity"], KEYWORDS: ["entrance"], SEMANTIC_QUERY: suspicious activity entrance

"""
            
            self.prompt = PromptTemplate(
                input_variables=["query", "current_time"],
                template=prompt_template
            )
            
            self.query_chain = LLMChain(
                llm=self.llm,
                prompt=self.prompt
            )
            
            logger.info("Query chain created successfully")
            return self.query_chain
            
        except Exception as e:
            logger.error(f"Error creating query chain: {e}")
            raise e
            
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process natural language query and return results"""
        try:
            # Update semantic search from database
            self.semantic_search.update_from_database()
            
            if self.query_chain:
                # Use LLM to parse query
                parsed_query = self.query_chain.run(
                    query=query,
                    current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            else:
                # Fallback to simple parsing
                parsed_query = self._simple_query_parse(query)
                
            # Search database
            results = self._search_events(parsed_query)
            
            # Enhance with semantic search
            if parsed_query.get('semantic_query'):
                semantic_results = self.semantic_search.search_similar_events(
                    parsed_query['semantic_query'], k=20
                )
                results = self._merge_results(results, semantic_results)
                
            return {
                'parsed_query': parsed_query,
                'results': results,
                'total_count': len(results)
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                'parsed_query': {'error': str(e)},
                'results': [],
                'total_count': 0
            }
            
    def _simple_query_parse(self, query: str) -> Dict[str, Any]:
        """Simple fallback query parsing without LLM"""
        query_lower = query.lower()
        
        # Extract time references
        time_range = None
        if 'yesterday' in query_lower:
            yesterday = datetime.now() - timedelta(days=1)
            time_range = (
                yesterday.replace(hour=0, minute=0, second=0),
                yesterday.replace(hour=23, minute=59, second=59)
            )
        elif 'today' in query_lower:
            today = datetime.now()
            time_range = (
                today.replace(hour=0, minute=0, second=0),
                today.replace(hour=23, minute=59, second=59)
            )
        elif 'last week' in query_lower:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            time_range = (start_date, end_date)
            
        # Extract event types
        event_types = []
        if 'fall' in query_lower:
            event_types.append('fall')
        if 'weapon' in query_lower or 'gun' in query_lower or 'knife' in query_lower:
            event_types.append('weapon_detected')
        if 'crash' in query_lower or 'accident' in query_lower:
            event_types.append('vehicle_crash')
        if 'suspicious' in query_lower:
            event_types.append('suspicious_activity')
            
        return {
            'time_range': time_range,
            'event_types': event_types,
            'keywords': [],
            'semantic_query': query
        }
        
    def _search_events(self, parsed_query: Dict[str, Any]) -> List[Dict]:
        """Search events in database based on parsed query"""
        try:
            start_time = None
            end_time = None
            
            if parsed_query.get('time_range'):
                start_time, end_time = parsed_query['time_range']
                
            # Search by event types
            all_results = []
            
            if parsed_query.get('event_types'):
                for event_type in parsed_query['event_types']:
                    events = db_manager.get_events(
                        start_time=start_time,
                        end_time=end_time,
                        event_type=event_type,
                        limit=100
                    )
                    all_results.extend(events)
            else:
                # Get all events in time range
                events = db_manager.get_events(
                    start_time=start_time,
                    end_time=end_time,
                    limit=100
                )
                all_results.extend(events)
                
            # Filter by keywords
            if parsed_query.get('keywords'):
                filtered_results = []
                for event in all_results:
                    for keyword in parsed_query['keywords']:
                        if keyword.lower() in event['description'].lower():
                            filtered_results.append(event)
                            break
                all_results = filtered_results
                
            # Remove duplicates
            seen_ids = set()
            unique_results = []
            for event in all_results:
                if event['id'] not in seen_ids:
                    seen_ids.add(event['id'])
                    unique_results.append(event)
                    
            return unique_results
            
        except Exception as e:
            logger.error(f"Error searching events: {e}")
            return []
            
    def _merge_results(self, db_results: List[Dict], 
                      semantic_results: List[Dict]) -> List[Dict]:
        """Merge database and semantic search results"""
        # Create lookup for database results
        db_lookup = {event['id']: event for event in db_results}
        
        # Add semantic scores to database results
        for semantic_result in semantic_results:
            event_id = semantic_result['event_id']
            if event_id in db_lookup:
                db_lookup[event_id]['semantic_score'] = semantic_result['similarity_score']
                
        # Get additional events from semantic search
        semantic_event_ids = {sr['event_id'] for sr in semantic_results}
        db_event_ids = {event['id'] for event in db_results}
        
        additional_event_ids = semantic_event_ids - db_event_ids
        
        # Fetch additional events
        for event_id in additional_event_ids:
            try:
                events = db_manager.get_events(limit=1000)  # Get all events
                for event in events:
                    if event['id'] == event_id:
                        # Find semantic score
                        for sr in semantic_results:
                            if sr['event_id'] == event_id:
                                event['semantic_score'] = sr['similarity_score']
                                break
                        db_results.append(event)
                        break
            except Exception as e:
                logger.error(f"Error fetching additional event {event_id}: {e}")
                
        # Sort by semantic score if available, otherwise by timestamp
        db_results.sort(key=lambda x: (
            -(x.get('semantic_score', 0)),
            -x['timestamp'].timestamp() if isinstance(x['timestamp'], datetime) else -float(x['timestamp'])
        ))
        
        return db_results

# Global instance
nlp_interface = NLPInterface()
