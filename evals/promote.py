"""
promote.py — Functions to promote real chat logs to the extended evaluation dataset.

Why this file exists:
  This module provides the bridge between real chat interactions (stored in SQLite)
  and the evaluation dataset system. It allows users to:
    - Convert chat logs to test case format
    - Add user-edited reference answers
    - Save to dataset_extended.json for future evaluations
    - Check for duplicate/similar queries

Key features:
  - User provides reference answers (no auto-generation)
  - Chat logs become single-turn test cases only
  - Duplicate detection with similarity checking
  - Preserves chat context and metadata
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import difflib
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXTENDED_DATASET_PATH = os.path.join(DATA_DIR, "dataset_extended.json")
DB_PATH = os.path.join(DATA_DIR, "eval_results.db")


def get_connection():
    """Get SQLite database connection."""
    return sqlite3.connect(DB_PATH)


def get_chat_logs_by_ids(chat_ids: List[int]) -> List[Dict]:
    """
    Retrieve chat logs by their IDs.
    
    Args:
        chat_ids: List of chat log IDs to retrieve
        
    Returns:
        List of chat log dictionaries with all fields
    """
    if not chat_ids:
        return []
    
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    placeholders = ','.join(['?'] * len(chat_ids))
    query = f"SELECT * FROM chat_logs WHERE id IN ({placeholders}) ORDER BY timestamp DESC"
    cursor.execute(query, chat_ids)
    
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def extract_context_from_trace(trace_data: str) -> Dict[str, Any]:
    """
    Extract context information from trace data JSON.
    
    Args:
        trace_data: JSON string of trace data
        
    Returns:
        Context dictionary with extracted information
    """
    try:
        trace = json.loads(trace_data)
        
        # Extract entities for context
        entities = trace.get('entities', {})
        order_id = None
        
        # Try to extract order ID from entities
        if isinstance(entities, dict):
            # Check common entity keys for order ID
            for key in ['order_id', 'order_number', 'order', 'id']:
                if key in entities and entities[key]:
                    order_id = entities[key]
                    break
        
        # Extract router intent for category
        router_intent = trace.get('router_intent', 'unknown')
        
        return {
            "order_id": order_id,
            "router_intent": router_intent,
            "agent_used": trace.get('agent_used'),
            "entities": entities,
            "original_timestamp": trace.get('timestamp')  # From trace, not chat log
        }
    except (json.JSONDecodeError, KeyError, AttributeError):
        return {
            "order_id": None,
            "router_intent": "unknown",
            "agent_used": None,
            "entities": {},
            "original_timestamp": None
        }


def generate_test_case_template(chat_log: Dict, reference_answer: str = "") -> Dict[str, Any]:
    """
    Generate a test case template from a chat log.
    
    Args:
        chat_log: Chat log dictionary from database
        reference_answer: User-provided reference answer (empty string if not provided yet)
        
    Returns:
        Test case dictionary in the format expected by the evaluation system
    """
    # Extract trace data and context
    trace_data = chat_log.get('trace_data', '{}')
    context_info = extract_context_from_trace(trace_data)
    
    # Generate test case ID
    chat_id = chat_log['id']
    test_case_id = f"CHAT-{chat_id}"
    
    # Use router intent as category, fallback to 'organic'
    category = context_info.get('router_intent', 'organic')
    if not category or category.lower() == 'unknown':
        category = 'organic'
    
    # Clean up category name (remove any special characters)
    category = re.sub(r'[^a-zA-Z0-9_-]', '_', category).lower()
    
    return {
        "id": test_case_id,
        "category": category,
        "query": chat_log['user_message'],
        "context": {
            "order_id": context_info.get('order_id'),
            "original_chat_id": chat_id,
            "timestamp": chat_log['timestamp'],
            "router_intent": context_info.get('router_intent'),
            "agent_used": context_info.get('agent_used')
        },
        "reference_answer": reference_answer,
        "tags": ["promoted", "organic"],
        "difficulty": "medium"
    }


def calculate_query_similarity(query1: str, query2: str) -> float:
    """
    Calculate similarity between two queries using difflib.
    
    Args:
        query1: First query string
        query2: Second query string
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    # Simple similarity using SequenceMatcher
    return difflib.SequenceMatcher(None, query1.lower(), query2.lower()).ratio()


def find_similar_queries(new_query: str, existing_queries: List[str], threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Find queries in existing dataset that are similar to the new query.
    
    Args:
        new_query: The new query to check
        existing_queries: List of existing query strings
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        List of tuples (query, similarity_score) for queries above threshold
    """
    similar = []
    for existing_query in existing_queries:
        similarity = calculate_query_similarity(new_query, existing_query)
        if similarity >= threshold:
            similar.append((existing_query, similarity))
    
    # Sort by similarity (highest first)
    similar.sort(key=lambda x: x[1], reverse=True)
    return similar


def load_extended_dataset() -> List[Dict]:
    """
    Load the current extended dataset.
    
    Returns:
        List of test cases in the extended dataset
    """
    try:
        with open(EXTENDED_DATASET_PATH, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            if not isinstance(dataset, list):
                return []
            return dataset
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_extended_dataset(test_cases: List[Dict]) -> bool:
    """
    Save test cases to the extended dataset file.
    
    Args:
        test_cases: List of test case dictionaries
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save to file
        with open(EXTENDED_DATASET_PATH, 'w', encoding='utf-8') as f:
            json.dump(test_cases, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving extended dataset: {e}")
        return False


def check_for_duplicates(new_test_cases: List[Dict], existing_dataset: List[Dict]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Check new test cases for duplicates/similar queries in existing dataset.
    
    Args:
        new_test_cases: List of new test cases to check
        existing_dataset: List of existing test cases
        
    Returns:
        Dictionary mapping new test case IDs to list of similar queries found
    """
    # Extract queries from existing dataset
    existing_queries = []
    for case in existing_dataset:
        if 'query' in case:
            existing_queries.append(case['query'])
    
    duplicates = {}
    
    for test_case in new_test_cases:
        new_query = test_case.get('query', '')
        if not new_query:
            continue
            
        similar = find_similar_queries(new_query, existing_queries, threshold=0.8)
        if similar:
            duplicates[test_case['id']] = similar
    
    return duplicates


def promote_chats_to_dataset(chat_ids: List[int], reference_answers: Dict[int, str]) -> Tuple[int, List[str]]:
    """
    Promote selected chat logs to the extended dataset.
    
    Args:
        chat_ids: List of chat log IDs to promote
        reference_answers: Dictionary mapping chat_id to reference answer
        
    Returns:
        Tuple of (success_count, error_messages)
    """
    if not chat_ids:
        return 0, ["No chat IDs provided"]
    
    # Load existing dataset
    existing_dataset = load_extended_dataset()
    
    # Get chat logs
    chat_logs = get_chat_logs_by_ids(chat_ids)
    if not chat_logs:
        return 0, ["No chat logs found for the provided IDs"]
    
    # Generate test cases
    new_test_cases = []
    for chat_log in chat_logs:
        chat_id = chat_log['id']
        reference_answer = reference_answers.get(chat_id, "")
        
        if not reference_answer.strip():
            # Skip if reference answer is empty
            continue
            
        test_case = generate_test_case_template(chat_log, reference_answer)
        new_test_cases.append(test_case)
    
    if not new_test_cases:
        return 0, ["No valid test cases to promote (all reference answers were empty)"]
    
    # Check for duplicates
    duplicates = check_for_duplicates(new_test_cases, existing_dataset)
    
    # Filter out exact duplicates (same query)
    filtered_test_cases = []
    for test_case in new_test_cases:
        query = test_case['query']
        
        # Check if exact query already exists
        exact_match = False
        for existing_case in existing_dataset:
            if existing_case.get('query') == query:
                exact_match = True
                break
        
        if not exact_match:
            filtered_test_cases.append(test_case)
    
    if not filtered_test_cases:
        return 0, ["All queries already exist in the extended dataset"]
    
    # Add to existing dataset
    updated_dataset = existing_dataset + filtered_test_cases
    
    # Save updated dataset
    if save_extended_dataset(updated_dataset):
        success_count = len(filtered_test_cases)
        messages = [f"Successfully promoted {success_count} chat(s) to extended dataset"]
        
        # Add duplicate warnings if any
        if duplicates:
            warning_msg = "⚠️ Some queries are similar to existing ones:"
            for chat_id, similar_list in duplicates.items():
                for similar_query, score in similar_list[:3]:  # Show top 3 matches
                    warning_msg += f"\n  - CHAT-{chat_id}: {similar_query[:50]}... (similarity: {score:.2f})"
            messages.append(warning_msg)
        
        return success_count, messages
    else:
        return 0, ["Failed to save extended dataset"]


def get_existing_queries() -> List[str]:
    """
    Get all queries from both core and extended datasets.
    
    Returns:
        List of all query strings from all datasets
    """
    from evals.cascade import load_dataset
    
    try:
        # Load all datasets
        core_data = load_dataset("core")
        extended_data = load_dataset("extended")
        
        all_queries = []
        
        # Extract queries from single-turn cases
        for case in core_data.get("single", []):
            if "query" in case:
                all_queries.append(case["query"])
        
        for case in extended_data.get("single", []):
            if "query" in case:
                all_queries.append(case["query"])
        
        # Extract queries from multi-turn cases
        for case in core_data.get("multi", []):
            if "turns" in case:
                for turn in case["turns"]:
                    if turn.get("role") == "user" and "content" in turn:
                        all_queries.append(turn["content"])
        
        return all_queries
    except Exception:
        # Fallback to just extended dataset
        extended_dataset = load_extended_dataset()
        return [case.get("query", "") for case in extended_dataset if case.get("query")]


def preview_promotion(chat_ids: List[int]) -> List[Dict]:
    """
    Generate preview of what test cases would be created from chat logs.
    
    Args:
        chat_ids: List of chat log IDs to preview
        
    Returns:
        List of test case templates (with empty reference answers)
    """
    chat_logs = get_chat_logs_by_ids(chat_ids)
    preview_cases = []
    
    for chat_log in chat_logs:
        test_case = generate_test_case_template(chat_log, "")
        preview_cases.append(test_case)
    
    return preview_cases


if __name__ == "__main__":
    # Test the promotion functions
    print("Testing promotion module...")
    
    # Load some sample chat logs (first 5)
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT id, user_message, bot_response, trace_data FROM chat_logs LIMIT 5")
    sample_chats = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    if sample_chats:
        print(f"Found {len(sample_chats)} sample chats")
        
        # Generate test case templates
        for i, chat in enumerate(sample_chats):
            test_case = generate_test_case_template(chat, "Sample reference answer")
            print(f"\nTest Case {i+1}:")
            print(f"  ID: {test_case['id']}")
            print(f"  Category: {test_case['category']}")
            print(f"  Query: {test_case['query'][:50]}...")
            print(f"  Tags: {test_case['tags']}")
    
    # Test duplicate detection
    existing_queries = ["What is your return policy?", "Can I return an opened phone?"]
    new_query = "What's your return policy for phones?"
    similar = find_similar_queries(new_query, existing_queries, threshold=0.7)
    print(f"\nDuplicate detection test:")
    print(f"  Query: '{new_query}'")
    print(f"  Similar matches: {similar}")
    
    print("\n✅ Promotion module ready!")