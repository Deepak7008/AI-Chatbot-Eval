"""
db.py — SQLite persistence for chat logs and evaluation results.

Why this file exists:
  All chat interactions and eval results are persisted to SQLite so they
  survive tab closes, app restarts, and can be queried for dashboards.
  This is the single source of truth for all stored data.

Tables:
  - chat_logs:    Every live chat interaction with full trace data
  - eval_runs:    One row per evaluation batch run
  - eval_results: One row per test case per run (with 6-dimension scores)
"""

import sqlite3
import os
import json
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_results.db')

def get_connection():
    # Ensure data directory exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table for live chat logs (including the trace info)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            user_message TEXT NOT NULL,
            bot_response TEXT NOT NULL,
            model_id TEXT,
            router_intent TEXT,
            router_confidence REAL,
            router_reasoning TEXT,
            agent_used TEXT,
            entities_json TEXT,
            guardrail_input_safe INTEGER,
            guardrail_input_reason TEXT,
            guardrail_bypassed INTEGER,
            tokens_used INTEGER,
            latency_ms REAL,
            user_feedback TEXT,
            escalated INTEGER DEFAULT 0,
            trace_data TEXT
        )
    ''')
    
    # Table for evaluation runs (upgraded with model tracking + stats)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            dataset_used TEXT NOT NULL,
            model TEXT,
            judge_model TEXT,
            total_cases INTEGER,
            passed_cases INTEGER,
            pass_rate REAL,
            avg_score REAL,
            total_cost REAL,
            total_tokens INTEGER,
            total_time_sec REAL,
            ci_lower REAL,
            ci_upper REAL
        )
    ''')
    
    # Table for individual evaluation results per run (upgraded with 6 dimensions)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eval_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            case_id TEXT NOT NULL,
            category TEXT,
            is_multi_turn INTEGER DEFAULT 0,
            query TEXT NOT NULL,
            reference_answer TEXT,
            actual_answer TEXT,
            cosine_similarity REAL,
            accuracy INTEGER,
            groundedness INTEGER,
            safety INTEGER,
            helpfulness INTEGER,
            relevance INTEGER,
            tone INTEGER,
            weighted_score REAL,
            pass_fail TEXT,
            tokens_used INTEGER,
            latency_ms REAL,
            judge_reasoning_json TEXT,
            FOREIGN KEY (run_id) REFERENCES eval_runs (id) ON DELETE CASCADE
        )
    ''')
    
    # Table for bias check runs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bias_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            sample_size INTEGER,
            avg_abs_delta REAL,
            max_abs_delta REAL,
            bias_direction TEXT,
            verdict TEXT,
            interpretation TEXT
        )
    ''')
    
    # Table for individual bias test case results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bias_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            case_id TEXT NOT NULL,
            category TEXT,
            base_score REAL,
            swapped_score REAL,
            delta REAL,
            FOREIGN KEY (run_id) REFERENCES bias_runs (id) ON DELETE CASCADE
        )
    ''')
    
    # ── INDEXES for query performance ──
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_logs (session_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_logs (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_intent ON chat_logs (router_intent)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_results_run ON eval_results (run_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_results_case ON eval_results (case_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_runs_timestamp ON eval_runs (timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_results_run ON bias_results (run_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_runs_timestamp ON bias_runs (timestamp)')
    
    conn.commit()
    conn.close()


# ── CHAT LOG FUNCTIONS ────────────────────────────────────────────────────────

def log_chat(session_id, user_message, bot_response, trace_data):
    """
    trace_data should be a dict containing:
    - router_intent, router_confidence, router_reasoning
    - agent_used
    - entities (dict)
    - guardrail_input_safe (bool), guardrail_input_reason, guardrail_bypassed (bool)
    - tokens_used, latency_ms
    - escalated (bool)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    entities_json = json.dumps(trace_data.get('entities', {})) if trace_data.get('entities') else None
    
    cursor.execute('''
        INSERT INTO chat_logs (
            session_id, timestamp, user_message, bot_response, model_id,
            router_intent, router_confidence, router_reasoning, agent_used,
            entities_json, guardrail_input_safe, guardrail_input_reason,
            guardrail_bypassed, tokens_used, latency_ms, escalated, trace_data
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id,
        datetime.now().isoformat(),
        user_message,
        bot_response,
        trace_data.get('model_id'),
        trace_data.get('router_intent'),
        trace_data.get('router_confidence'),
        trace_data.get('router_reasoning'),
        trace_data.get('agent_used'),
        entities_json,
        1 if trace_data.get('guardrail_input_safe', True) else 0,
        trace_data.get('guardrail_input_reason'),
        1 if trace_data.get('guardrail_bypassed', False) else 0,
        trace_data.get('tokens_used', 0),
        trace_data.get('latency_ms', 0.0),
        1 if trace_data.get('escalated', False) else 0,
        json.dumps(trace_data)
    ))
    
    conn.commit()
    log_id = cursor.lastrowid
    conn.close()
    return log_id

def update_chat_feedback(log_id, feedback):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE chat_logs SET user_feedback = ? WHERE id = ?', (feedback, log_id))
    conn.commit()
    conn.close()

def delete_chat_logs(log_ids):
    if not log_ids:
        return
    conn = get_connection()
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(log_ids))
    cursor.execute(f'DELETE FROM chat_logs WHERE id IN ({placeholders})', log_ids)
    conn.commit()
    conn.close()


# ── EVAL RUN FUNCTIONS ────────────────────────────────────────────────────────

def save_eval_run(run_data: dict) -> int:
    """
    Save an evaluation run summary.
    
    Args:
        run_data: Dict with keys matching eval_runs columns
        
    Returns:
        The auto-generated run_id
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO eval_runs (
            run_name, timestamp, dataset_used, model, judge_model,
            total_cases, passed_cases, pass_rate, avg_score,
            total_cost, total_tokens, total_time_sec, ci_lower, ci_upper
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        run_data.get('run_name', f"Run {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
        run_data.get('timestamp', datetime.now().isoformat()),
        run_data.get('dataset_used', 'core'),
        run_data.get('model'),
        run_data.get('judge_model'),
        run_data.get('total_cases', 0),
        run_data.get('passed_cases', 0),
        run_data.get('pass_rate', 0.0),
        run_data.get('avg_score', 0.0),
        run_data.get('total_cost', 0.0),
        run_data.get('total_tokens', 0),
        run_data.get('total_time_sec', 0.0),
        run_data.get('ci_lower'),
        run_data.get('ci_upper'),
    ))
    
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return run_id


def save_eval_result(result_data: dict):
    """
    Save a single test case evaluation result.
    
    Args:
        result_data: Dict with keys matching eval_results columns
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO eval_results (
            run_id, case_id, category, is_multi_turn,
            query, reference_answer, actual_answer, cosine_similarity,
            accuracy, groundedness, safety, helpfulness, relevance, tone,
            weighted_score, pass_fail, tokens_used, latency_ms,
            judge_reasoning_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        result_data.get('run_id'),
        result_data.get('case_id'),
        result_data.get('category'),
        1 if result_data.get('is_multi_turn', False) else 0,
        result_data.get('query'),
        result_data.get('reference_answer'),
        result_data.get('actual_answer'),
        result_data.get('cosine_similarity'),
        result_data.get('accuracy'),
        result_data.get('groundedness'),
        result_data.get('safety'),
        result_data.get('helpfulness'),
        result_data.get('relevance'),
        result_data.get('tone'),
        result_data.get('weighted_score'),
        result_data.get('pass_fail'),
        result_data.get('tokens_used', 0),
        result_data.get('latency_ms', 0.0),
        result_data.get('judge_reasoning_json'),
    ))
    
    conn.commit()
    conn.close()


# ── QUERY FUNCTIONS ───────────────────────────────────────────────────────────

def get_eval_runs():
    """Get all eval runs, most recent first."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM eval_runs ORDER BY timestamp DESC')
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_eval_results(run_id: int):
    """Get all eval results for a specific run."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM eval_results WHERE run_id = ? ORDER BY id', (run_id,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_chat_logs(limit=100, offset=0, filters=None):
    """
    Get chat logs with optional filtering.
    
    filters: dict with optional keys:
      - session_id, router_intent, user_feedback, escalated
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    query = 'SELECT * FROM chat_logs WHERE 1=1'
    params = []
    
    if filters:
        if filters.get('session_id'):
            query += ' AND session_id = ?'
            params.append(filters['session_id'])
        if filters.get('router_intent'):
            query += ' AND router_intent = ?'
            params.append(filters['router_intent'])
        if filters.get('user_feedback'):
            query += ' AND user_feedback = ?'
            params.append(filters['user_feedback'])
        if filters.get('escalated') is not None:
            query += ' AND escalated = ?'
            params.append(1 if filters['escalated'] else 0)
    
    query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
    params.extend([limit, offset])
    
    cursor.execute(query, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_eval_run(run_id: int, run_data: dict) -> bool:
    """
    Update an existing evaluation run with new statistics.
    
    Args:
        run_id: The ID of the run to update
        run_data: Dict with fields to update
        
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build UPDATE statement dynamically based on provided fields
        fields = []
        values = []
        
        for key, value in run_data.items():
            if key in ['total_cases', 'passed_cases', 'pass_rate', 'avg_score',
                      'total_cost', 'total_tokens', 'total_time_sec', 'ci_lower', 'ci_upper']:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False  # Nothing to update
            
        values.append(run_id)
        query = f"UPDATE eval_runs SET {', '.join(fields)} WHERE id = ?"
        
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error updating eval run {run_id}: {e}")
        return False

        return False


# ── BIAS CHECK FUNCTIONS ──────────────────────────────────────────────────────

def save_bias_run(run_data: dict) -> int:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bias_runs (
                timestamp, sample_size, avg_abs_delta, max_abs_delta,
                bias_direction, verdict, interpretation
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            run_data.get('timestamp', datetime.now().isoformat()),
            run_data.get('sample_size', 0),
            run_data.get('avg_abs_delta', 0.0),
            run_data.get('max_abs_delta', 0.0),
            run_data.get('bias_direction', ''),
            run_data.get('verdict', ''),
            run_data.get('interpretation', '')
        ))
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return run_id
    except Exception as e:
        print(f"Error saving bias run: {e}")
        return None

def update_bias_run(run_id: int, run_data: dict) -> bool:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        fields = []
        values = []
        
        for key, value in run_data.items():
            if key in ['avg_abs_delta', 'max_abs_delta', 'bias_direction', 'verdict', 'interpretation']:
                fields.append(f"{key} = ?")
                values.append(value)
        
        if not fields:
            return False
            
        values.append(run_id)
        query = f"UPDATE bias_runs SET {', '.join(fields)} WHERE id = ?"
        
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating bias run {run_id}: {e}")
        return False

def save_bias_result(result: dict) -> int:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bias_results (
                run_id, case_id, category, base_score, swapped_score, delta
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            result.get('run_id'),
            result.get('case_id'),
            result.get('category'),
            result.get('base_score'),
            result.get('swapped_score'),
            result.get('delta')
        ))
        result_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return result_id
    except Exception as e:
        print(f"Error saving bias result: {e}")
        return None


if __name__ == '__main__':
    # Delete old DB to force fresh schema creation
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print(f"Deleted old database: {DB_PATH}")
    init_db()
    print("Database initialized successfully with upgraded schema.")
    
    # Verify schema
    conn = get_connection()
    cursor = conn.cursor()
    for table in ['chat_logs', 'eval_runs', 'eval_results', 'bias_runs', 'bias_results']:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        print(f"  {table}: {', '.join(cols)}")
    conn.close()
else:
    # Ensure DB is initialized when imported
    init_db()
