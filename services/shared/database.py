# shared/database.py - FIXED VERSION (correct port handling)
import logging
from typing import List, Dict, Any, Optional
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
from datetime import datetime
import time

class ClickHouseClient:
    """Fixed ClickHouse database client with proper port handling"""
    
    def __init__(self, host: str, port: int, user: str, password: str, database: str):
        self.host = host
        
        # FIX: Handle port correctly
        # clickhouse-driver always uses native port (9000), not HTTP port (8123)
        if port == 8123:
            self.port = 9000  # Convert HTTP port to native port
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"Converting HTTP port 8123 to native port 9000")
        else:
            self.port = port
            
        self.user = user
        self.password = password
        self.database = database
        self.client = None
        self.logger = logging.getLogger(__name__)
        self.connected = False
    
    def connect(self):
        """Connect with proper port handling"""
        try:
            self.logger.info(f"Connecting to ClickHouse at {self.host}:{self.port} (native protocol)")
            
            # Create database first (connect without db specified)
            temp_client = Client(
                host=self.host,
                port=self.port,  # Use native port
                user=self.user,
                password=self.password,
                compression=False,
                connect_timeout=10,
                send_receive_timeout=30
            )
            
            # Test connection first
            try:
                temp_client.execute('SELECT 1')
                self.logger.info("✅ ClickHouse connection test successful")
            except Exception as e:
                self.logger.error(f"❌ ClickHouse connection test failed: {e}")
                raise
            
            # Create database
            try:
                temp_client.execute(f'CREATE DATABASE IF NOT EXISTS {self.database}')
                self.logger.info(f"✅ Database {self.database} ready")
            except Exception as e:
                self.logger.warning(f"⚠️ Database creation warning: {e}")
            
            # Connect to specific database
            self.client = Client(
                host=self.host,
                port=self.port,  # Use native port
                user=self.user,
                password=self.password,
                database=self.database,
                compression=False,
                connect_timeout=10,
                send_receive_timeout=30
            )
            
            # Test connection with database
            self.client.execute('SELECT 1')
            self.connected = True
            self.logger.info("✅ ClickHouse connection successful")
            
            # Create tables once
            self._create_tables_simple()
            
        except Exception as e:
            self.logger.error(f"❌ Connection failed: {e}")
            self.connected = False
            self.client = None
            # Don't raise exception - allow graceful degradation
    
    def _create_tables_simple(self):
        """Create tables without any recursion risk"""
        if not self.client:
            return
            
        tables = {
            'raw_logs': """
                CREATE TABLE IF NOT EXISTS raw_logs (
                    timestamp DateTime,
                    event_type String,
                    source_ip String,
                    destination_ip String,
                    port UInt16,
                    protocol String,
                    message String,
                    severity String,
                    threat_indicators Array(String),
                    event_id String,
                    session_id String,
                    user_agent String,
                    bytes_transferred UInt64,
                    duration_seconds UInt32
                ) ENGINE = MergeTree() ORDER BY timestamp
            """,
            'alerts': """
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp DateTime,
                    alert_id String,
                    severity String,
                    entity_id String,
                    message String,
                    aggregated_score Float64,
                    contributing_models Array(String),
                    status String DEFAULT 'open'
                ) ENGINE = MergeTree() ORDER BY timestamp
            """,
            'anomaly_scores': """
                CREATE TABLE IF NOT EXISTS anomaly_scores (
                    timestamp DateTime,
                    model_name String,
                    entity_id String,
                    score Float64,
                    features String,
                    is_anomaly UInt8
                ) ENGINE = MergeTree() ORDER BY timestamp
            """
        }
        
        for table_name, query in tables.items():
            try:
                self.client.execute(query)
                self.logger.info(f"✅ Created/verified table: {table_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to create table {table_name}: {e}")
    
    def is_connected(self) -> bool:
        """Simple connection check"""
        if not self.connected or not self.client:
            return False
        try:
            self.client.execute('SELECT 1')
            return True
        except:
            self.connected = False
            return False
    
    def get_log_count(self, hours: int = 24) -> int:
        """Get log count - simple version"""
        if not self.is_connected():
            return 0
        try:
            result = self.client.execute(f"SELECT count() FROM raw_logs WHERE timestamp >= now() - INTERVAL {hours} HOUR")
            return result[0][0] if result else 0
        except Exception as e:
            self.logger.error(f"Failed to get log count: {e}")
            return 0
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent alerts - enhanced version"""
        if not self.is_connected():
            return []
        try:
            result = self.client.execute(
                f"""
                SELECT timestamp, alert_id, severity, entity_id, message, aggregated_score, contributing_models, status
                FROM alerts 
                WHERE timestamp >= now() - INTERVAL {hours} HOUR
                ORDER BY timestamp DESC
                LIMIT 50
                """
            )
            return [dict(zip([
                'timestamp', 'alert_id', 'severity', 'entity_id', 
                'message', 'aggregated_score', 'contributing_models', 'status'
            ], row)) for row in result]
        except ClickHouseError as e:
            self.logger.error(f"Failed to get recent alerts: {e}")
            return []
    
    def insert_raw_logs(self, logs: List[Dict[str, Any]]):
        """Insert logs - enhanced version with better error handling"""
        if not self.is_connected() or not logs:
            return
        try:
            processed_logs = []
            for log in logs:
                try:
                    # Handle timestamp conversion properly
                    timestamp = log.get('timestamp')
                    if isinstance(timestamp, str):
                        if 'T' in timestamp:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        else:
                            timestamp = datetime.now()
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    processed_logs.append((
                        timestamp,
                        log.get('event_type', ''),
                        log.get('source_ip', ''),
                        log.get('destination_ip', ''),
                        int(log.get('port', 0)),
                        log.get('protocol', ''),
                        log.get('message', ''),
                        log.get('severity', ''),
                        log.get('threat_indicators', []),
                        log.get('event_id', ''),
                        log.get('session_id', ''),
                        log.get('user_agent', ''),
                        int(log.get('bytes_transferred', 0)),
                        int(log.get('duration_seconds', 0))
                    ))
                except Exception as e:
                    self.logger.debug(f"Skipping malformed log: {e}")
                    continue
            
            if processed_logs:
                self.client.execute("INSERT INTO raw_logs VALUES", processed_logs)
                self.logger.debug(f"✅ Inserted {len(processed_logs)} raw logs")
        except Exception as e:
            self.logger.error(f"❌ Failed to insert logs: {e}")
    
    def insert_anomaly_scores(self, scores: List[Dict[str, Any]]):
        """Insert anomaly scores - enhanced version"""
        if not self.is_connected() or not scores:
            return
        
        try:
            processed_scores = []
            for score in scores:
                try:
                    timestamp = score.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    processed_scores.append((
                        timestamp,
                        score.get('model_name', ''),
                        score.get('entity_id', ''),
                        float(score.get('score', 0.0)),
                        str(score.get('features', {})),
                        int(score.get('is_anomaly', False))
                    ))
                except Exception as e:
                    self.logger.debug(f"Skipping malformed score: {e}")
                    continue
            
            if processed_scores:
                self.client.execute("INSERT INTO anomaly_scores VALUES", processed_scores)
                self.logger.debug(f"✅ Inserted {len(processed_scores)} anomaly scores")
        except ClickHouseError as e:
            self.logger.error(f"❌ Failed to insert anomaly scores: {e}")
    
    def insert_alerts(self, alerts: List[Dict[str, Any]]):
        """Insert alerts - enhanced version"""
        if not self.is_connected() or not alerts:
            return
        
        try:
            processed_alerts = []
            for alert in alerts:
                try:
                    timestamp = alert.get('timestamp')
                    if isinstance(timestamp, str):
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif not isinstance(timestamp, datetime):
                        timestamp = datetime.now()
                    
                    processed_alerts.append((
                        timestamp,
                        alert.get('alert_id', ''),
                        alert.get('severity', ''),
                        alert.get('entity_id', ''),
                        alert.get('message', ''),
                        float(alert.get('aggregated_score', 0.0)),
                        alert.get('contributing_models', []),
                        alert.get('status', 'open')
                    ))
                except Exception as e:
                    self.logger.debug(f"Skipping malformed alert: {e}")
                    continue
            
            if processed_alerts:
                self.client.execute("INSERT INTO alerts VALUES", processed_alerts)
                self.logger.debug(f"✅ Inserted {len(processed_alerts)} alerts")
        except ClickHouseError as e:
            self.logger.error(f"❌ Failed to insert alerts: {e}")