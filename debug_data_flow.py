#!/usr/bin/env python3
# debug_data_flow.py - Comprehensive debugging for SOC Platform data flow

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_clickhouse_connection():
    """Check ClickHouse connection and table status"""
    logger = setup_logging()
    
    try:
        from clickhouse_driver import Client
        
        client = Client(
            host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
            port=int(os.getenv('CLICKHOUSE_PORT', '9000')),
            user=os.getenv('CLICKHOUSE_USER', 'default'),
            password=os.getenv('CLICKHOUSE_PASSWORD', 'secure_password'),
            database=os.getenv('CLICKHOUSE_DATABASE', 'soc_platform'),
            connect_timeout=5
        )
        
        # Test connection
        client.execute("SELECT 1")
        logger.info("✅ ClickHouse connection successful")
        
        # Check database
        databases = client.execute("SHOW DATABASES")
        logger.info(f"📊 Available databases: {[db[0] for db in databases]}")
        
        # Check tables
        tables = client.execute("SHOW TABLES")
        logger.info(f"📋 Available tables: {[table[0] for table in tables]}")
        
        # Check table structures
        for table_name in ['raw_logs', 'anomaly_scores', 'alerts']:
            try:
                result = client.execute(f"DESCRIBE {table_name}")
                logger.info(f"🔍 Table {table_name} structure:")
                for row in result:
                    logger.info(f"  - {row[0]}: {row[1]}")
                
                # Check row counts
                count_result = client.execute(f"SELECT count() FROM {table_name}")
                total_count = count_result[0][0]
                
                recent_result = client.execute(f"SELECT count() FROM {table_name} WHERE timestamp >= now() - INTERVAL 1 HOUR")
                recent_count = recent_result[0][0]
                
                logger.info(f"📊 {table_name}: {total_count} total rows, {recent_count} recent (1h)")
                
                if table_name == 'raw_logs' and total_count > 0:
                    # Show sample raw logs
                    sample = client.execute(f"SELECT timestamp, event_type, source_ip, severity FROM {table_name} ORDER BY timestamp DESC LIMIT 3")
                    logger.info(f"📝 Sample {table_name}:")
                    for row in sample:
                        logger.info(f"  - {row[0]} | {row[1]} | {row[2]} | {row[3]}")
                
                elif table_name == 'anomaly_scores' and total_count > 0:
                    # Show sample anomaly scores
                    sample = client.execute(f"SELECT timestamp, model_name, entity_id, score, is_anomaly FROM {table_name} ORDER BY timestamp DESC LIMIT 3")
                    logger.info(f"🤖 Sample {table_name}:")
                    for row in sample:
                        logger.info(f"  - {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")
                
                elif table_name == 'alerts' and total_count > 0:
                    # Show sample alerts
                    sample = client.execute(f"SELECT timestamp, severity, entity_id, aggregated_score FROM {table_name} ORDER BY timestamp DESC LIMIT 3")
                    logger.info(f"🚨 Sample {table_name}:")
                    for row in sample:
                        logger.info(f"  - {row[0]} | {row[1]} | {row[2]} | {row[3]}")
                        
            except Exception as e:
                logger.error(f"❌ Error checking table {table_name}: {e}")
        
        return client
        
    except Exception as e:
        logger.error(f"❌ ClickHouse connection failed: {e}")
        return None

def check_kafka_topics():
    """Check Kafka topics and message flow"""
    logger = setup_logging()
    
    try:
        from kafka import KafkaConsumer
        from kafka.errors import NoBrokersAvailable
        
        broker = os.getenv('KAFKA_BROKER', 'kafka:29092')
        logger.info(f"🔗 Connecting to Kafka broker: {broker}")
        
        # Check topics
        topics = ['raw-logs', 'anomaly-scores', 'alerts']
        
        for topic in topics:
            try:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=[broker],
                    auto_offset_reset='latest',
                    consumer_timeout_ms=5000,
                    value_deserializer=lambda m: m.decode('utf-8') if m else None
                )
                
                # Get topic info
                partitions = consumer.partitions_for_topic(topic)
                logger.info(f"📨 Topic {topic}: {len(partitions) if partitions else 0} partitions")
                
                # Try to read a few messages
                message_count = 0
                for message in consumer:
                    message_count += 1
                    logger.info(f"📩 {topic} message: {message.value[:100]}...")
                    if message_count >= 2:
                        break
                
                if message_count == 0:
                    logger.warning(f"⚠️ No recent messages in {topic}")
                
                consumer.close()
                
            except Exception as e:
                logger.error(f"❌ Error checking topic {topic}: {e}")
    
    except ImportError:
        logger.error("❌ kafka-python not available")
    except Exception as e:
        logger.error(f"❌ Kafka connection failed: {e}")

def check_service_logs():
    """Check Docker service logs for errors"""
    logger = setup_logging()
    
    services = [
        'soc-data-generator',
        'soc-ml-pipeline', 
        'soc-scoring-engine',
        'soc-alerting'
    ]
    
    for service in services:
        try:
            import subprocess
            
            # Get last 10 lines of logs
            result = subprocess.run(
                ['docker', 'logs', '--tail', '10', service],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                logger.info(f"📋 {service} logs (last 10 lines):")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
                        
                if result.stderr:
                    logger.warning(f"⚠️ {service} stderr:")
                    for line in result.stderr.strip().split('\n'):
                        if line.strip():
                            logger.warning(f"  {line}")
            else:
                logger.error(f"❌ Failed to get logs for {service}")
                
        except Exception as e:
            logger.error(f"❌ Error checking {service} logs: {e}")

def analyze_data_flow_issues(client):
    """Analyze specific data flow issues"""
    logger = setup_logging()
    
    if not client:
        logger.error("❌ No ClickHouse client available")
        return
    
    try:
        # Check if raw logs are being inserted
        recent_logs = client.execute("SELECT count() FROM raw_logs WHERE timestamp >= now() - INTERVAL 5 MINUTE")
        recent_log_count = recent_logs[0][0]
        
        logger.info(f"📊 Raw logs in last 5 minutes: {recent_log_count}")
        
        if recent_log_count == 0:
            logger.error("❌ ISSUE: No recent raw logs - Data Generator or ML Pipeline not working")
            logger.info("🔧 CHECK: Data Generator → Kafka → ML Pipeline → ClickHouse")
            return
        
        # Check if anomaly scores are being generated
        recent_scores = client.execute("SELECT count() FROM anomaly_scores WHERE timestamp >= now() - INTERVAL 5 MINUTE")
        recent_score_count = recent_scores[0][0]
        
        logger.info(f"🤖 Anomaly scores in last 5 minutes: {recent_score_count}")
        
        if recent_score_count == 0:
            logger.error("❌ ISSUE: Raw logs exist but no anomaly scores - ML Pipeline not generating/storing scores")
            logger.info("🔧 CHECK: ML Pipeline anomaly score generation and database insertion")
            
            # Check if ML models are enabled
            try:
                # This would require checking model status, but we can infer from the architecture
                logger.info("🔧 LIKELY CAUSES:")
                logger.info("  1. ML models not generating meaningful scores (threshold too high)")
                logger.info("  2. ML Pipeline not calling db.insert_anomaly_scores()")
                logger.info("  3. Database insertion failing silently")
            except Exception as e:
                logger.error(f"Error checking ML models: {e}")
            return
        
        # Check if alerts are being generated  
        recent_alerts = client.execute("SELECT count() FROM alerts WHERE timestamp >= now() - INTERVAL 5 MINUTE")
        recent_alert_count = recent_alerts[0][0]
        
        logger.info(f"🚨 Alerts in last 5 minutes: {recent_alert_count}")
        
        if recent_alert_count == 0:
            logger.warning("⚠️ Scores exist but no alerts - either no high-risk events or Scoring Engine not working")
            
            # Check score distribution
            score_stats = client.execute("""
                SELECT 
                    avg(score) as avg_score,
                    max(score) as max_score,
                    count() as total_scores,
                    countIf(score > 0.8) as high_scores
                FROM anomaly_scores 
                WHERE timestamp >= now() - INTERVAL 1 HOUR
            """)
            
            if score_stats:
                avg_score, max_score, total_scores, high_scores = score_stats[0]
                logger.info(f"📊 Score statistics (1h): avg={avg_score:.3f}, max={max_score:.3f}, high_scores={high_scores}/{total_scores}")
                
                if high_scores == 0:
                    logger.info("ℹ️ No high scores - system appears secure (no alerts expected)")
                else:
                    logger.error("❌ ISSUE: High scores exist but no alerts - Scoring Engine not working")
        else:
            logger.info("✅ Complete data flow working: Logs → Scores → Alerts")
            
            # Show alert distribution
            alert_stats = client.execute("""
                SELECT 
                    severity,
                    count() as alert_count
                FROM alerts 
                WHERE timestamp >= now() - INTERVAL 1 HOUR
                GROUP BY severity
                ORDER BY alert_count DESC
            """)
            
            logger.info("🚨 Alert distribution (1h):")
            for row in alert_stats:
                logger.info(f"  - {row[0]}: {row[1]} alerts")
    
    except Exception as e:
        logger.error(f"❌ Error analyzing data flow: {e}")

def check_ml_pipeline_status():
    """Check ML Pipeline specific issues"""
    logger = setup_logging()
    
    logger.info("🤖 Checking ML Pipeline Status...")
    
    # Check if ML Pipeline container is running
    try:
        import subprocess
        
        result = subprocess.run(
            ['docker', 'ps', '--filter', 'name=soc-ml-pipeline', '--format', 'table {{.Names}}\t{{.Status}}'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info(f"🐳 ML Pipeline container status:")
            logger.info(result.stdout)
        
        # Check recent ML Pipeline logs for key indicators
        result = subprocess.run(
            ['docker', 'logs', '--tail', '20', 'soc-ml-pipeline'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logs = result.stdout
            
            # Look for key indicators
            if "✅ Connected to ClickHouse" in logs:
                logger.info("✅ ML Pipeline connected to ClickHouse")
            else:
                logger.error("❌ ML Pipeline not connected to ClickHouse")
            
            if "✅ Started Kafka consumer" in logs:
                logger.info("✅ ML Pipeline consuming from Kafka")
            else:
                logger.error("❌ ML Pipeline not consuming from Kafka")
            
            if "💾 Stored" in logs and "raw logs" in logs:
                logger.info("✅ ML Pipeline storing raw logs")
            else:
                logger.error("❌ ML Pipeline not storing raw logs")
            
            if "💾 Stored" in logs and "anomaly scores" in logs:
                logger.info("✅ ML Pipeline storing anomaly scores")
            else:
                logger.error("❌ ML Pipeline not storing anomaly scores")
                logger.info("🔧 This is likely the main issue!")
    
    except Exception as e:
        logger.error(f"❌ Error checking ML Pipeline: {e}")

def main():
    """Main debugging function"""
    logger = setup_logging()
    
    logger.info("🔍 SOC Platform Data Flow Debugging")
    logger.info("="*50)
    
    # Step 1: Check ClickHouse
    logger.info("🗄️ Step 1: Checking ClickHouse Database...")
    client = check_clickhouse_connection()
    
    # Step 2: Check Kafka
    logger.info("\n📨 Step 2: Checking Kafka Topics...")
    check_kafka_topics()
    
    # Step 3: Check service logs
    logger.info("\n📋 Step 3: Checking Service Logs...")
    check_service_logs()
    
    # Step 4: Analyze data flow
    logger.info("\n🔄 Step 4: Analyzing Data Flow...")
    analyze_data_flow_issues(client)
    
    # Step 5: Check ML Pipeline specifically
    logger.info("\n🤖 Step 5: ML Pipeline Deep Check...")
    check_ml_pipeline_status()
    
    logger.info("\n" + "="*50)
    logger.info("🎯 Debugging Complete!")
    
    # Provide recommendations
    logger.info("\n💡 RECOMMENDATIONS:")
    logger.info("1. If only raw_logs have data:")
    logger.info("   → ML Pipeline is not storing anomaly scores")
    logger.info("   → Check ML Pipeline logs for database insertion errors")
    logger.info("2. If no anomaly_scores:")
    logger.info("   → Replace run_ml_pipeline.py with fixed microservices version")
    logger.info("3. If scores exist but no alerts:")
    logger.info("   → Check Scoring Engine Kafka consumption")
    logger.info("4. Check your original architecture diagram alignment")

if __name__ == "__main__":
    main()