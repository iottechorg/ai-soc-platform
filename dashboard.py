# # dashboard.py - ROBUST VERSION (handles config errors gracefully)
# import streamlit as st
# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# from datetime import datetime, timedelta
# import numpy as np
# import os
# import sys
# from pathlib import Path

# # Configure Streamlit first
# st.set_page_config(
#     page_title="SOC Platform Dashboard",
#     page_icon="🛡️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Add current directory to path
# sys.path.insert(0, str(Path(__file__).parent))

# # Robust imports with fallbacks
# try:
#     from shared.database import ClickHouseClient
#     DATABASE_AVAILABLE = True
# except ImportError as e:
#     st.sidebar.error(f"Database module not available: {e}")
#     DATABASE_AVAILABLE = False

# try:
#     from config.config_loader import ConfigLoader
#     CONFIG_AVAILABLE = True
# except ImportError as e:
#     st.sidebar.error(f"Config module not available: {e}")
#     CONFIG_AVAILABLE = False

# class RobustSOCDashboard:
#     """SOC Dashboard that handles configuration and connection errors gracefully"""
    
#     def __init__(self):
#         self.db = None
#         self.config = None
#         self.connection_status = "unknown"
        
#         # Try to initialize config
#         if CONFIG_AVAILABLE:
#             try:
#                 self.config = ConfigLoader()
#                 self.connection_status = "config_loaded"
#             except Exception as e:
#                 st.sidebar.error(f"Config loading failed: {str(e)[:100]}...")
#                 self.connection_status = "config_failed"
        
#         # Try to initialize database
#         if DATABASE_AVAILABLE and self.config:
#             try:
#                 self.db = ClickHouseClient(
#                     host=self.config.get('clickhouse.host', 'clickhouse'),
#                     port=9000,  # Force native port
#                     user=self.config.get('clickhouse.user', 'default'),
#                     password=self.config.get('clickhouse.password', ''),
#                     database=self.config.get('clickhouse.database', 'soc_platform')
#                 )
#                 self.db.connect()
#                 if self.db.is_connected():
#                     self.connection_status = "database_connected"
#                 else:
#                     self.connection_status = "database_failed"
#             except Exception as e:
#                 st.sidebar.error(f"Database connection failed: {str(e)[:100]}...")
#                 self.connection_status = "database_failed"
#                 self.db = None

# def main():
#     """Main dashboard function with comprehensive error handling"""
    
#     st.title("🛡️ AI-Driven SOC Platform Dashboard")
    
#     # Initialize dashboard with error handling
#     try:
#         dashboard = RobustSOCDashboard()
#     except Exception as e:
#         st.error(f"Failed to initialize dashboard: {e}")
#         st.info("Running in minimal mode...")
#         dashboard = None
    
#     # Connection status display
#     if dashboard:
#         if dashboard.connection_status == "database_connected":
#             st.success("✅ System Online - Real Data Available")
#         elif dashboard.connection_status == "config_failed":
#             st.error("❌ Configuration Error - Check settings.yaml")
#         elif dashboard.connection_status == "database_failed":
#             st.warning("⚠️ Database Offline - Showing Demo Data")
#         else:
#             st.info("ℹ️ Limited Mode - Basic Functionality Only")
#     else:
#         st.error("❌ Dashboard Initialization Failed")
    
#     # Sidebar with error handling
#     render_sidebar(dashboard)
    
#     # Main content tabs
#     tab1, tab2, tab3, tab4 = st.tabs([
#         "🔍 Real-time Monitoring",
#         "📊 Analytics", 
#         "🤖 ML Models",
#         "⚙️ System Status"
#     ])
    
#     with tab1:
#         render_realtime_monitoring(dashboard)
    
#     with tab2:
#         render_analytics(dashboard)
    
#     with tab3:
#         render_ml_models(dashboard)
    
#     with tab4:
#         render_system_status(dashboard)

# def render_sidebar(dashboard):
#     """Render sidebar with error handling"""
#     with st.sidebar:
#         st.header("Dashboard Controls")
        
#         # Basic controls that always work
#         time_range = st.selectbox(
#             "Time Range", 
#             ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days"], 
#             index=2
#         )
        
#         # Auto-refresh with error handling
#         auto_refresh = st.checkbox("Auto Refresh (30s)")
#         if st.button("🔄 Refresh"):
#             try:
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Refresh failed: {e}")
        
#         st.markdown("---")
        
#         # Quick stats
#         st.subheader("📊 Quick Stats")
#         if dashboard and dashboard.connection_status == "database_connected":
#             try:
#                 # Try to get real data
#                 hours = parse_time_range(time_range)
#                 log_count = get_real_log_count(dashboard.db, hours)
#                 alert_count = get_real_alert_count(dashboard.db, hours)
#                 active_alerts = get_real_active_alerts(dashboard.db)
                
#                 st.metric("Logs", f"{log_count:,}")
#                 st.metric("Alerts", alert_count)
#                 st.metric("Active Alerts", active_alerts)
#             except Exception as e:
#                 st.error(f"Stats error: {str(e)[:50]}...")
#                 render_demo_sidebar_stats()
#         else:
#             render_demo_sidebar_stats()
        
#         st.markdown("---")
        
#         # System status
#         st.subheader("⚙️ System Status")
#         render_system_status_sidebar(dashboard)

# def render_realtime_monitoring(dashboard):
#     """Render monitoring tab with comprehensive error handling"""
#     st.header("Real-time Security Monitoring")
    
#     # Key metrics
#     col1, col2, col3, col4 = st.columns(4)
    
#     if dashboard and dashboard.connection_status == "database_connected":
#         try:
#             # Real metrics
#             hours = 24  # Default
#             log_count = get_real_log_count(dashboard.db, hours)
#             alert_count = get_real_alert_count(dashboard.db, hours)
#             active_alerts = get_real_active_alerts(dashboard.db)
#             detection_rate = calculate_real_detection_rate(dashboard.db, hours)
            
#             with col1:
#                 st.metric("Logs Generated", f"{log_count:,}", delta=f"+{max(log_count//10, 1)}")
#             with col2:
#                 st.metric("Alerts Generated", alert_count, delta=f"+{max(alert_count//5, 1)}")
#             with col3:
#                 st.metric("Active Alerts", active_alerts, delta="2")
#             with col4:
#                 st.metric("Detection Rate", f"{detection_rate:.1f}%", delta="2.1%")
#         except Exception as e:
#             st.error(f"Metrics error: {e}")
#             render_demo_metrics(col1, col2, col3, col4)
#     else:
#         render_demo_metrics(col1, col2, col3, col4)
    
#     # Recent alerts
#     st.subheader("Recent High-Priority Alerts")
    
#     if dashboard and dashboard.connection_status == "database_connected":
#         try:
#             alerts = get_real_recent_alerts(dashboard.db, limit=10)
            
#             if alerts:
#                 for alert in alerts[:5]:  # Show top 5
#                     severity = alert.get('severity', 'low').lower()
#                     entity = alert.get('entity_id', 'unknown')
#                     message = alert.get('message', 'No details')
#                     score = alert.get('aggregated_score', 0.0)
#                     timestamp = alert.get('timestamp', datetime.now())
                    
#                     if isinstance(timestamp, str):
#                         try:
#                             timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
#                         except:
#                             timestamp = datetime.now()
                    
#                     time_str = timestamp.strftime('%H:%M:%S')
                    
#                     if severity == 'critical':
#                         st.error(f"🔴 **CRITICAL** - {entity} - Score: {score:.3f} - {message} ({time_str})")
#                     elif severity == 'high':
#                         st.warning(f"🟠 **HIGH** - {entity} - Score: {score:.3f} - {message} ({time_str})")
#                     else:
#                         st.info(f"🟡 **{severity.upper()}** - {entity} - {message} ({time_str})")
                
#                 # Show alerts table
#                 with st.expander("View All Recent Alerts"):
#                     try:
#                         df = pd.DataFrame(alerts)
#                         if not df.empty:
#                             display_cols = ['timestamp', 'severity', 'entity_id', 'message', 'aggregated_score']
#                             available_cols = [col for col in display_cols if col in df.columns]
#                             st.dataframe(df[available_cols], use_container_width=True)
#                     except Exception as e:
#                         st.error(f"Table display error: {e}")
#             else:
#                 st.success("🟢 No recent critical alerts - system appears secure")
#         except Exception as e:
#             st.error(f"Alerts error: {e}")
#             render_demo_alerts()
#     else:
#         render_demo_alerts()

# def render_analytics(dashboard):
#     """Render analytics tab"""
#     st.header("Security Analytics")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.subheader("Threat Detection Trends")
#         render_demo_timeline()
    
#     with col2:
#         st.subheader("Top Threat Sources")
#         render_demo_entities()

# def render_ml_models(dashboard):
#     """Render ML models tab"""
#     st.header("ML Model Management")
#     st.subheader("Current Model Status")
    
#     # Demo model status
#     models = [
#         {'name': 'Isolation Forest', 'status': 'Active', 'accuracy': '89%', 'last_update': '2 min ago'},
#         {'name': 'Clustering', 'status': 'Active', 'accuracy': '76%', 'last_update': '2 min ago'},
#         {'name': 'Time Series', 'status': 'Inactive', 'accuracy': '82%', 'last_update': '5 min ago'},
#         {'name': 'Forbidden Ratio', 'status': 'Inactive', 'accuracy': '71%', 'last_update': '5 min ago'}
#     ]
    
#     for model in models:
#         col1, col2, col3, col4 = st.columns(4)
#         status_icon = "🟢" if model['status'] == 'Active' else "🔴"
        
#         with col1:
#             st.write(f"{status_icon} **{model['name']}**")
#         with col2:
#             st.write(model['status'])
#         with col3:
#             st.write(model['accuracy'])
#         with col4:
#             st.write(model['last_update'])

# def render_system_status(dashboard):
#     """Render system status tab"""
#     st.header("System Status & Health")
    
#     # System metrics
#     col1, col2, col3 = st.columns(3)
    
#     with col1:
#         st.metric("CPU Usage", "67%", delta="5%")
#         st.metric("Memory Usage", "45%", delta="-2%")
    
#     with col2:
#         st.metric("Logs/Hour", "1,247")
#         st.metric("Processing Rate", "21/min")
    
#     with col3:
#         st.metric("Network I/O", "234 MB/s", delta="12")
#         st.metric("Error Rate", "0.3%", delta="-0.1%")
    
#     # Service status
#     st.subheader("Service Health")
#     render_service_status_table(dashboard)

# # Helper functions
# def parse_time_range(time_range: str) -> int:
#     """Parse time range to hours"""
#     mapping = {
#         "Last 1 hour": 1,
#         "Last 6 hours": 6,
#         "Last 24 hours": 24,
#         "Last 7 days": 168
#     }
#     return mapping.get(time_range, 24)

# def get_real_log_count(db, hours):
#     """Get real log count with error handling"""
#     try:
#         if db and db.is_connected():
#             result = db.client.execute(f"SELECT count() FROM raw_logs WHERE timestamp >= now() - INTERVAL {hours} HOUR")
#             return result[0][0] if result else 0
#     except:
#         pass
#     return 0

# def get_real_alert_count(db, hours):
#     """Get real alert count with error handling"""
#     try:
#         if db and db.is_connected():
#             result = db.client.execute(f"SELECT count() FROM alerts WHERE timestamp >= now() - INTERVAL {hours} HOUR")
#             return result[0][0] if result else 0
#     except:
#         pass
#     return 0

# def get_real_active_alerts(db):
#     """Get real active alert count with error handling"""
#     try:
#         if db and db.is_connected():
#             result = db.client.execute("SELECT count() FROM alerts WHERE status = 'open'")
#             return result[0][0] if result else 0
#     except:
#         pass
#     return 0

# def calculate_real_detection_rate(db, hours):
#     """Calculate real detection rate with error handling"""
#     try:
#         log_count = get_real_log_count(db, hours)
#         alert_count = get_real_alert_count(db, hours)
#         return (alert_count / max(log_count, 1)) * 100
#     except:
#         pass
#     return 0.0

# def get_real_recent_alerts(db, limit=10):
#     """Get real recent alerts with error handling"""
#     try:
#         if db and db.is_connected():
#             result = db.client.execute(f"""
#                 SELECT timestamp, alert_id, severity, entity_id, message, aggregated_score, status
#                 FROM alerts
#                 ORDER BY timestamp DESC
#                 LIMIT {limit}
#             """)
            
#             alerts = []
#             for row in result:
#                 alerts.append({
#                     'timestamp': row[0],
#                     'alert_id': row[1],
#                     'severity': row[2],
#                     'entity_id': row[3],
#                     'message': row[4],
#                     'aggregated_score': row[5],
#                     'status': row[6]
#                 })
#             return alerts
#     except Exception as e:
#         st.sidebar.error(f"Alert query error: {str(e)[:50]}...")
#     return []

# # Demo/fallback functions
# def render_demo_sidebar_stats():
#     st.metric("Logs", "12,470")
#     st.metric("Alerts", "45")
#     st.metric("Active Alerts", "8")

# def render_demo_metrics(col1, col2, col3, col4):
#     with col1: st.metric("Logs Generated", "12,470", delta="+1,247")
#     with col2: st.metric("Alerts Generated", "45", delta="+8")
#     with col3: st.metric("Active Alerts", "8", delta="+2")
#     with col4: st.metric("Detection Rate", "94.2%", delta="+2.1%")

# def render_demo_alerts():
#     st.warning("⚠️ Database offline - showing demo data")
#     st.error("🔴 **CRITICAL** - 192.168.1.100 - Score: 0.950 - Brute force attack detected (12:34:56)")

# def render_demo_timeline():
#     hours = pd.date_range(end=datetime.now(), periods=24, freq='H')
#     demo_data = pd.DataFrame({
#         'time': hours,
#         'detections': np.random.poisson(8, 24)
#     })
    
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=demo_data['time'], y=demo_data['detections'],
#                            mode='lines+markers', name='Demo Detections'))
#     fig.update_layout(title="Demo: Threat Timeline", height=400)
#     st.plotly_chart(fig, use_container_width=True)

# def render_demo_entities():
#     demo_data = pd.DataFrame({
#         'entity_id': ['192.168.1.100', '10.0.0.45', '172.16.0.23'],
#         'threat_score': [0.95, 0.82, 0.76]
#     })
#     fig = px.bar(demo_data, x='threat_score', y='entity_id', orientation='h')
#     fig.update_layout(title="Demo: Top Entities", height=400)
#     st.plotly_chart(fig, use_container_width=True)

# def render_system_status_sidebar(dashboard):
#     services = [
#         ("Data Generator", True),
#         ("ML Pipeline", True),
#         ("RL Orchestrator", True),
#         ("Scoring Engine", True),
#         ("Alerting Service", True),
#         ("ClickHouse", dashboard and dashboard.connection_status == "database_connected"),
#         ("Kafka", True),
#         ("Dashboard", True)
#     ]
    
#     for service, status in services:
#         icon = "🟢" if status else "🔴"
#         st.text(f"{icon} {service}")

# def render_service_status_table(dashboard):
#     services_data = {
#         'Service': ["Data Generator", "ML Pipeline", "RL Orchestrator", "Scoring Engine", "Alerting Service", "ClickHouse", "Kafka", "Dashboard"],
#         'Status': ["Running", "Running", "Running", "Running", "Running", 
#                   "Connected" if dashboard and dashboard.connection_status == "database_connected" else "Disconnected", 
#                   "Running", "Running"],
#         'Health': ["🟢", "🟢", "🟢", "🟢", "🟢", 
#                   "🟢" if dashboard and dashboard.connection_status == "database_connected" else "🔴", 
#                   "🟢", "🟢"]
#     }
    
#     df_services = pd.DataFrame(services_data)
#     st.dataframe(df_services, use_container_width=True)

# if __name__ == "__main__":
#     main()

# setup_database.py - Ensure ClickHouse tables exist with correct schema
#!/usr/bin/env python3

#!/usr/bin/env python3
# dashboard_working.py - WORKING dashboard that shows real ClickHouse data

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import numpy as np

# Configure Streamlit
st.set_page_config(
    page_title="SOC Platform - Live Data",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_clickhouse_client():
    """Get ClickHouse client with connection caching"""
    try:
        from clickhouse_driver import Client
        return Client(
            host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
            port=int(os.getenv('CLICKHOUSE_PORT', '9000')),
            user=os.getenv('CLICKHOUSE_USER', 'default'),
            password=os.getenv('CLICKHOUSE_PASSWORD', 'secure_password'),
            database=os.getenv('CLICKHOUSE_DATABASE', 'soc_platform'),
            connect_timeout=5,
            send_receive_timeout=10
        )
    except Exception as e:
        st.error(f"ClickHouse connection error: {e}")
        return None

def check_database_connection():
    """Check if we can connect to ClickHouse"""
    client = get_clickhouse_client()
    if not client:
        return False, "No client"
    
    try:
        client.execute("SELECT 1")
        return True, "Connected"
    except Exception as e:
        return False, str(e)

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_data_counts():
    """Get counts of data in different tables"""
    client = get_clickhouse_client()
    if not client:
        return None
    
    try:
        # Get log counts
        result = client.execute("SELECT count() FROM raw_logs")
        total_logs = result[0][0]
        
        result = client.execute("SELECT count() FROM raw_logs WHERE timestamp >= now() - INTERVAL 1 HOUR")
        recent_logs = result[0][0]
        
        # Get anomaly score counts
        result = client.execute("SELECT count() FROM anomaly_scores")
        total_scores = result[0][0]
        
        result = client.execute("SELECT count() FROM anomaly_scores WHERE timestamp >= now() - INTERVAL 1 HOUR")
        recent_scores = result[0][0]
        
        # Get alert counts
        result = client.execute("SELECT count() FROM alerts")
        total_alerts = result[0][0]
        
        result = client.execute("SELECT count() FROM alerts WHERE timestamp >= now() - INTERVAL 1 HOUR")
        recent_alerts = result[0][0]
        
        return {
            'total_logs': total_logs,
            'recent_logs': recent_logs,
            'total_scores': total_scores,
            'recent_scores': recent_scores,
            'total_alerts': total_alerts,
            'recent_alerts': recent_alerts
        }
    except Exception as e:
        st.error(f"Error getting data counts: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_recent_logs(limit=20):
    """Get recent logs from database"""
    client = get_clickhouse_client()
    if not client:
        return pd.DataFrame()
    
    try:
        result = client.execute(f"""
            SELECT timestamp, event_type, source_ip, destination_ip, port, protocol, severity, message
            FROM raw_logs 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """)
        
        if result:
            df = pd.DataFrame(result, columns=[
                'Timestamp', 'Event Type', 'Source IP', 'Dest IP', 
                'Port', 'Protocol', 'Severity', 'Message'
            ])
            # Convert timestamp to string for display
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%H:%M:%S')
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting recent logs: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_anomaly_scores(limit=20):
    """Get recent anomaly scores"""
    client = get_clickhouse_client()
    if not client:
        return pd.DataFrame()
    
    try:
        result = client.execute(f"""
            SELECT timestamp, model_name, entity_id, score, is_anomaly
            FROM anomaly_scores 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """)
        
        if result:
            df = pd.DataFrame(result, columns=[
                'Timestamp', 'Model', 'Entity', 'Score', 'Is Anomaly'
            ])
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%H:%M:%S')
            df['Is Anomaly'] = df['Is Anomaly'].map({1: '🔴 Yes', 0: '🟢 No'})
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting anomaly scores: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_model_activity():
    """Get ML model activity statistics"""
    client = get_clickhouse_client()
    if not client:
        return pd.DataFrame()
    
    try:
        result = client.execute("""
            SELECT 
                model_name,
                count() as total_scores,
                avg(score) as avg_score,
                sum(is_anomaly) as anomalies_detected
            FROM anomaly_scores 
            WHERE timestamp >= now() - INTERVAL 1 HOUR
            GROUP BY model_name 
            ORDER BY total_scores DESC
        """)
        
        if result:
            df = pd.DataFrame(result, columns=[
                'Model', 'Total Scores', 'Avg Score', 'Anomalies Detected'
            ])
            df['Avg Score'] = df['Avg Score'].round(3)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting model activity: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_alerts():
    """Get recent alerts"""
    client = get_clickhouse_client()
    if not client:
        return pd.DataFrame()
    
    try:
        result = client.execute("""
            SELECT timestamp, alert_id, severity, entity_id, message, aggregated_score, status
            FROM alerts 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        if result:
            df = pd.DataFrame(result, columns=[
                'Timestamp', 'Alert ID', 'Severity', 'Entity', 'Message', 'Score', 'Status'
            ])
            df['Timestamp'] = pd.to_datetime(df['Timestamp']).dt.strftime('%H:%M:%S')
            # Add severity icons
            severity_icons = {
                'critical': '🔴',
                'high': '🟠', 
                'medium': '🟡',
                'low': '🟢'
            }
            df['Severity'] = df['Severity'].map(lambda x: f"{severity_icons.get(x, '⚪')} {x.title()}")
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting alerts: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_timeline_data():
    """Get timeline data for charts"""
    client = get_clickhouse_client()
    if not client:
        return pd.DataFrame()
    
    try:
        result = client.execute("""
            SELECT 
                toStartOfHour(timestamp) as hour,
                count() as log_count,
                countIf(severity IN ('high', 'critical')) as threat_logs
            FROM raw_logs 
            WHERE timestamp >= now() - INTERVAL 24 HOUR
            GROUP BY hour 
            ORDER BY hour
        """)
        
        if result:
            df = pd.DataFrame(result, columns=['Hour', 'Log Count', 'Threat Logs'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting timeline data: {e}")
        return pd.DataFrame()

def main():
    """Main dashboard function"""
    
    st.title("🛡️ AI-Driven SOC Platform - Live Dashboard")
    
    # Check database connection
    is_connected, connection_msg = check_database_connection()
    
    if is_connected:
        st.success("✅ Connected to ClickHouse Database - Showing Real Data")
    else:
        st.error(f"❌ Database Connection Failed: {connection_msg}")
        st.info("Make sure ClickHouse is running and the enhanced ML pipeline is storing data")
        st.stop()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=True)
        
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Get data counts for sidebar
        counts = get_data_counts()
        if counts:
            st.subheader("📊 Data Status")
            st.metric("Total Logs", f"{counts['total_logs']:,}")
            st.metric("Recent Logs (1h)", f"{counts['recent_logs']:,}")
            st.metric("Total Scores", f"{counts['total_scores']:,}")
            st.metric("Recent Scores (1h)", f"{counts['recent_scores']:,}")
            st.metric("Total Alerts", f"{counts['total_alerts']:,}")
            st.metric("Recent Alerts (1h)", f"{counts['recent_alerts']:,}")
            
            # Data flow status
            st.markdown("---")
            st.subheader("🔄 Data Flow Status")
            
            if counts['recent_logs'] > 0:
                st.success("✅ Data Generator → ClickHouse")
            else:
                st.error("❌ No recent logs")
            
            if counts['recent_scores'] > 0:
                st.success("✅ ML Models → ClickHouse")
            else:
                st.error("❌ No recent scores")
            
            if counts['recent_alerts'] > 0:
                st.warning("⚠️ Active Alerts Detected")
            else:
                st.success("✅ No Recent Alerts")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-Time Overview",
        "🔍 Raw Logs", 
        "🤖 ML Model Activity",
        "🚨 Security Alerts"
    ])
    
    with tab1:
        st.header("📊 Real-Time Security Overview")
        
        counts = get_data_counts()
        if counts:
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "📝 Logs (1h)", 
                    f"{counts['recent_logs']:,}",
                    delta=f"Total: {counts['total_logs']:,}"
                )
            
            with col2:
                st.metric(
                    "🤖 ML Scores (1h)", 
                    f"{counts['recent_scores']:,}",
                    delta=f"Total: {counts['total_scores']:,}"
                )
            
            with col3:
                st.metric(
                    "🚨 Alerts (1h)", 
                    f"{counts['recent_alerts']:,}",
                    delta=f"Total: {counts['total_alerts']:,}"
                )
            
            with col4:
                if counts['recent_logs'] > 0:
                    detection_rate = (counts['recent_scores'] / counts['recent_logs']) * 100
                    st.metric("🎯 Detection Rate", f"{detection_rate:.1f}%")
                else:
                    st.metric("🎯 Detection Rate", "0%")
        
        # Timeline chart
        timeline_df = get_timeline_data()
        if not timeline_df.empty:
            st.subheader("📈 Activity Timeline (Last 24 Hours)")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timeline_df['Hour'],
                y=timeline_df['Log Count'],
                mode='lines+markers',
                name='Total Logs',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=timeline_df['Hour'],
                y=timeline_df['Threat Logs'],
                mode='lines+markers',
                name='Threat Logs',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title="Log Activity Over Time",
                xaxis_title="Time",
                yaxis_title="Log Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model activity
        model_df = get_model_activity()
        if not model_df.empty:
            st.subheader("🤖 ML Model Activity (Last Hour)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    model_df, 
                    x='Model', 
                    y='Total Scores',
                    title="Scores Generated by Model"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    model_df, 
                    x='Model', 
                    y='Anomalies Detected',
                    title="Anomalies Detected by Model",
                    color='Anomalies Detected',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🔍 Raw Security Logs")
        
        logs_df = get_recent_logs(50)
        if not logs_df.empty:
            st.subheader("📋 Most Recent Logs")
            
            # Filter controls
            col1, col2 = st.columns(2)
            with col1:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=logs_df['Severity'].unique(),
                    default=logs_df['Severity'].unique()
                )
            
            with col2:
                protocol_filter = st.multiselect(
                    "Filter by Protocol",
                    options=logs_df['Protocol'].unique(),
                    default=logs_df['Protocol'].unique()
                )
            
            # Apply filters
            filtered_df = logs_df[
                (logs_df['Severity'].isin(severity_filter)) &
                (logs_df['Protocol'].isin(protocol_filter))
            ]
            
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Summary stats
            st.subheader("📊 Log Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                severity_counts = logs_df['Severity'].value_counts()
                fig = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Logs by Severity"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                protocol_counts = logs_df['Protocol'].value_counts().head(10)
                fig = px.bar(
                    x=protocol_counts.values,
                    y=protocol_counts.index,
                    orientation='h',
                    title="Top Protocols"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                port_stats = logs_df['Port'].describe()
                st.write("**Port Statistics:**")
                st.write(f"Most Common: {logs_df['Port'].mode().iloc[0] if not logs_df['Port'].mode().empty else 'N/A'}")
                st.write(f"Average: {port_stats['mean']:.0f}")
                st.write(f"Range: {port_stats['min']:.0f} - {port_stats['max']:.0f}")
        else:
            st.warning("⚠️ No logs found in database")
            st.info("Make sure the enhanced ML pipeline is running and storing data")
    
    with tab3:
        st.header("🤖 ML Model Activity & Performance")
        
        # Model activity table
        model_df = get_model_activity()
        if not model_df.empty:
            st.subheader("📊 Model Performance (Last Hour)")
            st.dataframe(model_df, use_container_width=True)
        
        # Recent anomaly scores
        scores_df = get_anomaly_scores(30)
        if not scores_df.empty:
            st.subheader("🔍 Recent Anomaly Scores")
            st.dataframe(scores_df, use_container_width=True, height=400)
            
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    scores_df,
                    x='Score',
                    nbins=20,
                    title="Score Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                anomaly_counts = scores_df['Is Anomaly'].value_counts()
                fig = px.pie(
                    values=anomaly_counts.values,
                    names=anomaly_counts.index,
                    title="Anomaly Detection Results"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No anomaly scores found")
            st.info("Make sure ML models are running and generating scores")
    
    with tab4:
        st.header("🚨 Security Alerts & Incidents")
        
        alerts_df = get_alerts()
        if not alerts_df.empty:
            st.subheader("🔔 Recent Security Alerts")
            
            # Highlight critical alerts
            critical_alerts = alerts_df[alerts_df['Severity'].str.contains('Critical|High')]
            if not critical_alerts.empty:
                st.error(f"⚠️ {len(critical_alerts)} Critical/High Severity Alerts!")
                st.dataframe(critical_alerts, use_container_width=True)
            
            st.subheader("📋 All Recent Alerts")
            st.dataframe(alerts_df, use_container_width=True, height=400)
            
            # Alert statistics
            col1, col2 = st.columns(2)
            
            with col1:
                severity_dist = alerts_df['Severity'].str.extract(r'(Critical|High|Medium|Low)')[0].value_counts()
                fig = px.bar(
                    x=severity_dist.values,
                    y=severity_dist.index,
                    orientation='h',
                    title="Alerts by Severity"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                entity_counts = alerts_df['Entity'].value_counts().head(10)
                fig = px.bar(
                    x=entity_counts.values,
                    y=entity_counts.index,
                    orientation='h',
                    title="Top Alert Sources"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("✅ No Recent Security Alerts")
            st.info("System appears secure - no threats detected")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()