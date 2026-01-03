#!/usr/bin/env python3
# dashboard_fixed.py - FIXED dashboard with better ClickHouse connection handling

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import time
import numpy as np
import threading
from contextlib import contextmanager

# Configure Streamlit
st.set_page_config(
    page_title="SOC Platform - Live Data",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Connection pool to manage multiple connections
class ClickHouseConnectionPool:
    def __init__(self, max_connections=5):
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
    
    def get_connection(self):
        """Get a connection from the pool"""
        with self.lock:
            if self.connections:
                return self.connections.pop()
            else:
                return self._create_connection()
    
    def return_connection(self, connection):
        """Return a connection to the pool"""
        with self.lock:
            if len(self.connections) < self.max_connections:
                self.connections.append(connection)
    
    def _create_connection(self):
        """Create a new ClickHouse connection with optimized settings"""
        try:
            from clickhouse_driver import Client
            return Client(
                host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
                port=int(os.getenv('CLICKHOUSE_PORT', '9000')),
                user=os.getenv('CLICKHOUSE_USER', 'default'),
                password=os.getenv('CLICKHOUSE_PASSWORD', 'secure_password'),
                database=os.getenv('CLICKHOUSE_DATABASE', 'soc_platform'),
                connect_timeout=3,           # Reduced timeout
                send_receive_timeout=5,      # Reduced timeout
                sync_request_timeout=5,      # Add sync timeout
                # Optimize for dashboard queries
                settings={
                    'max_execution_time': 10,        # 10 second query timeout
                    'max_memory_usage': 1000000000,  # 1GB memory limit
                    'readonly': 1,                   # Read-only mode for dashboard
                    'max_rows_to_read': 100000,      # Limit rows for dashboard
                    'priority': 1                    # Lower priority than training
                }
            )
        except Exception as e:
            st.error(f"Failed to create ClickHouse connection: {e}")
            return None

# Global connection pool
@st.cache_resource
def get_connection_pool():
    return ClickHouseConnectionPool()

@contextmanager
def get_clickhouse_client():
    """Context manager for ClickHouse connections"""
    pool = get_connection_pool()
    client = pool.get_connection()
    try:
        yield client
    finally:
        if client:
            pool.return_connection(client)

def check_database_connection():
    """Check if we can connect to ClickHouse with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with get_clickhouse_client() as client:
                if client:
                    client.execute("SELECT 1")
                    return True, "Connected"
                else:
                    return False, "No client available"
        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"Max retries exceeded: {str(e)}"
            time.sleep(1)  # Wait before retry
    
    return False, "Unknown error"

def execute_query_with_retry(query, max_retries=2):
    """Execute query with retry logic and timeout handling"""
    for attempt in range(max_retries):
        try:
            with get_clickhouse_client() as client:
                if client:
                    return client.execute(query)
                else:
                    return None
        except Exception as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                if attempt == max_retries - 1:
                    st.warning(f"⚠️ Database query timed out (attempt {attempt + 1}): {query[:100]}...")
                    return None
                time.sleep(0.5)  # Brief wait before retry
            else:
                st.error(f"Database query error: {e}")
                return None
    return None

@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_data_counts():
    """Get counts of data in different tables with optimized queries"""
    try:
        # Use faster approximate counts for large tables
        queries = {
            'total_logs': "SELECT count() FROM raw_logs",
            'recent_logs': "SELECT count() FROM raw_logs WHERE timestamp >= now() - INTERVAL 1 HOUR",
            'total_scores': "SELECT count() FROM anomaly_scores",
            'recent_scores': "SELECT count() FROM anomaly_scores WHERE timestamp >= now() - INTERVAL 1 HOUR",
            'total_alerts': "SELECT count() FROM alerts",
            'recent_alerts': "SELECT count() FROM alerts WHERE timestamp >= now() - INTERVAL 1 HOUR"
        }
        
        results = {}
        for key, query in queries.items():
            result = execute_query_with_retry(query)
            if result:
                results[key] = result[0][0]
            else:
                results[key] = 0
        
        return results
    except Exception as e:
        st.error(f"Error getting data counts: {e}")
        return {
            'total_logs': 0,
            'recent_logs': 0,
            'total_scores': 0,
            'recent_scores': 0,
            'total_alerts': 0,
            'recent_alerts': 0
        }

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_recent_logs(limit=20):
    """Get recent logs from database with optimized query"""
    try:
        # Optimized query with LIMIT first
        query = f"""
            SELECT timestamp, event_type, source_ip, destination_ip, port, protocol, severity, message
            FROM raw_logs 
            WHERE timestamp >= now() - INTERVAL 2 HOUR
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        
        result = execute_query_with_retry(query)
        
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
    """Get recent anomaly scores with optimized query"""
    try:
        query = f"""
            SELECT timestamp, model_name, entity_id, score, is_anomaly
            FROM anomaly_scores 
            WHERE timestamp >= now() - INTERVAL 2 HOUR
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """
        
        result = execute_query_with_retry(query)
        
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
    """Get ML model activity statistics with optimized query"""
    try:
        query = """
            SELECT 
                model_name,
                count() as total_scores,
                round(avg(score), 3) as avg_score,
                sum(is_anomaly) as anomalies_detected
            FROM anomaly_scores 
            WHERE timestamp >= now() - INTERVAL 1 HOUR
            GROUP BY model_name 
            ORDER BY total_scores DESC
        """
        
        result = execute_query_with_retry(query)
        
        if result:
            df = pd.DataFrame(result, columns=[
                'Model', 'Total Scores', 'Avg Score', 'Anomalies Detected'
            ])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting model activity: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_alerts():
    """Get recent alerts with optimized query"""
    try:
        query = """
            SELECT timestamp, alert_id, severity, entity_id, message, aggregated_score, status
            FROM alerts 
            WHERE timestamp >= now() - INTERVAL 4 HOUR
            ORDER BY timestamp DESC 
            LIMIT 20
        """
        
        result = execute_query_with_retry(query)
        
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
    """Get timeline data for charts with optimized query"""
    try:
        query = """
            SELECT 
                toStartOfHour(timestamp) as hour,
                count() as log_count,
                countIf(severity IN ('high', 'critical')) as threat_logs
            FROM raw_logs 
            WHERE timestamp >= now() - INTERVAL 12 HOUR
            GROUP BY hour 
            ORDER BY hour
        """
        
        result = execute_query_with_retry(query)
        
        if result:
            df = pd.DataFrame(result, columns=['Hour', 'Log Count', 'Threat Logs'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting timeline data: {e}")
        return pd.DataFrame()

def show_connection_status():
    """Show database connection status with troubleshooting info"""
    is_connected, connection_msg = check_database_connection()
    
    if is_connected:
        st.success("✅ Connected to ClickHouse Database - Showing Real Data")
        return True
    else:
        st.error(f"❌ Database Connection Failed: {connection_msg}")
        
        # Show troubleshooting information
        with st.expander("🔧 Troubleshooting Database Connection"):
            st.markdown("""
            **Common Issues & Solutions:**
            
            1. **ML Training Blocking Database:**
               - Heavy training queries can block dashboard access
               - Try refreshing in a few minutes
               - Check if ML pipeline is running intensive training
            
            2. **Connection Settings:**
               - Host: `clickhouse` (in Docker)
               - Port: `9000` (native) or `8123` (HTTP)
               - Make sure ClickHouse container is running
            
            3. **Resource Constraints:**
               - ClickHouse might be under heavy load
               - Check container resource limits
               - Consider adding more memory/CPU
            
            4. **Network Issues:**
               - Verify containers are on same network
               - Check firewall/security settings
            
            **Quick Checks:**
            ```bash
            # Check ClickHouse container status
            docker ps | grep clickhouse
            
            # Check ClickHouse logs
            docker logs soc-clickhouse
            
            # Test connection manually
            docker exec -it soc-clickhouse clickhouse-client --query "SELECT 1"
            ```
            """)
        
        return False

def main():
    """Main dashboard function with improved error handling"""
    
    st.title("🛡️ AI-Driven SOC Platform - Live Dashboard")
    
    # Check database connection with status
    if not show_connection_status():
        st.info("🔄 Retrying connection in 10 seconds...")
        time.sleep(10)
        st.rerun()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Dashboard Controls")
        
        auto_refresh = st.checkbox("🔄 Auto Refresh (30s)", value=False)  # Disabled by default
        
        # if st.button("🔄 Refresh Now"):
        #     st.cache_data.clear()
        #     st.rerun()
        
        # if st.button("🔄 Force Reconnect"):
        #     st.cache_resource.clear()
        #     st.cache_data.clear()
        #     st.rerun()
        
        st.markdown("---")
        
        
        # Get data counts for sidebar
        with st.spinner("Loading data counts..."):
            counts = get_data_counts()
            
        if counts and any(counts.values()):
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
                st.warning("⚠️ No recent logs (may be normal)")
            
            if counts['recent_scores'] > 0:
                st.success("✅ ML Models → ClickHouse")
            else:
                st.warning("⚠️ No recent ML scores")
            
            if counts['recent_alerts'] > 0:
                st.warning("⚠️ Active Alerts Detected")
            else:
                st.success("✅ No Recent Alerts")
        else:
            st.warning("⚠️ No data found or connection issues")
    
    # Main content with error handling
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Real-Time Overview",
        "🔍 Raw Logs", 
        "🤖 ML Model Activity",
        "🚨 Security Alerts"
    ])
    
    with tab1:
        st.header("📊 Real-Time Security Overview")
        
        with st.spinner("Loading overview data..."):
            counts = get_data_counts()
            
        if counts and any(counts.values()):
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
        with st.spinner("Loading timeline data..."):
            timeline_df = get_timeline_data()
            
        if not timeline_df.empty:
            st.subheader("📈 Activity Timeline (Last 12 Hours)")
            
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
        else:
            st.info("📊 No timeline data available")
        
        # Model activity
        with st.spinner("Loading model activity..."):
            model_df = get_model_activity()
            
        if not model_df.empty:
            st.subheader("🤖 ML Model Activity (Last Hour)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not model_df.empty:
                    fig = px.bar(
                        model_df, 
                        x='Model', 
                        y='Total Scores',
                        title="Scores Generated by Model"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not model_df.empty:
                    fig = px.bar(
                        model_df, 
                        x='Model', 
                        y='Anomalies Detected',
                        title="Anomalies Detected by Model",
                        color='Anomalies Detected',
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🤖 No model activity data available")
    
    with tab2:
        st.header("🔍 Raw Security Logs")
        
        with st.spinner("Loading recent logs..."):
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
                if not logs_df.empty:
                    severity_counts = logs_df['Severity'].value_counts()
                    fig = px.pie(
                        values=severity_counts.values,
                        names=severity_counts.index,
                        title="Logs by Severity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not logs_df.empty:
                    protocol_counts = logs_df['Protocol'].value_counts().head(10)
                    fig = px.bar(
                        x=protocol_counts.values,
                        y=protocol_counts.index,
                        orientation='h',
                        title="Top Protocols"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                if not logs_df.empty:
                    port_stats = logs_df['Port'].describe()
                    st.write("**Port Statistics:**")
                    st.write(f"Most Common: {logs_df['Port'].mode().iloc[0] if not logs_df['Port'].mode().empty else 'N/A'}")
                    st.write(f"Average: {port_stats['mean']:.0f}")
                    st.write(f"Range: {port_stats['min']:.0f} - {port_stats['max']:.0f}")
        else:
            st.warning("⚠️ No logs found in database")
            st.info("This may be normal if ML training is running or no data is being generated")
    
    with tab3:
        st.header("🤖 ML Model Activity & Performance")
        
        # Model activity table
        with st.spinner("Loading model activity..."):
            model_df = get_model_activity()
            
        if not model_df.empty:
            st.subheader("📊 Model Performance (Last Hour)")
            st.dataframe(model_df, use_container_width=True)
        
        # Recent anomaly scores
        with st.spinner("Loading anomaly scores..."):
            scores_df = get_anomaly_scores(30)
            
        if not scores_df.empty:
            st.subheader("🔍 Recent Anomaly Scores")
            st.dataframe(scores_df, use_container_width=True, height=400)
            
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                if not scores_df.empty:
                    fig = px.histogram(
                        scores_df,
                        x='Score',
                        nbins=20,
                        title="Score Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not scores_df.empty:
                    anomaly_counts = scores_df['Is Anomaly'].value_counts()
                    fig = px.pie(
                        values=anomaly_counts.values,
                        names=anomaly_counts.index,
                        title="Anomaly Detection Results"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🤖 No anomaly scores available")
    
    with tab4:
        st.header("🚨 Security Alerts & Incidents")
        
        with st.spinner("Loading alerts..."):
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
                if not alerts_df.empty:
                    severity_dist = alerts_df['Severity'].str.extract(r'(Critical|High|Medium|Low)')[0].value_counts()
                    fig = px.bar(
                        x=severity_dist.values,
                        y=severity_dist.index,
                        orientation='h',
                        title="Alerts by Severity"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if not alerts_df.empty:
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
    
    # Auto-refresh logic (with warning)
    if auto_refresh:
        st.info("🔄 Auto-refresh enabled - page will refresh in 30 seconds")
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()