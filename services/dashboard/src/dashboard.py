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