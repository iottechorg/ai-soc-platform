# services/alerting.py 

import requests
import logging
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List
from datetime import datetime, timedelta
from collections import deque
import threading
import time
import uuid

class AlertingService:
    """Enhanced multi-channel alerting service for SOC platform"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced alert channels
        self.channels = {
            'slack': EnhancedSlackAlerter(config.get('channels', {}).get('slack', {})),
            'email': EnhancedEmailAlerter(config.get('channels', {}).get('email', {})),
            'webhook': EnhancedWebhookAlerter(config.get('channels', {}).get('webhook', {}))
        }
        
        # Alert management
        self.alert_queue = deque()
        self.sent_alerts = deque(maxlen=1000)
        self.alert_templates = self._load_enhanced_alert_templates()
        
        # Enhanced deduplication settings
        self.dedup_window = config.get('deduplication_window_minutes', 5)  # Reduced from 20
        self.dedup_cache = {}
        self.similarity_threshold = config.get('similarity_threshold', 0.7)  # Reduced from 0.8
        
        # Enhanced rate limiting
        self.rate_limits = config.get('rate_limits', {
            'critical': {'count': 15, 'window': 3600},   # Increased from 10
            'high': {'count': 30, 'window': 3600},       # Increased from 25
            'medium': {'count': 60, 'window': 3600},     # Increased from 50
            'low': {'count': 120, 'window': 3600}        # Increased from 100
        })
        self.rate_limit_buckets = {severity: deque() for severity in self.rate_limits.keys()}
        
        # Alert processing
        self.running = False
        self.processing_thread = None
        
        # Enhanced metrics
        self.metrics = {
            'alerts_sent': 0,
            'alerts_deduplicated': 0,
            'alerts_rate_limited': 0,
            'channel_failures': {},
            'alerts_by_severity': {},
            'processing_time': deque(maxlen=1000)
        }
    
    def _load_enhanced_alert_templates(self) -> Dict[str, str]:
        """Load enhanced alert message templates with rich formatting"""
        return {
            'critical': """
🚨 **CRITICAL SECURITY ALERT** 🚨

**Entity**: {entity_id}
**Threat Score**: {aggregated_score:.3f}
**Confidence**: {confidence:.3f}
**Risk Level**: {risk_level}
**Time**: {timestamp}

**Contributing Models**: {contributing_models}
**Model Scores**: {model_scores}
**Tags**: {tags}
**Actions**: {actions}

**Score Trend**: {score_trend}
**Alert Generation Reason**: {processing_reason}

⚠️ **This requires immediate attention!** ⚠️
            """.strip(),
            
            'high': """
⚠️ **HIGH PRIORITY SECURITY ALERT** ⚠️

**Entity**: {entity_id}
**Threat Score**: {aggregated_score:.3f}
**Confidence**: {confidence:.3f}
**Risk Level**: {risk_level}
**Time**: {timestamp}

**Contributing Models**: {contributing_models}
**Tags**: {tags}
**Actions**: {actions}
**Score Trend**: {score_trend}

**Recommended Actions**: {actions}
            """.strip(),
            
            'medium': """
📊 **MEDIUM PRIORITY SECURITY ALERT**

**Entity**: {entity_id}
**Threat Score**: {aggregated_score:.3f}
**Risk Level**: {risk_level}
**Time**: {timestamp}

**Contributing Models**: {contributing_models}
**Tags**: {tags}
**Score Trend**: {score_trend}
            """.strip(),
            
            'low': """
ℹ️ **LOW PRIORITY SECURITY ALERT**

**Entity**: {entity_id}
**Threat Score**: {aggregated_score:.3f}
**Time**: {timestamp}
**Score Trend**: {score_trend}
            """.strip()
        }
    
    def start(self):
        """Start enhanced alert processing"""
        self.running = True
        
        def enhanced_process_alerts():
            while self.running:
                try:
                    if self.alert_queue:
                        alert = self.alert_queue.popleft()
                        start_time = time.time()
                        self._process_enhanced_alert(alert)
                        processing_time = time.time() - start_time
                        self.metrics['processing_time'].append(processing_time)
                    else:
                        time.sleep(0.5)  # Reduced sleep for better responsiveness
                except Exception as e:
                    self.logger.error(f"Error in enhanced alert processing: {e}")
                    time.sleep(5)
        
        self.processing_thread = threading.Thread(target=enhanced_process_alerts, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("✅ Started enhanced alerting service")
    
    def stop(self):
        """Stop alert processing"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        self.logger.info("🛑 Stopped enhanced alerting service")
    
    def send_alert(self, alert_data: Dict[str, Any]):
        """Queue an enhanced alert for sending"""
        alert_id = str(uuid.uuid4())
        enhanced_alert = {
            'id': alert_id,
            'timestamp': datetime.now(),
            'data': alert_data
        }
        
        self.alert_queue.append(enhanced_alert)
        self.logger.debug(f"📥 Queued enhanced alert {alert_id}")
    
    def _process_enhanced_alert(self, alert: Dict[str, Any]):
        """Process a single alert with enhanced logic"""
        alert_data = alert['data']
        severity = alert_data.get('severity', 'low')
        entity_id = alert_data.get('entity_id', 'unknown')
        
        # Enhanced deduplication check
        if self._is_duplicate_enhanced(alert_data):
            self.metrics['alerts_deduplicated'] += 1
            self.logger.debug(f"🔄 Deduplicated alert for {entity_id}")
            return
        
        # Enhanced rate limiting check
        if self._is_rate_limited_enhanced(severity):
            self.metrics['alerts_rate_limited'] += 1
            self.logger.warning(f"🚫 Rate limited {severity} alert for {entity_id}")
            return
        
        # Format enhanced alert message
        message = self._format_enhanced_alert_message(alert_data)
        
        # Determine channels based on enhanced routing
        channels_to_use = self._determine_enhanced_channels(severity, alert_data)
        
        # Send to channels with enhanced error handling
        successful_channels = []
        for channel_name in channels_to_use:
            try:
                channel = self.channels.get(channel_name)
                if channel and channel.is_enabled():
                    channel.send_alert(message, alert_data)
                    successful_channels.append(channel_name)
                    self.logger.info(f"✅ Sent {severity} alert to {channel_name}")
            except Exception as e:
                self.logger.error(f"❌ Failed to send alert via {channel_name}: {e}")
                self.metrics['channel_failures'][channel_name] = \
                    self.metrics['channel_failures'].get(channel_name, 0) + 1
        
        # Update enhanced metrics and tracking
        self.metrics['alerts_sent'] += 1
        self.metrics['alerts_by_severity'][severity] = \
            self.metrics['alerts_by_severity'].get(severity, 0) + 1
        
        self.sent_alerts.append({
            'id': alert['id'],
            'timestamp': alert['timestamp'],
            'severity': severity,
            'entity_id': entity_id,
            'channels': successful_channels,
            'score': alert_data.get('aggregated_score', 0.0),
            'confidence': alert_data.get('confidence', 0.0)
        })
        
        # Update enhanced deduplication cache
        self._update_enhanced_dedup_cache(alert_data)
        
        # Update enhanced rate limiting
        self._update_enhanced_rate_limit(severity)
    
    def _is_duplicate_enhanced(self, alert_data: Dict[str, Any]) -> bool:
        """Enhanced duplicate detection with similarity checking"""
        entity_id = alert_data.get('entity_id')
        severity = alert_data.get('severity')
        score = alert_data.get('aggregated_score', 0.0)
        
        cache_key = f"{entity_id}:{severity}"
        
        if cache_key in self.dedup_cache:
            last_alert = self.dedup_cache[cache_key]
            time_diff = datetime.now() - last_alert['timestamp']
            
            if time_diff.total_seconds() < self.dedup_window * 60:
                # Check score similarity
                score_diff = abs(score - last_alert.get('score', 0.0))
                if score_diff < self.similarity_threshold:
                    return True
        
        return False
    
    def _update_enhanced_dedup_cache(self, alert_data: Dict[str, Any]):
        """Update enhanced deduplication cache with score tracking"""
        entity_id = alert_data.get('entity_id')
        severity = alert_data.get('severity')
        cache_key = f"{entity_id}:{severity}"
        
        self.dedup_cache[cache_key] = {
            'timestamp': datetime.now(),
            'score': alert_data.get('aggregated_score', 0.0),
            'confidence': alert_data.get('confidence', 0.0)
        }
        
        # Enhanced cleanup of old entries
        cutoff_time = datetime.now() - timedelta(minutes=self.dedup_window * 2)
        keys_to_remove = [
            key for key, data in self.dedup_cache.items()
            if data['timestamp'] < cutoff_time
        ]
        
        for key in keys_to_remove:
            del self.dedup_cache[key]
    
    def _is_rate_limited_enhanced(self, severity: str) -> bool:
        """Enhanced rate limiting with burst allowance"""
        if severity not in self.rate_limits:
            return False
        
        limit_config = self.rate_limits[severity]
        bucket = self.rate_limit_buckets[severity]
        
        # Clean old entries
        cutoff_time = datetime.now() - timedelta(seconds=limit_config['window'])
        while bucket and bucket[0] < cutoff_time:
            bucket.popleft()
        
        # Check if under limit
        return len(bucket) >= limit_config['count']
    
    def _update_enhanced_rate_limit(self, severity: str):
        """Update enhanced rate limiting bucket"""
        if severity in self.rate_limit_buckets:
            self.rate_limit_buckets[severity].append(datetime.now())
    
    def _format_enhanced_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert message using enhanced templates"""
        severity = alert_data.get('severity', 'low')
        template = self.alert_templates.get(severity, self.alert_templates['low'])
        
        # Enhanced template variables with defaults
        template_vars = {
            'entity_id': alert_data.get('entity_id', 'unknown'),
            'aggregated_score': alert_data.get('aggregated_score', 0.0),
            'confidence': alert_data.get('confidence', 0.0),
            'risk_level': alert_data.get('risk_level', 'unknown'),
            'timestamp': alert_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S UTC'),
            'contributing_models': ', '.join(alert_data.get('contributing_models', [])),
            'model_scores': str(alert_data.get('model_scores', {})),
            'tags': ', '.join(alert_data.get('tags', [])),
            'actions': ', '.join(alert_data.get('actions', [])),
            'score_trend': alert_data.get('score_trend', 'unknown'),
            'processing_reason': alert_data.get('processing_reason', 'standard_detection')
        }
        
        try:
            return template.format(**template_vars)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            return f"Enhanced Alert for {template_vars['entity_id']} at {template_vars['timestamp']} - Score: {template_vars['aggregated_score']:.3f}"
    
    def _determine_enhanced_channels(self, severity: str, alert_data: Dict[str, Any]) -> List[str]:
        """Determine channels using enhanced routing logic"""
        # Base channel mapping
        base_mapping = {
            'critical': ['slack', 'email', 'webhook'],
            'high': ['slack', 'email'],
            'medium': ['slack'],
            'low': ['slack']
        }
        
        channels = base_mapping.get(severity, ['slack'])
        
        # Enhanced routing based on alert characteristics
        score = alert_data.get('aggregated_score', 0.0)
        confidence = alert_data.get('confidence', 0.0)
        tags = alert_data.get('tags', [])
        
        # Escalate medium alerts with high confidence to email
        if severity == 'medium' and confidence > 0.8 and score > 0.75:
            if 'email' not in channels:
                channels.append('email')
        
        # Priority tags always get full channel treatment
        priority_tags = {'critical_threat', 'persistent_threat', 'escalating_threat'}
        if any(tag in priority_tags for tag in tags):
            for channel in ['slack', 'email', 'webhook']:
                if channel not in channels:
                    channels.append(channel)
        
        return channels
    
    def get_enhanced_alert_stats(self) -> Dict[str, Any]:
        """Get enhanced alerting statistics"""
        recent_alerts = [
            alert for alert in self.sent_alerts
            if (datetime.now() - alert['timestamp']).total_seconds() < 3600
        ]
        
        # Enhanced statistics
        severity_counts = {}
        score_stats = []
        confidence_stats = []
        
        for alert in recent_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            score_stats.append(alert.get('score', 0.0))
            confidence_stats.append(alert.get('confidence', 0.0))
        
        enhanced_stats = {
            'total_alerts_sent': self.metrics['alerts_sent'],
            'alerts_deduplicated': self.metrics['alerts_deduplicated'],
            'alerts_rate_limited': self.metrics['alerts_rate_limited'],
            'recent_alerts_1h': len(recent_alerts),
            'severity_distribution_1h': severity_counts,
            'channel_failures': self.metrics['channel_failures'],
            'queue_size': len(self.alert_queue),
            'alerts_by_severity_total': self.metrics['alerts_by_severity']
        }
        
        # Add score and confidence statistics
        if score_stats:
            enhanced_stats.update({
                'recent_score_stats': {
                    'avg': sum(score_stats) / len(score_stats),
                    'min': min(score_stats),
                    'max': max(score_stats)
                },
                'recent_confidence_stats': {
                    'avg': sum(confidence_stats) / len(confidence_stats),
                    'min': min(confidence_stats),
                    'max': max(confidence_stats)
                }
            })
        
        # Add processing time statistics
        if self.metrics['processing_time']:
            recent_times = list(self.metrics['processing_time'])[-100:]
            enhanced_stats['avg_processing_time_ms'] = (sum(recent_times) / len(recent_times)) * 1000
        
        return enhanced_stats


class EnhancedSlackAlerter:
    """Enhanced Slack alerting implementation with rich formatting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get('webhook_url')
        self.channel = config.get('channel', '#soc-alerts')
        self.username = config.get('username', 'SOC-Enhanced-Bot')
        self.enabled = bool(self.webhook_url) and config.get('enabled', False)
        self.logger = logging.getLogger(__name__)
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def send_alert(self, message: str, alert_data: Dict[str, Any]):
        """Send enhanced alert to Slack with rich formatting"""
        if not self.webhook_url:
            raise ValueError("Slack webhook URL not configured")
        
        severity = alert_data.get('severity', 'low')
        entity_id = alert_data.get('entity_id', 'unknown')
        score = alert_data.get('aggregated_score', 0.0)
        confidence = alert_data.get('confidence', 0.0)
        risk_level = alert_data.get('risk_level', 'unknown')
        
        # Enhanced color mapping
        color_map = {
            'critical': '#FF0000',  # Bright Red
            'high': '#FF4500',      # Orange Red
            'medium': '#FFA500',    # Orange
            'low': '#32CD32'        # Lime Green
        }
        
        # Enhanced emoji mapping
        emoji_map = {
            'critical': '🚨',
            'high': '⚠️',
            'medium': '📊',
            'low': 'ℹ️'
        }
        
        # Create enhanced Slack attachment
        attachment = {
            'color': color_map.get(severity, '#32CD32'),
            'title': f"{emoji_map.get(severity, 'ℹ️')} {severity.upper()} Security Alert",
            'text': message,
            'fields': [
                {
                    'title': 'Entity',
                    'value': entity_id,
                    'short': True
                },
                {
                    'title': 'Threat Score',
                    'value': f'{score:.3f}',
                    'short': True
                },
                {
                    'title': 'Confidence',
                    'value': f'{confidence:.3f}',
                    'short': True
                },
                {
                    'title': 'Risk Level',
                    'value': risk_level.title(),
                    'short': True
                },
                {
                    'title': 'Contributing Models',
                    'value': ', '.join(alert_data.get('contributing_models', [])),
                    'short': False
                }
            ],
            'footer': 'SOC Platform Enhanced',
            'ts': int(datetime.now().timestamp())
        }
        
        # Add additional fields for high-priority alerts
        if severity in ['critical', 'high']:
            tags = alert_data.get('tags', [])
            actions = alert_data.get('actions', [])
            
            if tags:
                attachment['fields'].append({
                    'title': 'Tags',
                    'value': ', '.join(tags),
                    'short': False
                })
            
            if actions:
                attachment['fields'].append({
                    'title': 'Recommended Actions',
                    'value': ', '.join(actions),
                    'short': False
                })
        
        payload = {
            'channel': self.channel,
            'username': self.username,
            'attachments': [attachment]
        }
        
        response = requests.post(self.webhook_url, json=payload, timeout=10)
        
        if response.status_code != 200:
            raise Exception(f"Slack API error: {response.status_code} - {response.text}")
        
        self.logger.info(f"✅ Sent enhanced {severity} alert to Slack for {entity_id}")


class EnhancedEmailAlerter:
    """Enhanced Email alerting implementation with HTML formatting"""
    
    def __init__(self, config: Dict[str, Any]):
        self.smtp_server = config.get('smtp_server')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username')
        self.password = config.get('password')
        self.from_email = config.get('from_email')
        self.to_emails = config.get('to_emails', [])
        self.enabled = (bool(self.smtp_server and self.username and self.password and 
                           self.from_email and self.to_emails) and 
                       config.get('enabled', False))
        self.logger = logging.getLogger(__name__)
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def send_alert(self, message: str, alert_data: Dict[str, Any]):
        """Send enhanced alert via email with HTML formatting"""
        if not self.enabled:
            raise ValueError("Email not properly configured")
        
        severity = alert_data.get('severity', 'low')
        entity_id = alert_data.get('entity_id', 'unknown')
        score = alert_data.get('aggregated_score', 0.0)
        confidence = alert_data.get('confidence', 0.0)
        risk_level = alert_data.get('risk_level', 'unknown')
        
        # Create enhanced email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'[SOC Enhanced Alert - {severity.upper()}] {entity_id} - Score: {score:.3f}'
        msg['From'] = self.from_email
        msg['To'] = ', '.join(self.to_emails)
        
        # Enhanced HTML content with better styling
        severity_colors = {
            'critical': '#ff4444',
            'high': '#ff6600', 
            'medium': '#ffa500',
            'low': '#32cd32'
        }
        
        color = severity_colors.get(severity, '#ffa500')
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .alert-container {{ 
                    border: 2px solid {color}; 
                    border-radius: 8px; 
                    overflow: hidden; 
                    max-width: 600px; 
                    margin: 0 auto;
                }}
                .alert-header {{ 
                    background-color: {color}; 
                    color: white; 
                    padding: 20px; 
                    text-align: center;
                }}
                .alert-body {{ 
                    padding: 20px; 
                    background-color: #f9f9f9;
                }}
                .alert-field {{ 
                    margin: 10px 0; 
                    padding: 8px; 
                    background-color: white; 
                    border-radius: 4px;
                }}
                .field-label {{ 
                    font-weight: bold; 
                    color: #333;
                }}
                .field-value {{ 
                    margin-left: 10px; 
                    color: #666;
                }}
                .priority-{severity} {{
                    border-left: 4px solid {color};
                    padding-left: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="alert-container">
                <div class="alert-header">
                    <h1>🚨 {severity.upper()} Security Alert</h1>
                </div>
                <div class="alert-body">
                    <div class="alert-field priority-{severity}">
                        <span class="field-label">Entity:</span>
                        <span class="field-value">{entity_id}</span>
                    </div>
                    <div class="alert-field">
                        <span class="field-label">Threat Score:</span>
                        <span class="field-value">{score:.3f}</span>
                    </div>
                    <div class="alert-field">
                        <span class="field-label">Confidence:</span>
                        <span class="field-value">{confidence:.3f}</span>
                    </div>
                    <div class="alert-field">
                        <span class="field-label">Risk Level:</span>
                        <span class="field-value">{risk_level.title()}</span>
                    </div>
                    <div class="alert-field">
                        <span class="field-label">Detection Models:</span>
                        <span class="field-value">{', '.join(alert_data.get('contributing_models', []))}</span>
                    </div>
                    <div class="alert-field">
                        <span class="field-label">Time:</span>
                        <span class="field-value">{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</span>
                    </div>
        """
        
        # Add additional fields for high-priority alerts
        if severity in ['critical', 'high']:
            tags = alert_data.get('tags', [])
            actions = alert_data.get('actions', [])
            score_trend = alert_data.get('score_trend', 'unknown')
            
            if tags:
                html_content += f"""
                    <div class="alert-field">
                        <span class="field-label">Tags:</span>
                        <span class="field-value">{', '.join(tags)}</span>
                    </div>
                """
            
            if actions:
                html_content += f"""
                    <div class="alert-field">
                        <span class="field-label">Recommended Actions:</span>
                        <span class="field-value">{', '.join(actions)}</span>
                    </div>
                """
            
            html_content += f"""
                    <div class="alert-field">
                        <span class="field-label">Score Trend:</span>
                        <span class="field-value">{score_trend.title()}</span>
                    </div>
            """
        
        html_content += """
                    <div class="alert-field priority-{severity}">
                        <span class="field-label">Message:</span><br>
                        <div style="margin-top: 10px; white-space: pre-line;">{message}</div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """.format(severity=severity, message=message)
        
        # Add both text and HTML versions
        text_part = MIMEText(message, 'plain')
        html_part = MIMEText(html_content, 'html')
        msg.attach(text_part)
        msg.attach(html_part)
        
        # Send via SMTP with enhanced error handling
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
            
            self.logger.info(f"✅ Sent enhanced {severity} alert email for {entity_id}")
        except Exception as e:
            self.logger.error(f"❌ Failed to send email: {e}")
            raise


class EnhancedWebhookAlerter:
    """Enhanced generic webhook alerting implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.webhook_url = config.get('webhook_url')
        self.headers = config.get('headers', {'Content-Type': 'application/json'})
        self.enabled = bool(self.webhook_url) and config.get('enabled', False)
        self.timeout = config.get('timeout', 10)
        self.retry_count = config.get('retry_count', 3)
        self.logger = logging.getLogger(__name__)
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def send_alert(self, message: str, alert_data: Dict[str, Any]):
        """Send enhanced alert to webhook with retry logic"""
        if not self.webhook_url:
            raise ValueError("Webhook URL not configured")
        
        # Create enhanced webhook payload
        enhanced_payload = {
            'alert_type': 'soc_security_alert',
            'version': '2.0',
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'alert_data': {
                **alert_data,
                'processing_timestamp': datetime.now().isoformat(),
                'alert_source': 'soc_platform_enhanced'
            },
            'metadata': {
                'severity': alert_data.get('severity', 'low'),
                'entity_id': alert_data.get('entity_id', 'unknown'),
                'threat_score': alert_data.get('aggregated_score', 0.0),
                'confidence': alert_data.get('confidence', 0.0),
                'risk_level': alert_data.get('risk_level', 'unknown'),
                'contributing_models': alert_data.get('contributing_models', []),
                'tags': alert_data.get('tags', []),
                'actions': alert_data.get('actions', [])
            }
        }
        
        # Enhanced retry logic
        last_exception = None
        for attempt in range(self.retry_count):
            try:
                response = requests.post(
                    self.webhook_url,
                    json=enhanced_payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code in [200, 201, 202]:
                    self.logger.info(f"✅ Sent enhanced webhook alert (attempt {attempt + 1})")
                    return
                else:
                    raise Exception(f"Webhook error: {response.status_code} - {response.text}")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.retry_count - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    self.logger.warning(f"⚠️ Webhook attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"❌ All webhook attempts failed: {e}")
        
        if last_exception:
            raise last_exception