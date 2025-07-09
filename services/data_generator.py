# enhanced_data_generator.py - Realistic threat distribution for balanced ML training

import random
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
from dataclasses import dataclass
import threading
import json
import numpy as np

@dataclass
class EnhancedThreatPattern:
    """Enhanced threat pattern with realistic characteristics"""
    name: str
    base_probability: float
    severity_distribution: Dict[str, float]
    indicators: List[str]
    port_ranges: List[tuple]
    protocols: List[str]
    ip_patterns: List[str]
    temporal_patterns: Dict[str, float]  # Hour-based probability multipliers
    persistence: float  # Probability of repeat from same source

@dataclass 
class TrafficProfile:
    """Normal traffic profile for realistic baseline"""
    name: str
    probability: float
    port_range: tuple
    protocols: List[str]
    typical_sizes: tuple
    duration_range: tuple

class EnhancedSyntheticDataGenerator:
    """Enhanced data generator with realistic threat and normal traffic patterns"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Enhanced threat patterns with realistic distributions
        self.threat_patterns = self._create_enhanced_threat_patterns()
        self.normal_profiles = self._create_enhanced_normal_profiles()
        
        # Source IP management for persistence
        self.known_threat_sources = {}  # IP -> {last_seen, threat_type, persistence_score}
        self.normal_sources = set()
        
        # Temporal patterns (some threats more common at certain times)
        self.temporal_multipliers = self._create_temporal_patterns()
        
        # Statistics tracking
        self.generation_stats = {
            'total_generated': 0,
            'threats_generated': 0,
            'normal_generated': 0,
            'threat_breakdown': {},
            'hourly_stats': {}
        }
        
        self.running = False
    
    def _create_enhanced_threat_patterns(self) -> List[EnhancedThreatPattern]:
        """Create realistic threat patterns with proper distributions"""
        
        # Get configuration or use defaults
        patterns_config = self.config.get('threat_patterns', {})
        
        return [
            EnhancedThreatPattern(
                name="brute_force_ssh",
                base_probability=patterns_config.get('brute_force', {}).get('probability', 0.015),
                severity_distribution={'high': 0.6, 'critical': 0.4},
                indicators=["multiple_failed_logins", "suspicious_source", "credential_attack"],
                port_ranges=[(22, 22), (2222, 2222)],
                protocols=["ssh", "tcp"],
                ip_patterns=["external", "tor_exit", "known_bad"],
                temporal_patterns={  # More common during off-hours
                    'night': 1.5, 'day': 0.7, 'weekend': 1.3
                },
                persistence=0.8  # Brute force attacks often persist
            ),
            
            EnhancedThreatPattern(
                name="brute_force_rdp", 
                base_probability=patterns_config.get('brute_force', {}).get('probability', 0.015) * 0.6,
                severity_distribution={'high': 0.7, 'critical': 0.3},
                indicators=["multiple_failed_logins", "rdp_attack", "suspicious_source"],
                port_ranges=[(3389, 3389), (3390, 3399)],
                protocols=["rdp", "tcp"],
                ip_patterns=["external", "cloud_provider"],
                temporal_patterns={'night': 1.4, 'day': 0.8, 'weekend': 1.2},
                persistence=0.7
            ),
            
            EnhancedThreatPattern(
                name="port_scan_comprehensive",
                base_probability=patterns_config.get('port_scan', {}).get('probability', 0.012),
                severity_distribution={'medium': 0.7, 'high': 0.3},
                indicators=["port_scanning", "reconnaissance", "multiple_ports"],
                port_ranges=[(1, 1024), (8000, 8999), (9000, 9999)],
                protocols=["tcp", "udp"],
                ip_patterns=["external", "scanner"],
                temporal_patterns={'night': 1.2, 'day': 0.9, 'weekend': 1.0},
                persistence=0.5
            ),
            
            EnhancedThreatPattern(
                name="web_application_attack",
                base_probability=0.010,
                severity_distribution={'medium': 0.4, 'high': 0.5, 'critical': 0.1},
                indicators=["sql_injection", "xss_attempt", "web_attack", "suspicious_payload"],
                port_ranges=[(80, 80), (443, 443), (8080, 8080)],
                protocols=["http", "https"],
                ip_patterns=["external", "bot_network"],
                temporal_patterns={'night': 0.8, 'day': 1.2, 'weekend': 0.9},
                persistence=0.3
            ),
            
            EnhancedThreatPattern(
                name="data_exfiltration",
                base_probability=patterns_config.get('data_exfiltration', {}).get('probability', 0.008),
                severity_distribution={'high': 0.4, 'critical': 0.6},
                indicators=["large_data_transfer", "unusual_destination", "data_exfiltration", "suspicious_timing"],
                port_ranges=[(80, 80), (443, 443), (21, 21), (22, 22)],
                protocols=["https", "ftp", "sftp"],
                ip_patterns=["external", "cloud_storage", "suspicious_geo"],
                temporal_patterns={'night': 1.8, 'day': 0.5, 'weekend': 1.4},
                persistence=0.6
            ),
            
            EnhancedThreatPattern(
                name="malware_c2_communication",
                base_probability=patterns_config.get('malware_communication', {}).get('probability', 0.006),
                severity_distribution={'high': 0.3, 'critical': 0.7},
                indicators=["c2_communication", "malware_signature", "beacon_pattern", "encrypted_tunnel"],
                port_ranges=[(80, 80), (443, 443), (8080, 8999), (53, 53)],
                protocols=["http", "https", "dns", "tcp"],
                ip_patterns=["external", "bulletproof_hosting", "fast_flux"],
                temporal_patterns={'night': 1.1, 'day': 1.0, 'weekend': 1.0},
                persistence=0.9  # C2 traffic is very persistent
            ),
            
            EnhancedThreatPattern(
                name="insider_threat_data_access",
                base_probability=0.003,  # Rare but critical
                severity_distribution={'high': 0.2, 'critical': 0.8},
                indicators=["privilege_escalation", "unauthorized_access", "after_hours_access", "bulk_data_access"],
                port_ranges=[(1433, 1433), (3306, 3306), (5432, 5432), (445, 445)],
                protocols=["mssql", "mysql", "postgresql", "smb"],
                ip_patterns=["internal", "privileged_network"],
                temporal_patterns={'night': 2.0, 'day': 0.3, 'weekend': 1.8},
                persistence=0.4
            ),
            
            EnhancedThreatPattern(
                name="ddos_attack",
                base_probability=0.008,
                severity_distribution={'medium': 0.3, 'high': 0.6, 'critical': 0.1},
                indicators=["high_volume_traffic", "ddos_pattern", "amplification_attack"],
                port_ranges=[(80, 80), (443, 443), (53, 53)],
                protocols=["http", "https", "udp", "tcp"],
                ip_patterns=["botnet", "multiple_sources"],
                temporal_patterns={'night': 0.8, 'day': 1.3, 'weekend': 1.0},
                persistence=0.2  # DDoS sources change frequently
            )
        ]
    
    def _create_enhanced_normal_profiles(self) -> List[TrafficProfile]:
        """Create realistic normal traffic profiles"""
        
        normal_config = self.config.get('normal_patterns', {})
        
        return [
            TrafficProfile(
                name="web_browsing",
                probability=normal_config.get('web_traffic', 0.65),
                port_range=(80, 443),
                protocols=["http", "https"],
                typical_sizes=(512, 51200),  # 512B to 50KB
                duration_range=(1, 30)
            ),
            
            TrafficProfile(
                name="database_operations", 
                probability=normal_config.get('database_queries', 0.15),
                port_range=(1433, 5432),
                protocols=["mssql", "mysql", "postgresql"],
                typical_sizes=(256, 8192),  # Smaller database queries
                duration_range=(1, 10)
            ),
            
            TrafficProfile(
                name="file_sharing",
                probability=normal_config.get('file_access', 0.12),
                port_range=(445, 445),
                protocols=["smb", "ftp", "sftp"],
                typical_sizes=(1024, 1048576),  # 1KB to 1MB
                duration_range=(5, 120)
            ),
            
            TrafficProfile(
                name="dns_queries",
                probability=normal_config.get('dns_queries', 0.08),
                port_range=(53, 53),
                protocols=["dns", "udp"],
                typical_sizes=(64, 512),  # Small DNS packets
                duration_range=(1, 3)
            )
        ]
    
    def _create_temporal_patterns(self) -> Dict[str, Dict[str, float]]:
        """Create hour-based threat probability multipliers"""
        return {
            'brute_force': {
                # More attacks during off-hours
                **{str(h): 1.5 for h in range(0, 6)},   # Night: 12am-6am
                **{str(h): 0.7 for h in range(6, 18)},  # Day: 6am-6pm  
                **{str(h): 1.2 for h in range(18, 24)}  # Evening: 6pm-12am
            },
            'port_scan': {
                # Fairly consistent but slightly more at night
                **{str(h): 1.2 for h in range(0, 8)},
                **{str(h): 0.9 for h in range(8, 20)},
                **{str(h): 1.1 for h in range(20, 24)}
            },
            'data_exfiltration': {
                # Much more common during off-hours
                **{str(h): 2.0 for h in range(0, 6)},
                **{str(h): 0.4 for h in range(6, 18)},
                **{str(h): 1.3 for h in range(18, 24)}
            }
        }
    
    def _generate_realistic_ip(self, pattern: str) -> str:
        """Generate IP addresses based on threat patterns"""
        
        if pattern == "internal":
            # Internal networks
            networks = ["192.168.", "10.", "172.16."]
            network = random.choice(networks)
            if network == "192.168.":
                return f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}"
            elif network == "10.":
                return f"10.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
            else:  # 172.16.x.x
                return f"172.{random.randint(16, 31)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        elif pattern == "external":
            # External IPs (avoid private ranges)
            while True:
                ip = f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
                # Avoid private ranges
                if not (ip.startswith("192.168.") or ip.startswith("10.") or 
                       ip.startswith("172.16.") or ip.startswith("127.")):
                    return ip
        
        elif pattern == "tor_exit":
            # Known Tor exit node ranges (simplified)
            tor_ranges = ["185.220.", "199.87.", "176.10.", "162.247."]
            base = random.choice(tor_ranges)
            return f"{base}{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        elif pattern == "cloud_provider":
            # AWS, Azure, GCP ranges (simplified)
            cloud_ranges = ["3.208.", "13.107.", "35.186.", "52.96."]
            base = random.choice(cloud_ranges)
            return f"{base}{random.randint(1, 254)}.{random.randint(1, 254)}"
        
        else:
            # Default external
            return f"{random.randint(1, 223)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
    
    def _get_temporal_multiplier(self, threat_type: str) -> float:
        """Get current temporal multiplier for threat type"""
        current_hour = datetime.now().hour
        
        # Simplified temporal patterns
        if threat_type in ["brute_force_ssh", "brute_force_rdp", "data_exfiltration"]:
            if 0 <= current_hour <= 6 or 22 <= current_hour <= 23:  # Night
                return 1.5
            elif 9 <= current_hour <= 17:  # Business hours
                return 0.6
            else:  # Evening
                return 1.1
        elif threat_type in ["web_application_attack"]:
            if 9 <= current_hour <= 17:  # Business hours (more web traffic)
                return 1.3
            else:
                return 0.8
        else:
            return 1.0  # No temporal variation
    
    def _should_persist_source(self, source_ip: str, threat_type: str, persistence: float) -> bool:
        """Determine if a threat source should persist"""
        if source_ip in self.known_threat_sources:
            threat_data = self.known_threat_sources[source_ip]
            time_since_last = (datetime.now() - threat_data['last_seen']).total_seconds()
            
            # More likely to persist if recent and matches threat type
            if (time_since_last < 3600 and  # Within last hour
                threat_data['threat_type'] == threat_type and 
                random.random() < persistence):
                return True
        
        return False
    
    def _generate_enhanced_threat_log(self, pattern: EnhancedThreatPattern) -> Dict[str, Any]:
        """Generate realistic threat log with enhanced characteristics"""
        
        # Check for persistent threat sources
        source_ip = None
        existing_sources = [ip for ip, data in self.known_threat_sources.items() 
                           if data['threat_type'] == pattern.name]
        
        if existing_sources and self._should_persist_source(
            random.choice(existing_sources), pattern.name, pattern.persistence):
            source_ip = random.choice(existing_sources)
            self.known_threat_sources[source_ip]['last_seen'] = datetime.now()
        else:
            # Generate new threat source
            ip_pattern = random.choice(pattern.ip_patterns)
            source_ip = self._generate_realistic_ip(ip_pattern)
            
            # Track new threat source
            self.known_threat_sources[source_ip] = {
                'last_seen': datetime.now(),
                'threat_type': pattern.name,
                'persistence_score': pattern.persistence
            }
        
        # Generate realistic destination
        dest_ip = self._generate_realistic_ip("internal")
        
        # Select port from pattern ranges
        port_range = random.choice(pattern.port_ranges)
        if port_range[0] == port_range[1]:
            port = port_range[0]
        else:
            port = random.randint(port_range[0], port_range[1])
        
        # Select protocol
        protocol = random.choice(pattern.protocols)
        
        # Determine severity based on distribution
        severity_roll = random.random()
        cumulative_prob = 0
        severity = "medium"  # default
        
        for sev, prob in pattern.severity_distribution.items():
            cumulative_prob += prob
            if severity_roll <= cumulative_prob:
                severity = sev
                break
        
        # Enhanced threat-specific characteristics
        if pattern.name.startswith("brute_force"):
            bytes_transferred = random.randint(64, 512)  # Small failed login attempts
            duration = random.randint(1, 5)
            session_id = f"brute_{random.randint(1000, 9999)}"
        
        elif pattern.name == "data_exfiltration":
            bytes_transferred = random.randint(1048576, 104857600)  # 1MB to 100MB
            duration = random.randint(300, 3600)  # 5 minutes to 1 hour
            session_id = f"exfil_{random.randint(1000, 9999)}"
        
        elif pattern.name == "port_scan_comprehensive":
            bytes_transferred = random.randint(64, 256)  # Small packets
            duration = random.randint(1, 2)
            session_id = f"scan_{random.randint(1000, 9999)}"
        
        elif pattern.name.startswith("malware"):
            bytes_transferred = random.randint(1024, 51200)  # 1KB to 50KB
            duration = random.randint(10, 60)
            session_id = f"c2_{random.randint(1000, 9999)}"
        
        else:
            # Default threat characteristics
            bytes_transferred = random.randint(512, 8192)
            duration = random.randint(5, 30)
            session_id = f"threat_{random.randint(1000, 9999)}"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "event_type": pattern.name,
            "source_ip": source_ip,
            "destination_ip": dest_ip,
            "port": port,
            "protocol": protocol,
            "message": f"Detected {pattern.name} from {source_ip} targeting {dest_ip}:{port}",
            "severity": severity,
            "threat_indicators": pattern.indicators.copy(),
            "event_id": f"evt_{int(time.time())}_{random.randint(1000, 9999)}",
            "session_id": session_id,
            "user_agent": self._generate_threat_user_agent(pattern.name),
            "bytes_transferred": bytes_transferred,
            "duration_seconds": duration
        }
    
    def _generate_threat_user_agent(self, threat_type: str) -> str:
        """Generate realistic user agents for different threat types"""
        
        if "brute_force" in threat_type:
            agents = [
                "ssh-2.0-libssh2_1.4.3",
                "RDP/7.0",
                "hydra/8.6",
                "ncrack/0.6"
            ]
        elif "web" in threat_type:
            agents = [
                "sqlmap/1.6.2",
                "Mozilla/5.0 (compatible; MSIE 6.0; Windows NT 5.1)",  # Old browser (suspicious)
                "python-requests/2.25.1",
                "curl/7.68.0"
            ]
        elif "scan" in threat_type:
            agents = [
                "nmap/7.80",
                "masscan/1.3.2", 
                "zmap/2.1.1",
                "Mozilla/5.0 (compatible; Nmap Scripting Engine)"
            ]
        else:
            agents = [
                "python-requests/2.28.1",
                "curl/7.68.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ]
        
        return random.choice(agents)
    
    def _generate_enhanced_normal_log(self) -> Dict[str, Any]:
        """Generate realistic normal traffic log"""
        
        # Select traffic profile based on probabilities
        profile_roll = random.random()
        cumulative_prob = 0
        selected_profile = self.normal_profiles[0]  # Default
        
        for profile in self.normal_profiles:
            cumulative_prob += profile.probability
            if profile_roll <= cumulative_prob:
                selected_profile = profile
                break
        
        # Generate realistic normal source (mix of internal and some external)
        if random.random() < 0.8:  # 80% internal traffic
            source_ip = self._generate_realistic_ip("internal")
        else:  # 20% legitimate external
            source_ip = self._generate_realistic_ip("external")
        
        dest_ip = self._generate_realistic_ip("internal")
        
        # Generate realistic port
        if selected_profile.port_range[0] == selected_profile.port_range[1]:
            port = selected_profile.port_range[0]
        else:
            port = random.randint(selected_profile.port_range[0], selected_profile.port_range[1])
        
        protocol = random.choice(selected_profile.protocols)
        
        # Realistic normal traffic sizes and durations
        bytes_transferred = random.randint(selected_profile.typical_sizes[0], selected_profile.typical_sizes[1])
        duration = random.randint(selected_profile.duration_range[0], selected_profile.duration_range[1])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "event_type": selected_profile.name,
            "source_ip": source_ip,
            "destination_ip": dest_ip,
            "port": port,
            "protocol": protocol,
            "message": f"Normal {selected_profile.name} from {source_ip} to {dest_ip}:{port}",
            "severity": "info",
            "threat_indicators": [],  # No threat indicators for normal traffic
            "event_id": f"evt_{int(time.time())}_{random.randint(1000, 9999)}",
            "session_id": f"sess_{random.randint(100000, 999999)}",
            "user_agent": self._generate_normal_user_agent(selected_profile.name),
            "bytes_transferred": bytes_transferred,
            "duration_seconds": duration
        }
    
    def _generate_normal_user_agent(self, traffic_type: str) -> str:
        """Generate realistic user agents for normal traffic"""
        
        if traffic_type == "web_browsing":
            agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0"
            ]
        elif traffic_type == "database_operations":
            agents = [
                "Microsoft SQL Server Management Studio",
                "MySQL Workbench 8.0",
                "pgAdmin 4",
                "application/database-client"
            ]
        else:
            agents = [
                "Windows-File-Explorer",
                "application/system-service",
                "internal-application/1.0",
                "corporate-software/2.1"
            ]
        
        return random.choice(agents)
    
    def generate_enhanced_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Generate enhanced batch with realistic threat distribution"""
        logs = []
        
        # Get base threat probability and apply temporal multipliers
        base_threat_prob = self.config.get('threat_probability', 0.04)
        
        for _ in range(batch_size):
            try:
                # Apply temporal multipliers to threat probability
                current_threat_prob = base_threat_prob
                
                # Randomly select a threat type to check temporal pattern
                sample_threat = random.choice(self.threat_patterns)
                temporal_mult = self._get_temporal_multiplier(sample_threat.name)
                adjusted_prob = current_threat_prob * temporal_mult
                
                if random.random() < adjusted_prob:
                    # Generate threat log
                    # Select threat pattern based on individual probabilities
                    pattern_roll = random.random()
                    cumulative_prob = 0
                    selected_pattern = self.threat_patterns[0]  # Default
                    
                    for pattern in self.threat_patterns:
                        cumulative_prob += pattern.base_probability
                        if pattern_roll <= cumulative_prob:
                            selected_pattern = pattern
                            break
                    
                    threat_log = self._generate_enhanced_threat_log(selected_pattern)
                    logs.append(threat_log)
                    
                    # Update statistics
                    self.generation_stats['threats_generated'] += 1
                    threat_name = selected_pattern.name
                    self.generation_stats['threat_breakdown'][threat_name] = \
                        self.generation_stats['threat_breakdown'].get(threat_name, 0) + 1
                    
                    self.logger.debug(f"Generated {selected_pattern.name} threat from {threat_log['source_ip']}")
                
                else:
                    # Generate normal log
                    normal_log = self._generate_enhanced_normal_log()
                    logs.append(normal_log)
                    self.generation_stats['normal_generated'] += 1
                
                self.generation_stats['total_generated'] += 1
                
            except Exception as e:
                self.logger.error(f"Error generating individual log: {e}")
                continue
        
        # Log generation statistics periodically
        if self.generation_stats['total_generated'] % 500 == 0:
            self._log_generation_statistics()
        
        return logs
    
    def _log_generation_statistics(self):
        """Log current generation statistics"""
        total = self.generation_stats['total_generated']
        threats = self.generation_stats['threats_generated']
        normal = self.generation_stats['normal_generated']
        
        if total > 0:
            threat_rate = (threats / total) * 100
            self.logger.info(f"📊 Generation Stats: {total} total, {threats} threats ({threat_rate:.1f}%), {normal} normal")
            
            # Log threat breakdown
            if self.generation_stats['threat_breakdown']:
                breakdown = ", ".join([f"{name}: {count}" for name, count in 
                                     self.generation_stats['threat_breakdown'].items()])
                self.logger.info(f"🎯 Threat Breakdown: {breakdown}")
    
    def start_enhanced_continuous_generation(self, kafka_client, topic: str):
        """Start enhanced continuous generation with better error handling and statistics"""
        self.running = True
        batch_size = self.config.get('batch_size', 80)
        interval = self.config.get('interval_seconds', 12)
        
        def enhanced_generation_loop():
            consecutive_errors = 0
            max_consecutive_errors = 5
            generation_count = 0
            
            self.logger.info(f"🚀 Starting enhanced data generation: {batch_size} logs every {interval}s")
            
            while self.running:
                try:
                    # Generate enhanced batch
                    start_time = time.time()
                    logs = self.generate_enhanced_batch(batch_size)
                    generation_time = time.time() - start_time
                    
                    if not logs:
                        self.logger.warning("No logs generated in batch")
                        time.sleep(interval)
                        continue
                    
                    # Send logs to Kafka
                    sent_count = 0
                    failed_count = 0
                    
                    for log in logs:
                        try:
                            kafka_client.send_message(topic, log)
                            sent_count += 1
                        except Exception as e:
                            failed_count += 1
                            self.logger.debug(f"Failed to send individual log: {e}")
                            continue
                    
                    generation_count += 1
                    
                    # Calculate current threat rate for this batch
                    threat_logs = [log for log in logs if log.get('threat_indicators', [])]
                    batch_threat_rate = (len(threat_logs) / len(logs)) * 100 if logs else 0
                    
                    self.logger.info(
                        f"📦 Batch {generation_count}: {sent_count}/{len(logs)} sent to {topic} "
                        f"({batch_threat_rate:.1f}% threats, {generation_time:.2f}s generation)"
                    )
                    
                    if failed_count > 0:
                        self.logger.warning(f"⚠️ {failed_count} messages failed to send")
                    
                    consecutive_errors = 0  # Reset error counter on success
                    
                    # Adaptive interval based on performance
                    if generation_time > interval * 0.8:  # If generation takes most of interval
                        adaptive_interval = max(interval, generation_time + 2)
                        self.logger.debug(f"Adaptive interval: {adaptive_interval}s (generation took {generation_time:.2f}s)")
                    else:
                        adaptive_interval = interval
                    
                    time.sleep(adaptive_interval)
                    
                except Exception as e:
                    consecutive_errors += 1
                    self.logger.error(f"Error in enhanced generation loop (attempt {consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        self.logger.error(f"Too many consecutive errors ({consecutive_errors}). Stopping generation.")
                        break
                    
                    # Exponential backoff on errors
                    error_sleep = min(interval * (2 ** consecutive_errors), 60)
                    self.logger.info(f"Sleeping {error_sleep} seconds before retry...")
                    time.sleep(error_sleep)
            
            self.logger.info("Enhanced data generation loop ended")
        
        thread = threading.Thread(target=enhanced_generation_loop, daemon=True)
        thread.start()
        self.logger.info(f"🎯 Enhanced continuous data generation started")
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive generation statistics"""
        total = self.generation_stats['total_generated']
        threats = self.generation_stats['threats_generated']
        normal = self.generation_stats['normal_generated']
        
        stats = {
            'total_generated': total,
            'threats_generated': threats,
            'normal_generated': normal,
            'threat_rate': (threats / total * 100) if total > 0 else 0,
            'threat_breakdown': self.generation_stats['threat_breakdown'].copy(),
            'active_threat_sources': len(self.known_threat_sources),
            'generation_running': self.running
        }
        
        # Calculate threat source persistence
        now = datetime.now()
        recent_sources = sum(1 for data in self.known_threat_sources.values() 
                           if (now - data['last_seen']).total_seconds() < 3600)
        stats['recent_threat_sources'] = recent_sources
        
        return stats
    
    def cleanup_old_threat_sources(self, hours: int = 24):
        """Clean up old threat sources to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        sources_to_remove = [
            ip for ip, data in self.known_threat_sources.items()
            if data['last_seen'] < cutoff_time
        ]
        
        for ip in sources_to_remove:
            del self.known_threat_sources[ip]
        
        if sources_to_remove:
            self.logger.info(f"🧹 Cleaned up {len(sources_to_remove)} old threat sources")
    
    def stop(self):
        """Stop enhanced data generation"""
        self.running = False
        self.logger.info("🛑 Enhanced data generation stopped")
        
        # Log final statistics
        self._log_generation_statistics()
        
        # Cleanup
        self.cleanup_old_threat_sources(1)  # Clean up sources older than 1 hour

class EnhancedDataGeneratorRunner:
    """Runner for the enhanced data generator with monitoring"""
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        try:
            from config.config_loader import ConfigLoader
            self.config = ConfigLoader()
        except ImportError:
            self.logger.warning("Config loader not available, using defaults")
            self.config = self._get_default_config()
        
        # Initialize enhanced generator
        self.generator = EnhancedSyntheticDataGenerator(
            self.config.get_section('data_generator') if hasattr(self.config, 'get_section') 
            else self.config.get('data_generator', {})
        )
        
        # Initialize Kafka client
        try:
            from shared.kafka_client import KafkaClient
            broker = (self.config.get('kafka.broker') if hasattr(self.config, 'get') 
                     else self.config.get('kafka', {}).get('broker', 'kafka:29092'))
            self.kafka_client = KafkaClient(broker)
        except ImportError:
            self.logger.error("Kafka client not available")
            self.kafka_client = None
        
        self.running = False
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if config loader fails"""
        return {
            'data_generator': {
                'batch_size': 80,
                'interval_seconds': 12,
                'threat_probability': 0.04,
                'threat_patterns': {
                    'brute_force': {'probability': 0.015},
                    'port_scan': {'probability': 0.012},
                    'data_exfiltration': {'probability': 0.008},
                    'malware_communication': {'probability': 0.006}
                },
                'normal_patterns': {
                    'web_traffic': 0.65,
                    'database_queries': 0.15,
                    'file_access': 0.12,
                    'dns_queries': 0.08
                }
            },
            'kafka': {
                'broker': 'kafka:29092',
                'topics': {
                    'raw_logs': 'raw-logs'
                }
            }
        }
    
    def start(self):
        """Start the enhanced data generator"""
        if not self.kafka_client:
            self.logger.error("Cannot start without Kafka client")
            return False
        
        try:
            # Get topic name
            topic = (self.config.get('kafka.topics.raw_logs') if hasattr(self.config, 'get')
                    else self.config.get('kafka', {}).get('topics', {}).get('raw_logs', 'raw-logs'))
            
            # Start enhanced generation
            self.generator.start_enhanced_continuous_generation(self.kafka_client, topic)
            self.running = True
            
            self.logger.info("✅ Enhanced Data Generator started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start enhanced data generator: {e}")
            return False
    
    def stop(self):
        """Stop the enhanced data generator"""
        self.running = False
        if self.generator:
            self.generator.stop()
        self.logger.info("Enhanced Data Generator stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get generator status and statistics"""
        if not self.generator:
            return {'error': 'Generator not initialized'}
        
        stats = self.generator.get_generation_statistics()
        stats['runner_status'] = 'running' if self.running else 'stopped'
        
        return stats

def main():
    """Main entry point for enhanced data generator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Enhanced Data Generator...")
        
        # Initialize and start
        runner = EnhancedDataGeneratorRunner()
        
        if runner.start():
            logger.info("✅ Enhanced Data Generator running with realistic threat distribution")
            
            # Monitor and report statistics
            while True:
                time.sleep(60)  # Report every minute
                
                try:
                    stats = runner.get_status()
                    threat_rate = stats.get('threat_rate', 0)
                    total = stats.get('total_generated', 0)
                    
                    logger.info(f"📊 Generated {total} logs, {threat_rate:.1f}% threats")
                    
                    # Report threat breakdown every 5 minutes
                    if total > 0 and total % 300 == 0:  # Every ~5 minutes at 80 logs/12s
                        breakdown = stats.get('threat_breakdown', {})
                        if breakdown:
                            logger.info(f"🎯 Threat types: {breakdown}")
                        
                        active_sources = stats.get('recent_threat_sources', 0)
                        logger.info(f"🔍 Active threat sources: {active_sources}")
                    
                except Exception as e:
                    logger.error(f"Error getting statistics: {e}")
        else:
            logger.error("❌ Failed to start Enhanced Data Generator")
            
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down Enhanced Data Generator...")
        if 'runner' in locals():
            runner.stop()
    except Exception as e:
        logger.error(f"💥 Enhanced Data Generator error: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()