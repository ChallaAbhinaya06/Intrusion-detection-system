"""
Attack Knowledge Base
---------------------
Maps ML prediction class IDs to attack semantics.

NOW DYNAMIC: reads actual label names from the dataset's .meta file
instead of using hardcoded attack types.

Supports both CIC-IDS2017 and 5G-NIDD datasets with comprehensive
attack pattern recognition.
"""

# Global variable to hold the current label mapping
# Will be populated by set_label_mapping() before predictions are logged
_CURRENT_LABEL_MAPPING = {}


def set_label_mapping(label_mapping):
    """
    Set the global label mapping for the current dataset.
    Should be called after loading a model and its .meta file.
    
    Args:
        label_mapping: dict mapping class_id (int) -> original_label_name (str)
                       Example: {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan', ...}
    """
    global _CURRENT_LABEL_MAPPING
    _CURRENT_LABEL_MAPPING = label_mapping


def get_label_mapping():
    """
    Get the current label mapping.
    
    Returns:
        Dictionary mapping class_id -> label_name
    """
    return _CURRENT_LABEL_MAPPING.copy()


def clear_label_mapping():
    """Clear the global label mapping (useful for testing or cleanup)."""
    global _CURRENT_LABEL_MAPPING
    _CURRENT_LABEL_MAPPING = {}


def _infer_severity(label_name):
    """
    Infer severity level based on attack label name.
    
    Returns: "None", "Low", "Medium", "High", or "Critical"
    """
    label_lower = label_name.lower()
    
    # Benign/Normal traffic
    if "benign" in label_lower or "normal" in label_lower:
        return "None"
    
    # Critical severity attacks
    if any(keyword in label_lower for keyword in [
        "ddos", "ransom", "backdoor", "infiltration", "botnet", "heartbleed"
    ]):
        return "Critical"
    
    # High severity attacks
    if any(keyword in label_lower for keyword in [
        "brute", "force", "sql", "injection", "xss", "csrf", "mitm", 
        "man-in-the-middle", "password", "web attack", "ftp-patator",
        "ssh-patator", "uploading"
    ]):
        return "High"
    
    # Low severity attacks
    if any(keyword in label_lower for keyword in [
        "probe", "scan", "reconnaissance"
    ]):
        return "Low"
    
    # Medium severity (default for other attacks)
    return "Medium"


def _infer_pattern_and_signature(label_name):
    """
    Infer attack pattern name and signature based on label.
    
    Returns: (pattern_name, signature) tuple
    """
    label_lower = label_name.lower()
    
    # ==================== BENIGN/NORMAL ====================
    if "benign" in label_lower or "normal" in label_lower:
        return (
            "None",
            "No malicious activity detected"
        )
    
    # ==================== DDoS ATTACKS ====================
    if "ddos" in label_lower:
        if "udp" in label_lower:
            return (
                "UDP Flood Attack",
                "High-volume UDP packet flooding to exhaust network resources"
            )
        elif "syn" in label_lower:
            return (
                "SYN Flood Attack",
                "TCP SYN flood to exhaust server connection resources"
            )
        elif "http" in label_lower:
            return (
                "HTTP Flood Attack",
                "Application-layer DDoS using HTTP requests"
            )
        else:
            return (
                "Distributed Denial of Service",
                "Coordinated traffic flooding from multiple sources"
            )
    
    # ==================== FLOODING ATTACKS ====================
    if "flood" in label_lower:
        if "ping" in label_lower or "icmp" in label_lower:
            return (
                "ICMP Flood",
                "ICMP echo request flooding to overwhelm target"
            )
        elif "udp" in label_lower:
            return (
                "UDP Flood",
                "User Datagram Protocol packet flooding attack"
            )
        else:
            return (
                "Network Flooding",
                "High-volume packet transmission to saturate bandwidth"
            )
    
    # ==================== RECONNAISSANCE ====================
    if "scan" in label_lower:
        if "port" in label_lower:
            return (
                "Port Scanning",
                "Systematic probing of ports to identify open services"
            )
        elif "vuln" in label_lower or "vulnerability" in label_lower:
            return (
                "Vulnerability Scanning",
                "Automated scanning to detect system vulnerabilities"
            )
        else:
            return (
                "Network Reconnaissance",
                "Information gathering about target network topology"
            )
    
    if "probe" in label_lower:
        return (
            "Network Probing",
            "Active reconnaissance to map network services"
        )
    
    # ==================== BRUTE FORCE / PASSWORD ATTACKS ====================
    if "brute" in label_lower or "force" in label_lower:
        if "ftp" in label_lower:
            return (
                "FTP Brute Force",
                "Automated FTP password guessing attack"
            )
        elif "ssh" in label_lower:
            return (
                "SSH Brute Force",
                "Repeated SSH authentication attempts"
            )
        else:
            return (
                "Brute Force Attack",
                "Systematic password enumeration through trial and error"
            )
    
    if "password" in label_lower:
        return (
            "Credential Attack",
            "Unauthorized attempts to obtain or use passwords"
        )
    
    # ==================== WEB ATTACKS ====================
    if "sql" in label_lower and "injection" in label_lower:
        return (
            "SQL Injection",
            "Database exploitation through malicious SQL query injection"
        )
    
    if "xss" in label_lower or "cross-site scripting" in label_lower:
        return (
            "Cross-Site Scripting (XSS)",
            "Client-side code injection through web application"
        )
    
    if "csrf" in label_lower:
        return (
            "Cross-Site Request Forgery",
            "Unauthorized commands transmitted from trusted user"
        )
    
    if "web attack" in label_lower:
        return (
            "Web Application Attack",
            "HTTP-based exploitation of web services"
        )
    
    # ==================== MITM / INTERCEPTION ====================
    if "mitm" in label_lower or "man-in-the-middle" in label_lower or "man in the middle" in label_lower:
        return (
            "Man-in-the-Middle Attack",
            "Interception and manipulation of communication between parties"
        )
    
    # ==================== MALWARE ====================
    if "ransom" in label_lower:
        return (
            "Ransomware Attack",
            "Malicious encryption of data for extortion"
        )
    
    if "bot" in label_lower and "net" in label_lower:
        return (
            "Botnet Activity",
            "Coordinated attack from network of compromised devices"
        )
    
    if "backdoor" in label_lower:
        return (
            "Backdoor Installation",
            "Covert remote access mechanism implanted in system"
        )
    
    if "heartbleed" in label_lower:
        return (
            "Heartbleed Exploit",
            "OpenSSL vulnerability exploitation for memory disclosure"
        )
    
    # ==================== INFILTRATION ====================
    if "infiltrat" in label_lower:
        return (
            "Network Infiltration",
            "Unauthorized access and lateral movement within network"
        )
    
    if "uploading" in label_lower or "upload" in label_lower:
        return (
            "Malicious Upload",
            "Unauthorized file upload to compromise system"
        )
    
    # ==================== IoT / 5G SPECIFIC ====================
    if "slowloris" in label_lower:
        return (
            "Slowloris Attack",
            "Low-bandwidth HTTP DoS by holding connections open"
        )
    
    if "goldeneye" in label_lower:
        return (
            "GoldenEye Attack",
            "HTTP DoS tool using randomized headers"
        )
    
    # ==================== GENERIC FALLBACK ====================
    return (
        "Unclassified Attack Pattern",
        f"Network intrusion detected: {label_name}"
    )


def get_attack_info(class_id):
    """
    Get attack information for a given class ID.
    Uses the actual dataset label names when available,
    falls back to generic names when mapping is not set.
    
    Args:
        class_id: The predicted class ID (int)
    
    Returns:
        Dictionary with attack information:
        {
            "attack_type": str,       # The actual label name from dataset
            "pattern_name": str,      # Specific attack pattern description
            "signature": str,         # Detailed signature description
            "severity": str           # None/Low/Medium/High/Critical
        }
    """
    # Try to get the actual label name from the current mapping
    if class_id in _CURRENT_LABEL_MAPPING:
        label_name = _CURRENT_LABEL_MAPPING[class_id]
    else:
        # Fallback when no mapping is set (shouldn't happen in production)
        label_name = f"Unknown_Class_{class_id}"
    
    # Infer severity, pattern, and signature from the label name
    severity = _infer_severity(label_name)
    pattern_name, signature = _infer_pattern_and_signature(label_name)
    
    return {
        "attack_type": label_name,
        "pattern_name": pattern_name,
        "signature": signature,
        "severity": severity
    }


def get_dynamic_attack_map(num_classes, label_mapping=None):
    """
    Generate attack map for any number of classes.
    
    Args:
        num_classes: Number of classes in the dataset
        label_mapping: Optional dict mapping class_id -> label_name
                      If not provided, uses the global mapping
    
    Returns:
        Dictionary mapping class IDs to attack info
    """
    if label_mapping is not None:
        # Temporarily set the mapping for this call
        old_mapping = _CURRENT_LABEL_MAPPING.copy()
        set_label_mapping(label_mapping)
    
    attack_map = {}
    for class_id in range(num_classes):
        attack_map[class_id] = get_attack_info(class_id)
    
    if label_mapping is not None:
        # Restore old mapping
        set_label_mapping(old_mapping)
    
    return attack_map