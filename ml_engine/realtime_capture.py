"""
ml_engine/realtime_capture.py
==============================
Real-time network traffic capture for CIC-IDS2017 and 5G-NIDD formats.
Called by the Django view — captures from the local machine's network
interface using Scapy and writes a CSV directly into MEDIA_ROOT/datasets/.

Requirements (install once):
    pip install scapy
    Windows: install Npcap from https://npcap.com/#download
             (tick "WinPcap API-compatible mode" during install)
"""

import os
import sys
import math
import time
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# EXACT column orders — must match the project's default_datasets exactly
# ─────────────────────────────────────────────────────────────────────────────
CIC_COLUMNS = [
    ' Destination Port', ' Flow Duration', ' Total Fwd Packets',
    ' Total Backward Packets', 'Total Length of Fwd Packets',
    ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
    ' Fwd Packet Length Min', ' Fwd Packet Length Mean',
    ' Fwd Packet Length Std', 'Bwd Packet Length Max',
    ' Bwd Packet Length Min', ' Bwd Packet Length Mean',
    ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s',
    ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
    'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std',
    ' Fwd IAT Max', ' Fwd IAT Min',
    'Bwd IAT Total', ' Bwd IAT Mean', ' Bwd IAT Std',
    ' Bwd IAT Max', ' Bwd IAT Min',
    'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags',
    ' Fwd Header Length', ' Bwd Header Length',
    'Fwd Packets/s', ' Bwd Packets/s',
    ' Min Packet Length', ' Max Packet Length',
    ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance',
    'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count',
    ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count',
    ' CWE Flag Count', ' ECE Flag Count',
    ' Down/Up Ratio', ' Average Packet Size',
    ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
    ' Fwd Header Length.1',
    'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate',
    ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
    'Subflow Fwd Packets', ' Subflow Fwd Bytes',
    ' Subflow Bwd Packets', ' Subflow Bwd Bytes',
    'Init_Win_bytes_forward', ' Init_Win_bytes_backward',
    ' act_data_pkt_fwd', ' min_seg_size_forward',
    'Active Mean', ' Active Std', ' Active Max', ' Active Min',
    'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min',
    ' Label'
]

NIDD_COLUMNS = [
    'Unnamed: 0', 'Seq', 'Dur', 'RunTime', 'Mean', 'Sum', 'Min', 'Max',
    'Proto', 'sTos', 'dTos', 'sDSb', 'dDSb', 'sTtl', 'dTtl',
    'sHops', 'dHops', 'Cause', 'TotPkts', 'SrcPkts', 'DstPkts',
    'TotBytes', 'SrcBytes', 'DstBytes', 'Offset',
    'sMeanPktSz', 'dMeanPktSz', 'Load', 'SrcLoad', 'DstLoad',
    'Loss', 'SrcLoss', 'DstLoss', 'pLoss',
    'SrcGap', 'DstGap', 'Rate', 'SrcRate', 'DstRate',
    'State', 'SrcWin', 'DstWin', 'sVid', 'dVid',
    'SrcTCPBase', 'DstTCPBase', 'TcpRtt', 'SynAck', 'AckDat',
    'Label', 'Attack Type', 'Attack Tool'
]

PROTO_MAP = {1: 'icmp', 6: 'tcp', 17: 'udp', 0: 'other'}

FLOW_TIMEOUT = 5.0   # seconds idle before a flow is closed


# ─────────────────────────────────────────────────────────────────────────────
# SHARED PACKET HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _import_scapy():
    """Import scapy lazily so Django can load even if scapy isn't installed."""
    try:
        from scapy.all import IP, TCP, UDP, ICMP, sniff
        return IP, TCP, UDP, ICMP, sniff
    except ImportError:
        raise ImportError(
            "Scapy is not installed. Run:  pip install scapy\n"
            "Windows: also install Npcap from https://npcap.com/#download"
        )


def _flow_key(pkt, IP, TCP, UDP, ICMP):
    if IP not in pkt:
        return None
    src, dst = pkt[IP].src, pkt[IP].dst
    sport = dport = 0
    if TCP in pkt:
        sport, dport, proto = pkt[TCP].sport, pkt[TCP].dport, 6
    elif UDP in pkt:
        sport, dport, proto = pkt[UDP].sport, pkt[UDP].dport, 17
    elif ICMP in pkt:
        proto = 1
    else:
        proto = pkt[IP].proto
    return (src, dst, sport, dport, proto)


def _safe_stats(values):
    if not values:
        return 0.0, 0.0, 0, 0
    a = np.array(values, dtype=float)
    return float(a.mean()), float(a.std()), int(a.max()), int(a.min())


def _iat_us(timestamps):
    ts = sorted(timestamps)
    if len(ts) < 2:
        return []
    return [int((ts[i+1] - ts[i]) * 1_000_000) for i in range(len(ts) - 1)]


# ─────────────────────────────────────────────────────────────────────────────
# CIC FLOW
# ─────────────────────────────────────────────────────────────────────────────
class _CICFlow:
    def __init__(self, key, ts):
        self.key = key
        self.first_ts = self.last_ts = ts
        self.fwd, self.bwd = [], []
        self.init_win_fwd = self.init_win_bwd = 0
        self._fwd_win_set = self._bwd_win_set = False

    def add(self, pkt, ts, IP, TCP, UDP):
        self.last_ts = ts
        length = len(pkt[IP]) if IP in pkt else len(pkt)
        hlen = 0
        if IP in pkt:
            hlen += pkt[IP].ihl * 4
        flags = win = 0
        if TCP in pkt:
            hlen += pkt[TCP].dataofs * 4
            flags = int(pkt[TCP].flags)
            win   = pkt[TCP].window
        elif UDP in pkt:
            hlen += 8
        is_fwd = (IP in pkt and pkt[IP].src == self.key[0])
        entry = (ts, length, hlen, flags, win)
        if is_fwd:
            self.fwd.append(entry)
            if not self._fwd_win_set:
                self.init_win_fwd = win
                self._fwd_win_set = True
        else:
            self.bwd.append(entry)
            if not self._bwd_win_set:
                self.init_win_bwd = win
                self._bwd_win_set = True


def _cic_features(flow, label):
    fwd, bwd = flow.fwd, flow.bwd
    all_p = fwd + bwd
    if not all_p:
        return None

    dur_us = int((flow.last_ts - flow.first_ts) * 1_000_000)
    dur_s  = max(dur_us / 1_000_000, 1e-9)

    fl = [p[1] for p in fwd]; bl = [p[1] for p in bwd]; al = [p[1] for p in all_p]
    fm, fs, fx, fn = _safe_stats(fl)
    bm, bs, bx, bn = _safe_stats(bl)
    am, as_, _, _  = _safe_stats(al)

    f_iat = _iat_us([p[0] for p in all_p])
    fwd_iat = _iat_us([p[0] for p in fwd])
    bwd_iat = _iat_us([p[0] for p in bwd])

    fi_m, fi_s, fi_x, fi_n = _safe_stats(f_iat)
    fwi_m, fwi_s, fwi_x, fwi_n = _safe_stats(fwd_iat)
    bwi_m, bwi_s, bwi_x, bwi_n = _safe_stats(bwd_iat)

    FIN=0x01; SYN=0x02; RST=0x04; PSH=0x08; ACK=0x10; URG=0x20; ECE=0x40; CWR=0x80
    fc = lambda pkts, b: sum(1 for p in pkts if p[3] & b)

    tb = sum(al); fb = sum(fl); bb = sum(bl)
    tp = len(all_p)

    return {
        ' Destination Port':           flow.key[3],
        ' Flow Duration':              dur_us,
        ' Total Fwd Packets':          len(fwd),
        ' Total Backward Packets':     len(bwd),
        'Total Length of Fwd Packets': fb,
        ' Total Length of Bwd Packets':bb,
        ' Fwd Packet Length Max':      fx,
        ' Fwd Packet Length Min':      fn,
        ' Fwd Packet Length Mean':     round(fm, 6),
        ' Fwd Packet Length Std':      round(fs, 6),
        'Bwd Packet Length Max':       bx,
        ' Bwd Packet Length Min':      bn,
        ' Bwd Packet Length Mean':     round(bm, 6),
        ' Bwd Packet Length Std':      round(bs, 6),
        'Flow Bytes/s':                round(tb / dur_s, 6),
        ' Flow Packets/s':             round(tp / dur_s, 6),
        ' Flow IAT Mean':              round(fi_m, 6),
        ' Flow IAT Std':               round(fi_s, 6),
        ' Flow IAT Max':               fi_x,
        ' Flow IAT Min':               fi_n,
        'Fwd IAT Total':               int(sum(fwd_iat)),
        ' Fwd IAT Mean':               round(fwi_m, 6),
        ' Fwd IAT Std':                round(fwi_s, 6),
        ' Fwd IAT Max':                fwi_x,
        ' Fwd IAT Min':                fwi_n,
        'Bwd IAT Total':               int(sum(bwd_iat)),
        ' Bwd IAT Mean':               round(bwi_m, 6),
        ' Bwd IAT Std':                round(bwi_s, 6),
        ' Bwd IAT Max':                bwi_x,
        ' Bwd IAT Min':                bwi_n,
        'Fwd PSH Flags':               fc(fwd, PSH),
        ' Bwd PSH Flags':              fc(bwd, PSH),
        ' Fwd URG Flags':              fc(fwd, URG),
        ' Bwd URG Flags':              fc(bwd, URG),
        ' Fwd Header Length':          sum(p[2] for p in fwd),
        ' Bwd Header Length':          sum(p[2] for p in bwd),
        'Fwd Packets/s':               round(len(fwd) / dur_s, 6),
        ' Bwd Packets/s':              round(len(bwd) / dur_s, 6),
        ' Min Packet Length':          min(al) if al else 0,
        ' Max Packet Length':          max(al) if al else 0,
        ' Packet Length Mean':         round(am, 6),
        ' Packet Length Std':          round(as_, 6),
        ' Packet Length Variance':     round(float(np.var(al)) if al else 0, 6),
        'FIN Flag Count':              fc(all_p, FIN),
        ' SYN Flag Count':             fc(all_p, SYN),
        ' RST Flag Count':             fc(all_p, RST),
        ' PSH Flag Count':             fc(all_p, PSH),
        ' ACK Flag Count':             fc(all_p, ACK),
        ' URG Flag Count':             fc(all_p, URG),
        ' CWE Flag Count':             fc(all_p, CWR),
        ' ECE Flag Count':             fc(all_p, ECE),
        ' Down/Up Ratio':              round(len(bwd) / len(fwd), 6) if fwd else 0,
        ' Average Packet Size':        round(am, 6),
        ' Avg Fwd Segment Size':       round(fm, 6),
        ' Avg Bwd Segment Size':       round(bm, 6),
        ' Fwd Header Length.1':        sum(p[2] for p in fwd),
        'Fwd Avg Bytes/Bulk':          0,
        ' Fwd Avg Packets/Bulk':       0,
        ' Fwd Avg Bulk Rate':          0,
        ' Bwd Avg Bytes/Bulk':         0,
        ' Bwd Avg Packets/Bulk':       0,
        'Bwd Avg Bulk Rate':           0,
        'Subflow Fwd Packets':         len(fwd),
        ' Subflow Fwd Bytes':          fb,
        ' Subflow Bwd Packets':        len(bwd),
        ' Subflow Bwd Bytes':          bb,
        'Init_Win_bytes_forward':      flow.init_win_fwd,
        ' Init_Win_bytes_backward':    flow.init_win_bwd,
        ' act_data_pkt_fwd':           sum(1 for p in fwd if p[1] > p[2]),
        ' min_seg_size_forward':       min((p[2] for p in fwd), default=0),
        'Active Mean':                 dur_us, ' Active Std': 0,
        ' Active Max':                 dur_us, ' Active Min': dur_us,
        'Idle Mean':                   0,      ' Idle Std':   0,
        ' Idle Max':                   0,      ' Idle Min':   0,
        ' Label':                      label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5G FLOW
# ─────────────────────────────────────────────────────────────────────────────
class _5GFlow:
    def __init__(self, key, seq_id, ts):
        self.key = key; self.seq_id = seq_id
        self.first_ts = self.last_ts = self.runtime = ts
        self.pkts = []
        self.syn_ts = self.synack_ts = self.ack_ts = None
        self.has_syn = self.has_fin = self.has_rst = False
        self.cause = "Start"

    def add(self, pkt, ts, IP, TCP, UDP, ICMP):
        self.last_ts = ts
        length = len(pkt[IP]) if IP in pkt else len(pkt)
        ttl  = pkt[IP].ttl if IP in pkt else 0
        tos  = pkt[IP].tos if IP in pkt else 0
        is_src = IP in pkt and pkt[IP].src == self.key[0]
        flags = win = seq = 0
        if TCP in pkt:
            flags = int(pkt[TCP].flags)
            win   = pkt[TCP].window
            seq   = pkt[TCP].seq
            if flags & 0x02 and not (flags & 0x10):
                self.has_syn = True; self.syn_ts = ts
            if flags & 0x02 and flags & 0x10:
                self.synack_ts = ts
            if flags & 0x10 and self.synack_ts and not self.ack_ts:
                self.ack_ts = ts
            if flags & 0x01:
                self.has_fin = True; self.cause = "End"
            if flags & 0x04:
                self.has_rst = True; self.cause = "End"
        self.pkts.append((ts, length, ttl, tos, is_src, flags, win, seq))


def _dsb(tos):
    return f"0x{tos:02x}"


def _hops(ttl):
    for init in (64, 128, 255):
        if ttl <= init:
            return init - ttl
    return 0


def _5g_features(flow, row_idx, label, attack_type, attack_tool):
    pkts = flow.pkts
    if not pkts:
        return None
    proto_num = flow.key[4]
    proto_str = PROTO_MAP.get(proto_num, 'other')

    dur = flow.last_ts - flow.first_ts
    dur_s = max(dur, 1e-9)

    src_p = [p for p in pkts if p[4]]
    dst_p = [p for p in pkts if not p[4]]
    sl = [p[1] for p in src_p]; dl = [p[1] for p in dst_p]; al = [p[1] for p in pkts]

    tb = sum(al); sb = sum(sl); db = sum(dl)
    tp = len(pkts); sc = len(src_p); dc = len(dst_p)

    def gaps(pl):
        ts = sorted(p[0] for p in pl)
        if len(ts) < 2: return 0.0
        return float(np.mean([ts[i+1]-ts[i] for i in range(len(ts)-1)]))

    s_ttl = int(src_p[0][2]) if src_p else 0
    d_ttl = int(dst_p[0][2]) if dst_p else 0

    if proto_num == 6:
        a_seqs = [p[7] for p in pkts]
        loss = len(a_seqs) - len(set(a_seqs))
        s_seqs = [p[7] for p in src_p]; d_seqs = [p[7] for p in dst_p]
        sl_ = len(s_seqs) - len(set(s_seqs)); dl_ = len(d_seqs) - len(set(d_seqs))
    else:
        loss = sl_ = dl_ = 0

    syn_ack_t = ack_dat_t = tcp_rtt = 0
    if flow.syn_ts and flow.synack_ts:
        syn_ack_t = int((flow.synack_ts - flow.syn_ts) * 1_000_000)
    if flow.synack_ts and flow.ack_ts:
        ack_dat_t = int((flow.ack_ts - flow.synack_ts) * 1_000_000)
    if syn_ack_t and ack_dat_t:
        tcp_rtt = syn_ack_t + ack_dat_t

    if proto_num == 6:
        state = "RST" if flow.has_rst else "FIN" if flow.has_fin else "CON" if flow.has_syn else "INT"
    elif proto_num == 17:
        state = "CON"
    elif proto_num == 1:
        state = "ECO"
    else:
        state = "CON"

    s_tos = int(src_p[0][3]) if src_p else 0
    d_tos = int(dst_p[0][3]) if dst_p else 0

    return {
        'Unnamed: 0':  row_idx,
        'Seq':         flow.seq_id,
        'Dur':         round(dur, 6),
        'RunTime':     round(flow.last_ts - flow.runtime, 6),
        'Mean':        round(float(np.mean(al)) if al else 0, 6),
        'Sum':         float(tb),
        'Min':         float(min(al)) if al else 0.0,
        'Max':         float(max(al)) if al else 0.0,
        'Proto':       proto_str,
        'sTos':        s_tos,
        'dTos':        float(d_tos),
        'sDSb':        _dsb(s_tos),
        'dDSb':        _dsb(d_tos),
        'sTtl':        s_ttl,
        'dTtl':        float(d_ttl),
        'sHops':       _hops(s_ttl),
        'dHops':       float(_hops(d_ttl)),
        'Cause':       flow.cause,
        'TotPkts':     tp,
        'SrcPkts':     sc,
        'DstPkts':     dc,
        'TotBytes':    tb,
        'SrcBytes':    sb,
        'DstBytes':    db,
        'Offset':      row_idx,
        'sMeanPktSz':  round(float(np.mean(sl)) if sl else 0, 6),
        'dMeanPktSz':  round(float(np.mean(dl)) if dl else 0, 6),
        'Load':        round(tb / dur_s, 6),
        'SrcLoad':     round(sb / dur_s, 6),
        'DstLoad':     round(db / dur_s, 6),
        'Loss':        loss,
        'SrcLoss':     sl_,
        'DstLoss':     dl_,
        'pLoss':       round((loss / tp * 100) if tp else 0, 6),
        'SrcGap':      round(gaps(src_p), 6),
        'DstGap':      round(gaps(dst_p), 6),
        'Rate':        round(tp / dur_s, 6),
        'SrcRate':     round(sc / dur_s, 6),
        'DstRate':     round(dc / dur_s, 6),
        'State':       state,
        'SrcWin':      float(src_p[0][6]) if src_p and proto_num == 6 else 0.0,
        'DstWin':      float(dst_p[0][6]) if dst_p and proto_num == 6 else 0.0,
        'sVid':        0.0,
        'dVid':        0.0,
        'SrcTCPBase':  float(src_p[0][7]) if src_p and proto_num == 6 else 0.0,
        'DstTCPBase':  float(dst_p[0][7]) if dst_p and proto_num == 6 else 0.0,
        'TcpRtt':      tcp_rtt,
        'SynAck':      syn_ack_t,
        'AckDat':      ack_dat_t,
        'Label':       label,
        'Attack Type': attack_type,
        'Attack Tool': attack_tool,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API  — called by the Django view
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_iface(iface):
    # Normalize Windows GUID forms to the full path Scapy needs.
    # Bare GUID: {xxxx} -> \\Device\\NPF_{xxxx}
    # NPF_{xxxx}        -> \\Device\\NPF_{xxxx}
    # Already full path -> unchanged
    # eth0 / Wi-Fi      -> unchanged (Linux/macOS)
    NPF = '\\Device\\NPF_'
    s = iface.strip()
    if s.replace('\\\\', '\\').startswith(NPF):
        return s
    if s.startswith('{') and s.endswith('}'):
        return NPF + s
    if s.upper().startswith('NPF_'):
        return NPF + s[4:]
    return s
def capture(dataset_type, iface, packet_count, label, output_path,
            attack_type=None, attack_tool="Live-Capture"):
    """
    Capture live traffic and save as a CSV in the correct dataset format.

    Parameters
    ----------
    dataset_type  : "cic" | "5g"
    iface         : network interface name  (e.g. "eth0" or Windows NPF GUID)
    packet_count  : number of packets to capture
    label         : label string to assign to all flows
    output_path   : full path where the output CSV will be saved
    attack_type   : (5G only) Attack Type column value; defaults to label
    attack_tool   : (5G only) Attack Tool column value

    Returns
    -------
    (success: bool, message: str, flow_count: int)
    """
    IP, TCP, UDP, ICMP, sniff = _import_scapy()

    # "ALL" or blank = capture on every interface at once (recommended on Windows)
    if iface.strip().upper() in ("ALL", "", "0"):
        iface_arg = None   # Scapy captures on all interfaces when iface=None
    else:
        iface_arg = _normalize_iface(iface.strip())

    if attack_type is None:
        attack_type = label

    flows   = {}
    records = []
    seq_id  = [1]   # mutable for closure

    def _expire(flows, ts, force=False):
        expired = [k for k, f in flows.items()
                   if force or (ts - f.last_ts > FLOW_TIMEOUT)]
        for k in expired:
            if dataset_type == "cic":
                rec = _cic_features(flows[k], label)
            else:
                rec = _5g_features(flows[k], seq_id[0], label, attack_type, attack_tool)
                seq_id[0] += 1
            if rec:
                records.append(rec)
            del flows[k]

    def handle(pkt):
        if IP not in pkt:
            return
        key = _flow_key(pkt, IP, TCP, UDP, ICMP)
        if not key:
            return
        ts = float(pkt.time)
        _expire(flows, ts)
        if key not in flows:
            if dataset_type == "cic":
                flows[key] = _CICFlow(key, ts)
            else:
                flows[key] = _5GFlow(key, seq_id[0], ts)
        if dataset_type == "cic":
            flows[key].add(pkt, ts, IP, TCP, UDP)
        else:
            flows[key].add(pkt, ts, IP, TCP, UDP, ICMP)

    # Hard timeout = 3 seconds per packet, minimum 60s, maximum 300s
    # This prevents hanging forever on an idle interface
    hard_timeout = max(60, min(packet_count * 3, 300))

    try:
        sniff(iface=iface_arg, count=packet_count, prn=handle,
              store=False, timeout=hard_timeout)
    except Exception as e:
        return False, f"Capture failed: {str(e)}", 0

    # Flush remaining open flows
    _expire(flows, time.time(), force=True)

    if not records:
        return False, "No flows captured. Try increasing packet count.", 0

    cols = CIC_COLUMNS if dataset_type == "cic" else NIDD_COLUMNS
    df = pd.DataFrame(records)
    for col in cols:
        if col not in df.columns:
            df[col] = 0
    df = df[cols]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return True, f"Captured {len(df)} flows saved to {os.path.basename(output_path)}", len(df)


def list_interfaces():
    # Returns [(value_for_sniff, display_label), ...]
    # First entry is always "ALL" = capture on every interface at once (safest default).
    NPF = '\\Device\\NPF_'
    default = [("ALL", "ALL interfaces (recommended — captures Wi-Fi + Ethernet together)")]
    try:
        from scapy.all import get_if_list
        try:
            from scapy.arch.windows import get_windows_if_list
            ifaces = get_windows_if_list()
            result = []
            for i in ifaces:
                guid = i.get('guid', '')
                name = i.get('name', '')
                desc = i.get('description', '')
                npf_path = (NPF + guid) if guid else name
                # Build a friendly label: prefer description, fall back to name
                friendly = desc if desc else name
                label = f"{friendly}  [{name}]" if (desc and name and desc != name) else friendly
                result.append((npf_path, label))
            return default + result
        except Exception:
            ifaces = get_if_list()
            return default + [(i, i) for i in ifaces]
    except ImportError:
        return default
