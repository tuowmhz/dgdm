import base64
import csv
import io
import serial
import time
import threading
from collections import deque
from datetime import datetime

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from serial.tools import list_ports

# === Alicat Setup ===
# Ten-unit order as requested
UNIT_ORDER = ["A", "B", "C", "D", "E", "M", "G", "H", "I", "J"]

# Split across two adapters: first five on port1, last five on port2.
# Update PORT1/PORT2 if you know the exact paths; otherwise we try to auto-detect.
PORT1 = "/dev/tty.usbserial-AU0124RG"   # A, B, C, D, E
PORT2 = "/dev/tty.usbserial-AU0122XL"  # M, G, H, I, J

def _discover_ports():
    """Return (candidates, all_ports). Candidates are likely USB serial devices."""
    try:
        all_ports = [p.device for p in list_ports.comports()]
    except Exception:
        all_ports = []
    print(f"Available serial ports: {all_ports}")
    # Likely USB serial names across macOS/Linux
    candidates = [p for p in all_ports if any(s in p for s in ("usbserial", "usbmodem", "ttyUSB", "ttyACM"))]
    # Exclude macOS built-ins
    candidates = [p for p in candidates if "Bluetooth" not in p and "debug" not in p]
    return candidates, all_ports

_candidates, _all_ports = _discover_ports()
port1 = PORT1 if PORT1 in _all_ports else (_candidates[0] if len(_candidates) >= 1 else PORT1)
port2 = PORT2 if PORT2 in _all_ports else (_candidates[1] if len(_candidates) >= 2 else PORT2)

COM_PORTS = {
    port1: ["A", "B", "C", "D", "E"],
    port2: ["M", "G", "H", "I", "J"],
}
BAUD_RATE = 19200
SETPOINTS = {u: 0.0 for u in UNIT_ORDER}
MAX_POINTS = 100  # How many points to keep in the plot

# === Data Buffers ===
pressure_data = {addr: deque(maxlen=MAX_POINTS) for addr in SETPOINTS.keys()}
time_data = deque(maxlen=MAX_POINTS)

# === Serial Connection ===
ser_connections = {}
for port in COM_PORTS:
    try:
        ser_connections[port] = serial.Serial(port, BAUD_RATE, timeout=0.5)
        print(f"Connected to {port}")
    except serial.SerialException as e:
        print(f"Error connecting to {port}: {e} — skipping this port")
        continue

if not ser_connections:
    print("Warning: no serial ports opened; UI will run but hardware I/O is disabled.")

# === Helper Functions ===
def set_pressure(port, address, pressure_value):
    cmd = f"{address}S{pressure_value:.2f}\r"
    print(f"Setting {address} to {pressure_value:.2f} psi with command: {cmd.strip()}")
    ser_connections[port].write(cmd.encode())

def read_pressure(port, address):
    cmd = f"{address}\r"
    try:
        ser_connections[port].write(cmd.encode())
        time.sleep(0.1)
        response = ser_connections[port].readline().decode().strip()
        print(f"[{address}] response: '{response}'")
        return response
    except Exception as e:
        print(f"Read error for {address}: {e}")
        return ""

def extract_pressure(response):
    if not response:
        return None
    try:
        tokens = response.split()
        for i, token in enumerate(tokens):
            if "PSI" in token.upper() and i > 0:
                return float(tokens[i - 1])
        # Fallback: try second token
        return float(tokens[1])
    except Exception as e:
        print(f"Could not parse pressure from: '{response}' — {e}")
        return None

# === CSV Parsing ===
def parse_csv_10(values_bytes: bytes):
    """Return a list of exactly 10 floats from CSV bytes (row or column)."""
    text = values_bytes.decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(text))
    nums = []
    for row in reader:
        for cell in row:
            c = cell.strip()
            if not c:
                continue
            try:
                nums.append(float(c))
            except ValueError:
                # ignore headers/strings
                continue
            if len(nums) == 10:
                return nums
    if len(nums) != 10:
        raise ValueError(f"CSV must contain exactly 10 numeric values; found {len(nums)}.")
    return nums

# === Background Thread ===
def update_pressure_loop():
    while True:
        timestamp = time.time()
        time_data.append(timestamp)
        for port, addresses in COM_PORTS.items():
            if port not in ser_connections:
                continue
            for addr in addresses:
                response = read_pressure(port, addr)
                pressure = extract_pressure(response)
                pressure_data[addr].append(pressure if pressure is not None else 0)
        time.sleep(0.5)

threading.Thread(target=update_pressure_loop, daemon=True).start()

# === Initial Pressure Set ===
for port, addresses in COM_PORTS.items():
    if port not in ser_connections:
        continue
    for addr in addresses:
        if addr in SETPOINTS:
            set_pressure(port, addr, SETPOINTS[addr])
            time.sleep(0.1)

# ───────── UI: Upload CSV for ten set‑points ─────────
upload_ui = html.Div(
    [
        html.Label(
            "Upload a CSV with 10 numbers for units A, B, C, D, E, M, G, H, I, J (in that order):",
            style={"marginRight": "8px"},
        ),
        dcc.Upload(
            id="upload-csv",
            children=html.Div(["Drag and Drop or ", html.A("Select File")]),
            multiple=False,
            style={
                "width": "360px",
                "lineHeight": "32px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "4px",
                "textAlign": "center",
                "margin": "8px 0",
            },
        ),
        html.Div(id="upload-filename", style={"marginTop": "6px"}),
        html.Button("Apply from CSV", id="apply-csv-btn", n_clicks=0, style={"marginLeft": "0px"}),
        html.Span(id="apply-status", style={"marginLeft": "12px", "fontWeight": "bold"}),
    ],
    style={"margin": "10px 0"},
)

# === Dash App ===
app = dash.Dash(__name__)
app.title = "Alicat Real-Time Pressure Monitor"

app.layout = html.Div(
    [
        html.H2("Real‑Time Alicat Pressure Monitor"),
        dcc.Graph(id="live-graph", animate=False),
        html.Hr(),
        html.H3("Set new pressures (psi) from CSV"),
        upload_ui,
        html.Hr(),
        html.H4("Set‑point History (latest first)"),
        html.Pre(
            id="log-display",
            style={
                "maxHeight": "200px",
                "overflowY": "auto",
                "background": "#111",
                "color": "#0f0",
                "padding": "8px",
                "border": "1px solid #444",
            },
        ),
        html.Button("Download Log", id="download-btn", n_clicks=0, style={"marginTop": "12px"}),
        dcc.Download(id="download-log"),
        dcc.Interval(id="interval-component", interval=1000, n_intervals=0),
        dcc.Store(id="log-store", data=[]),  # hidden store for log records
    ],
    style={"font-family": "sans-serif", "margin": "0 20px"},
)

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    fig = go.Figure()
    for addr, data in pressure_data.items():
        fig.add_trace(go.Scatter(
            x=list(time_data),
            y=list(data),
            mode='lines+markers',
            name=f'Unit {addr}'
        ))
    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Pressure (psi)',
        title='Live Pressure Readings',
        uirevision=True,
        template='plotly_dark',
        height=600
    )
    return fig

# --- Apply from CSV ---
@app.callback(
    Output("apply-status", "children"),
    Output("upload-filename", "children"),
    Output("log-store", "data"),
    Input("apply-csv-btn", "n_clicks"),
    State("upload-csv", "contents"),
    State("upload-csv", "filename"),
    State("log-store", "data"),
    prevent_initial_call=True,
)
def apply_from_csv(n_clicks, contents, filename, log_data):
    log_data = log_data or []
    if not contents:
        return "✖  Please upload a CSV file first.", "", log_data

    try:
        # contents format: "data:text/csv;base64,<b64>"
        header, b64data = contents.split(",", 1)
        file_bytes = base64.b64decode(b64data)
        values = parse_csv_10(file_bytes)
    except Exception as e:
        return f"✖  Failed to parse CSV: {e}", filename or "", log_data

    # Map values to units in UNIT_ORDER
    assignments = dict(zip(UNIT_ORDER, values))

    # Send to hardware and log
    for port, addresses in COM_PORTS.items():
        if port not in ser_connections:
            # Log but skip if port is not open
            for addr in addresses:
                val = assignments[addr]
                SETPOINTS[addr] = val
                log_entry = f"{datetime.now().strftime('%H:%M:%S')} | {addr} -> {val:.2f} psi (SKIPPED: {port} not connected)"
                log_data.insert(0, log_entry)
            continue
        for addr in addresses:
            val = assignments[addr]
            SETPOINTS[addr] = val
            set_pressure(port, addr, val)
            log_entry = f"{datetime.now().strftime('%H:%M:%S')} | {addr} -> {val:.2f} psi"
            log_data.insert(0, log_entry)
            time.sleep(0.03)

    status = f"✓ Applied CSV to units {', '.join(UNIT_ORDER)} at {datetime.now().strftime('%H:%M:%S')}"
    shown_name = filename or "(uploaded)"
    return status, f"Loaded: {shown_name}", log_data

@app.callback(
    Output("log-display", "children"),
    Input("log-store", "data")
)
def update_log_display(log_data):
    return "\n".join(log_data[:20])  # show max 20 recent entries

@app.callback(
    Output("download-log", "data"),
    Input("download-btn", "n_clicks"),
    State("log-store", "data"),
    prevent_initial_call=True
)
def trigger_download(n_clicks, log_data):
    if not log_data:
        return dash.no_update
    # Combine log lines into single string (newline-separated)
    content = "\n".join(log_data)
    filename = f"setpoint_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    return dict(content=content, filename=filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=False)