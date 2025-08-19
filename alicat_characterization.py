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

# ───────── UI: Manual set‑points per unit ─────────
controls_ui = html.Div(
    [
        html.Label("Set pressures (psi) – changes apply instantly"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(f"Unit {u}", style={"fontWeight": "bold"}),
                        dcc.Input(
                            id=f"input-{u}",
                            type="number",
                            value=SETPOINTS[u],
                            step=0.01,
                            debounce=True,
                            style={"width": "100%"}
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "4px",
                        "padding": "6px",
                        "border": "1px solid #444",
                        "borderRadius": "4px",
                    },
                )
                for u in UNIT_ORDER
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(120px, 1fr))",
                "gap": "8px",
                "marginTop": "8px",
            },
        ),
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
        html.H3("Set pressures (psi) per unit"),
        controls_ui,
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
        dcc.Store(id="setpoints-store", data=SETPOINTS),  # hidden store for current set-points
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

# --- Apply from per-unit inputs (instant apply on change) ---
@app.callback(
    Output("setpoints-store", "data"),
    Output("log-store", "data"),
    Input("input-A", "value"),
    Input("input-B", "value"),
    Input("input-C", "value"),
    Input("input-D", "value"),
    Input("input-E", "value"),
    Input("input-M", "value"),
    Input("input-G", "value"),
    Input("input-H", "value"),
    Input("input-I", "value"),
    Input("input-J", "value"),
    State("setpoints-store", "data"),
    State("log-store", "data"),
    prevent_initial_call=True,
)
def apply_from_inputs(valA, valB, valC, valD, valE, valM, valG, valH, valI, valJ, setpoints_data, log_data):
    log_data = log_data or []
    prev = setpoints_data or {u: 0.0 for u in UNIT_ORDER}
    new_map = {
        "A": valA, "B": valB, "C": valC, "D": valD, "E": valE,
        "M": valM, "G": valG, "H": valH, "I": valI, "J": valJ
    }

    def _find_port_for_unit(unit):
        for p, addrs in COM_PORTS.items():
            if unit in addrs:
                return p
        return None

    for unit, new_val in new_map.items():
        if new_val is None:
            continue
        try:
            new_val = float(new_val)
        except (TypeError, ValueError):
            continue

        if abs(new_val - float(prev.get(unit, 0.0))) < 1e-9:
            continue

        port = _find_port_for_unit(unit)
        if port not in ser_connections:
            SETPOINTS[unit] = new_val
            log_entry = f"{datetime.now().strftime('%H:%M:%S')} | {unit} -> {new_val:.2f} psi (SKIPPED: {port} not connected)"
            log_data.insert(0, log_entry)
            continue

        SETPOINTS[unit] = new_val
        set_pressure(port, unit, new_val)
        log_entry = f"{datetime.now().strftime('%H:%M:%S')} | {unit} -> {new_val:.2f} psi"
        log_data.insert(0, log_entry)
        time.sleep(0.03)

    return new_map, log_data

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

# Run the app
if __name__ == '__main__':
    app.run(debug=False)