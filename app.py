import csv
import cv2
from datetime import datetime
from functools import wraps
import io
import json
import os
import traceback
import socket
import uuid

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
import pandas as pd

app = Flask(__name__)
app.secret_key = "your_secret_key"

# ─── PATHS & CONFIG ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER = os.path.join(BASE_DIR, "templates", "csv_files")
os.makedirs(CSV_FOLDER, exist_ok=True)

CAMERA_CSV = os.path.join(CSV_FOLDER, "camera_data.csv")
ANPR_FILE = os.path.join(CSV_FOLDER, "anpr_data.csv")
WHITELIST_FILE = os.path.join(CSV_FOLDER, "whitelist.csv")
PARKING_CONFIG_FILE = os.path.join(CSV_FOLDER, "parking_config.json")
SITE_FILE = os.path.join(CSV_FOLDER, "site_ids.csv")
VEHICLE_IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "vehicle_images")
ENTRY_IMAGE_FOLDER = os.path.join(VEHICLE_IMAGE_FOLDER,"entry")
EXIT_IMAGE_FOLDER = os.path.join(VEHICLE_IMAGE_FOLDER,"exit")
os.makedirs(ENTRY_IMAGE_FOLDER, exist_ok=True)
os.makedirs(EXIT_IMAGE_FOLDER, exist_ok=True)
DEFAULT_PARKING_CAPACITY = 500
SITE_CAPACITY = 10

USERS = {
    "admin": "admin123@",
    "operator": "opr@2026",
    "viewer": "view@2026",
    "user": "user456#",
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_parking_capacity():
    """Return the client-configured parking capacity."""
    try:
        if os.path.exists(PARKING_CONFIG_FILE):
            with open(PARKING_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            capacity = int(data.get("capacity", DEFAULT_PARKING_CAPACITY))
            return max(0, capacity)
    except Exception:
        pass
    return DEFAULT_PARKING_CAPACITY


def save_parking_capacity_value(capacity):
    capacity = max(0, int(capacity))
    with open(PARKING_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"capacity": capacity}, f)


def login_required(f):
  @wraps(f)
  def decorated_function(*args, **kwargs):
    if "user" not in session:
      return redirect(url_for("login"))
    return f(*args, **kwargs)

  return decorated_function


def update_entry_exit_csvs():
  """Reads the main ANPR file, cleans NaN rows, and updates both entry and exit CSV files."""
  if not os.path.exists(ANPR_FILE):
    return

  try:
    df = pd.read_csv(ANPR_FILE)
    df = df.dropna(subset=["plate_number"])

    # Clean headers
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace("\ufeff", "")
        .str.lower()
    )

    valid_cols = [c for c in df.columns if not c.startswith("unnamed")]
    df = df[valid_cols]

    entry_df = df[
        df["location"].astype(str).str.contains("entry", case=False, na=False)
    ]
    exit_df = df[
        df["location"].astype(str).str.contains("exit", case=False, na=False)
    ]

    os.makedirs(CSV_FOLDER, exist_ok=True)
    entry_df.to_csv(os.path.join(CSV_FOLDER, "entry_logs.csv"), index=False)
    exit_df.to_csv(os.path.join(CSV_FOLDER, "exit_logs.csv"), index=False)
  except Exception as e:
    print("Error updating Entry/Exit CSVs:", e)


def init_anpr_data():
  if not os.path.exists(ANPR_FILE):
    with open(ANPR_FILE, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["plate_number", "date", "time", "location", "image"])


def read_anpr_data():
  try:
    df = pd.read_csv(ANPR_FILE)
    df = df.dropna(how="all")
    df = df.fillna("")
    df = df[df["plate_number"].astype(str).str.strip() != ""]
    return df
  except Exception:
    return pd.DataFrame()


def save_anpr_data(plate_number, location, frame=None):
    """
    Save ANPR detection information and vehicle image.
    frame:
        OpenCV image/frame captured from the camera.
    """
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H:%M:%S")

    plate_number = str(plate_number).strip().upper()
    location = str(location).strip()

    image_path = ""
    if frame is not None:
        safe_plate = "".join(
            c for c in plate_number
            if c.isalnum()
        )
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_plate}_{timestamp}.jpg"

        # Entry / Exit folder
        if "entry" in location.lower():
            image_folder = ENTRY_IMAGE_FOLDER
            relative_folder = "vehicle_images/entry"

        elif "exit" in location.lower():
            image_folder = EXIT_IMAGE_FOLDER
            relative_folder = "vehicle_images/exit"

        else:
            image_folder = VEHICLE_IMAGE_FOLDER
            relative_folder = "vehicle_images"
        os.makedirs(image_folder, exist_ok=True)

        full_image_path = os.path.join(
            static_folder := os.path.join(BASE_DIR, "static"),
            filename
        )
        # Save OpenCV frame
        cv2.imwrite(full_image_path, frame)

        # Path used by browser
        image_path = f"{relative_folder}/{filename}"

    file_exists = os.path.isfile(ANPR_FILE)

    with open(
        ANPR_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists or os.path.getsize(ANPR_FILE) == 0:
            writer.writerow([
                "plate_number",
                "date",
                "time",
                "location",
                "image"
            ])

        writer.writerow([
            plate_number,
            date,
            time,
            location,
            image_path
        ])


def get_anpr_stats(df):
  cameras = load_cameras()
  if df.empty:
    return {
        "total_records": 0,
        "unique_plates": 0,
        "cameras_used": len(cameras),
        "locations": {},
    }

  return {
      "total_records": len(df),
      "unique_plates": df["plate_number"].nunique(),
      "cameras_used": len(cameras),
      "locations": df["location"].value_counts().to_dict(),
  }

def init_camera_csv():
    if not os.path.exists(CAMERA_CSV):
        with open(CAMERA_CSV, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "address",
                "location",
                "camera_type",
                "ip",
                "username",
                "password",
                "status"
            ])

def load_cameras():
    cameras = []

    if not os.path.exists(CAMERA_CSV):
        return cameras

    try:
        df = pd.read_csv(CAMERA_CSV)
        df.columns = df.columns.str.strip().str.lower()
        df = df.dropna(how="all")

        for idx, row in df.iterrows():

            ip_val = str(row.get("ip", "")).strip()

            # Ignore deleted cameras
            if not ip_val or ip_val.upper() == "N":
                continue

            location = str(row.get("location", "")).strip()

            # Detect Entry / Exit
            location_upper = location.upper()

            if location_upper.startswith("ENTRY"):
                camera_type = "Entry"
            elif location_upper.startswith("EXIT"):
                camera_type = "Exit"
            else:
                camera_type = str(
                    row.get("camera_type", "Unknown")
                ).strip()

            # Extract camera number
            camera_number = None

            parts = location.split()

            if len(parts) >= 3 and parts[-1].isdigit():
                camera_number = int(parts[-1])

            cameras.append({
                "id": idx,
                "number": camera_number,
                "name": location,
                "type": camera_type,
                "ip": ip_val,
                "username": str(row.get("username", "")).strip(),
                "password": str(row.get("password", "")).strip(),
                "status": str(
                    row.get("status", "OFF")
                ).strip().upper(),
            })

    except Exception as e:
        print("Error loading cameras:", e)

    return cameras

def get_next_address():
  if not os.path.exists(CAMERA_CSV):
    return 1001

  with open(CAMERA_CSV, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    addresses = []
    for row in reader:
      address = row.get("address", "").strip()
      if address.isdigit():
        addresses.append(int(address))

  return max(addresses) + 1 if addresses else 1001


def get_parking_stats():
    capacity = get_parking_capacity()

    if not os.path.exists(CAMERA_CSV):
        return {"total": capacity, "occupied": 0, "vacant": capacity}

    try:
        df = pd.read_csv(CAMERA_CSV)
    except Exception:
        return {"total": capacity, "occupied": 0, "vacant": capacity}

    if df.empty:
        return {"total": capacity, "occupied": 0, "vacant": capacity}

    df.columns = df.columns.str.strip().str.lower()

    if "ip" in df.columns:
        df = df[df["ip"].fillna("").astype(str).str.strip().str.upper() != "N"]

    df["location"] = df["location"].fillna("").astype(str).str.strip().str.upper()
    entry_count = df["location"].str.startswith("ENTRY").sum()
    exit_count = df["location"].str.startswith("EXIT").sum()

    occupied = min(capacity, max(0, entry_count - exit_count))
    vacant = max(0, capacity - occupied)

    return {"total": capacity, "occupied": occupied, "vacant": vacant}

def init_whitelist():
    if not os.path.exists(WHITELIST_FILE):
        with open(WHITELIST_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "plate_number",
                "vehicle_type",
                "owner_name",
                "timestamp"
            ])

def read_whitelist():
    columns = [
        "plate_number",
        "vehicle_type",
        "owner_name",
        "timestamp"
    ]

    if not os.path.exists(WHITELIST_FILE):
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(WHITELIST_FILE)

        df.columns = (
            df.columns.astype(str)
            .str.strip()
            .str.replace("\ufeff", "")
            .str.lower()
            .str.replace(" ", "_")
        )

        # Add missing columns for compatibility
        if "plate_number" not in df.columns:
            return pd.DataFrame(columns=columns)

        if "vehicle_type" not in df.columns:
            df["vehicle_type"] = ""

        if "owner_name" not in df.columns:
            df["owner_name"] = ""

        if "timestamp" not in df.columns:
            df["timestamp"] = ""

        df = df.dropna(how="all")

        for col in columns:
            df[col] = (
                df[col]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        # Remove empty plate numbers
        df = df[
            (df["plate_number"] != "") &
            (df["plate_number"].str.lower() != "nan")
        ]

        return df[columns]

    except Exception as e:
        print("Error reading whitelist:", e)
        return pd.DataFrame(columns=columns)

def init_site_csv():
    if not os.path.exists(SITE_FILE):
        pd.DataFrame(
          columns=["site_id"]
        ).to_csv(SITE_FILE,index=False
        )

def load_site_ids():
    init_site_csv()
    try:
        df = pd.read_csv(SITE_FILE)
        if "site_id" not in df.columns:
            return []
        return (
            df["site_id"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
    except Exception as e:
        print("Error loading Site IDs:", e)
        return []


# ─── ROUTES ───────────────────────────────────────────────────────────────────
@app.route("/")
def home():
  return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
  if request.method == "POST":
    user_id = request.form.get("user_id")
    password = request.form.get("password")

    if user_id in USERS and USERS[user_id] == password:
      session["user"] = user_id
      return redirect(url_for("dashboard"))
    else:
      flash("Invalid user_id or password")
      return redirect(url_for("login"))

  return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
  try:
    anpr_df = pd.read_csv(ANPR_FILE)
  except Exception:
    anpr_df = pd.DataFrame(columns=["plate_number", "date", "time", "location"])

  try:
    cam_df = pd.read_csv(CAMERA_CSV)
  except Exception:
    cam_df = pd.DataFrame(
        columns=["address", "location", "ip", "username", "password", "status"]
    )

  if "location" in anpr_df.columns:
    anpr_df["location"] = (
        anpr_df["location"].astype(str).str.strip().str.upper()
    )
  if "location" in cam_df.columns:
    cam_df["location"] = (
        cam_df["location"].astype(str).str.strip().str.upper()
    )
  if "ip" in cam_df.columns:
      cam_df = cam_df[cam_df["ip"].fillna("").astype(str).str.strip().str.upper() != "N"]

  whitelist_df = read_whitelist()
  whitelisted = whitelist_df.to_dict(orient="records")

  records = []
  for _, row in cam_df.reset_index().iterrows():
    records.append({
        "camera_name": row.get("location", "N/A"),
        "camera_ip": row.get("ip", "N/A"),
        "camera_status": row.get("status", "OFF"),
    })

  total_entries = 0
  total_exits = 0
  occupied = 0
  parking_capacity = get_parking_capacity()
  vacant = parking_capacity

  if os.path.exists(ANPR_FILE):
    try:
      df = pd.read_csv(ANPR_FILE).dropna(subset=["plate_number"])
      location_series = df["location"].astype(str)
      total_entries = (
          location_series.str.contains("entry", case=False, na=False).sum()
      )
      total_exits = (
          location_series.str.contains("exit", case=False, na=False).sum()
      )
      occupied = max(0, total_entries - total_exits)
      vacant = max(0, parking_capacity - occupied)
    except Exception as e:
      print("Error calculating parking stats:", e)

  stats = {
      "total": parking_capacity,
      "vacant": vacant,
      "occupied": occupied,
      "entries": total_entries,
      "exits": total_exits,
  }

  return render_template(
      "dashboard.html",
      records=records,
      parking=get_parking_stats(),
      stats=stats,
      whitelist=whitelist_df,
      whitelisted=whitelisted,
      total=len(whitelisted),
  )


@app.route("/parking")
@login_required
def parking():
  return render_template("parking.html", stats=get_parking_stats())


@app.route("/records")
@login_required
def records():
  df = read_anpr_data()
  whitelist_df = read_whitelist()

  whitelist = set(
      whitelist_df["plate_number"].fillna("").astype(str).str.upper().str.strip()
  )

  if not df.empty:
    df["plate_number"] = (
        df["plate_number"].fillna("").astype(str).str.upper().str.strip()
    )
    df["whitelisted"] = df["plate_number"].apply(
        lambda x: "Yes" if x in whitelist else "No"
    )
    records_list = df.to_dict(orient="records")
  else:
    records_list = []

  return render_template(
      "records.html",
      records=records_list,
      stats=get_anpr_stats(df),
  )


@app.route("/whitelisted")
@login_required
def whitelisted():
  df = read_whitelist()
  return render_template("white_listed.html", data=df)


@app.route("/add_to_whitelist", methods=["POST"])
@login_required
def add_to_whitelist():
    data = request.get_json() or {}

    plate_number = str(data.get("plate", "")).strip().upper()
    vehicle_type = str(data.get("vehicle_type", "")).strip()
    owner_name = str(data.get("owner_name", "")).strip()

    if not plate_number:
        return jsonify({
            "status": "error",
            "message": "Plate number is required"
        })

    existing_numbers = set()

    if os.path.exists(WHITELIST_FILE):
        with open(
            WHITELIST_FILE,
            "r",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            for row in reader:
                existing_plate = str(
                    row.get("plate_number", "")
                ).strip().upper()

                if existing_plate:
                    existing_numbers.add(existing_plate)

    if plate_number in existing_numbers:
        return jsonify({
            "status": "error",
            "message": "Vehicle already exists"
        })

    timestamp = datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )

    file_exists = (
        os.path.exists(WHITELIST_FILE)
        and os.path.getsize(WHITELIST_FILE) > 0
    )

    with open(
        WHITELIST_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "plate_number",
                "vehicle_type",
                "owner_name",
                "timestamp"
            ])

        writer.writerow([
            plate_number,
            vehicle_type,
            owner_name,
            timestamp
        ])

    return jsonify({
        "status": "success",
        "message": "Vehicle added successfully ✅"
    })

@app.route("/remove_white_listed", methods=["POST"])
@login_required
def remove_white_listed():
    """
    Remove a vehicle from the whitelist.

    Uses read_whitelist() instead of csv.DictReader directly so that
    headers such as "Plate Number", BOM characters, spaces, etc. are
    normalized consistently.
    """
    try:
        data = request.get_json(silent=True) or {}
        plate = str(data.get("plate", "")).strip().upper()

        if not plate:
            return jsonify({
                "status": "error",
                "message": "Plate number is required."
            }), 400

        df = read_whitelist()

        if df.empty:
            return jsonify({
                "status": "error",
                "message": "Vehicle not found."
            })

        # Normalize stored plate numbers before comparing.
        df["plate_number"] = (
            df["plate_number"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )

        # Remove every matching occurrence.
        match = df["plate_number"] == plate
        removed = bool(match.any())

        if not removed:
            return jsonify({
                "status": "error",
                "message": f"Vehicle {plate} not found."
            })

        df = df.loc[~match].copy()

        # Always write the whitelist back using the application's
        # canonical column names.
        columns = ["plate_number", "vehicle_type", "owner_name", "timestamp"]

        for col in columns:
            if col not in df.columns:
                df[col] = ""

        df = df[columns]

        df.to_csv(
            WHITELIST_FILE,
            index=False,
            encoding="utf-8"
        )

        return jsonify({
            "status": "success",
            "message": f"Vehicle {plate} removed successfully ✅"
        })

    except Exception as e:
        print("Error removing whitelisted vehicle:", e)
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": f"Failed to remove vehicle: {str(e)}"
        }), 500


@app.route("/upload_whitelist_csv", methods=["POST"])
@login_required
def upload_whitelist_csv():

    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded"
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "status": "error",
            "message": "No selected file"
        }), 400

    try:

        stream = io.StringIO(
            file.stream.read().decode("utf-8-sig"),
            newline=None
        )

        reader = csv.DictReader(stream)

        if not reader.fieldnames:
            return jsonify({
                "status": "error",
                "message": (
                    "CSV must contain: "
                    "Plate Number, Vehicle Type, "
                    "Owner Name, Timestamp"
                )
            }), 400

        # Normalize headers
        header_map = {
            h.strip().lower().replace(" ", "_"): h
            for h in reader.fieldnames
        }

        plate_key = header_map.get("plate_number")
        vehicle_key = header_map.get("vehicle_type")
        owner_key = header_map.get("owner_name")
        timestamp_key = header_map.get("timestamp")

        if not all([
            plate_key,
            vehicle_key,
            owner_key,
            timestamp_key
        ]):
            return jsonify({
                "status": "error",
                "message": (
                    "Invalid CSV format. Required columns are: "
                    "Plate Number, Vehicle Type, "
                    "Owner Name, Timestamp"
                )
            }), 400

        # Existing plates
        existing = set()

        if os.path.exists(WHITELIST_FILE):

            with open(
                WHITELIST_FILE,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as f:

                for row in csv.DictReader(f):

                    plate = str(
                        row.get("plate_number", "")
                    ).strip().upper()

                    if plate:
                        existing.add(plate)

        new_rows = []
        duplicate_count = 0

        for row in reader:

            plate = str(
                row.get(plate_key, "")
            ).strip().upper()

            vehicle_type = str(
                row.get(vehicle_key, "")
            ).strip()

            owner_name = str(
                row.get(owner_key, "")
            ).strip()

            timestamp = str(
                row.get(timestamp_key, "")
            ).strip()

            if not plate:
                continue

            if plate in existing:
                duplicate_count += 1
                continue

            # Generate timestamp if CSV timestamp is blank
            if not timestamp:
                timestamp = datetime.now().strftime(
                    "%d-%m-%Y %I:%M:%S %p"
                )

            existing.add(plate)

            new_rows.append([
                plate,
                vehicle_type,
                owner_name,
                timestamp
            ])

        if not new_rows:
            return jsonify({
                "status": "warning",
                "message": (
                    f"No new vehicles added. "
                    f"({duplicate_count} duplicate(s) skipped)"
                )
            })

        file_exists = (
            os.path.exists(WHITELIST_FILE)
            and os.path.getsize(WHITELIST_FILE) > 0
        )

        with open(
            WHITELIST_FILE,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "plate_number",
                    "vehicle_type",
                    "owner_name",
                    "timestamp"
                ])

            writer.writerows(new_rows)

        message = (
            f"Successfully added "
            f"{len(new_rows)} vehicle(s) to whitelist!"
        )

        if duplicate_count:
            message += (
                f" ({duplicate_count} duplicate(s) skipped)"
            )

        return jsonify({
            "status": "success",
            "message": message
        })

    except Exception as e:

        return jsonify({
            "status": "error",
            "message": (
                f"Failed to process whitelist CSV: {str(e)}"
            )
        }), 500


@app.route("/download_whitelist_csv")
@login_required
def download_whitelist_csv():
  if os.path.exists(WHITELIST_FILE):
    return send_file(
        WHITELIST_FILE,
        mimetype="text/csv",
        as_attachment=True,
        download_name="whitelist.csv",
    )
  flash("Whitelist CSV file does not exist yet.", "error")
  return redirect(url_for("config"))


@app.route("/config")
@login_required
def config():
  whitelist_df = read_whitelist()
  whitelisted = whitelist_df.to_dict(orient="records")
  whitelisted.reverse()
  site_ids = load_site_ids()

  return render_template(
      "config.html",
      whitelisted=whitelisted,
      total_count=len(whitelisted),
      cameras=load_cameras(),
      parking_capacity=get_parking_capacity(),
      site_ids=site_ids
  )


@app.route("/add_camera", methods=["POST"])
@login_required
def add_camera():
  entry_count = int(request.form.get("entry_count", 0))
  exit_count = int(request.form.get("exit_count", 0))

  entry_ips = [
      ip
      for i in range(entry_count)
      if (ip := request.form.get(f"entry_ip_{i}"))
  ]
  exit_ips = [
      ip for i in range(exit_count) if (ip := request.form.get(f"exit_ip_{i}"))
  ]

  if not entry_ips and not exit_ips:
    flash("No cameras added!", "error")
  else:
    with open(CAMERA_CSV, "a", newline="") as f:
      writer = csv.writer(f)
      for i, ip in enumerate(entry_ips):
        writer.writerow(
            [get_next_address(), f"Entry Cam {i+1}", ip, "", "", "OFF"]
        )
      for i, ip in enumerate(exit_ips):
        writer.writerow(
            [get_next_address(), f"Exit Cam {i+1}", ip, "", "", "OFF"]
        )
    flash("Cameras added successfully!", "success")

  return redirect(url_for("config"))


@app.route("/upload_camera_csv", methods=["POST"])
@login_required
def upload_camera_csv():
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No camera CSV uploaded"
        }), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({
            "status": "error",
            "message": "No selected CSV file"
        }), 400

    try:
        # Read uploaded CSV
        stream = io.StringIO(
            file.stream.read().decode("utf-8-sig"),
            newline=None
        )

        reader = csv.DictReader(stream)

        if not reader.fieldnames:
            return jsonify({
                "status": "error",
                "message": (
                    "CSV must contain only: "
                    "IP Address, Cam Username, Cam Password, Cam_Type"
                )
            }), 400

        # Normalize uploaded headers
        header_map = {
            str(h).strip().lower().replace(" ", "_"): h
            for h in reader.fieldnames
        }

        ip_key = header_map.get("ip_address")
        username_key = header_map.get("cam_username")
        password_key = header_map.get("cam_password")
        type_key = header_map.get("cam_type")

        # Validate required columns
        if not all([
            ip_key,
            username_key,
            password_key,
            type_key
        ]):
            return jsonify({
                "status": "error",
                "message": (
                    "Invalid camera CSV format. Required columns are: "
                    "IP Address, Cam Username, Cam Password, Cam_Type"
                )
            }), 400

        # ---------------------------------------------------------
        # Existing IP addresses
        # ---------------------------------------------------------
        existing_ips = set()

        if os.path.exists(CAMERA_CSV):
            with open(
                CAMERA_CSV,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as f:

                csv_reader = csv.DictReader(f)

                for row in csv_reader:
                    ip = str(
                        row.get("ip", "")
                    ).strip().lower()

                    # Ignore empty and deleted cameras
                    if ip and ip != "n":
                        existing_ips.add(ip)

        # ---------------------------------------------------------
        # Existing camera names
        # ---------------------------------------------------------
        existing_entry_numbers = []
        existing_exit_numbers = []

        if os.path.exists(CAMERA_CSV):
            with open(
                CAMERA_CSV,
                "r",
                newline="",
                encoding="utf-8-sig"
            ) as f:

                csv_reader = csv.DictReader(f)

                for row in csv_reader:
                    location = str(
                        row.get("location", "")
                    ).strip()

                    location_upper = location.upper()

                    # Entry Cam 1
                    if location_upper.startswith("ENTRY CAM"):
                        try:
                            number = int(location.split()[-1])
                            existing_entry_numbers.append(number)
                        except ValueError:
                            pass

                    # Exit Cam 1
                    elif location_upper.startswith("EXIT CAM"):
                        try:
                            number = int(location.split()[-1])
                            existing_exit_numbers.append(number)
                        except ValueError:
                            pass

        # Next Entry / Exit camera numbers
        next_entry_number = (
            max(existing_entry_numbers) + 1
            if existing_entry_numbers
            else 1
        )

        next_exit_number = (
            max(existing_exit_numbers) + 1
            if existing_exit_numbers
            else 1
        )

        new_rows = []
        duplicate_count = 0
        invalid_type_count = 0

        # ---------------------------------------------------------
        # Process uploaded CSV
        # ---------------------------------------------------------
        for row in reader:

            ip = str(
                row.get(ip_key, "")
            ).strip()

            username = str(
                row.get(username_key, "")
            ).strip()

            password = str(
                row.get(password_key, "")
            ).strip()

            camera_type = str(
                row.get(type_key, "")
            ).strip()

            # Skip completely empty rows
            if not ip:
                continue

            # Validate camera type
            if camera_type.lower() not in ["entry", "exit"]:
                invalid_type_count += 1
                continue

            camera_type = (
                "Entry"
                if camera_type.lower() == "entry"
                else "Exit"
            )

            # Check duplicate IP
            if ip.lower() in existing_ips:
                duplicate_count += 1
                continue

            # -----------------------------------------------------
            # Generate camera name automatically
            # -----------------------------------------------------
            if camera_type == "Entry":
                location = f"Entry Cam {next_entry_number}"
                next_entry_number += 1
            else:
                location = f"Exit Cam {next_exit_number}"
                next_exit_number += 1

            # Get next internal address
            address = get_next_address() + len(new_rows)

            # Add to new rows
            new_rows.append([
                address,
                location,
                camera_type,
                ip,
                username,
                password,
                "OFF"
            ])

            # Prevent duplicate IP within the uploaded CSV itself
            existing_ips.add(ip.lower())

        # ---------------------------------------------------------
        # Nothing to add
        # ---------------------------------------------------------
        if not new_rows:
            message = "No new cameras added."

            if duplicate_count:
                message += f" {duplicate_count} duplicate IP(s) skipped."

            if invalid_type_count:
                message += (
                    f" {invalid_type_count} row(s) skipped because "
                    "Cam_Type must be Entry or Exit."
                )

            return jsonify({
                "status": "warning",
                "message": message
            })

        # ---------------------------------------------------------
        # Save to camera_data.csv
        # ---------------------------------------------------------
        file_exists = (
            os.path.exists(CAMERA_CSV)
            and os.path.getsize(CAMERA_CSV) > 0
        )

        with open(
            CAMERA_CSV,
            "a",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            if not file_exists:
                writer.writerow([
                    "address",
                    "location",
                    "camera_type",
                    "ip",
                    "username",
                    "password",
                    "status"
                ])

            writer.writerows(new_rows)

        # ---------------------------------------------------------
        # Success message
        # ---------------------------------------------------------
        message = (
            f"Successfully added {len(new_rows)} camera(s) from CSV! ✅"
        )

        if duplicate_count:
            message += (
                f" {duplicate_count} duplicate IP(s) skipped."
            )

        if invalid_type_count:
            message += (
                f" {invalid_type_count} invalid Cam_Type row(s) skipped."
            )

        return jsonify({
            "status": "success",
            "message": message
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "status": "error",
            "message": f"Failed to process camera CSV: {str(e)}"
        }), 500


@app.route("/anpr")
@login_required
def anpr():
  logs = []
  camera_map = {}
  update_entry_exit_csvs()
  entry_count = sum(
    1 for log in logs
    if "entry" in str(log.get("location", "")).lower()
  )

  exit_count = sum(
    1 for log in logs
    if "exit" in str(log.get("location", "")).lower()
  )

  if os.path.exists(CAMERA_CSV):
    try:
      cam_df = pd.read_csv(CAMERA_CSV)
      cam_df.columns = cam_df.columns.str.strip().str.lower()
      for _, row in cam_df.iterrows():
        loc = str(row.get("location", "")).strip().lower()
        if loc:
          camera_map[loc] = {
              "address": str(row.get("address", "N/A")).strip(),
              "ip": str(row.get("ip", "N/A")).strip(),
              "status": str(row.get("status", "OFF")).strip().upper(),
          }
    except Exception as e:
      print("Error loading camera_data.csv:", e)

  if os.path.exists(ANPR_FILE):
    try:
      anpr_df = pd.read_csv(ANPR_FILE).dropna(subset=["plate_number"])
      anpr_df.columns = (
          anpr_df.columns.astype(str)
          .str.strip()
          .str.replace("\ufeff", "")
          .str.lower()
      )

      for _, row in anpr_df.iterrows():
        loc_raw = str(row.get("location", "")).strip()
        cam_info = camera_map.get(
            loc_raw.lower(),
            {"address": "N/A", "ip": "N/A", "status": "UNKNOWN"},
        )

        timestamp = str(row.get("timestamp", "")).strip()
        if timestamp:
          parts = timestamp.split(maxsplit=1)
          date_val = parts[0] if len(parts) > 0 else ""
          time_val = parts[1] if len(parts) > 1 else ""
        else:
          date_val = str(row.get("date", "")).strip()
          time_val = str(row.get("time", "")).strip()

        logs.append({
            "plate_number": str(row.get("plate_number", "")).strip().upper(),
            "location": loc_raw if loc_raw else "Unknown",
            "camera_status": cam_info["status"],
            "date": date_val,
            "time": time_val,
            "image": str(row.get("image", "")).strip(),
        })
    except Exception as e:
      print("Error parsing anpr_data.csv:", e)

  return render_template("anpr.html", logs=logs, entry_count=entry_count, exit_count=exit_count)


@app.route("/download/<log_type>")
@login_required
def download_anpr_csv(log_type):
  update_entry_exit_csvs()
  filename = "entry_logs.csv" if log_type == "entry" else "exit_logs.csv"
  file_path = os.path.join(CSV_FOLDER, filename)

  if os.path.exists(file_path):
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype="text/csv",
    )
  return "File not found", 404


@app.route("/api/add_detection", methods=["POST"])
def add_detection():
  update_entry_exit_csvs()
  return {"status": "success", "message": "Log added and CSVs updated"}


@app.route("/save_parking_capacity", methods=["POST"])
@login_required
def save_parking_capacity():
  try:
    data = request.get_json() or {}
    capacity = int(data.get("capacity", 0))

    if capacity < 0:
      return jsonify({
          "status": "error",
          "message": "Parking capacity cannot be negative."
      }), 400

    save_parking_capacity_value(capacity)

    return jsonify({
        "status": "success",
        "message": f"Parking capacity saved as {capacity}."
    })
  except (TypeError, ValueError):
    return jsonify({
        "status": "error",
        "message": "Please enter a valid whole-number parking capacity."
    }), 400
  except Exception as e:
    return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/save_all_cameras", methods=["POST"])
@login_required
def save_all_cameras():
  try:
    data = request.get_json()
    cameras_to_save = data.get("cameras", [])

    if not cameras_to_save:
      return jsonify({"status": "error", "message": "No camera data provided."})

    existing_ips = set()
    if os.path.exists(CAMERA_CSV):
      with open(CAMERA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
          ip_val = row.get("ip", "").strip().lower()
          if ip_val:
            existing_ips.add(ip_val)

    for cam in cameras_to_save:
      ip = cam.get("ip", "").strip().lower()
      if ip in existing_ips:
        return jsonify({
            "status": "error",
            "message": (
                f" ⚠️ IP {cam.get('ip')} is already assigned to an existing"
                " camera."
            ),
        })

    current_address = get_next_address()
    file_exists = os.path.exists(CAMERA_CSV)

    with open(CAMERA_CSV, "a", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      if not file_exists or os.stat(CAMERA_CSV).st_size == 0:
        writer.writerow(
            ["address", "location","camera_type", "ip", "username", "password", "status"]
        )

      for cam in cameras_to_save:
        writer.writerow([
            current_address,
            cam.get("location"),
            cam.get("camera_type"),
            cam.get("ip"),
            cam.get("username"),
            cam.get("password"),
            "OFF",
        ])
        current_address += 1

    return jsonify({
        "status": "success",
        "message": f"Successfully saved {len(cameras_to_save)} camera(s)! ✅",
    })

  except Exception as e:
    traceback.print_exc()
    return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/save_camera", methods=["POST"])
@login_required
def save_camera():
  try:
    data = request.get_json()
    location = data.get("location")
    ip = data.get("ip")
    username = data.get("username")
    password = data.get("password")
    status = data.get("status", "OFF")

    if os.path.exists(CAMERA_CSV):
      with open(CAMERA_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
          existing_ip = row.get("ip", "").strip()
          if existing_ip and existing_ip.upper() == ip.upper():
            return jsonify({
                "status": "error",
                "message": (
                    f" ⚠️ IP {ip} is already assigned to another camera."
                ),
            })

    with open(CAMERA_CSV, "a", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(
          [
            get_next_address(),
            location,
            data.get("camera_type", ""),
            ip,
            username,
            password,
            status
          ]
      )

    return jsonify(
        {"status": "success", "message": "Camera saved successfully ✅"}
    )

  except Exception as e:
    traceback.print_exc()
    return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/delete_camera", methods=["POST"])
@login_required
def delete_camera():
    try:
        data = request.get_json()
        cam_name = data.get("name", "").strip().lower()

        if not os.path.exists(CAMERA_CSV):
            return jsonify({"status": "error", "message": "Camera CSV file not found"}), 404

        df = pd.read_csv(CAMERA_CSV)
        
        # Clean column names to ensure case matching works properly
        df.columns = df.columns.str.strip().str.lower()

        # Check if 'location' column exists
        if "location" not in df.columns or "ip" not in df.columns:
            return jsonify({"status": "error", "message": "Invalid CSV structure"}), 400

        # Create mask to target the camera row
        mask = df["location"].fillna("").astype(str).str.strip().str.lower() == cam_name

        if not mask.any():
            return jsonify({"status": "error", "message": "Camera not found"}), 404

        # Replace only the IP column with 'N' for matching rows
        df.loc[mask, "ip"] = "N"

        # Save back to CSV keeping all other columns and rows intact
        df.to_csv(CAMERA_CSV, index=False)

        return jsonify({"status": "success", "message": "Camera IP updated to 'N'"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/toggle_camera", methods=["POST"])
@login_required
def toggle_camera():
  data = request.json
  cam_id = int(data["id"])
  new_status = data["status"]

  df = pd.read_csv(CAMERA_CSV)
  df.columns = df.columns.str.strip().str.lower()

  if "status" not in df.columns:
    df["status"] = "OFF"

  df = df.reset_index(drop=True)

  if cam_id < len(df):
    df.loc[cam_id, "status"] = new_status
    df.to_csv(CAMERA_CSV, index=False)
    return jsonify({"status": "success"})

  return jsonify({"status": "error", "message": "Invalid camera ID"})

@app.route("/ping_camera", methods=["POST"])
@login_required
def ping_camera():
    """Performs a real-time TCP socket check on the camera IP/port."""
    try:
        data = request.get_json()
        raw_ip = data.get("ip", "").strip()

        # Ignore deleted ('N') or empty IPs
        if not raw_ip or raw_ip.upper() == "N":
            return jsonify({"status": "OFF", "message": "Invalid or deleted camera IP"})

        # Clean IP / Hostname / Stream strings
        # Handles formats: "192.168.1.50", "192.168.1.50:8080", "http://192.168.1.50", "rtsp://192.168.1.50:554"
        clean_host = raw_ip.replace("http://", "").replace("https://", "").replace("rtsp://", "").split("/")[0]
        
        # Extract port if provided, otherwise default to HTTP port 80 (or standard RTSP port 554)
        if ":" in clean_host:
            host, port_str = clean_host.split(":")
            port = int(port_str)
        else:
            host = clean_host
            port = 80

        # Test socket connection with a 2-second timeout
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        result = sock.connect_ex((host, port))
        sock.close()

        if result == 0:
            return jsonify({"status": "ON", "message": "Online (Reachable)"})
        else:
            return jsonify({"status": "OFF", "message": "Offline (Unreachable)"})

    except Exception as e:
        return jsonify({"status": "OFF", "message": f"Connection error: {str(e)}"})

@app.route("/add_site_id", methods=["POST"])
@login_required
def add_site_id():
    data = request.get_json()
    site_id = str(
        data.get("site_id", "")
    ).strip()
    if not site_id:
        return jsonify({
            "status": "error",
            "message": "Please enter a Site ID."
        })
    site_ids = load_site_ids()
    # Capacity check
    if len(site_ids) >= SITE_CAPACITY:
        return jsonify({
            "status": "error",
            "message": "⚠️ Site ID capacity reached. Maximum 10 sites allowed."
        })
    # Duplicate check
    if site_id.upper() in [
        x.upper() for x in site_ids
    ]:
        return jsonify({
            "status": "error",
            "message": f"⚠️ Site ID '{site_id}' already exists."
        })
    site_ids.append(site_id)
    pd.DataFrame({
        "site_id": site_ids
    }).to_csv(
        SITE_FILE,
        index=False
    )
    return jsonify({
        "status": "success",
        "message": f"✅ Site ID '{site_id}' added successfully."
    })

@app.route("/upload_site_csv", methods=["POST"])
@login_required
def upload_site_csv():
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No CSV file selected."
        })
    file = request.files["file"]
    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please select a CSV file."
        })
    try:
        uploaded_df = pd.read_csv(file)
        # Accept Site ID / site_id
        uploaded_df.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in uploaded_df.columns
        ]
        if "site_id" not in uploaded_df.columns:
            return jsonify({
                "status": "error",
                "message": "CSV must contain a 'Site ID' column."
            })
        uploaded_sites = (
            uploaded_df["site_id"]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )
        existing_sites = load_site_ids()
        # Remove duplicates
        for site in uploaded_sites:
            if site and site.upper() not in [
                x.upper() for x in existing_sites
            ]:
                if len(existing_sites) >= SITE_CAPACITY:
                    break
                existing_sites.append(site)
        pd.DataFrame({
            "site_id": existing_sites
        }).to_csv(
            SITE_FILE,
            index=False
        )
        return jsonify({
            "status": "success",
            "message":
                f"✅ Site CSV uploaded successfully. "
                f"{len(existing_sites)} / {SITE_CAPACITY} sites currently configured."
        })
    except Exception as e:
        print("Site CSV upload error:", e)
        return jsonify({
            "status": "error",
            "message": "❌ Invalid Site ID CSV format."
        })


@app.route("/get_cameras")
@login_required
def get_cameras():
  cameras = []
  if os.path.exists(CAMERA_CSV):
    with open(CAMERA_CSV, newline="", encoding="utf-8-sig") as file:
      reader = csv.DictReader(file)
      for row in reader:
        cameras.append({
            "address": row.get("address"),
            "name": row.get("location"),
            "ip": row.get("ip"),
            "username": row.get("username"),
            "password": row.get("password"),
        })
  return jsonify(cameras)


@app.route("/get_camera_names")
@login_required
def get_camera_names():
  return jsonify([camera["name"] for camera in load_cameras()])


@app.route("/csv_files")
@login_required
def csv_files():
  files = [
      file for file in os.listdir(CSV_FOLDER) if file.endswith(".csv")
  ] if os.path.exists(CSV_FOLDER) else []
  return render_template("csv_files.html", files=files)


@app.route("/download_csv/<filename>")
@login_required
def download_csv(filename):
  return send_from_directory(CSV_FOLDER, filename, as_attachment=True)


@app.route("/api/stats")
@login_required
def api_stats():
  records_df = read_anpr_data()
  return jsonify(get_anpr_stats(records_df))


@app.route("/api/camera-status")
@login_required
def api_camera_status():
  return jsonify({"status": "ok"})


@app.route("/service")
@login_required
def service():
  cameras = load_cameras()
  for cam in cameras:
    if not cam.get("status"):
      cam["status"] = "OFF"
  return render_template("service.html", cameras=cameras)


@app.route("/logout")
def logout():
  session.clear()
  return redirect(url_for("login"))


if __name__ == "__main__":
  init_anpr_data()
  init_camera_csv()
  init_whitelist()

app.run(host="0.0.0.0", port=5000, debug=True)