# app/server.py

import os
import subprocess
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# Load passkey from .env
PASSKEY = os.getenv("PASSKEY", "secret123")

app = Flask(__name__)


@app.route("/run", methods=["POST"])
def run_pipeline():
    """
    Expects a JSON body with { "passkey": "..." }
    or a form/URL param. If correct, runs main.py.
    """
    data = request.get_json(force=True) if request.is_json else request.form
    provided_key = data.get("passkey")

    if provided_key != PASSKEY:
        return jsonify({"error": "Invalid passkey"}), 401

    # Here, we call main.py. You can also run your pipeline in-process if you want.
    # This example uses a subprocess call for demonstration.
    try:
        result = subprocess.run(
            ["python", "main.py"],
            cwd=os.path.dirname(__file__),
            capture_output=True,
            text=True,
            check=False,
        )
        output = result.stdout + result.stderr
        return jsonify({"status": "Pipeline executed", "output": output}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return "Welcome to the ADB OCR App. POST to /run with { passkey: 'your_password' } to execute."


if __name__ == "__main__":
    # Run on all interfaces (0.0.0.0) so phones on the same network can reach it
    app.run(host="0.0.0.0", port=8000, debug=True)
