#!/usr/bin/env python3
import sys, os, time, struct, socket, subprocess

# Bootstrap a virtual environment and install requirements, then re-exec
script_dir = os.path.dirname(os.path.abspath(__file__))
venv_dir = os.path.join(script_dir, '.venv')

def _in_venv():
    return os.path.realpath(sys.executable).startswith(os.path.realpath(venv_dir))

if not _in_venv():
    if not os.path.isdir(venv_dir):
        print('Server: creating virtualenv at', venv_dir)
        subprocess.check_call([sys.executable, '-m', 'venv', venv_dir])
    venv_python = os.path.join(venv_dir, 'Scripts' if os.name == 'nt' else 'bin', 'python')
    requirements = os.path.join(script_dir, 'requirements.txt')
    if os.path.isfile(requirements):
        print('Server: installing requirements from', requirements)
        subprocess.check_call([venv_python, '-m', 'pip', 'install', '-r', requirements])
    else:
        print('Server: requirements.txt not found at', requirements)
    os.execv(venv_python, [venv_python] + sys.argv)

import cv2

video_path = sys.argv[1] if len(sys.argv) > 1 else "./Videos/traffictrim.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Server: cannot open", video_path); sys.exit(1)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", 9999))
srv.listen(1)
print("Server: listening on 127.0.0.1:9999")
conn, addr = srv.accept()
print("Server: client connected", addr)

try:
    fps = max(1, int(cap.get(cv2.CAP_PROP_FPS) or 25))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            break
        data = buf.tobytes()
        conn.sendall(struct.pack('!I', len(data)))
        conn.sendall(data)
        time.sleep(1.0 / fps)
finally:
    conn.close()
    srv.close()
    cap.release()