#!/usr/bin/env python3
import sys, time, struct, socket
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