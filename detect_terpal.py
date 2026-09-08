#!/usr/bin/env python3
"""
Deteksi objek real-time untuk Orange Pi 4 Pro (8-core, tanpa NPU aktif).
Model: hasil export NCNN dari /root/fiinal2 (2).pt (kelas: Terpal-Orange).

Desain untuk efisiensi di hardware ini:
- Backend NCNN (NEON-optimized), bukan PyTorch murni -> jauh lebih ringan di CPU ARM.
- Model NCNN sudah di-export sekali di awal (bukan convert .pt tiap start) -> start cepat, tanpa overhead torch tracing.
- Capture kamera jalan di thread terpisah yang selalu menyimpan HANYA frame terbaru
  (bukan antrian) -> saat inferensi jadi bottleneck, program tidak memproses
  backlog frame basi, cocok untuk target bergerak (drone).
- Loop utama tidak pernah menunggu I/O kamera; ia hanya mengambil frame terbaru yang tersedia.
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import yaml

from ultralytics import YOLO

DEFAULT_MODEL = "/root/yolo/terpal_orange_ncnn_model_640"


def timestamped_path(path_str: str) -> Path:
    """Sisipkan tanggal+waktu MULAI REKAM ke nama file (mis. hasil.mp4 ->
    hasil_20260904_162304.mp4), supaya tiap sesi rekam dapat nama unik dan
    tidak menimpa rekaman sebelumnya -- penting untuk yolo.service yang
    Restart=always dan bisa mulai ulang beberapa kali per hari."""
    p = Path(path_str)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{p.stem}_{stamp}{p.suffix or '.mp4'}")


def exported_imgsz(model_dir: Path):
    """Baca imgsz yang dibakukan (fixed) saat export NCNN, dari metadata.yaml.

    Model NCNN hasil export ultralytics itu FIXED-SHAPE: PNNX men-trace graph
    memakai imgsz tertentu, dan layer reshape/view di dalamnya hardcode ke
    ukuran itu. Kalau inferensi dijalankan di imgsz lain, reshape jadi salah
    dan outputnya jadi sampah (bukan error, jadi mudah tidak ketahuan) --
    misal keluar ratusan box palsu dengan conf=1.00 semua.
    """
    meta = model_dir / "metadata.yaml"
    if not meta.exists():
        return None
    data = yaml.safe_load(meta.read_text())
    sz = data.get("imgsz")
    if isinstance(sz, list) and sz:
        return int(sz[0])
    return None


class LatestFrameGrabber:
    """Capture di background thread, selalu simpan cuma frame paling baru."""

    def __init__(self, source, width=1280, height=720, fourcc=None):
        self.source = source
        cap_source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(cap_source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(cap_source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Tidak bisa membuka kamera/source '{source}'. "
                "Kalau ini /dev/video0 dan sedang dipakai proses lain "
                "(mis. krti-sender.service), matikan dulu:\n"
                "  systemctl stop krti-watchdog.service krti-sender.service"
            )
        # /dev/video0 (real camera) only streams reliably as MJPEG -- forcing
        # it avoids the driver defaulting to raw YUYV, which collapses FPS
        # above 480p on this USB2.0 camera (see profiles.conf). The
        # video10/video11 v4l2loopback relay devices carry RAW YUY2 instead
        # (see camera-relay.sh), so forcing MJPG there breaks negotiation --
        # only force a fourcc for the real device, auto-negotiate otherwise.
        is_real_camera = str(source) in ("0", "/dev/video0")
        if fourcc is None:
            fourcc = "MJPG" if is_real_camera else None
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._ok = False
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._ok = True

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2)
        self.cap.release()


def main():
    ap = argparse.ArgumentParser(description="YOLO NCNN inference efisien untuk Orange Pi 4 Pro")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="path folder model NCNN (default: %(default)s)")
    ap.add_argument("--source", default="0", help="'0' untuk /dev/video0, atau path gambar/video/RTSP")
    ap.add_argument("--imgsz", type=int, default=None, help="resolusi inferensi; HARUS sama dengan imgsz saat export NCNN (default: baca otomatis dari metadata.yaml)")
    ap.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    ap.add_argument("--width", type=int, default=1280, help="lebar capture kamera")
    ap.add_argument("--height", type=int, default=720, help="tinggi capture kamera")
    ap.add_argument("--fourcc", default=None, help="paksa fourcc capture (mis. MJPG); default: auto (MJPG utk /dev/video0, negosiasi otomatis utk device lain)")
    ap.add_argument("--save-dir", default=None, help="jika diisi, simpan frame ber-anotasi tiap deteksi ke folder ini")
    ap.add_argument("--max-frames", type=int, default=0, help="stop otomatis setelah N frame (0 = jalan terus)")
    ap.add_argument("--fps-log-every", type=int, default=30, help="cetak FPS rata-rata tiap N frame")
    ap.add_argument("--record", default=None, help="path video output (mis. run.mp4); merekam SEMUA frame ber-anotasi dari awal sampai program berhenti. Tanggal+jam mulai rekam otomatis disisipkan ke nama file (run.mp4 -> run_20260904_162304.mp4) supaya tiap sesi tidak saling timpa")
    ap.add_argument("--record-fps", type=float, default=None, help="fps video hasil rekam (default: dihitung otomatis dari FPS rata-rata aktual di akhir run)")
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model tidak ditemukan: {model_path}")

    fixed_sz = exported_imgsz(model_path)
    if args.imgsz is None:
        if fixed_sz is None:
            sys.exit("Tidak menemukan imgsz di metadata.yaml, wajib isi --imgsz manual.")
        args.imgsz = fixed_sz
        print(f"[init] imgsz otomatis dari metadata.yaml: {args.imgsz}")
    elif fixed_sz is not None and args.imgsz != fixed_sz:
        sys.exit(
            f"--imgsz {args.imgsz} tidak cocok dengan shape hasil export ({fixed_sz}).\n"
            f"Model NCNN ini fixed-shape ke {fixed_sz}x{fixed_sz} -- jalankan dengan "
            f"--imgsz {fixed_sz}, atau export ulang model di imgsz {args.imgsz} dulu:\n"
            f'  yolo export model="/root/fiinal2 (2).pt" format=ncnn imgsz={args.imgsz}'
        )

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    is_image = str(args.source).lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    print(f"[init] loading model: {model_path}")
    model = YOLO(str(model_path), task="detect")
    print(f"[init] classes: {model.names}")

    if is_image:
        frame = cv2.imread(args.source)
        if frame is None:
            sys.exit(f"Gagal baca gambar: {args.source}")
        r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        print(f"[result] {len(r.boxes)} deteksi")
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            print(f"  {model.names[int(b.cls)]}  conf={float(b.conf):.3f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
        if save_dir:
            out = save_dir / "result.jpg"
            cv2.imwrite(str(out), r.plot())
            print(f"[save] {out}")
        return

    grabber = LatestFrameGrabber(args.source, args.width, args.height, fourcc=args.fourcc)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # warmup: buang beberapa frame pertama (auto-exposure belum stabil) + warmup model
    # sekalian dipakai buat estimasi FPS aktual (dasar default --record-fps)
    time.sleep(0.5)
    ok, frame = grabber.read()
    warmup_fps_estimate = 8.0
    if ok:
        # panggilan pertama memicu lazy-load backend NCNN (mahal, sekali saja) --
        # jangan ikut dihitung, atau estimasi FPS jadi jauh meleset (bisa ~1 fps)
        model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        t0 = time.time()
        for _ in range(5):
            model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        dt = time.time() - t0
        if dt > 0:
            warmup_fps_estimate = 5 / dt

    video_writer = None
    record_path = None
    if args.record:
        record_path = timestamped_path(args.record)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_fps = args.record_fps or warmup_fps_estimate
        fourcc = cv2.VideoWriter_fourcc(*("XVID" if record_path.suffix.lower() == ".avi" else "mp4v"))
        h, w = frame.shape[:2] if ok else (args.height, args.width)
        video_writer = cv2.VideoWriter(str(record_path), fourcc, record_fps, (w, h))
        if not video_writer.isOpened():
            sys.exit(f"Gagal membuka video writer untuk: {record_path}")
        print(f"[record] merekam ke {record_path} @ {record_fps:.2f} fps (estimasi dari warmup)")

    print("[run] mulai deteksi, Ctrl+C untuk berhenti")
    n = 0
    t_window_start = time.time()
    t_run_start = time.time()
    last_frame_id = None

    try:
        while not stop.is_set():
            ok, frame = grabber.read()
            if not ok:
                time.sleep(0.005)
                continue

            # skip kalau frame belum berubah (grabber lebih lambat dari loop ini)
            frame_id = id(frame)
            if frame_id == last_frame_id:
                time.sleep(0.001)
                continue
            last_frame_id = frame_id

            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            n += 1

            annotated = None
            if len(r.boxes) > 0:
                dets = ", ".join(
                    f"{model.names[int(b.cls)]}:{float(b.conf):.2f}" for b in r.boxes
                )
                print(f"[frame {n}] {len(r.boxes)} deteksi -> {dets}")
                if save_dir:
                    annotated = r.plot()
                    out = save_dir / f"det_{n:06d}.jpg"
                    cv2.imwrite(str(out), annotated)

            if video_writer is not None:
                if annotated is None:
                    annotated = r.plot()
                video_writer.write(annotated)

            if args.fps_log_every and n % args.fps_log_every == 0:
                dt = time.time() - t_window_start
                fps = args.fps_log_every / dt if dt > 0 else 0.0
                print(f"[fps] {fps:.2f} FPS (rata-rata {args.fps_log_every} frame terakhir)")
                t_window_start = time.time()

            if args.max_frames and n >= args.max_frames:
                break
    finally:
        grabber.release()
        if video_writer is not None:
            video_writer.release()
            print(f"[record] selesai, tersimpan di {record_path}")
        total_dt = time.time() - t_run_start
        avg_fps = n / total_dt if total_dt > 0 else 0.0
        print(f"[done] total {n} frame diproses dalam {total_dt:.1f}s (rata-rata {avg_fps:.2f} FPS)")


if __name__ == "__main__":
    main()
