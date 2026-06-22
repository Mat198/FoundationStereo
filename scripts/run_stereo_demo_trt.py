#!/usr/bin/env python3
"""Run live stereo demo using FoundationStereo with PyTorch, ONNX or TensorRT.

Captures from a stereo camera, splits into left/right frames, runs inference,
visualizes disparity live and optionally updates a 3D point cloud.

Usage examples:
  # PyTorch checkpoint
  python scripts/run_stereo_demo_trt.py --pretrained weights/model_best_bp2.pth

  # ONNX model
  python scripts/run_stereo_demo_trt.py --pretrained pretrained_models/foundation_stereo.onnx --height 448 --width 672

  # TensorRT engine
  python scripts/run_stereo_demo_trt.py --pretrained pretrained_models/foundation_stereo.plan --height 448 --width 672
"""
import os
import sys
import argparse
import logging
import time

import imageio
import cv2
import numpy as np
import torch
import threading

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo


def split_stereo_images(frame):
    h, w = frame.shape[:2]
    mid = w // 2
    imgL = frame[:, 0:mid].copy()
    imgR = frame[:, mid:w].copy()
    return imgL, imgR


def build_pytorch_model(ckpt_path):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), 'cfg.yaml'))
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    return model


def get_onnx_model(pretrained_path):
    import onnxruntime as ort
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(pretrained_path, sess_options=session_options, providers=['CUDAExecutionProvider'])
    return model


def get_engine_model(pretrained_path):
    import tensorrt as trt
    from onnx_tensorrt import tensorrt_engine
    with open(pretrained_path, 'rb') as file:
        engine_data = file.read()
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_data)
    engine = tensorrt_engine.Engine(engine)
    return engine


def load_model(args):
    pretrained = args.pretrained
    ext = os.path.splitext(pretrained)[1].lower()

    if ext in ['.pth', '.pt']:
        model = build_pytorch_model(pretrained)
        return model, 'pytorch'
    elif ext == '.onnx':
        model = get_onnx_model(pretrained)
        return model, 'onnx'
    elif ext in ['.plan', '.engine']:
        model = get_engine_model(pretrained)
        return model, 'trt'
    else:
        raise ValueError(f'Unsupported pretrained format: {pretrained}')


def prepare_inputs(imgL_bgr, imgR_bgr, args, model_type):
    imgL_rgb = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2RGB)
    imgR_rgb = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2RGB)

    if model_type == 'pytorch':
        if args.scale != 1.0:
            imgL_rgb = cv2.resize(imgL_rgb, dsize=None, fx=args.scale, fy=args.scale)
            imgR_rgb = cv2.resize(imgR_rgb, dsize=None, fx=args.scale, fy=args.scale)

        imgL_t = torch.as_tensor(imgL_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        imgR_t = torch.as_tensor(imgR_rgb).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(imgL_t.shape, divis_by=32, force_square=False)
        imgL_t, imgR_t = padder.pad(imgL_t, imgR_t)
        return imgL_rgb, imgR_rgb, imgL_t, imgR_t, padder

    resized_left = cv2.resize(imgL_rgb, (args.trt_width, args.trt_height))
    resized_right = cv2.resize(imgR_rgb, (args.trt_width, args.trt_height))
    left_tensor = torch.as_tensor(resized_left.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    right_tensor = torch.as_tensor(resized_right.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
    return resized_left, resized_right, left_tensor, right_tensor, None


def run_inference(model, model_type, left_tensor, right_tensor, args):
    if model_type == 'pytorch':
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.cuda.amp.autocast(True):
                if not args.hiera:
                    disp = model.forward(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True)
                else:
                    disp = model.run_hierachical(left_tensor, right_tensor, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        disp = disp.cpu().float()
        return disp, (t1 - t0) * 1000.0

    left_np = left_tensor.cpu().numpy()
    right_np = right_tensor.cpu().numpy()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if model_type == 'onnx':
        disp = model.run(None, {'left': left_np, 'right': right_np})[0]
    else:
        disp = model.run([left_np, right_np])[0]
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return torch.from_numpy(disp).float(), (t1 - t0) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description='Live FoundationStereo demo with ONNX/TensorRT support')
    parser.add_argument('--device', default='/dev/video0', help='camera device')
    parser.add_argument('--width', type=int, default=2560, help='capture width')
    parser.add_argument('--height', type=int, default=720, help='capture height')
    parser.add_argument('--fourcc', default='MJPG')
    parser.add_argument('--pretrained', default=os.path.join(code_dir, 'weights', 'model_best_bp2.pth'), help='Path to .pth, .onnx, .plan or .engine model')
    parser.add_argument('--intrinsic_file', default=os.path.join(code_dir, 'assets', 'stereo_camera.txt'))
    parser.add_argument('--out_dir', default=os.path.join(code_dir, 'output'))
    parser.add_argument('--every', type=int, default=1, help='run inference every N frames')
    parser.add_argument('--hiera', type=int, default=0, help='use hierarchical inference for PyTorch model')
    parser.add_argument('--valid_iters', type=int, default=32, help='iterations for PyTorch model')
    parser.add_argument('--scale', type=float, default=1.0, help='scale factor for PyTorch model input')
    parser.add_argument('--trt_height', type=int, default=448, help='model height for ONNX/TRT inference')
    parser.add_argument('--trt_width', type=int, default=672, help='model width for ONNX/TRT inference')
    parser.add_argument('--display_scale', type=float, default=0.5, help='scale factor for display window')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.height = args.height
    args.width = args.width

    set_logging_format()
    set_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info(f"Opening camera {args.device} ({args.width}x{args.height}, {args.fourcc})")

    class CameraReader(threading.Thread):
        def __init__(self, device, width, height, fourcc):
            super().__init__(daemon=True)
            self.device = device
            self.width = width
            self.height = height
            self.fourcc = fourcc
            self.cap = None
            self.lock = threading.Lock()
            self.frame = None
            self.running = False

        def open(self):
            try:
                self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            except Exception:
                self.cap = cv2.VideoCapture(self.device)
            if not self.cap or not self.cap.isOpened():
                return False
            fourcc_code = cv2.VideoWriter_fourcc(*self.fourcc)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_code)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            return True

        def run(self):
            if self.cap is None:
                ok = self.open()
                if not ok:
                    logging.error(f"Cannot open camera {self.device}")
                    return
            self.running = True
            while self.running:
                try:
                    ret, frame = self.cap.read()
                except Exception:
                    ret = False
                    frame = None
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self.frame = frame
                else:
                    time.sleep(0.005)

        def read(self):
            with self.lock:
                if self.frame is None:
                    return None
                return self.frame.copy()

        def close(self):
            self.running = False
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass

    cam_reader = CameraReader(args.device, args.width, args.height, args.fourcc)
    cam_reader.start()

    model = None
    model_type = None
    try:
        logging.info(f"Loading pretrained model {args.pretrained}")
        model, model_type = load_model(args)
        logging.info(f"Using model type: {model_type}")
    except Exception as e:
        logging.warning(f"Failed to load model {args.pretrained}: {e}. Running in preview-only mode.")
        model = None

    import open3d as o3d
    vis3d = o3d.visualization.Visualizer()
    vis3d.create_window(window_name="3D Point Cloud", width=640, height=480)

    pcd = o3d.geometry.PointCloud()
    vis3d.add_geometry(pcd)
    first_pcd_frame = True

    frame_id = 0
    last_display_time = None
    last_inference_ms = 0.0

    try:
        while True:
            frame = cam_reader.read()
            if frame is None:
                time.sleep(0.005)
                continue

            imgL_bgr, imgR_bgr = split_stereo_images(frame)
            if imgL_bgr is None or imgR_bgr is None or imgR_bgr.size == 0:
                logging.warning("Received empty left/right image after split, skipping frame")
                time.sleep(0.005)
                continue

            disp_vis = None
            if model is not None and (frame_id % args.every == 0):
                try:
                    imgL_rgb, imgR_rgb, left_tensor, right_tensor, padder = prepare_inputs(imgL_bgr, imgR_bgr, args, model_type)
                    disp, last_inference_ms = run_inference(model, model_type, left_tensor, right_tensor, args)

                    if disp is not None:
                        if padder is not None:
                            disp = padder.unpad(disp)
                        disp = disp.squeeze().cpu().numpy()
                        vis = vis_disparity(disp)
                        disp_vis = vis

                        if os.path.isfile(args.intrinsic_file):
                            with open(args.intrinsic_file, 'r') as f:
                                lines = f.readlines()
                                K = np.array(list(map(float, lines[0].rstrip().split())), dtype=np.float32).reshape(3, 3)
                                baseline = float(lines[1])
                            source_h, source_w = imgL_bgr.shape[:2]
                            target_h, target_w = imgL_rgb.shape[:2]
                            K[0, :] *= target_w / source_w
                            K[1, :] *= target_h / source_h
                            xyz_map = depth2xyzmap(disp, K)
                            current_pcd = toOpen3dCloud(xyz_map, imgL_rgb)
                            R = current_pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
                            current_pcd.rotate(R, center=(0, 0, 0))
                            pcd.points = current_pcd.points
                            pcd.colors = current_pcd.colors
                            vis3d.update_geometry(pcd)
                            if first_pcd_frame:
                                vis3d.reset_view_point(True)
                                first_pcd_frame = False
                except cv2.error as e:
                    logging.warning(f"cv2.cvtColor failed: {e}; skipping frame")
                except Exception as e:
                    logging.warning(f"Inference failed: {e}; skipping frame")

            left_display = imgL_bgr.copy()
            ds = float(args.display_scale)
            left_show = cv2.resize(left_display, dsize=None, fx=ds, fy=ds)
            if disp_vis is None:
                display = left_show
            else:
                if disp_vis.ndim == 2:
                    disp_vis_col = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
                else:
                    disp_vis_col = disp_vis.copy()
                th = left_show.shape[0]
                if disp_vis_col.shape[0] == 0:
                    display = left_show
                else:
                    new_w = max(1, int(disp_vis_col.shape[1] * (th / float(disp_vis_col.shape[0]))))
                    disp_show = cv2.resize(disp_vis_col, (new_w, th))
                    display = np.concatenate([left_show, disp_show], axis=1)

            now = time.perf_counter()
            if last_display_time is None:
                fps = 0
            else:
                dt = now - last_display_time
                fps = int(round(1.0 / dt)) if dt > 0 else 0
            last_display_time = now

            fps_text = f"FPS: {fps}"
            inf_text = f"Inf: {last_inference_ms:.1f} ms"
            font = cv2.FONT_HERSHEY_SIMPLEX
            org_fps = (10, 25)
            org_inf = (10, 55)
            font_scale = 0.8
            thickness = 2

            cv2.putText(display, fps_text, org_fps, font, font_scale, (0, 0, 0), thickness=thickness+2, lineType=cv2.LINE_AA)
            cv2.putText(display, inf_text, org_inf, font, font_scale, (0, 0, 0), thickness=thickness+2, lineType=cv2.LINE_AA)
            cv2.putText(display, fps_text, org_fps, font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(display, inf_text, org_inf, font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)

            cv2.imshow('stereo_live_trt', display)
            vis3d.poll_events()
            vis3d.update_renderer()

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = int(time.time())
                cv2.imwrite(os.path.join(args.out_dir, f'left_{timestamp}.png'), imgL_bgr)
                cv2.imwrite(os.path.join(args.out_dir, f'right_{timestamp}.png'), imgR_bgr)
                if disp_vis is not None:
                    imageio.imwrite(os.path.join(args.out_dir, f'disp_vis_{timestamp}.png'), cv2.cvtColor(disp_vis, cv2.COLOR_BGR2RGB))
                    logging.info(f"Saved outputs to {args.out_dir}")

            frame_id += 1

    finally:
        try:
            cam_reader.close()
            cam_reader.join(timeout=1.0)
        except Exception:
            pass
        cv2.destroyAllWindows()
        try:
            vis3d.destroy_window()
        except Exception:
            pass


if __name__ == '__main__':
    main()
