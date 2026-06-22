#!/usr/bin/env python3
"""Run live stereo demo from USB camera with side-by-side left/right frames.

Captures from a camera (default /dev/video0) with MJPG encoding at 2560x720,
splits into left/right (1280x720 each), runs the `FoundationStereo` model and
visualizes disparity live. Press `q` to quit, `s` to save current outputs, 
and `m` to generate and save a 3D mesh.
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
    # frame expected shape (H, W, C) where W = 2560 and left/right are 1280 each
    h, w = frame.shape[:2]
    mid = 1280
    imgL = frame[:, 0:mid].copy()
    imgR = frame[:, mid:w].copy()
    return imgL, imgR


def build_model(ckpt_path):
    cfg = OmegaConf.load(os.path.join(os.path.dirname(ckpt_path), 'cfg.yaml'))
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    args = OmegaConf.create(cfg)
    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt['model'])
    model.cuda()
    model.eval()
    return model, args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='/dev/video0', help='camera device')
    parser.add_argument('--width', type=int, default=2560)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--fourcc', default='MJPG')
    parser.add_argument('--ckpt', default=os.path.join(code_dir, 'weights', 'model_best_bp2.pth'))
    parser.add_argument('--intrinsic_file', default=os.path.join(code_dir, 'assets', 'stereo_camera.txt'))
    parser.add_argument('--out_dir', default=os.path.join(code_dir, 'output'))
    parser.add_argument('--every', type=int, default=1, help='run inference every N frames')
    parser.add_argument('--hiera', type=int, default=0)
    parser.add_argument('--valid_iters', type=int, default=32)
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--display_scale', type=float, default=0.5, help='scale factor for display window')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.info(f"Opening camera {args.device} ({args.width}x{args.height}, {args.fourcc})")

    # Use a background thread to continuously read frames and keep only the latest one.
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
    try:
        logging.info(f"Loading model checkpoint {args.ckpt}")
        model, cfg_args = build_model(args.ckpt)
    except Exception as e:
        logging.warning(f"Failed to load model: {e}. Running in preview-only mode.")
        model = None

    import open3d as o3d
    vis3d = o3d.visualization.Visualizer()
    vis3d.create_window(window_name="3D Point Cloud", width=640, height=480)
    
    # Initialize an empty point cloud and add it to the visualizer
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

            # split dynamically in case device returns slightly different width
            imgL_bgr, imgR_bgr = split_stereo_images(frame)
            if imgL_bgr is None or imgR_bgr is None or imgR_bgr.size == 0:
                logging.warning("Received empty left/right image after split, skipping frame")
                time.sleep(0.005)
                continue

            disp_vis = None

            if model is not None and (frame_id % args.every == 0):
                # convert BGR->RGB and to tensor
                try:
                    img0 = cv2.cvtColor(imgL_bgr, cv2.COLOR_BGR2RGB)
                    img1 = cv2.cvtColor(imgR_bgr, cv2.COLOR_BGR2RGB)
                except cv2.error as e:
                    logging.warning(f"cv2.cvtColor failed: {e}; skipping frame")
                    frame_id += 1
                    continue
                if args.scale != 1.0:
                    img0 = cv2.resize(img0, dsize=None, fx=args.scale, fy=args.scale)
                    img1 = cv2.resize(img1, dsize=None, fx=args.scale, fy=args.scale)

                H, W = img0.shape[:2]
                img0_t = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
                img1_t = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
                padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
                img0_t, img1_t = padder.pad(img0_t, img1_t)

                # run inference and measure time
                t0 = time.perf_counter()
                with torch.no_grad():
                    with torch.cuda.amp.autocast(True):
                        if not args.hiera:
                            disp = model.forward(img0_t, img1_t, iters=args.valid_iters, test_mode=True)
                        else:
                            disp = model.run_hierachical(img0_t, img1_t, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
                t1 = time.perf_counter()
                last_inference_ms = (t1 - t0) * 1000.0

                disp = padder.unpad(disp.float())
                disp = disp.data.cpu().numpy().reshape(H, W)
                vis = vis_disparity(disp)
                disp_vis = vis

                with open(args.intrinsic_file, 'r') as f:
                    lines = f.readlines()
                    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
                    baseline = float(lines[1])
                K[:2] *= args.scale
                depth = K[0,0]*baseline/disp
                xyz_map = depth2xyzmap(depth, K) 
                current_pcd = toOpen3dCloud(xyz_map, img0)
                pcd.points = current_pcd.points
                pcd.colors = current_pcd.colors
                
                # Update geometry in renderer
                vis3d.update_geometry(pcd)
                
                # Auto-center the camera view only on the very first frame
                if first_pcd_frame:
                    vis3d.reset_view_point(True)
                    first_pcd_frame = False

            # build display frame: left image and disparity side-by-side scaled by display_scale
            left_display = imgL_bgr.copy()
            ds = float(args.display_scale)
            left_show = cv2.resize(left_display, dsize=None, fx=ds, fy=ds)
            if disp_vis is None:
                display = left_show
            else:
                # ensure disp_vis is 3-channel uint8
                if disp_vis.ndim == 2:
                    disp_vis_col = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
                else:
                    disp_vis_col = disp_vis.copy()

                # resize disparity to match left_show height to avoid mismatched shapes
                th = left_show.shape[0]
                if disp_vis_col.shape[0] == 0:
                    display = left_show
                else:
                    new_w = max(1, int(disp_vis_col.shape[1] * (th / float(disp_vis_col.shape[0]))))
                    disp_show = cv2.resize(disp_vis_col, (new_w, th))
                    display = np.concatenate([left_show, disp_show], axis=1)

            # overlay FPS and inference time
            now = time.perf_counter()
            if last_display_time is None:
                fps = 0
            else:
                dt = now - last_display_time
                fps = int(round(1.0 / dt)) if dt > 0 else 0
            last_display_time = now

            # prepare text strings
            fps_text = f"FPS: {fps}"
            inf_text = f"Inf: {last_inference_ms:.1f} ms"

            # draw text with black outline
            font = cv2.FONT_HERSHEY_SIMPLEX
            org_fps = (10, 25)
            org_inf = (10, 55)
            font_scale = 0.8
            thickness = 2
            
            cv2.putText(display, fps_text, org_fps, font, font_scale, (0, 0, 0), thickness=thickness+2, lineType=cv2.LINE_AA)
            cv2.putText(display, inf_text, org_inf, font, font_scale, (0, 0, 0), thickness=thickness+2, lineType=cv2.LINE_AA)
            cv2.putText(display, fps_text, org_fps, font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)
            cv2.putText(display, inf_text, org_inf, font, font_scale, (255, 255, 255), thickness=thickness, lineType=cv2.LINE_AA)

            cv2.imshow('stereo_live', display)

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

            elif key == ord('m'):
                if pcd.is_empty():
                    logging.warning("Point cloud is empty, cannot generate mesh.")
                else:
                    logging.info("Generating mesh from point cloud...")
                    timestamp = int(time.time())
                    
                    # 1. Estimate normals (required for surface reconstruction methods)
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
                    )
                    # Align normals toward the camera location so surfaces face the right way
                    pcd.orient_normals_towards_camera_location(camera_location=np.array([0, 0, 0]))
                    
                    # 2. Compute dynamic radii parameters based on point distances
                    distances = pcd.compute_nearest_neighbor_distance()
                    avg_dist = np.mean(distances) if len(distances) > 0 else 0.01
                    radii = [avg_dist, avg_dist * 2]
                    
                    # 3. Generate mesh using Ball Pivoting algorithm
                    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii)
                    )
                    
                    # 4. Save to disk
                    mesh_path = os.path.join(args.out_dir, f'mesh_{timestamp}.ply')
                    o3d.io.write_triangle_mesh(mesh_path, mesh)
                    logging.info(f"Successfully saved 3D mesh to {mesh_path}")

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