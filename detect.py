import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size,
    check_requirements,
    check_imshow,
    non_max_suppression,
    apply_classifier,
    scale_coords,
    xyxy2xywh,
    strip_optimizer,
    set_logging,
    increment_path,
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device,
    load_classifier,
    time_synchronized,
    TracedModel,
)

import pybboxes as pbx


class YoloInferenece:
    def __init__(self, weights, imgsz) -> None:
        self.init_model(weights, imgsz)
        print("Gamed ya ged3an")

    def init_model(
        self,
        weights,
        imgsz,
        exist_ok=True,
        device="",
        conf_thres=0.25,
        iou_thres=0.45,
        classes=0,
        agnostic_nms=True,
        save_conf=True,
    ):
        # init
        # self.half = False  # half precision only supported on CUDA
        self.exit_ok = exist_ok
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.save_conf = save_conf
        self.device = select_device(device)

        # load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size

        set_logging()
        self.half = self.device.type != "cpu"  # half precision only supported on CUDA

        if self.half:
            self.model.half()  # to FP16

    def inference(self, source):
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride)

        # Get names and colors
        names = (
            self.model.module.names
            if hasattr(self.model, "module")
            else self.model.names
        )
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if self.device.type != "cpu":
            self.model(
                torch.zeros(1, 3, self.imgsz, self.imgsz)
                .to(self.device)
                .type_as(next(self.model.parameters()))
            )  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()

        to_return = {"predictions": []}
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if self.device.type != "cpu" and (
                old_img_b != img.shape[0]
                or old_img_h != img.shape[2]
                or old_img_w != img.shape[3]
            ):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=False)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(
                pred,
                self.conf_thres,
                self.iou_thres,
                classes=self.classes,
                agnostic=self.agnostic_nms,
            )
            t3 = time_synchronized()

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape
                    ).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (
                            (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                            .view(-1)
                            .tolist()
                        )  # normalized xywh
                        line = (
                            (cls, *xywh, conf) if self.save_conf else (cls, *xywh)
                        )  # label format
                        predd = {
                            "x": xywh[0],
                            "y": xywh[1],
                            "w": xywh[2],
                            "h": xywh[3],
                            "confidence": f"{conf:.2f}",
                        }
                        to_return["predictions"].append(predd)

                return to_return

    def get_plate_xywh(self, img: str):
        try:
            pred = self.inference(img)
        except:
            return None
        if len(pred["predictions"]) == 1:
            xywh_yolo = (
                pred["predictions"][0]["x"],
                pred["predictions"][0]["y"],
                pred["predictions"][0]["w"],
                pred["predictions"][0]["h"],
            )
        elif len(pred["predictions"]) > 1:
            max = 0.0
            highest_conf = {}
            for i in range(len(pred["predictions"])):
                if float(pred["predictions"][i]["confidence"]) > max:
                    max = float(pred["predictions"][i]["confidence"])
                    highest_conf = pred["predictions"][i]
            xywh_yolo = (
                highest_conf["x"],
                highest_conf["y"],
                highest_conf["w"],
                highest_conf["h"],
            )
        else:
            return None

        img_cv = cv2.imread(img)
        H, W = img_cv.shape[:2]
        xywh = pbx.convert_bbox(
            xywh_yolo, from_type="yolo", to_type="voc", image_size=(W, H)
        )
        return xywh
