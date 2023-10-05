# Ultralytics YOLO 🚀, GPL-3.0 license

import hydra
import torch
import easyocr
import cv2
import numpy as np
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import imutils

def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(result[1]) > 6 and result[2] > conf:
            ocr = result[1]

    return str(ocr)

class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

            if len(pred) > 0:
                for box in pred:
                    x, y, w, h, conf, cls = box.tolist()
                    cv2.rectangle(orig_img[i], (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

                    # Get the region of interest (ROI) for further processing
                    roi = orig_img[i][int(y):int(h), int(x):int(w)]

                    # Check if roi is not empty
                    if roi.size == 0:
                        continue  # Skip empty regions

                    # Convert the ROI to grayscale
                    if roi.ndim == 3:  # Check if the image has three channels
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                    else:
                        roi_gray = roi

                    bfilter = cv2.bilateralFilter(roi_gray, 11, 11, 17)
                    edged = cv2.Canny(bfilter, 30, 200)

                    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    contours = imutils.grab_contours(keypoints)
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
                    location = None

                    for contour in contours:
                        approx = cv2.approxPolyDP(contour, 10, True)
                        if len(approx) == 4:
                            location = approx
                            break

                    if location is not None:
                        mask = np.zeros(roi_gray.shape, np.uint8)
                        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
                        new_image = cv2.bitwise_and(roi_gray, roi_gray, mask=mask)

                        result = reader.readtext(new_image)
                        if result:
                            text = result[0][1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            orig_img[i] = cv2.putText(orig_img[i], text=text, org=(int(x + location[0][0][0]), int(y + location[1][0][1] + 60)),
                                                    fontFace=font, fontScale=1, color=(0, 255, 0), thickness=5)
                            print("Detected License Plate:", text)

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor(im0.shape)[[1, 0, 1, 0]]).view(-1).tolist()
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)
                if ocr != "":
                    label = ocr
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
