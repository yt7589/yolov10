#
import argparse
import os
from ultralytics import YOLOv10
import onnx
import onnxscript
import onnxruntime
import cv2
import torch
from ultralytics.data.augment import LetterBox

def main(args:argparse.Namespace = {}) -> None:
    print(f'Yolov10车路协同应用 v0.0.3')
    train()
    # export_onnx()
    # run_onnx()
    # prepare_ds()

def prepare_ds():
    cls_ids = set()
    for root, dirs, files in os.walk('D:/awork/zjkj/datasets/GC10-DET-yolo/labels/train'):
        for fn in files:
            print(f'### {root}/{fn}...')
            with open(f'{root}/{fn}', 'r', encoding='utf-8') as fd:
                for row in fd:
                    row = row.strip()
                    arrs = row.split(' ')
                    cls_id = int(arrs[0])
                    cls_ids.add(cls_id)
    for cls_id in cls_ids:
        print(f'@@@ {cls_id};')

def run_onnx() -> None:
    onnx_fn = 'runs/detect/train2/weights/best.onnx'
    ort_session = onnxruntime.InferenceSession(onnx_fn, providers=["CPUExecutionProvider"])
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    # 产生数据
    img = cv2.imread('ultralytics/assets/bus.jpg')
    cvrt = LetterBox()
    img = cvrt(image=img)
    print(f'@@@ img: {img.shape};')
    img = img.transpose(2,0,1)/255.0
    # x = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    x = torch.from_numpy(img).unsqueeze(0).float()
    print(f'img:{img.shape}; {img[0][0][0]}; x: {x.shape};')
    # x = torch.randn(batch_size, 128, requires_grad=True)
    # x = x.to('cpu')
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    y_hat = ort_session.run(None, ort_inputs)
    print(f'y_hat: {len(y_hat)}; {y_hat[0].shape}; \n{y_hat};????')

    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

def export_onnx() -> None:
    pt_fn = 'runs/detect/train2/weights/best.pt'
    model = YOLOv10(pt_fn)
    model.export(format='onnx')

def train() -> None:
    model = YOLOv10('work/ckpts/yolov10s.pt')
    model.train(data='ultralytics/cfg/datasets/gc10.yaml', epochs=3, batch=4, imgsz=640)

def predict() -> None:
    model = YOLOv10('work/ckpts/yolov10s.pt')
    results = model.predict(source='ultralytics/assets/bus.jpg')
    results[0].plot(save=True, filename='a001.png')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--run_mode', action='store',
        type=int, default=1, dest='run_mode',
        help='run mode'
    )
    return parser.parse_args()

if '__main__' == __name__:
    args = parse_args()
    main(args=args)