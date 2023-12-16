import gc
import multiprocessing as mp
import os
import shutil
import sys
import time
from os import path

import cv2
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from PIL import Image

import ape
import detectron2.data.transforms as T
import gradio as gr
from ape.model_zoo import get_config_file
from demo_lazy import get_parser, setup_cfg
from detectron2.config import CfgNode
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo

ckpt_repo_id = "shenyunhang/APE"



def load_APE_A():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj_cp_720k_20230504_002019/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj_cp_720k_20230504_002019/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VG/ape_deta/ape_deta_vitl_eva02_lsj1024_cp_720k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["fusedLN"] = False

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_A"] = demo
    all_cfg["APE_A"] = cfg


def load_APE_B():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_225418/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_225418/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VGR_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["fusedLN"] = False

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_B"] = demo
    all_cfg["APE_B"] = cfg


def load_APE_C():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_210950/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj_cp_1080k_20230702_210950/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO/ape_deta/ape_deta_vitl_eva02_vlf_lsj1024_cp_1080k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1
    ape.modeling.text.eva01_clip.eva_clip._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["fusedLN"] = False

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_C"] = demo
    all_cfg["APE_C"] = cfg


def load_APE_D():
    # init_checkpoint= "output2/APE/configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth"
    init_checkpoint = "configs/LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k_mdl_20230829_162438/model_final.pth"
    init_checkpoint = hf_hub_download(repo_id=ckpt_repo_id, filename=init_checkpoint)

    args = get_parser().parse_args()
    args.config_file = get_config_file(
        "LVISCOCOCOCOSTUFF_O365_OID_VGR_SA1B_REFCOCO_GQA_PhraseCut_Flickr30k/ape_deta/ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k.py"
    )
    args.confidence_threshold = 0.01
    args.opts = [
        "train.init_checkpoint='{}'".format(init_checkpoint),
        "model.model_language.cache_dir=''",
        "model.model_vision.select_box_nums_for_evaluation=500",
        "model.model_vision.text_feature_bank_reset=True",
        "model.model_vision.backbone.net.xattn=False",
        "model.model_vision.transformer.encoder.pytorch_attn=True",
        "model.model_vision.transformer.decoder.pytorch_attn=True",
    ]
    if running_device == "cpu":
        args.opts += [
            "model.model_language.dtype='float32'",
        ]
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva02_clip.factory._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1

    demo = VisualizationDemo(cfg, args=args)
    if save_memory:
        demo.predictor.model.to("cpu")
        # demo.predictor.model.half()
    else:
        demo.predictor.model.to(running_device)

    all_demo["APE_D"] = demo
    all_cfg["APE_D"] = cfg


def setup_model(name):
    gc.collect()
    torch.cuda.empty_cache()

    if save_memory:
        pass
    else:
        return

    for key, demo in all_demo.items():
        if key == name:
            demo.predictor.model.to(running_device)
        else:
            demo.predictor.model.to("cpu")

    gc.collect()
    torch.cuda.empty_cache()


def run_on_image(
    input_image_path,
    input_text,
    output_type,
    demo,
    cfg,
):
    with_box = False
    with_mask = False
    with_sseg = False
    if "object detection" in output_type:
        with_box = True
    if "instance segmentation" in output_type:
        with_mask = True
    if "semantic segmentation" in output_type:
        with_sseg = True

    if isinstance(input_image_path, dict):
        input_mask_path = input_image_path["mask"]
        input_image_path = input_image_path["image"]
        print("input_image_path", input_image_path)
        print("input_mask_path", input_mask_path)
    else:
        input_mask_path = None

    print("input_text", input_text)

    if isinstance(cfg, CfgNode):
        input_format = cfg.INPUT.FORMAT
    else:
        if "model_vision" in cfg.model:
            input_format = cfg.model.model_vision.input_format
        else:
            input_format = cfg.model.input_format

    input_image = read_image(input_image_path, format="BGR")
    # img = cv2.imread(input_image_path)
    # cv2.imwrite("tmp.jpg", img)
    # # input_image = read_image("tmp.jpg", format=input_format)
    # input_image = read_image("tmp.jpg", format="BGR")

    if input_mask_path is not None:
        input_mask = read_image(input_mask_path, "L").squeeze(2)
        print("input_mask", input_mask)
        print("input_mask", input_mask.shape)
    else:
        input_mask = None

    if not with_box and not with_mask and not with_sseg:
        return input_image[:, :, ::-1]

    if input_image.shape[0] > 1024 or input_image.shape[1] > 1024:
        transform = aug.get_transform(input_image)
        input_image = transform.apply_image(input_image)
    else:
        transform = None

    start_time = time.time()
    predictions, visualized_output, _, metadata = demo.run_on_image(
        input_image,
        text_prompt=input_text,
        mask_prompt=input_mask,
        with_box=with_box,
        with_mask=with_mask,
        with_sseg=with_sseg,
    )

    logger.info(
        "{} in {:.2f}s".format(
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    output_image = visualized_output.get_image()
    print("output_image", output_image.shape)
    # if input_format == "RGB":
    #     output_image = output_image[:, :, ::-1]
    if transform:
        output_image = transform.inverse().apply_image(output_image)
    print("output_image", output_image.shape)

    output_image = Image.fromarray(output_image)

    gc.collect()
    torch.cuda.empty_cache()

    json_results = instances_to_coco_json(predictions["instances"].to(demo.cpu_device), 0)
    for json_result in json_results:
        json_result["category_name"] = metadata.thing_classes[json_result["category_id"]]
        del json_result["image_id"]

    return output_image, json_results


def run_on_image_A(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image")

    setup_model("APE_A")
    demo = all_demo["APE_A"]
    cfg = all_cfg["APE_A"]
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    return run_on_image(
        input_image_path,
        input_text,
        output_type,
        demo,
        cfg,
    )


def run_on_image_C(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image_C")

    setup_model("APE_C")
    demo = all_demo["APE_C"]
    cfg = all_cfg["APE_C"]
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    return run_on_image(
        input_image_path,
        input_text,
        output_type,
        demo,
        cfg,
    )


def run_on_image_D(input_image_path, input_text, score_threshold, output_type):
    logger.info("run_on_image_D")

    setup_model("APE_D")
    demo = all_demo["APE_D"]
    cfg = all_cfg["APE_D"]
    demo.predictor.model.model_vision.test_score_thresh = score_threshold

    return run_on_image(
        input_image_path,
        input_text,
        output_type,
        demo,
        cfg,
    )



if __name__ == '__main__':
    available_memory = [
        torch.cuda.mem_get_info(i)[0] / 1024 ** 3 for i in range(torch.cuda.device_count())
    ]

    global running_device
    max_available_memory = max(available_memory)
    device_id = available_memory.index(max_available_memory)

    running_device = "cuda:" + str(device_id)

    global save_memory
    save_memory = False
    # if max_available_memory > 0 and max_available_memory < 40:
    #     save_memory = True

    print("available_memory", available_memory)
    print("max_available_memory", max_available_memory)
    print("running_device", running_device)
    print("save_memory", save_memory)

    # ==========================================================================================

    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    setup_logger(name="ape")
    global logger
    logger = setup_logger()

    global aug
    aug = T.ResizeShortestEdge([1024, 1024], 1024)

    global all_demo
    all_demo = {}
    all_cfg = {}

    # load_APE_A()
    # load_APE_B()
    # load_APE_C()
    save_memory = False
    load_APE_D()

    promot_txt = ','.join(
        ['person', 'mouse', 'cat', 'dog', 'cigarette',
         'person who is smoking', 'person who is shirtless', 'person without clothing'])
    classes = [6, 3, 4, 5, 1, 1, 2, 2]

    classes_chn = ['无', '抽烟', '赤膊', '老鼠', '猫', '狗', '人']

    # 保存csv
    f = open('result_video.csv', 'w')
    f.write('filename,result\n')

    # 创建文件夹
    save_dir = './res_video'
    if os.path.exists(save_dir):
        # 删除文件夹，并创建新的
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for i in classes_chn:
        os.makedirs(os.path.join(save_dir, i))


    video_dir = '/data/gulingrui/code/mclz/data/video'
    for i in tqdm(os.listdir(video_dir)):
        if not i.endswith('.ts'):
            continue

        video_path = os.path.join(video_dir, i)

        # 使用 VideoCapture 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print("无法打开视频文件")
            exit()

        # 使用循环读取和展示视频的每一帧
        ind = 0
        video_classes = []
        while True:
            # 读取一帧视频
            ret, frame = cap.read()

            # 如果帧没有被正确的读取，那么 read 返回的 ret 将会是 False
            if not ret:
                print("不能接收到帧（流可能结束了 ...）")
                break

            if (ind - 1) % 5 == 0:
                # 保存图片
                cv2.imwrite('tmp.jpg', frame)
                # 处理图片
                res = run_on_image_D(
                    "tmp.jpg",
                    promot_txt,
                    0.25,
                    ["object detection",],
                )[1]
                category_set = list(set([classes[k['category_id']] for k in res]))
                video_classes += category_set

            ind += 1
        video_classes = set(video_classes)

        # 后处理
        # 如果人在视频中，那就默认不存在老鼠
        if 6 in video_classes:
            video_classes.discard(6)
            if 3 in video_classes:
                video_classes.discard(3)

        res_class = 0

        for cat_ind in video_classes:
            res_class += 2 ** (cat_ind - 1)
            shutil.copy(video_path, os.path.join(save_dir, classes_chn[cat_ind], i))
        if len(video_classes) == 0:
            shutil.copy(video_path, os.path.join(save_dir, '无', i))
        f.write(i + ',' + str(res_class) + '\n')


