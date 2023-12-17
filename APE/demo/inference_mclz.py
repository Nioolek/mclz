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
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

import ape
import detectron2.data.transforms as T
from ape.model_zoo import get_config_file
from demo_lazy import get_parser, setup_cfg
from detectron2.config import CfgNode
from detectron2.data.detection_utils import read_image
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.utils.logger import setup_logger
from predictor_lazy import VisualizationDemo

# from blip_models.blip import blip_decoder
# from blip_models.blip_vqa import blip_vqa

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

    ape_model = VisualizationDemo(cfg, args=args)
    ape_model.predictor.model.to(running_device)

    return ape_model, cfg


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

    ape_model = VisualizationDemo(cfg, args=args)
    ape_model.predictor.model.to(running_device)

    return ape_model, cfg


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

    ape_model = VisualizationDemo(cfg, args=args)
    ape_model.predictor.model.to(running_device)

    return ape_model, cfg


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
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    cfg.model.model_vision.criterion[0].use_fed_loss = False
    cfg.model.model_vision.criterion[2].use_fed_loss = False
    cfg.train.device = running_device

    ape.modeling.text.eva02_clip.factory._MODEL_CONFIGS[cfg.model.model_language.clip_model][
        "vision_cfg"
    ]["layers"] = 1

    ape_model = VisualizationDemo(cfg, args=args)
    ape_model.predictor.model.to(running_device)

    return ape_model, cfg

# def load_blip_vqa():
#     model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth'
#     model = blip_vqa(pretrained=model_url, image_size=480, vit='base')
#     model.eval()
#     model = model.to(running_device)
#     return model

# blip_img_transform = transforms.Compose([
#         transforms.Resize((480, 480), interpolation=InterpolationMode.BICUBIC),
#         transforms.ToTensor(),
#         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#         ])


# def transform_blip_img(img, bbox):
#     x1, y1, w, h = bbox
#     # 按照坐标，从原图中截取
#
#     img_h, img_w, _ = img.shape
#     x1 = max(0, int(x1 - w * 0.6))
#     y1 = max(0, int(y1 - h * 0.3))
#     y2 = min(img_h, int(y1 + h * 1.6))
#     x2 = min(img_w, int(x1 + w * 1.3))
#     img = img[y1:y2, x1:x2]
#     # 将img转为RGB，再转为PIL
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img).convert('RGB')
#
#     image = blip_img_transform(img).unsqueeze(0).to(running_device)
#     return image


def run_on_image(
    input_image_path,
    input_text,
    demo,
    cfg,
):
    input_image = read_image(input_image_path, format="BGR")


    # if input_mask_path is not None:
    #     input_mask = read_image(input_mask_path, "L").squeeze(2)
    #     print("input_mask", input_mask)
    #     print("input_mask", input_mask.shape)
    # else:
    input_mask = None

    if input_image.shape[0] > 1024 or input_image.shape[1] > 1024:
        transform = aug.get_transform(input_image)
        input_image = transform.apply_image(input_image)
    # elif input_image.shape[0] < 640 and input_image.shape[1] < 640:
    #     transform = short_aug.get_transform(input_image)
    #     input_image = transform.apply_image(input_image)
    else:
        transform = None
    # print('!!!', input_image.shape)

    start_time = time.time()
    print('input_text', input_text)
    predictions, metadata = demo.run_on_mclz(
        input_image,
        text_prompt=input_text
    )

    logger.info(
        "{} in {:.2f}s".format(
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )

    gc.collect()
    torch.cuda.empty_cache()

    json_results = instances_to_coco_json(predictions["instances"].to(demo.cpu_device), 0)
    for json_result in json_results:
        json_result["category_name"] = metadata.thing_classes[json_result["category_id"]]
        del json_result["image_id"]

    return json_results


def is_grey_scale(img):
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False


def inference_img():
    # 保存csv
    f = open('result.csv', 'w')
    f.write('filename,result\n')

    # 处理图片
    # picture_dir = '/data/gulingrui/code/mclz/data/picture'
    picture_dir = '/data/gulingrui/code/mmpretrain_tianchi/rotated_1'

    # 创建文件夹
    save_dir = './res'
    if os.path.exists(save_dir):
        # 删除文件夹，并创建新的
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    for i in classes_chn:
        os.makedirs(os.path.join(save_dir, i))

    for i in tqdm(os.listdir(picture_dir)):
        if not i.endswith('.jpg'):
            continue

        # if i not in [
        #     'ele_318dbbae3d7b9243d34ba038ed523be1.jpg',
        #     'ele_0b593e76f1bf2a21f9eaead5a0855473.jpg',
        #     'ele_0d0e3a492c553922273d216ab30b61e7.jpg',
        #     'ele_1b81e944593b335937097795c81adc8c.jpg',
        #     'ele_2b2a0492f282af6594c0ac615425fe21.jpg'
        # ]:
        #     continue
        img_path = os.path.join(picture_dir, i)
        # img = cv2.imread(img_path)
        # is_grey = is_grey_scale(img)


        res = run_on_image(img_path, promot_txt, ape_model, cfg)

        # 没穿衣服的分数按照0.3
        new_res = []
        for res_i in res:
            score = res_i['score']
            category_id = res_i['category_id']
            if (category_id in [6, 7]) and score< 0.3:
                continue

            # 抽烟
            if category_id in [4, ]:
                new_res.append(res_i)
            # 没穿衣服
            elif category_id in [6, 7]:
                if score>=0.3:
                    new_res.append(res_i)
            # 其他
            else:
                if score>=0.25:
                    new_res.append(res_i)

        res = new_res
        print(res)

        # 统计当前图片所有包含的类别
        img_class_list = []

        # 先处理人
        # person_res = [k for k in res if k['category_id'] == 0]

        # for p in person_res:
        #     bbox = p['bbox']
        #     crop_img = transform_blip_img(img, bbox)
        #     question = 'Is the person smoking? Please answer yes,no or not sure.'
        #     # question = 'Is the person smoking?'
        #     # question = 'Where is the cigarette?'
        #     with torch.no_grad():
        #         answer = blip_model(crop_img, question, train=False, inference='generate')
        #         print(i, 'answer: ' + answer[0])
        #         if 'yes' in answer[0].lower():
        #             img_class_list.append(1)

        res_class = 0
        category_set = set([classes[k['category_id']] for k in res] + img_class_list)

        # 后处理
        # 如果人在视频中，那就默认不存在老鼠
        if 6 in category_set:
            category_set.discard(6)
            if 3 in category_set:
                category_set.discard(3)

        for cat_ind in category_set:
            res_class += 2 ** (cat_ind - 1)
            # 保存可视化
            shutil.copy(img_path, os.path.join(save_dir, classes_chn[cat_ind], i))
        if len(category_set) == 0:
            shutil.copy(img_path, os.path.join(save_dir, '无', i))
        f.write(i + ',' + str(res_class) + '\n')

    f.close()

def inference_video():
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
                res = run_on_image('tmp.jpg', promot_txt, ape_model, cfg)

                # 过滤一下结果
                new_res = []

                # # 先处理人
                # person_res = [k for k in res if k['category_id'] == 0]
                # for p in person_res:
                #     bbox = p['bbox']
                #     crop_img = transform_blip_img(frame, bbox)
                #     question = 'Is the person smoking?'
                #     with torch.no_grad():
                #         answer = blip_model(crop_img, question, train=False, inference='generate')
                #         print('answer: ' + answer[0])
                #         if 'yes' in answer[0].lower():
                #             video_classes.append(1)

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


if __name__ == '__main__':
    flag = 'img'

    available_memory = [
        torch.cuda.mem_get_info(i)[0] / 1024 ** 3 for i in range(torch.cuda.device_count())
    ]

    global running_device
    max_available_memory = max(available_memory)
    device_id = available_memory.index(max_available_memory)

    running_device = "cuda:" + str(device_id)

    print("available_memory", available_memory)
    print("max_available_memory", max_available_memory)
    print("running_device", running_device)

    # ==========================================================================================

    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    setup_logger(name="ape")
    global logger
    logger = setup_logger()

    global aug, short_aug
    aug = T.ResizeShortestEdge([1024, 1024], 1024)
    short_aug = T.ResizeShortestEdge([640, 640], 640)

    # 加载APE模型
    # ape_model, cfg = load_APE_A()
    # ape_model, cfg = load_APE_B()
    # ape_model, cfg = load_APE_C()
    ape_model, cfg = load_APE_D()
    # 抽烟按照0.15
    ape_model.predictor.model.model_vision.test_score_thresh = 0.15

    # person, mouse, cat, dog, cigarette, person who is smoking, person who is shirtless, person without clothing
    promot_txt = ','.join(
        ['person', 'mouse', 'cat', 'dog', 'cigarette',
         'person who is smoking',
         'person who is shirtless', 'person without clothing'])
    classes = [6, 3, 4, 5,
               1,
               1, 2, 2]
    classes_chn = ['无', '抽烟', '赤膊', '老鼠', '猫', '狗', '人']

    # 加载BLIP模型
    # blip_model = load_blip_vqa()


    if flag == 'video':
        inference_video()
    else:
        inference_img()


