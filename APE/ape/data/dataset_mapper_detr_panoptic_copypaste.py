import copy
import logging
import os
import random
from typing import List, Optional, Union

import cv2
import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import batched_nms
from detectron2.structures import BitMasks, Boxes, Instances, PolygonMasks

from . import mapper_utils

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper_detr_panoptic_copypaste"]


class DatasetMapper_detr_panoptic_copypaste:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        augmentations_with_crop: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        ignore_label: int = 255,
        stuff_classes_offset: int = 80,
        stuff_classes_decomposition: bool = False,
        copypaste_prob: float = 0.5,
        output_dir: str = None,
        vis_period: int = 0,
        dataset_names: tuple = (),
        max_num_phrase: int = 0,
        nms_thresh_phrase: float = 0.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.augmentations_with_crop = T.AugmentationList(augmentations_with_crop)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.ignore_label           = ignore_label
        self.stuff_classes_offset   = stuff_classes_offset
        self.stuff_classes_decomposition   = stuff_classes_decomposition
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations_with_crop}")

        if output_dir is not None:
            self.output_dir = os.path.join(output_dir, "vis_mapper")
            os.makedirs(self.output_dir, exist_ok=True)

        self.copypaste_prob = copypaste_prob
        self.vis_period = vis_period
        self.iter = 0
        self.dataset_names = dataset_names

        self.metatada_list = []
        for dataset_name in self.dataset_names:
            metadata = MetadataCatalog.get(dataset_name)
            self.metatada_list.append(metadata)

        self.max_num_phrase = max_num_phrase
        self.nms_thresh_phrase = nms_thresh_phrase

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        raise NotImplementedError(self.__class__.__name__)

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        copypaste = [
            obj.get("copypaste", 0)
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]

        phrases = [
            obj.get("phrase", "")
            for obj in dataset_dict["annotations"]
            if obj.get("iscrowd", 0) == 0
        ]

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        instances.copypaste = torch.tensor(copypaste)

        if sum([len(x) for x in phrases]) > 0:
            instances.phrase_idxs = torch.tensor(range(len(phrases)))

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes and instances.has("gt_masks"):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        if sum([len(x) for x in phrases]) > 0:
            phrases_filtered = []
            for x in dataset_dict["instances"].phrase_idxs.tolist():
                phrases_filtered.append(phrases[x])
            dataset_dict["instances"].phrases = mapper_utils.transform_phrases(
                phrases_filtered, transforms
            )
            dataset_dict["instances"].remove("phrase_idxs")
            # dataset_dict["instances"].gt_classes = torch.tensor(range(len(phrases_filtered)))

    def __call__(self, dataset_dict, dataset_dict_bg):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        try:
            image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"read_image fails: {dataset_dict['file_name']}")
            logger.error(f"read_image fails: {e}")
            return None
        utils.check_image_size(dataset_dict, image)

        # ------------------------------------------------------------------------------------
        if "dataset_id" in dataset_dict:
            dataset_id = dataset_dict["dataset_id"]
        else:
            dataset_id = 0
        metadata = self.metatada_list[dataset_id]
        if "sa1b" in self.dataset_names[dataset_id]:
            metadata = None
        if (
            self.is_train
            and "annotations" in dataset_dict
            and (
                len(dataset_dict["annotations"]) == 0
                or any(["bbox" not in anno for anno in dataset_dict["annotations"]])
            )
        ):
            dataset_dict = mapper_utils.maybe_load_annotation_from_file(dataset_dict, meta=metadata)

            for anno in dataset_dict["annotations"]:
                if "bbox" not in anno:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Box not found: {dataset_dict}")
                    return None
                if "category_id" not in anno:
                    anno["category_id"] = 0
        # ------------------------------------------------------------------------------------

        # ------------------------------------------------------------------------------------
        if dataset_dict["copypaste"] and self.copypaste_prob > random.uniform(0, 1):
            image_cp, dataset_dict_cp = mapper_utils.copypaste(
                dataset_dict, dataset_dict_bg, self.image_format, self.instance_mask_format
            )

            if dataset_dict_cp is None or image_cp is None:
                pass
            else:
                for key in dataset_dict.keys():
                    if key in dataset_dict_cp:
                        continue
                    dataset_dict_cp[key] = dataset_dict[key]
                dataset_dict = dataset_dict_cp
                image = image_cp
        # ------------------------------------------------------------------------------------

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            try:
                sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"read_image fails: {e}")
                logger.error(f"read_image fails: {dataset_dict}")
                return None

            if "copypaste_mask" in dataset_dict:
                # assume thing class is 0
                sem_seg_gt = sem_seg_gt.copy()
                sem_seg_gt[dataset_dict["copypaste_mask"]] = 0
        else:
            sem_seg_gt = None

        # ordinal numbers
        disable_crop = False
        if (
            "annotations" in dataset_dict
            and len(dataset_dict["annotations"]) > 0
            and "phrase" in dataset_dict["annotations"][0]
        ):
            disable_crop = disable_crop or mapper_utils.has_ordinal_num(
                [anno["phrase"] for anno in dataset_dict["annotations"]]
            )
        if "expressions" in dataset_dict:
            disable_crop = disable_crop or mapper_utils.has_ordinal_num(dataset_dict["expressions"])

        if self.augmentations_with_crop is None or disable_crop:
            augmentations = self.augmentations
        else:
            if np.random.rand() > 0.5:
                augmentations = self.augmentations
            else:
                augmentations = self.augmentations_with_crop

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        # transforms = self.augmentations(aug_input)
        transforms = augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if "expressions" in dataset_dict:
            dataset_dict["expressions"] = mapper_utils.transform_expressions(
                dataset_dict["expressions"], transforms
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            dataset_dict.pop("pan_seg_file_name", None)
            dataset_dict.pop("segments_info", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

            dataset_dict["instances"].is_thing = torch.tensor(
                [True for _ in range(len(dataset_dict["instances"]))], dtype=torch.bool
            )

        if "instances" in dataset_dict and dataset_dict["instances"].has("phrases"):
            num_instances = len(dataset_dict["instances"])

            if self.nms_thresh_phrase > 0:
                boxes = dataset_dict["instances"].gt_boxes.tensor
                scores = torch.rand(num_instances)
                classes = torch.zeros(num_instances)
                keep = batched_nms(boxes, scores, classes, self.nms_thresh_phrase)
            else:
                keep = torch.randperm(num_instances)

            if self.max_num_phrase > 0:
                keep = keep[: self.max_num_phrase]

            phrases = dataset_dict["instances"].phrases
            phrases_filtered = []
            for x in keep:
                phrases_filtered.append(phrases[x])

            dataset_dict["instances"].remove("phrases")
            dataset_dict["instances"] = dataset_dict["instances"][keep]
            dataset_dict["instances"].phrases = phrases_filtered

        # Prepare per-category binary masks
        if sem_seg_gt is not None and not self.stuff_classes_decomposition:
            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt).astype(np.int64)
            # remove ignored region
            classes = classes[classes != self.ignore_label]

            if self.stuff_classes_offset > 0:
                classes = classes[classes != 0]
                instances.gt_classes = torch.tensor(
                    classes + self.stuff_classes_offset - 1, dtype=torch.int64
                )
            else:
                instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # # Some image does not have annotation (all ignored)
                # instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
                masks = BitMasks(torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1])))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )

            instances.gt_masks = masks
            instances.gt_boxes = masks.get_bounding_boxes()

            instances.is_thing = torch.tensor(
                [False for _ in range(len(instances))], dtype=torch.bool
            )

            if "instances" in dataset_dict and dataset_dict["instances"].has("copypaste"):
                instances.copypaste = torch.tensor([False for _ in range(len(instances))])

            if len(instances) > 0:
                if "instances" in dataset_dict and len(dataset_dict["instances"]) > 0:
                    dataset_dict["instances"] = Instances.cat(
                        [dataset_dict["instances"], instances]
                    )
                else:
                    dataset_dict["instances"] = instances

        # Prepare per-category binary masks
        if sem_seg_gt is not None and self.stuff_classes_decomposition:
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]

            if self.stuff_classes_offset > 0:
                classes = classes[classes != 0]

            gt_masks = []
            gt_classes = []
            for class_id in classes:
                bitmask = sem_seg_gt == class_id
                pygmask, _ = mapper_utils.mask_to_polygons_2(bitmask)
                for mask in pygmask:
                    gt_masks.append([mask])
                    gt_classes.append(class_id)

            # if len(gt_masks) == 0:
            #     return None

            instances = Instances(image_shape)
            instances.gt_classes = torch.tensor(gt_classes, dtype=torch.int64)
            if self.stuff_classes_offset > 0:
                instances.gt_classes += self.stuff_classes_offset - 1
            if self.instance_mask_format == "polygon":
                instances.gt_masks = PolygonMasks(gt_masks)
            else:
                assert self.instance_mask_format == "bitmask", self.instance_mask_format
                instances.gt_masks = BitMasks.from_polygon_masks(
                    gt_masks, image_shape[0], image_shape[1]
                )
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()

            if self.instance_mask_format == "polygon":
                area = instances.gt_masks.area()
            else:
                assert self.instance_mask_format == "bitmask", self.instance_mask_format
                area = instances.gt_masks.tensor.sum((1, 2))
            instances = instances[area > 8 * 8]

            instances.is_thing = torch.tensor(
                [False for _ in range(len(instances))], dtype=torch.bool
            )

            if "instances" in dataset_dict and dataset_dict["instances"].has("copypaste"):
                instances.copypaste = torch.tensor([False for _ in range(len(instances))])

            if len(instances) > 0:
                if "instances" in dataset_dict and len(dataset_dict["instances"]) > 0:
                    dataset_dict["instances"] = Instances.cat(
                        [dataset_dict["instances"], instances]
                    )
                else:
                    dataset_dict["instances"] = instances

        if "pan_seg_file_name" in dataset_dict and not self.stuff_classes_decomposition:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    classes.append(class_id)
                    masks.append(pan_seg_gt == segment_info["id"])

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            if "instances" in dataset_dict and dataset_dict["instances"].has("copypaste"):
                instances.copypaste = torch.tensor([False for _ in range(len(instances))])

            dataset_dict["instances"] = instances

        if "pan_seg_file_name" in dataset_dict and self.stuff_classes_decomposition:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")
            segments_info = dataset_dict["segments_info"]

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt)

            instances = Instances(image_shape)
            classes = []
            masks = []
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    if class_id in metadata.thing_dataset_id_to_contiguous_id.values():
                        classes.append(class_id)
                        masks.append(pan_seg_gt == segment_info["id"])
                    else:
                        bitmask = pan_seg_gt == segment_info["id"]
                        pygmask, _ = mapper_utils.mask_to_polygons_2(bitmask)
                        for mask in pygmask:
                            mask = (
                                BitMasks.from_polygon_masks(
                                    [[mask]], image_shape[0], image_shape[1]
                                )
                                .tensor[0, ...]
                                .numpy()
                            )
                            classes.append(class_id)
                            masks.append(mask)

            classes = np.array(classes)
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((0, 4)))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()

            if "instances" in dataset_dict and dataset_dict["instances"].has("copypaste"):
                instances.copypaste = torch.tensor([False for _ in range(len(instances))])

            dataset_dict["instances"] = instances

        if "instances" in dataset_dict and len(dataset_dict["instances"]) > 0:
            pass
        else:
            return None

        # ------------------------------------------------------------------------------------
        if self.vis_period > 0 and self.iter % self.vis_period == 0:
            self.visualize_training(dataset_dict)
        # ------------------------------------------------------------------------------------
        self.iter += 1

        return dataset_dict

    def visualize_training(self, dataset_dict, prefix="", suffix=""):
        if self.output_dir is None:
            return
        if dataset_dict is None:
            return
        # if "instances" not in dataset_dict:
        #     return
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog

        if "dataset_id" in dataset_dict:
            dataset_id = dataset_dict["dataset_id"]
        else:
            dataset_id = 0
        dataset_name = self.dataset_names[dataset_id]
        metadata = MetadataCatalog.get(dataset_name)
        class_names = (
            metadata.get("thing_classes", []) + metadata.get("stuff_classes", ["thing"])[1:]
        )

        if "instances" in dataset_dict and dataset_dict["instances"].has("phrases"):
            labels = dataset_dict["instances"].phrases
        elif "expressions" in dataset_dict:
            labels = [dataset_dict["expressions"]]
        else:
            labels = [class_names[i] for i in dataset_dict["instances"].gt_classes]

        img = dataset_dict["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.image_format)
        image_shape = img.shape[:2]  # h, w
        vis = Visualizer(img, metadata=metadata)
        if "instances" in dataset_dict:
            vis = vis.overlay_instances(
                boxes=dataset_dict["instances"].gt_boxes,
                masks=dataset_dict["instances"].gt_masks
                if dataset_dict["instances"].has("gt_masks")
                else None,
                labels=labels,
            )
        else:
            vis = vis.overlay_instances(
                boxes=None,
                masks=None,
                labels=None,
            )
        vis_gt = vis.get_image()

        if "instances_phrase" in dataset_dict:
            vis = Visualizer(img, metadata=metadata)
            vis = vis.overlay_instances(
                boxes=dataset_dict["instances_phrase"].gt_boxes,
                masks=dataset_dict["instances_phrase"].gt_masks
                if dataset_dict["instances_phrase"].has("gt_masks")
                else None,
                labels=dataset_dict["instances_phrase"].phrases,
            )
            vis_phrase = vis.get_image()
            vis_gt = np.concatenate((vis_gt, vis_phrase), axis=1)

        if "captions" in dataset_dict:
            vis = Visualizer(img, metadata=metadata)
            vis = vis.overlay_instances(
                boxes=Boxes(
                    np.array(
                        [
                            [
                                0 + i * 20,
                                0 + i * 20,
                                image_shape[1] - 1 - i * 20,
                                image_shape[0] - 1 - i * 20,
                            ]
                            for i in range(len(dataset_dict["captions"]))
                        ]
                    )
                ),
                masks=None,
                labels=dataset_dict["captions"],
            )
            vis_cap = vis.get_image()
            vis_gt = np.concatenate((vis_gt, vis_cap), axis=1)

        if "sem_seg" in dataset_dict:
            vis = Visualizer(img, metadata=metadata)
            vis = vis.draw_sem_seg(dataset_dict["sem_seg"], area_threshold=0, alpha=0.5)
            vis_sem_gt = vis.get_image()
            vis_gt = np.concatenate((vis_gt, vis_sem_gt), axis=1)

        concat = np.concatenate((vis_gt, img), axis=1)

        image_name = os.path.basename(dataset_dict["file_name"]).split(".")[0]

        save_path = os.path.join(
            self.output_dir,
            prefix
            + str(self.iter)
            + "_"
            + image_name
            + "_g"
            + str(comm.get_rank())
            + suffix
            + ".png",
        )
        concat = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, concat)

        return

        import pickle

        save_path = os.path.join(
            self.output_dir,
            prefix
            + str(self.iter)
            + "_"
            + str(dataset_dict["image_id"])
            + "_g"
            + str(comm.get_rank())
            + suffix
            + ".pkl",
        )
        with open(save_path, "wb") as save_file:
            pickle.dump(dataset_dict, save_file)
