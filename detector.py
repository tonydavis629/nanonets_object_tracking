import torch
import torchvision
from config import device, coco_CLASSES, abo_seashipsclasses
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2
# from torchvision.transforms import transforms as T
# import torchvision.transforms.functional as TF
# from tqdm import tqdm
# from evalforward import eval_forward
# # from torchvision.models.detection.roi_heads import fastrcnn_loss
from typing import Tuple, List, Dict, Optional
# from collections import OrderedDict
# from torchvision.models.detection.roi_heads import fastrcnn_loss
# from torchvision.models.detection.rpn import concat_box_prediction_layers

class teacher():
  def __init__(self, model_path=None,classes=coco_CLASSES,threshold=0.5):
    self.threshold = threshold
    self.model = self.load_model(model_path)
    self.classes = classes
  def load_model(self, model_path=None):
    """   
    Loads a pretrained model and state_dict if desired 
    """

    print("Loading model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    print("Loading model...done")

    if model_path is not None:
      print("Loading model from:", model_path)
      model.load_state_dict(torch.load(model_path)['model_state_dict'])

    model.to(device)
    return model

  def detect(self, image: torch.Tensor, threshold=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs inference on a single image.

    """

    image = transform(image)

    if len(image.shape) == 3:   #if input is single image, wrap in a batch
      image = image.unsqueeze(0)

    image = image.to(device)

    if threshold is None:
      threshold = self.threshold
    self.model.eval()
    with torch.no_grad():
      pred = self.model(image)
    boxes = pred[0]['boxes'].detach().cpu().numpy()
    if boxes.shape[0] != 0:
      max_box_y = max(boxes[:, 3])
      max_box_x = max(boxes[:, 2])
      min_box_y = min(boxes[:, 1])
      min_box_x = min(boxes[:, 0])
      new_boxes = torch.tensor([[min_box_x, min_box_y, max_box_x, max_box_y]])
      pred[0]['boxes'] = new_boxes.squeeze(0)
    return pred[0]['boxes'], pred[0]['labels'], pred[0]['scores']

def transform(image):

  if not torch.is_tensor(image):
    img_tensor = T.ToTensor()(image)
  else:
    img_tensor = image

  # norm_tensor = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img_tensor.type(torch.FloatTensor))
  norm_tensor = img_tensor

  scaled_tensor = (norm_tensor - torch.min(norm_tensor))/(torch.max(norm_tensor) - torch.min(norm_tensor))*(1 - 0) + 0 # min max scaling from 0 to 1

  return scaled_tensor

# class detector():
#   def __init__(self, model_path=None,classes=coco_CLASSES,threshold=0.5, finetune=False):
#     self.threshold = threshold
#     self.model = self.load_model(model_path)
#     self.classes = classes
#     self.finetune = finetune
#   def load_model(self, model_path=None, finetune=False):
#     """   
#     Loads a pretrained model and state_dict if desired 
#     """

#     print("Loading model...")
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
#     print("Loading model...done")

#     if model_path is not None:
#       print("Loading model from:", model_path)
#       model.load_state_dict(torch.load(model_path)['model_state_dict'])

#     if finetune:
#       num_classes = len(self.classes)
#       in_features = model.roi_heads.box_predictor.cls_score.in_features
#       model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     model.to(device)
#     return model

#   def train(self, train_data_loader, optimizer, train_loss_hist, metric):
#       print('Training...')
#       prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
      
#       for i, data in enumerate(prog_bar):
#           optimizer.zero_grad()
#           images, targets = data
          
#           images = list(image.to(device) for image in images)
#           targetsgpu = [{k: v.to(device) for k, v in t.items()} for t in targets]
          
#           losses, dets = self.eval_forward(images, targetsgpu)
#           targetscpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
#           detscpu = [{k: v.detach().cpu() for k, v in t.items()} for t in dets]

#           losssum = sum(loss for loss in losses.values())
#           loss_value = losssum.detach().item()
#           metric.update(preds=detscpu,target=targetscpu)
          
#           # loss_dict = self.model(images, targets)
#           # losses = sum(loss for loss in loss_dict.values())
#           # loss_value = losses.item()
          
#           train_loss_hist.send(loss_value)
#           losssum.backward()
#           optimizer.step()
#           optimizer.zero_grad(set_to_none=True)

#           prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
#       # return 


#   def validate(self,valid_data_loader,val_loss_hist,metric):
#       print('Validating...')
#       prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
#       with torch.no_grad():      
#         for i, data in enumerate(prog_bar):
#             images, targets_dict = data
            
#             images = list(image.to(device) for image in images)
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets_dict]
            
#             losses, dets = self.eval_forward(images, targets)
#             targetscpu = [{k: v.detach().cpu() for k, v in t.items()} for t in targets]
#             detscpu = [{k: v.detach().cpu() for k, v in t.items()} for t in dets]
#             losssum = sum(loss for loss in losses.values())
#             loss_value = losssum.detach().item()

#             metric.update(preds=detscpu,target=targetscpu)
            
#             # loss_dict = self.model(images, targets)
#             # losses = sum(loss for loss in loss_dict.values())
#             # loss_value = losses.item()
#             val_loss_hist.send(loss_value)

#             prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
#       # return val_loss_list, metric.compute()

#   def detect(self, image):
#     """
#     image: tensor of shape C H W
#     """

#     image = transform(image)

#     if len(image.shape) == 3:   #if input is single image, wrap in a batch
#       image = image.unsqueeze(0)

#     image = image.to(device)

#     with torch.no_grad():
#       self.model.eval()
#       outputs = self.model(image)

#     outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs] #move outputs to cpu

#     boxes = outputs[0]['boxes'] #[x1, y1, x2, y2]
#     scores = outputs[0]['scores']
#     labels = outputs[0]['labels']

#     # filter out boxes according to threshold
#     conf_mask = scores > self.threshold
#     boxes = boxes[conf_mask]
#     labels = labels[conf_mask]
#     scores = scores[conf_mask]

#     # get all the predicited class names
#     pred_classes = [self.classes[int(i)] for i in labels.cpu()]

#     return boxes, pred_classes, scores

#   def eval_forward(self, images, targets):
#     """
#     This function only exists because pytorch gives no way to return predictions and losses from a forward pass. It slightly modifies the model to do so.

#         # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
#     Args:
#         images (list[Tensor]): images to be processed
#         targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
#     Returns:
#         result (list[BoxList] or dict[Tensor]): the output from the model.
#             It returns list[BoxList] contains additional fields
#             like `scores`, `labels` and `mask` (for Mask R-CNN models).
#     """
#     self.model.eval()

#     original_image_sizes: List[Tuple[int, int]] = []
#     for img in images:
#         val = img.shape[-2:]
#         assert len(val) == 2
#         original_image_sizes.append((val[0], val[1]))

#     images, targets = self.model.transform(images, targets)

#     # Check for degenerate boxes
#     # TODO: Move this to a function
#     if targets is not None:
#         for target_idx, target in enumerate(targets):
#             boxes = target["boxes"]
#             degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
#             if degenerate_boxes.any():
#                 # print the first degenerate box
#                 bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
#                 degen_bb: List[float] = boxes[bb_idx].tolist()
#                 raise ValueError(
#                     "All bounding boxes should have positive height and width."
#                     f" Found invalid box {degen_bb} for target at index {target_idx}."
#                 )

#     features = self.model.backbone(images.tensors)
#     if isinstance(features, torch.Tensor):
#         features = OrderedDict([("0", features)])
#     self.model.rpn.training=True
#     #model.roi_heads.training=True


#     #####proposals, proposal_losses = model.rpn(images, features, targets)
#     features_rpn = list(features.values())
#     objectness, pred_bbox_deltas = self.model.rpn.head(features_rpn)
#     anchors = self.model.rpn.anchor_generator(images, features_rpn)

#     num_images = len(anchors)
#     num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
#     num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
#     objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
#     # apply pred_bbox_deltas to anchors to obtain the decoded proposals
#     # note that we detach the deltas because Faster R-CNN do not backprop through
#     # the proposals
#     proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
#     proposals = proposals.view(num_images, -1, 4)
#     proposals, scores = self.model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

#     proposal_losses = {}
#     assert targets is not None
#     labels, matched_gt_boxes = self.model.rpn.assign_targets_to_anchors(anchors, targets)
#     regression_targets = self.model.rpn.box_coder.encode(matched_gt_boxes, anchors)
#     loss_objectness, loss_rpn_box_reg = self.model.rpn.compute_loss(
#         objectness, pred_bbox_deltas, labels, regression_targets
#     )
#     proposal_losses = {
#         "loss_objectness": loss_objectness,
#         "loss_rpn_box_reg": loss_rpn_box_reg,
#     }

#     #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
#     image_shapes = images.image_sizes
#     proposals, matched_idxs, labels, regression_targets = self.model.roi_heads.select_training_samples(proposals, targets)
#     box_features = self.model.roi_heads.box_roi_pool(features, proposals, image_shapes)
#     box_features = self.model.roi_heads.box_head(box_features)
#     class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

#     result: List[Dict[str, torch.Tensor]] = []
#     detector_losses = {}
#     loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
#     detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
#     boxes, scores, labels = self.model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
#     num_images = len(boxes)
#     for i in range(num_images):
#         result.append(
#             {
#                 "boxes": boxes[i],
#                 "labels": labels[i],
#                 "scores": scores[i],
#             }
#         )
#     detections = result
#     detections = self.model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
#     self.model.rpn.training=False
#     self.model.roi_heads.training=False
#     losses = {}
#     losses.update(detector_losses)
#     losses.update(proposal_losses)
#     return losses, detections





# class Averager:
#     def __init__(self):
#         self.current_total = 0.0
#         self.iterations = 0.0
        
#     def send(self, value):
#         self.current_total += value
#         self.iterations += 1
    
#     @property
#     def value(self):
#         if self.iterations == 0:
#             return 0
#         else:
#             return 1.0 * self.current_total / self.iterations
    
#     def reset(self):
#         self.current_total = 0.0
#         self.iterations = 0.0
        
# class mask_detector():
#   def __init__(self,model_path=None,classes=coco_CLASSES,threshold=0.5):
#     self.threshold = threshold
#     self.model = self.load_model(model_path)
#     self.classes = classes
#   def load_model(self, model_path=None):
#     if model_path is None:
#       model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
#       model.cuda()
#       return model
#     else:
#       model = maskrcnn_resnet50_fpn_v2(pretrained=False)
#       model.load_state_dict(torch.load(model_path))
#       model.cuda()
#       return model
#   def detect(self,img):
#     img = transform(img)
#     if len(img.shape) == 3:   #if input is single image, wrap in a batch
#       img = img.unsqueeze(0)
#     img = img.to(device)
#     self.model.eval()
#     pred = self.model(img)

#     masks = pred[0]['masks'].detach().cpu()
#     masks = masks > .5
#     masks = masks.to(torch.bool)

#     pred_boxes = pred[0]['boxes'].detach().cpu().numpy()
#     labels = pred[0]['labels'].detach().cpu().numpy()
#     pred_scores = pred[0]['scores'].detach().cpu().numpy()

#     conf_mask = pred_scores > self.threshold
    
#     masks = masks[conf_mask]
#     labels = labels[conf_mask]
#     pred_scores = pred_scores[conf_mask]

#     pred_classes = [self.classes[i] for i in labels]

#     return pred_classes,pred_boxes,pred_scores,masks