import torch

BATCH_SIZE = 10 # increase / decrease according to GPU memeory
RESIZE_TO_WIDTH = 720 # resize the image for training and transforms
RESIZE_TO_HEIGHT = 1280 # resize the image for training and transforms
NUM_EPOCHS = 10 # number of epochs to train for

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

coco_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

seashipsclasses = {'ore carrier': 1, 'bulk cargo carrier': 2, 'general cargo ship': 2, 'container ship': 3, 'fishing boat': 4, 'passenger ship': 5}
inv_seashipsclasses = {v: k for k, v in seashipsclasses.items()}

aboshipsclasses = {'Seamark': 6, 'Motorboat': 7, 'Sailboat': 8, 'Passengership': 5, 'Cargoship': 2, 'Militaryship': 9, 'Ferry': 10, 'Cruiseship':11, 'Miscboat': 12, 'Boat':12, 'Miscellaneous': 13}
inv_aboshipsclasses = {v: k for k, v in aboshipsclasses.items()}

inv_seashipsclasses_copy = inv_seashipsclasses.copy()
inv_abo_seashipsclasses = inv_aboshipsclasses.copy()
inv_abo_seashipsclasses.update(inv_seashipsclasses_copy)
abo_seashipsclasses = {v: k for k, v in inv_abo_seashipsclasses.items()}

SMDclasses = {'Ferry':10, 'Buoy':6, 'Vessel/ship':12, 'Speed boat':14, 'Boat':12, 'Kayak':15, 'Sail boat':8, 'Swimming person':16, 'Flying bird/plane':17, 'Other':13}
inv_SMDclasses = {v: k for k, v in SMDclasses.items()}

#COCO boat bird plane person
COCO_bbpp = {'person':15, 'bird':17, 'boat':11, 'plane':18} # maybe just bird, boat, plane

inv_total = {1:'Ore Carrier', 2:'Cargo Carrier', 3:'Container Ship', 4:'Fishing Boat', 5:'Passenger Ship', 6:'Seamark', 7:'Motorboat', 8:'Sailboat', 9:'Militaryship', 10:'Ferry', 11:'Cruiseship',12:'Boat', 13:'Miscellaneous', 14:'Speed boat', 15:'Kayak', 16:'Person', 17:'Flying Object', 18:'Bird',19:'Plane'}
all_classes = {v:k for k,v in inv_total.items()}

