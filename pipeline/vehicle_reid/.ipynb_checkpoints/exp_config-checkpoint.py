import os.path as osp
class Validate:
    def __init__(self):
        self.QUE_EMB_FILE="track2_validate_query_embedding.h5"
        self.GAL_EMB_FILE="track2_validate_embedding.h5"
        self.QUE_FILE="data/track2_validate_query_v3.csv"
        self.GAL_FILE="data/track2_validate_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/val_results/"
        self.QUE_VIEW_POINT="val_que_view_point.h5"
        self.GAL_VIEW_POINT="val_view_point.h5"
        self.FLIP = False
        self.CROP = False

class Validate2:
    def __init__(self):
        self.QUE_EMB_FILE="track2_validate_query_embedding.h5"
        self.GAL_EMB_FILE="track2_validate_embedding.h5"
        self.QUE_FILE="data/track2_validate_query_v3.csv"
        self.GAL_FILE="data/track2_validate_new_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/new_validate/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/val_results/"
        self.QUE_VIEW_POINT="val_que_view_point.h5"
        self.GAL_VIEW_POINT="val_view_point.h5"
        self.FLIP = False
        self.CROP = False
 

class ValidateCrop:
    def __init__(self):
        self.QUE_EMB_FILE="track2_validate_crop_query_embedding.h5"
        self.GAL_EMB_FILE="track2_validate_crop_embedding.h5"
        self.QUE_FILE="data/track2_validate.csv"
        self.GAL_FILE="data/track2_validate_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train_crop/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train_crop/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/val_results/"
        self.QUE_VIEW_POINT="val_que_view_point.h5"
        self.GAL_VIEW_POINT="val_view_point.h5"
        self.FLIP = False
        self.CROP = False
        
class Test:
    def __init__(self):
        self.QUE_EMB_FILE="track2_query_embedding.h5"
        self.GAL_EMB_FILE="track2_test_embedding.h5"
        self.QUE_FILE="data/track2_query.csv"
        self.GAL_FILE="data/track2_test_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_test/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/test_results/"
        self.QUE_VIEW_POINT="que_view_point.h5"
        self.GAL_VIEW_POINT="test_view_point.h5"
        self.FLIP = False
        self.CROP = False
        
class CustomEmbed:
    def __init__(self):
        self.QUE_EMB_FILE="track2_custom_query_embedding.h5"
        self.GAL_EMB_FILE="track2_custom_test_embedding.h5"
        self.QUE_FILE="data/track2_test_best_imgs_que.csv"
        self.GAL_FILE="data/track2_test_best_imgs.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_test/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_test/"  
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/custom_results/"
        self.QUE_VIEW_POINT="best_img_que_view_point.h5"
        self.GAL_VIEW_POINT="best_img_view_point.h5"

class QueryQuery:
    def __init__(self):
        self.QUE_EMB_FILE="track2_query_embedding.h5"
        self.GAL_EMB_FILE="track2_query_embedding.h5"
        self.QUE_FILE="data/track2_query.csv"
        self.GAL_FILE="data/track2_query.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/queryquery_results/"
        self.QUE_VIEW_POINT="que_view_point.h5"
        self.GAL_VIEW_POINT="test_view_point.h5"

class QueryTrain:
    def __init__(self):
        self.QUE_EMB_FILE="track2_query_embedding.h5"
        self.GAL_EMB_FILE="track2_train_embedding.h5"
        self.QUE_FILE="data/track2_query.csv"
        self.GAL_FILE="data/track2_train_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/query_train_results/"
        self.QUE_VIEW_POINT="que_view_point.h5"
        self.GAL_VIEW_POINT="test_view_point.h5"
        self.FLIP = False
        self.CROP = False
        
class VehiType0:
    def __init__(self):
        self.QUE_EMB_FILE="track2_type_0_embedding.h5"
        self.GAL_EMB_FILE="track2_type_0_embedding.h5"
        self.QUE_FILE="data/track2_query.csv"
        self.GAL_FILE="data/track2_query.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.RESULTS_ROOT="/home/hthieu/AICityChallenge2019/queryquery_results/"
        
class ViewClassifier:
    def __init__(self):
        self.MODEL_ROOT = "/home/hthieu/AICityChallenge2019/track2_classifier_experiments/pretrained_resnet_v1_101_vehi_views_v2_classify"
        self.NUM_CLASSES = 5
        self.IMG_SIZE = 224
        self.MODEL_NAME = "resnet_v1_101"
        self.info_id = "instance_view"

EXP_ROOT = "/home/hthieu/AICityChallenge2019/track2_experiments/"
class FrontEmbedder:
    EXP_ID = osp.join(EXP_ROOT, "260419_triplet-reid_pre-trained_densenet161_track2_full_front/")
    
class RearEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"260419_triplet-reid_pre-trained_densenet161_track2_full_rear/")
        
class SideEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"260419_triplet-reid_pre-trained_densenet161_track2_full_side/")
    
class BusEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"200419_triplet-reid_pre-trained_densenet161_track2_small_512_vehi_type_2/")
    
class TruckEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"200419_triplet-reid_pre-trained_densenet161_track2_small_512_vehi_type_3/")
    
class Track2Embedder:
    EXP_ID = osp.join(EXP_ROOT,"180419_triplet-reid_pre-trained_densenet161_track2_small_512/") #in use
#     EXP_ID = osp.join(EXP_ROOT,"210319_triplet-reid_pre-trained_densenet161_veri_train_test+track2_full_512")
#     EXP_ID = osp.join(EXP_ROOT, "090519_triplet-reid_pre-trained_densenet161_track2_veri_finetune_done")
    
class Track2VeriEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"190319_triplet-reid_pre-trained_densenet161_veri+small_512/")
    
        
class WheelEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"100519_triplet-reid_wheel/")

class FrontLightEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"100519_triplet-reid_front_light/")

class RearLightEmbedder:
    EXP_ID = osp.join(EXP_ROOT,"100519_triplet-reid_rear_light/")
