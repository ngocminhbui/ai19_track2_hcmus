class Validate:
    def __init__(self):
        self.QUE_EMB_FILE="track2_validate_query_embedding.h5"
        self.GAL_EMB_FILE="track2_validate_embedding.h5"
        self.QUE_FILE="data/track2_validate_query_v3.csv"
        self.GAL_FILE="data/track2_validate_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_train/"
        
class Test:
    def __init__(self):
        self.QUE_EMB_FILE="track2_query_embedding.h5"
        self.GAL_EMB_FILE="track2_test_embedding.h5"
        self.QUE_FILE="data/track2_query.csv"
        self.GAL_FILE="data/track2_test_v3.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_test/"
        
class CustomEmbed:
    def __init__(self):
        self.QUE_EMB_FILE="track2_query_embedding.h5"
        self.GAL_EMB_FILE="track2_test_embedding.h5"
        self.QUE_FILE="data/track2_test_best_imgs_que.csv"
        self.GAL_FILE="data/track2_test_best_imgs.csv"
        self.QUE_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_query/"
        self.GAL_IMG_ROOT="/home/hthieu/AICityChallenge2019/data/Track2Data/image_test/"  

