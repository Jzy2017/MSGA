#training dataset path
# TRAIN_FOLDER = '../../ImageAlignment/output/training'
TRAIN_FOLDER = '/home/jiangzhiying/zzx/TIP2021_per_vis/ImageAlignment/output/training_vis_pre'
# TRAIN_FOLDER='/home/zhangzx/experiment/experiment_data/vis'
#testing dataset path
TEST_FOLDER = '/home/jiangzhiying/zzx//TIP2021_per_vis/ImageAlignment/output/testing_new'
# TEST_FOLDER='/home/zhangzx/experiment/experiment_data/vis'
                        
#GPU index
GPU = '0'

#batch size for training
TRAIN_BATCH_SIZE = 8

#batch size for testing
TEST_BATCH_SIZE = 1

#num of iters
ITERATIONS = 200000

# checkpoints path
# SNAPSHOT_DIR = "./checkpoints"
SNAPSHOT_DIR = "snapshot"

#sumary path
SUMMARY_DIR = "./summary"
