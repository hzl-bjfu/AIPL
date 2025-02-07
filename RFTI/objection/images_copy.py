import shutil

# test_1000_cfy_path = './data/test_1000_cfy'
test_negative_pathList = './test_negative_hard_samples.txt'
with open(test_negative_pathList, 'r') as f:
    test_negative_imageName = [x.strip().split(',,,')[0] for x in f]
for target in test_negative_imageName:
    # shutil.copy('F:\Dataset\ACCV_Datesets/test/' + target, './data/prec_negative_imgs/')
    shutil.copy(target, './data/prec_negative_imgs/')





# shutil.copy('F:\Dataset\ACCV_Datesets/test/' + target, './data/test')