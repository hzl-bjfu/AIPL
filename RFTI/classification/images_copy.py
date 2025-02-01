import shutil

# test_1000_cfy_path = './data/test_1000_cfy'
test_negative_pathList = './data/allCamera/test_negative_allCamera_focal_50_0.5_1000.txt'
with open(test_negative_pathList, 'r') as f:
    # test_negative_imageName = [x.strip().split(',,,')[0] for x in f if x.strip().split(',,,')[-1]=='1']
    test_negative_imageName = [x.strip().split(',,,')[0] for x in f]
    # test_negative_imageName = [x.strip() for x in f]
for target in test_negative_imageName:
    # shutil.copy(target, './data/test_original - 副本')
    shutil.copy(target, './data/allCamera/negative_images')
    # shutil.copyfile(target.split('_')[0]+'.JPG',
    #                 'D:\ScienceResearch\SoftwareProgram\Data\dataset/snow leopard\original/randomImg/'+
    #                 target.split('_')[-1].split('.')[0] + '_' +target.split('\\')[-1])





# shutil.copy('F:\CCV_Datesets/test/' + target, './data/test')Dataset\A