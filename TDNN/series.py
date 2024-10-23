import os

# os.system('python train_tdnn.py')
# os.rename('best_network.pth', 'crossS1_BN_4.pth')
# os.system('python train_plda.py')


# print('BN-----------------------------------------------------')
# for i in range(3):
#     num = i+1
#     os.system('python train_tdnn.py --tdnn_method tdnn_BN')
#     tem_name = 'S2_PCEN_' + str(num)
#     os.rename('best_network.pth', tem_name + '.pth')
#     os.system('python train_plda.py --model_name ' + tem_name)
#
# print('IFN----------------------------------------------------')
# for i in range(1):
#     num = i+2
#     os.system('python train_tdnn.py --tdnn_method tdnn_IFN')
#     tem_name = 'S2_IFN_' + str(num)
#     os.rename('best_network.pth', tem_name + '.pth')
#     os.system('python train_plda.py --model_name ' + tem_name)
#
# print('LSTM--------------------------------------------------')
# for i in range(1):
#     num = i+2
#     os.system('python train_tdnn.py --tdnn_method tdnn_LSTM')
#     tem_name = 'S2_LSTM_' + str(num)
#     os.rename('best_network.pth', tem_name + '.pth')
#     os.system('python train_plda.py --model_name ' + tem_name)
#
print('both--------------------------------------------------')
for i in range(1):
    num = i+1
    os.system('python train_tdnn.py --tdnn_method tdnn_both')
    tem_name = 'D1_both' + str(num)
    os.rename('best_network.pth', tem_name + '.pth')
    os.system('python train_plda.py --model_name ' + tem_name)
#
# print('GW  --------------------------------------------------')
# print('GW')
# for i in range(3):
#     num = i+1
#     os.system('python train_tdnn.py --tdnn_method tdnn_GW')
#     tem_name = 'S1_GW_' + str(num)
#     os.rename('best_network.pth', tem_name + '.pth')
#     os.system('python train_plda.py --model_name ' + tem_name)
#
# print('TN  --------------------------------------------------')
# print('TN')
# for i in range(3):
#     num = i+1
#     os.system('python train_tdnn.py --tdnn_method tdnn_TN')
#     tem_name = 'S1_TN_' + str(num)
#     os.rename('best_network.pth', tem_name + '.pth')
#     os.system('python train_plda.py --model_name ' + tem_name)

