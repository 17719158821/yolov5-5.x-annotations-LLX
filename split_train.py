import pandas as pd
import os
import numpy as np

def read_patient_name(dir_path):
    file_list = []
    for file in os.listdir(dir_path):
        file_list.append(file)
    return file_list

def read_patient_name2(dir_path):
    file_list = []
    for file in os.listdir(dir_path):
        file_list.append(file)
    final_file_list=list(set(file_list))
    return final_file_list

# 实验n折交叉验证数据划分
def cross_n_fold(data, data_nums, start_index, end_index):
    df_test = data[start_index * data_nums:end_index * data_nums]  # 数据刚好可以做n折交叉验证。
    df_test_index = list(df_test.index)
    df_test_flag = data.index.isin(df_test_index)  # 都转换为list来判定成员资格
    diff_flag = [not f for f in df_test_flag]  # 不是df_test_flag里面的索引的索引记为df_train_index的索引集合
    df_train = data[diff_flag]
    return df_train, df_test

if __name__ == '__main__':

    # data_dir = r'G:\YOLO_data\orig_image_and_txt\images\train'
    # data_dir1=r'G:\YOLO_data\orig_image_and_txt\images\val'
    # save_dir = r'G:\YOLO_data\orig_image_and_txt\dataset_split'
    # data_list1= np.array(read_patient_name2(data_dir))
    # data_list2 = np.array(read_patient_name2(data_dir1))
    # data_list=np.append(data_list1,data_list2)
    data_dir = r"E:\2_dataset\YoloDataSet\img"
    data_list = np.array(read_patient_name2(data_dir))
    save_dir = r"E:\2_dataset\YoloDataSet\data_split"
    folds = 5
    data_nums = int(data_list.shape[0] / folds)  # 对轻症患者数据实现N折交叉验证划分
    # 10次N折交叉验证的代码
    for folder in range(1, 11):
        folder_name = 'ImageSets' + str('%02d' % folder)
        folder_save_path = os.path.join(save_dir, folder_name)
        if not os.path.exists(folder_save_path):
            os.mkdir(folder_save_path)
        #随机抽样
        all_data = pd.DataFrame(data_list).sample(frac=1)
        #
        for i in range(1, folds + 1):
            df_train, df_vaild = cross_n_fold(all_data, data_nums, i - 1, i)  # 起始和尾部索引
            np.savetxt('{}/train{}.txt'.format(folder_save_path, str('%02d' % i)), df_train,
                       delimiter=" ", fmt='%s')
            np.savetxt('{}/valid{}.txt'.format(folder_save_path, str('%02d' % i)), df_vaild, delimiter=" ",
                       fmt='%s')
        print(folder_name + '_DONE')