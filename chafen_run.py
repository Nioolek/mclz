import os
import shutil

import cv2
import numpy as np
import pickle
import mmcv

# video_dir = r'D:\aliyun\video'
video_dir = r'/data/gulingrui/code/mclz/data/video'

label = """ele_0d046ae1124f7c2eedaec936eb3f01cc,0
ele_1a5466c642f33c15946052526100a448,1
ele_1d93a49ef10b321c8aba4fe2a242552b,0
ele_1fc64e89e70fe09b32ec4b1105dee329,0
ele_2c9ec3db67fadfc83f99ba0aa4b8b8be,0
ele_2e646d92f7fe8dcb52fefd454b379a3a,0
ele_2f51638441fbbf6202b3da8581b1f4f6,0
ele_3cc35ec55bb962f1c2220873e200d5b5,0
ele_3ff6b73c71fc992518fb68363afb53dc,0
ele_4f8aa6cc35fecc4dc7991288ef1f7bd6,0
ele_06be12e8c6c4ca9ecb83fa5e1f0cc5e2,0
ele_6c9c0881eaf16b4e80735afe69e90131,1
ele_6e531165cf18528b59a9ae62629d5b42,1
ele_6fb9024282e67b3cc368f4ec01ba594b,1
ele_7abae06a45c2d9171677a133d5ff7ed3,0
ele_08c17569367ca63e2b66f936fbdfd638,1
ele_09c6157bf89d1439207bca33b2c517bf,1
ele_9c2897e0869d7705457f26233d19a678,1
ele_12eeecbe7c19f323296d4edbcb7ba8dc,0
ele_25cacd19ffa6e9e3cc78f8cdcfbe9490,0
ele_47c8c1de09a9bd239901d0bbf5641048,0
ele_50ed6dbaa696b168b442bc01562e734c,1
ele_60b55ddf9da54fc102160d209030e6b8,0
ele_67f040b9c0815690a85fca2dfb6e3ce0,1
ele_78a543a9845bb856666444c061d3e562,0
ele_95ccc739208f174a3a52be276c64b2ae,0
ele_97a3d855ae65bee568373e84555ac61e,0
ele_417d8aab6f223be87d5a1d0e120cc554,0
ele_1169eb35349239475ee4660d02455b81,1
ele_2904e42d9d8352b96b81d0e416fe43bf,1
ele_3356cc0bf1530185f4357b7ede315ef7,0
ele_52446a577d115a411ababf1b2df27e9e,0
ele_119033b1a30eab893f4004047df6a0a5,0
ele_6906129f93cdcbf02fc360fe0a02d277,0
ele_20868775fca1fc5e0d58c46cf38799b8,0
ele_32849114247dcf9619a02b9644b5d274,0
ele_a0b1a6c552a1fe68e6e336af724518fd,0
ele_a707e6d5e35589b9f004e82a5ff6aa6d,1
ele_a84856960381c012756cca8b33f4abe9,0
ele_aa26e62901aa915663886c296e90f460,0
ele_adc9ef47a94205ad6e6bea3b0f08433a,0
ele_af717d6ea74519facf092ab2995dcad9,0
ele_b5c577ba8fef48d5d9e1d73b757f1cc9,0
ele_b8a20cbfd25abff7fd30ae4ef5c715a5,1
ele_b432d23e67a6cbbd564da4df38a42cce,0
ele_bc26e30eeea4a7a43d4cc3907fda3a70,0
ele_c0c20d812a5d0494e689c088ae4f0696,0
ele_c1eb331bfaa9cd3368984cd3c1c478e7,0
ele_d9a990c1c407d83442232971b58fab5d,0
ele_da8c8d304b2e8a4a2f6169487781f5f2,0
ele_dddc3cf237afefbad6c171301f37436d,0
ele_df1951e3ae85e55aeb6c774f3ff7dfb0,0
ele_dfa10ee47b7705c786947aa90479cbb9,1
ele_e9ed30fcb9a96d5cf84f414c1b8e0f2c,0
ele_ef1129355837032f6ce5fe076339d2a5,0
ele_f7a645cd8dd9266480f006e71507c876,0
ele_f253bc5e753af623420a507265f8595e,0
ele_fc5d7be39e0af690622d772fde5448c0,1
""".strip().split('\n')
label_dict = {}
for i in label:
    video_name, label = i.split(',')
    label_dict[video_name] = int(label)


def is_grey_scale(img):
    if len(img.shape) < 3: return True
    if img.shape[2] == 1: return True
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all(): return True
    return False


def panduan_video():
    grey_video_list = []
    for i in os.listdir(video_dir):
        if i.endswith('.ts'):
            # if i == 'ele_08c17569367ca63e2b66f936fbdfd638.ts':
            #
            #     print(i)
            video_path = os.path.join(video_dir, i)

            # 读取视频
            cap = cv2.VideoCapture(video_path)

            # 判断视频是不是黑白
            ret, frame1 = cap.read()
            ret, frame2 = cap.read()
            if is_grey_scale(frame1) and is_grey_scale(frame2):
                print(i, 'is grey')
                grey_video_list.append(i)
    return grey_video_list


def chafen(video_path, show=False):
    res = []

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))

    cap = cv2.VideoCapture(video_path)

    # 获取三个连续的帧
    frame1 = cap.read()[1][:, :, 0]
    frame2 = cap.read()[1][:, :, 0]
    frame3 = cap.read()[1][:, :, 0]

    v_h, v_w = frame1.shape[:2]
    if v_h > 512:
        # 蒙掉时间区域
        mask_flag = True
        mask_h = int(v_h * 0.08)
    else:
        mask_flag = False
        mask_h = 0

    if mask_flag:
        frame1[:mask_h, :] = 0
        frame2[:mask_h, :] = 0
        frame3[:mask_h, :] = 0

    while True:
        # 计算帧间差异
        diff1 = cv2.absdiff(frame1, frame2)
        diff2 = cv2.absdiff(frame2, frame3)

        # 对差异进行二值化处理
        diff = cv2.bitwise_and(diff1, diff2)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 对二值化后的图像进行膨胀, 并且使用原型的kernel
        diff = cv2.dilate(diff, kernel)

        # 将diff转成unit8
        diff = diff.astype(np.uint8)
        print(diff.shape)

        # 获取到diff中的连通域
        contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if show:
            show_img = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)

        fil_contours = []

        # 过滤掉面积小于100的连通域
        for c in contours:

            if cv2.contourArea(c) < 150:
                continue
            else:
                print(cv2.contourArea(c))
                # 计算连通域的外接矩形框
                (x, y, w, h) = cv2.boundingRect(c)
                # 在原始图像中绘制出外接矩形框
                fil_contours.append(c)

                if show:
                    cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        res.append(fil_contours)

        if show:
            # 将show_img resize到长边为512，以进行展示
            w, h = show_img.shape[:2]
            resize_w = 512
            resize_h = int(512 * (h / w))

            show_img = cv2.resize(show_img, (resize_h, resize_w))

            # # 显示图像
            # cv2.imshow("Moving Object Detection", show_img)

        # 准备下一轮迭代，更新帧
        frame1 = frame2
        frame2 = frame3
        ret, frame3 = cap.read()
        # 判断 frame3 是不是None
        if not ret:
            break
        frame3 = frame3[:, :, 0]
        if mask_flag:
            frame3[:mask_h, :] = 0
    return res


def find_mouse(video_path, chafen_res):
    """
    返回1有老鼠，0没有老鼠
    :param video_path:
    :param chafen_res:
    :return:
    """

    length_list = np.array([len(i) for i in chafen_res])
    length_bool = (length_list > 0).astype(int)
    filter = np.array([1, 1, 1, 1, 1, 1, 1])
    res = np.convolve(length_bool, filter, 'valid')

    # print(length_bool)
    # print(res)
    # 现在是要把预测为1的当中，很多转为0
    if np.max(res) > 3:
        if length_list.mean() > 4:
            return 0
        if length_bool.mean() > 0.90:
            return 0
        return 1
    else:
        return 0




if __name__ == '__main__':
    # 获取黑白视频列表
    grey_video_dir = 'grey_video'
    if not os.path.exists('grey_video_list.txt'):
        grey_video_list = panduan_video()
        print(len(grey_video_list))
        with open('grey_video_list.txt', 'w') as f:
            for i in grey_video_list:
                f.write(i + '\n')

        if not os.path.exists(grey_video_dir):
            os.mkdir(grey_video_dir)
        for i in grey_video_list:
            # print(i)
            video_path = os.path.join(video_dir, i)
            video = mmcv.VideoReader(video_path)
            video.cvt2frames(os.path.join(grey_video_dir, os.path.splitext(i)[0]),
                             )

    else:
        with open('grey_video_list.txt', 'r') as f:
            grey_video_list = [i.strip() for i in f.readlines()]

    # 获取差分结果
    if os.path.exists('video_chafen_res.pkl'):
        with open('video_chafen_res.pkl', 'rb') as f:
            video_chafen_res = pickle.load(f)
    else:
        video_chafen_res = []
        for i in grey_video_list:
            # print(i)
            video_path = os.path.join(video_dir, i)
            res = chafen(video_path)
            video_chafen_res.append(res)

        with open('video_chafen_res.pkl', 'wb') as f:
            pickle.dump(video_chafen_res, f)

    # 有老鼠的示例
    # video_ind = grey_video_list.index('ele_1a5466c642f33c15946052526100a448.ts')
    # print(video_chafen_res[video_ind])

    mouse_predict_list = []

    matrix = np.zeros((2, 2))
    if os.path.exists('save_00/'):
        shutil.rmtree('save_00/')
        shutil.rmtree('save_01/')
        shutil.rmtree('save_11/')
        shutil.rmtree('save_10/')

    for i in range(len(grey_video_list)):
        video_path = os.path.join(video_dir, grey_video_list[i])
        # print(grey_video_list[i])
        predict_res = find_mouse(video_path, video_chafen_res[i])
        if predict_res:
            mouse_predict_list.append(grey_video_list[i])

        label_res = label_dict[grey_video_list[i].rstrip('.ts')]
        matrix[label_res, predict_res] += 1

        video_img_dir = os.path.join(grey_video_dir, os.path.splitext(grey_video_list[i])[0])
        cp_dir = 'save_%s%s/' % (predict_res, label_res)
        if not os.path.exists(cp_dir):
            os.mkdir(cp_dir)
        shutil.copytree(video_img_dir, os.path.join(cp_dir, os.path.splitext(grey_video_list[i])[0]))

    print('acc:', np.sum(np.diag(matrix)) / np.sum(matrix))
    # tn ,tp
    # fn ,tp
    # recall = tp / (tp + fn)  召回出的老鼠比例
    print('recall:', matrix[1, 1] / np.sum(matrix[1, :]))
    # precision = tp / (tp + fp)  预测出的老鼠中真正有老鼠的比例
    print('precision', matrix[1, 1] / np.sum(matrix[:, 1]))

    print('mouse_predict_list', mouse_predict_list)

    # 写入结果
    org_csv_path = 'result_video3.csv'
    new_csv_path = 'result_video_mouse.csv'
    with open(org_csv_path, 'r') as f:
        data = f.read().rstrip().split('\n')[1:]
    with open(new_csv_path, 'w') as f:
        f.write('filename,result\n')
        for i in data:
            video_name, label = i.split(',')
            if video_name in mouse_predict_list:
                # 按位或
                label = int(label) | 4
                f.write('%s,%s\n' % (video_name, label))
            else:
                f.write(i + '\n')
