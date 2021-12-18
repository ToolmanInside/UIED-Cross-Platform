import cv2
import json
import os
from os.path import join as pjoin
from random import randint as rint
import matplotlib.pyplot as plt
from logzero import logger
import sys
import base64
import numpy as np
sys.path.append("..")

from match_elements.Element import Element
import match_elements.matching as match

class GUIPair:
    def __init__(self, input_figure_1, input_figure_2):

        self.base64_figure_1 = input_figure_1
        self.base64_figure_2 = input_figure_2
        # self.ui_name = ui_name
        self.figure_1 = cv2.imdecode(np.frombuffer(self.parse_base64_img(self.base64_figure_1), np.uint8), cv2.IMREAD_COLOR)
        self.figure_2 = cv2.imdecode(np.frombuffer(self.parse_base64_img(self.base64_figure_2), np.uint8), cv2.IMREAD_COLOR)
        
        self.det_result_imgs_1 = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data_1 = None  # {'elements':[], 'img_shape'}
        self.det_result_imgs_2 = {'text': None, 'non-text': None, 'merge': None}  # image visualization for different stages
        self.det_result_data_2 = None  # {'elements':[], 'img_shape'}

        self.elements_1 = []          # list of Element objects for android UI
        self.elements_2 = []              # list of Element objects for ios UI
        self.elements_mapping = {}          # {'id': Element}
        self.element_matching_pairs = []    # list of matching similar element pairs: [(ele_android, ele_ios)]
        self.reverse_ratio = None

    '''
    *******************************
    *** Detect or Load Elements ***
    *******************************
    '''
    def parse_base64_img(self, basestr):
        img = base64.b64decode(bytes(basestr, encoding='utf-8'))
        return img

    def element_detection(self, is_text=True, is_nontext=True, is_merge=True):
        if is_text:
            import detect_text.text_detection as text
            # from paddleocr import PaddleOCR
            # paddle_cor = PaddleOCR(use_angle_cls=True, lang="ch")
            # self.det_result_imgs_android['text'] = text.text_detection_paddle(self.img_path_android, self.output_dir, paddle_cor=paddle_cor)
            # self.det_result_imgs_ios['text'] = text.text_detection_paddle(self.img_path_ios, self.output_dir, paddle_cor=paddle_cor)
            self.det_result_imgs_1['text'] = text.text_detection_longce(self.base64_figure_1)
            self.det_result_imgs_2['text'] = text.text_detection_longce(self.base64_figure_2)
        if is_nontext:
            import detect_compo.ip_region_proposal as ip
            key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 100, 'merge-contained-ele': False, 'resize_by_height': 900}
            # key_params = {'min-grad': 6, 'ffl-block': 5, 'min-ele-area': 100, 'merge-contained-ele': False}
            self.det_result_imgs_1['non-text'] = ip.compo_detection(self.figure_1, 'data/output', key_params)
            with open("det1.txt", 'w') as f:
                print(self.det_result_imgs_1['non-text'], file = f)
            # logger.debug(f"{self.det_result_imgs_1['non-text']}")
            self.det_result_imgs_2['non-text'] = ip.compo_detection(self.figure_2, 'data/output', key_params)
            # logger.debug(f"{self.det_result_imgs_2['non-text']}")
            
        if is_merge:
            import detect_merge.merge as merge
            # for android GUI
            # compo_path = pjoin(self.output_dir, 'ip', 'A' + str(self.ui_name) + '.json')
            # ocr_path = pjoin(self.output_dir, 'ocr', 'A' + str(self.ui_name) + '.json')
            # logger.debug(f"{self.det_result_imgs_1['text']}")
            self.det_result_imgs_1['merge'], self.det_result_data_1 = merge.merge(self.figure_1, self.det_result_imgs_1['non-text'], \
            self.det_result_imgs_1['text'], merge_root=None, is_remove_bar=True, is_paragraph=False)
            # logger.debug(f"{self.det_result_imgs_1['merge']}")
            with open("merge1.txt", 'w') as f:
                print(self.det_result_imgs_1['non-text'], file = f)
            # for ios GUI
            # compo_path = pjoin(self.output_dir, 'ip', 'I' + str(self.ui_name) + '.json')
            # ocr_path = pjoin(self.output_dir, 'ocr', 'I' + str(self.ui_name) + '.json')
            self.det_result_imgs_2['merge'], self.det_result_data_2 = merge.merge(self.figure_2, self.det_result_imgs_2['non-text'], \
            self.det_result_imgs_2['text'], merge_root=None, is_remove_bar=True, is_paragraph=False)
            # convert elements as Element objects
            self.draw_detection_result()
            self.cvt_elements()

    def load_detection_result(self, data_path_android=None, data_path_ios=None):
        if not data_path_android:
            data_path_android = pjoin(self.output_dir, 'merge', 'A' + self.ui_name + '.json')
        if not data_path_ios:
            data_path_ios = pjoin(self.output_dir, 'merge', 'I' + self.ui_name + '.json')
        self.det_result_data_android = json.load(open(data_path_android))
        self.det_result_data_ios = json.load(open(data_path_ios))
        # convert elements as Element objects
        self.cvt_elements()

    '''
    **************************************
    *** Operations for Element Objects ***
    **************************************
    '''
    def cvt_elements(self):
        '''
        Convert detection result to Element objects
        @ det_result_data: {'elements':[], 'img_shape'}
        '''
        class_map = {'Text': 't', 'Compo': 'c', 'Block': 'b'}
        # logger.debug(f"{self.det_result_data_1['compos']}")
        for i, element in enumerate(self.det_result_data_1['compos']):
            # logger.warning(f"{element['class']}")
            e = Element('a' + str(i) + class_map[element['class']], '1', element['class'], element['position'], self.det_result_data_1['img_shape'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
            e.get_clip(self.figure_1)
            self.elements_1.append(e)
            self.elements_mapping[e.id] = e

        # logger.debug(f"{self.det_result_data_2['compos']}")
        for i, element in enumerate(self.det_result_data_2['compos']):
            e = Element('i' + str(i) + class_map[element['class']], '2', element['class'], element['position'], self.det_result_data_2['img_shape'])
            if element['class'] == 'Text':
                e.text_content = element['text_content']
            e.get_clip(self.figure_2)
            self.elements_2.append(e)
            self.elements_mapping[e.id] = e

    def save_element_clips(self):
        clip_dir = pjoin(self.output_dir, 'clip')
        clip_dir_android = pjoin(clip_dir, 'android')
        clip_dir_ios = pjoin(clip_dir, 'ios')
        os.makedirs(clip_dir, exist_ok=True)
        os.makedirs(clip_dir_android, exist_ok=True)
        os.makedirs(clip_dir_ios, exist_ok=True)

        for element in self.elements_android:
            name = pjoin(clip_dir_android, element.id + '.jpg')
            cv2.imwrite(name, element.clip)
        for element in self.elements_ios:
            name = pjoin(clip_dir_ios, element.id + '.jpg')
            cv2.imwrite(name, element.clip)

    '''
    ******************************
    *** Match Similar Elements ***
    ******************************
    '''
    def match_similar_elements(self, model=None, min_similarity_img=0.75, min_similarity_text=0.8, del_prev = True, img_sim_method='resnet'):
        '''
        @min_similarity_img: similarity threshold for Non-text elements
        @min_similarity_text: similarity threshold for Text elements
        @img_sim_method: the method used to calculate the similarity between two images
            options: 'dhash', 'ssim', 'sift', 'surf'
        '''
        if del_prev:
            self.element_matching_pairs = []
            for ele in self.elements_2 + self.elements_1:
                ele.matched_element = None

        logger.debug("Encoding Elements")
        clips = []
        no_type_1 = 0
        for ele in self.elements_1 + self.elements_2:
            if ele.category == "Compo":
                clips.append(cv2.resize(ele.clip, (32,32)))
                if ele.ui_type == '1':
                    no_type_1 += 1
        encodings = model.predict(np.array(clips))
        encodings = encodings.reshape((encodings.shape[0], -1))
        # tmp_x = [x for x in self.elements_1 if x.category == "Compo"]
        # tmp_y = [x for x in self.elements_2 if x.category == "Compo"]
        # logger.debug(f"Element_android length: {len(tmp_x)}")   
        # logger.debug(f"Element_ios length: {len(tmp_y)}") 
        encodings_1 = encodings[:no_type_1]
        # logger.debug(f"Encoding_1 length: {len(encodings_1)}")
        encodings_2 = encodings[no_type_1:]
        # logger.debug(f"Encoding_2 length: {len(encodings_2)}")
        logger.debug("Encoding Complete")

        logger.debug("Start Matching")
        mark = np.full(len(self.elements_2), False)
        n_compos = 0
        n_texts = 0
        for i, ele_a in enumerate(self.elements_1):
            compo_list = list()
            text_list = list()
            for j, ele_b in enumerate(self.elements_2):
                # only match elements in the same category
                if ele_b.matched_element is not None or ele_a.category != ele_b.category:
                    continue
                height_ratio = max(ele_a.height, ele_b.height) / min(ele_a.height, ele_b.height)
                width_ratio = max(ele_a.width, ele_b.width) / min(ele_a.width, ele_b.width)
                aspect_ratio = max(ele_a.aspect_ratio, ele_b.aspect_ratio) / min(ele_a.aspect_ratio, ele_b.aspect_ratio)
                if mark[j] or height_ratio > 1.5 or width_ratio > 1.5 or aspect_ratio > 1.5:
                    continue
                # use different method to calc the similarity of of images and texts
                if ele_a.category == 'Compo':
                    if j >= len(encodings_2):
                    # logger.warning("Index out of range")
                        continue
                    # match non-text clip through image similarity
                    # compo_similarity = match.image_similarity(ele_a.clip, ele_b.clip, model = model, method=img_sim_method)
                    # logger.warning((i, j))
                    compo_similarity = match.resnet_similarity(encodings_1[i], encodings_2[j])
                    if compo_similarity > min_similarity_img:
                        n_compos += 1
                        compo_list.append((ele_a, ele_b, j, round(compo_similarity, 3)))

                elif ele_a.category == 'Text':
                    # match text by through string similarity
                    text_similarity = match.text_similarity(ele_a.text_content, ele_b.text_content)
                    if text_similarity > min_similarity_text:
                        n_texts += 1
                        text_list.append((ele_a, ele_b, j, round(text_similarity, 3)))

            # compo_list: [(ele_a, ele_b, j, round(compo_similarity, 3)), ...]
            # text_list: [(ele_a, ele_b, j, round(text_similarity, 3)), ...]
            compo_list.sort(key = lambda x: x[3], reverse = True)
            text_list.sort(key = lambda x: x[3], reverse = True)
            if len(compo_list) > 0:
                self.element_matching_pairs.append((compo_list[0][0], compo_list[0][1]))
                compo_list[0][0].matched_element = compo_list[0][1]
                compo_list[0][1].matched_element = compo_list[0][0]
                mark[compo_list[0][2]] = True
            if len(text_list) > 0:
                self.element_matching_pairs.append((text_list[0][0], text_list[0][1]))
                text_list[0][0].matched_element = text_list[0][1]
                text_list[0][1].matched_element = text_list[0][0]
                mark[text_list[0][2]] = True

        logger.debug('[Similar Elements Matching] Method:%s Paired Text:%d, Paired Compos:%d' % (img_sim_method, n_texts, n_compos))

    # def match_similar_elements(self, model = None, min_similarity_img=0.7, min_similarity_text=0.8):
    #     logger.debug("[Matching Similar Elements]")
    #     for ele_a in self.elements_1:
    #         for ele_b in self.elements_2:
    #             # only match elements in the same category
    #             if ele_b.matched_element is not None or ele_a.category != ele_b.category:
    #                 continue
    #             # use different method to calc the similarity of of images and texts
    #             if ele_a.category == 'Compo':
    #                 # match non-text clip through image similarity
    #                 compo_similarity = match.image_similarity(ele_a.clip, ele_b.clip, model=model, method='resnet')
    #                 if compo_similarity > min_similarity_img:
    #                     self.element_matching_pairs.append((ele_a, ele_b))
    #                     ele_a.matched_element = ele_b
    #                     ele_b.matched_element = ele_a
    #             elif ele_a.category == 'Text':
    #                 # match text by through string similarity
    #                 text_similarity = match.text_similarity(ele_a.text_content, ele_b.text_content)
    #                 if text_similarity > min_similarity_text:
    #                     self.element_matching_pairs.append((ele_a, ele_b))
    #                     ele_a.matched_element = ele_b
    #                     ele_b.matched_element = ele_a
    #     logger.debug("[Matching Similar Elements Complete]")

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_detection_result(self):
        if self.det_result_imgs_android['merge'] is not None:
            plt.imshow(cv2.resize(self.det_result_imgs_1['merge'], (int(self.figure_1.shape[1] * (800 / self.figure_1.shape[0])), 800)))
            plt.imshow(cv2.resize(self.det_result_imgs_2['merge'], (int(self.figure_2.shape[1] * (800 / self.figure_2.shape[0])), 800)))
        elif self.det_result_data_android is not None:
            self.draw_detection_result()
            plt.imshow(cv2.resize(self.det_result_imgs_1['merge'], (int(self.figure_1.shape[1] * (800 / self.figure_1.shape[0])), 800)))
            plt.imshow(cv2.resize(self.det_result_imgs_2['merge'], (int(self.figure_2.shape[1] * (800 / self.figure_2.shape[0])), 800)))
        else:
            print('No detection result, run element_detection() or load_detection_result() first')
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    def draw_detection_result(self, show_id=True):
        '''
        Draw detected elements based on det_result_data
        '''
        color_map = {'Compo': (0,255,0), 'Text': (0,0,255)}
        ratio = self.figure_1.shape[0] / self.det_result_data_1['img_shape'][0]
        board = self.figure_1.copy()
        # logger.debug(f"Reverse Ratio: {ratio}")
        self.reverse_ratio = ratio
        # logger.debug(f"{self.elements_1}")
        for i, element in enumerate(self.elements_1):
            logger.debug("draw_1")
            element.draw_element(board, ratio, color_map[element.category], show_id=show_id)
        self.det_result_imgs_1['merge'] = board.copy()
        cv2.imwrite("img_1.png", self.det_result_imgs_1['merge'])

        ratio = self.figure_2.shape[0] / self.det_result_data_2['img_shape'][0]
        board = self.figure_2.copy()
        # logger.debug(f"{self.elements_2}")
        for i, element in enumerate(self.elements_2):
            logger.debug("draw_2")
            element.draw_element(board, ratio, color_map[element.category], show_id=show_id)
        self.det_result_imgs_2['merge'] = board.copy()
        cv2.imwrite("img_2.png", self.det_result_imgs_2['merge'])

    def visualize_matched_element_pairs(self, line=-1):
        board_android = self.img_android.copy()
        board_ios = self.img_ios.copy()
        for pair in self.element_matching_pairs:
            color = (rint(0,255), rint(0,255), rint(0,255))
            pair[0].draw_element(board_android, color=color, line=line, show_id=False)
            pair[1].draw_element(board_ios, color=color, line=line, show_id=False)
        # logger.debug(board_android)
        plt.imshow(cv2.resize(board_android, (int(board_android.shape[1] * (800 / board_android.shape[0])), 800))) # android
        plt.imshow(cv2.resize(board_ios, (int(board_ios.shape[1] * (800 / board_ios.shape[0])), 800))) # ios
        # cv2.imshow('android', cv2.resize(board_android, (int(board_android.shape[1] * (800 / board_android.shape[0])), 800)))
        # cv2.imshow('ios', cv2.resize(board_ios, (int(board_ios.shape[1] * (800 / board_ios.shape[0])), 800)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()
