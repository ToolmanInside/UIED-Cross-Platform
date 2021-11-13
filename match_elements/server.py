from flask import Flask, request
import json
import base64
from gui_matching import GUIPair

app = Flask("GUI-Matching")

def parse_base64_img(basestr):
    img = base64.b64decode(basestr)
    return img

def result_processing(result):
    ele_pairs = {
        "ele_pairs": list()
    }
    for pair in result:
        pair_dict = dict()
        ele_1 = pair[0]
        ele_2 = pair[1]
        ele_1_info = dict()
        ele_1_info['id'] = ele_1.id
        ele_1_info['category'] = ele_1.category
        ele_1_info['col_min'] = ele_1.col_min
        ele_1_info['row_min'] = ele_1.row_min
        ele_1_info['col_max'] = ele_1.col_max
        ele_1_info['row_max'] = ele_1.row_max
        ele_1_info['width'] = ele_1.width
        ele_1_info['height'] = ele_1.height
        ele_1_info['area'] = ele_1.area
        ele_1_info['text_content'] = ele_1.text_content
        ele_2_info = dict()
        ele_2_info['id'] = ele_2.id
        ele_2_info['category'] = ele_2.category
        ele_2_info['col_min'] = ele_2.col_min
        ele_2_info['row_min'] = ele_2.row_min
        ele_2_info['col_max'] = ele_2.col_max
        ele_2_info['row_max'] = ele_2.row_max
        ele_2_info['width'] = ele_2.width
        ele_2_info['height'] = ele_2.height
        ele_2_info['area'] = ele_2.area
        ele_2_info['text_content'] = ele_2.text_content
        pair_dict['ele_1'] = ele_1_info
        pair_dict['ele_2'] = ele_2_info
        ele_pairs['ele_pairs'].append(pair_dict)
    return ele_pairs

@app.route('/guimatching', methods=['POST'])
def matching():
    request_data = json.loads(request.data)
    figure_1_base64 = request_data['figure1']
    figure_2_base64 = request_data['figure2']
    # figure_1 = parse_base64_img(figure_1_base64)
    # figure_2 = parse_base64_img(figure_2_base64)
    gui_pair = GUIPair(figure_1_base64, figure_2_base64)
    gui_pair.element_detection(True, True, True)
    gui_pair.match_similar_elements()
    result = result_processing(gui_pair.element_matching_pairs)
    return json.dumps(result).encode('utf-8').decode("unicode-escape")

def main():
    app.run("192.168.50.94", port = 6785)

if __name__ == "__main__":
    main()
