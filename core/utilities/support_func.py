import re
import pandas as pd
import json
import os
import random


def preprocess(raw_text):
    raw_text = raw_text.strip()
    raw_text = raw_text.replace("\xa0", " ")
    raw_text = raw_text.replace("\n", " ")
    raw_text = raw_text.replace("\t", " ")
    raw_text = raw_text.replace("\r", " ")
    # raw_text = raw_text.replace(";", "")
    # raw_text = raw_text.replace(".", "")
    # raw_text = raw_text.replace(":", "")
    return " ".join(raw_text.split())

def list_preprocess(raw_list):
    output = []
    for i in range(len(raw_list)):
        if raw_list[i] == "":
            continue
        output.append(preprocess(raw_list[i]))
    return output

def html_preprocess(raw_text):
    raw_text = raw_text.strip()
    raw_text = raw_text.replace("\xa0", " ")
    raw_text = raw_text.replace("\n", " ")
    raw_text = raw_text.replace("\t", " ")
    raw_text = raw_text.replace("\r", " ")
    return raw_text


def html_list_preprocess(raw_list):
    output = []
    for i in range(len(raw_list)):
        if raw_list[i] == "":
            continue
        output.append(preprocess(raw_list[i]))
    return output


def read_json(input_path):
    """
    Read json file from input path
    :param input_path: path to json file
    :return: list of json object
    """
    # print(input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(output_path, data):
    """
    Write json file to output path
    :param output_path: path to json file
    :param data: list of json object
    :return:
    """
    file_exists = os.path.exists(output_path)
    file_exists = False

    if not file_exists:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        # print("File created: " + output_path)
    else:
        print("File already exists!")


def write_csv(output_path, data):
    """
    Write csv file to output path
    :param output_path: path to csv file
    :param data: list of json object
    :return:
    """
    # file_exists = os.path.exists(output_path)

    file_exists = False

    if not file_exists:
        data.to_csv(output_path, index=False)
        # print("File created: " + output_path)
    else:
        print("File already exists!")


def get_list(input_path):
    list_path = os.listdir(input_path)
    for i in range(len(list_path)):
        list_path[i] = os.path.join(input_path, list_path[i])
    return list_path


def random_num(end_ran):
    return random.randint(0, end_ran - 1)

def random_list(input_list, num):
    trial_time = 0
    output_list = []
    while len(output_list) < num:
        ran = random_num(len(input_list))
        if ran not in output_list:
            output_list.append(ran)
        trial_time += 1
        if trial_time == 1000:
            break
    return output_list


def concat_csv(csv_list):
    df = pd.concat(csv_list, ignore_index=True)
    return df


def read_json(input_path):
    """
    Read json file from input path
    :param input_path: path to json file
    :return: list of json object
    """
    # print(input_path)
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def write_json(output_path, data):
    """
    Write json file to output path
    :param output_path: path to json file
    :param data: list of json object
    :return:
    """
    print("Writing json file to: " + output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def pre_processing(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())


def min_max_scale(data, esp=1e-10):
    local_max = max(data)
    local_min = min(data)
    output_list = []
    for local_data in data:
        try:
            output_list.append((local_data - local_min) / (local_max - local_min + esp))
        except:
            output_list.append(-1)

    return output_list