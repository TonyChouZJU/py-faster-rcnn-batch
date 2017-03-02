#------------------------------------------------------------
#Tools for get sysnets of the full dataset
#written by wl-algorithm
#------------------------------------------------------------
import json
import os
import xml.etree.ElementTree as ET

def parse_args():
    """
    parse input arguments
    """
    parser = argparse.ArgumentParser(description='Pre-process the xml dataset')
    parser.add_argument('--xml_json', dest='xml_json',
                        help='xml json to be preprocessed',
                        default=None, type=str)
    parser.add_argument('--output', dest='output',
                        help='output dir to save sysnet',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_sysnets_set(xml_list):
    sysnets = set()
    for xml_file in xml_list:
        tree = ET.parse(xml_file)
        objs = tree.findall('object')
        for ix,obj in enumerat(objs):
            objname = obj.find('name').text.lower().strip()
            sysnets.add(objname)

def get_xml_list(xml_dir_list):
    xml_nums = 0
    xml_list = []
    for xml_dir in xml_dir_list:
        this_xml_list = []
        for xml_file in os.listdir(xml_dir):
            postfix = xml_file.split('.')[-1]
            if not postfix == 'xml':
                continue
            this_xml_list.append(xml_file)
            xml_nums +=1
        xml_list += this_xml_list


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    
    xml_dir_json = json.load(args.xml_json)
    xml_list = get_xml_list(xml_dir_json['xml_dir_list'])
    sysnets = get_sysnets_set(xml_list)
    sysnets_json = {'sysnets':sysnets}
    with open(os.path.join(args.output,'sysnets.json'),'w+') as f:
        json.dumps(sysnets_json,f)
