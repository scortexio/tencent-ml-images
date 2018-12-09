#!/usr/bin/env python
"""
Tencent is pleased to support the open source community by making Tencent ML-Images available.
Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
Licensed under the BSD 3-Clause License (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
https://opensource.org/licenses/BSD-3-Clause
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
"""

import os
import sys
import argparse
import threading,signal
import time
import uuid
import socket
socket.setdefaulttimeout(10.0)
import urllib2 as ul
from urllib2 import URLError, HTTPError

debug = False


def resize_if_needed(mem_img, max_size):
    import numpy as np
    import cv2
    buf = np.fromstring(mem_img, dtype='uint8')
    decoded = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        # Could not decode
        return None
    height,width,_ = decoded.shape
    # Do we have to resize?
    if width <= max_size and height <= max_size:
        return mem_img
    # We have to resize. How?
    ratio = float(width) / height
    dim= (max_size, int(max_size / ratio)) if ratio >= 1.0 else (int(max_size * ratio), max_size)

    resized = cv2.resize(decoded, dim, interpolation = cv2.INTER_AREA)
    return resized

def downloadImg(i, url_file_mem,start, end, url_list, save_dir, max_size):
    def do_log(status, url, save_name, labels):
        ''' status => 'OK', NOK-<http code != 200>, NOK-<content type not image>, 'NOK-DECODE'
        '''
        im_file_Record.write('{}\t{}\t{}\t{}\n'.format(status, url, save_name, labels))

    import hashlib
    import cv2

    print('Thread {} starting, set to download from {} to {}'.format(i, start, end))
    global record, count, count_invalid, count_already_done, is_exit
    for line in url_file_mem[start:end]:
        sp = line.rstrip('\n').split('\t')
        url = sp[0]
        # We save the image as the md5 of the url
        im_name = '{}.jpg'.format(hashlib.md5(url).hexdigest())
        save_path_name = os.path.join(save_dir, im_name)
        # Remember labels as we keep them together with the file name
        labels = sp[1:]

        if os.path.isfile(save_path_name):
            count_already_done += 1
            if debug: print('T', i, count_already_done, "already done" , save_path_name)
            continue

        def rec(status):
            global record, count_invalid
            if status != 'OK':
                count_invalid += 1
                if debug: print('Failed {}: {}'.format(url, status))
            else:
                record += 1
            do_log(status, url, im_name, labels)

        try:
            if debug: print('T', i, "fetching", url)
            # Fetch resource
            r = ul.urlopen(url)
            code = r.getcode()
            if code is not 200:
                # failed!
                rec('NOK-{}'.format(code))
                continue
            # Check content type for image
            ct = r.info().typeheader
            if not ct.startswith('image/'):
                rec('NOK-{}'.format(ct))
                continue
            # Read it!
            img = r.read()
            # Resize
            img = resize_if_needed(img, max_size)
            if img is None:
                # Decoding failed
                rec('NOK-DECODE')
                continue
            # Write file with a unique name
            id = uuid.uuid4()
            cv2.imwrite(save_path_name, img)
            rec('OK')
        except HTTPError as e:
            rec('NOK-{}'.format(e.code))
        except URLError as e:
            rec('NOK-{}'.format(e.reason))
        except Exception as e:
            rec('NOK-Exception-{}'.format(e))
    if debug: print('T', i, 'exiting')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_list', type=str, help='the url list file')
    parser.add_argument('--im_list', type=str, default='img.txt',help='the image list file')
    parser.add_argument('--num_threads', type=int, default=8, help='the num of processing')
    parser.add_argument('--save_dir', type=str, default='./images', help='the directory to save images')
    parser.add_argument('--max_size', type=int, default=500, help='maximum image size in width or height')
    args = parser.parse_args()

    url_list = args.url_list
    im_list = args.im_list
    num_threads = args.num_threads
    save_dir = args.save_dir
    max_size = args.max_size
    # create savedir
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    count = 0 # the num of urls
    count_invalid = 0 # the num of invalid urls
    record = 0
    count_already_done = 0

    with open(url_list, 'r')  as url_f: # Dont // the memory taken !!!
        url_file_mem = url_f.readlines()
        count = len(url_file_mem)
        print("loaded list of %d urls"%count)

    #count = 176089752 # calculated once above.
    part = int(count/num_threads)

    with open(im_list, 'w') as im_file_Record: # record the downloaded imgs
        thread_list = []

        t0 = time.time()
        for i in range(num_threads):
            if(i == num_threads-1):
                t = threading.Thread(name='Downloader %d'%i, target = downloadImg, kwargs={"i":i,"url_file_mem":url_file_mem,'start':i*part, 'end':count, 'url_list':url_list, 'save_dir':save_dir, 'max_size': max_size})
            else:
                t = threading.Thread(name='Downloader %d'%i, target = downloadImg, kwargs={"i":i,"url_file_mem":url_file_mem,'start':i*part, 'end':(i+1)*part, 'url_list':url_list, 'save_dir':save_dir, 'max_size': max_size})
            t.setDaemon(True)
            thread_list.append(t)
            t.start()

        for i in range(num_threads):
            try:
                t = thread_list[i]
                while t.is_alive():
                    t.join(.25)
                    t1 = time.time()
                    if t1 - t0 > 1:
                        print('{} threads. invalid={}, already={}, record={}'.format(threading.active_count(),count_invalid,count_already_done,record))
                        t0 = t1
                print('Thread', t.name, ' is done')
            except KeyboardInterrupt:
                break

        if count_invalid==0:
            print ("all {} imgs have been downloaded!".format(count))
        else:
            print("{}/{} imgs have been downloaded, {} URLs are invalid".format(record, count, count_invalid))
