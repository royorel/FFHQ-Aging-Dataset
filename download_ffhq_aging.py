# Copyright (c) 2020, Roy Or-El. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

# This code is a modification of the download_ffhq.py file from the original FFHQ dataset.
# Here we download an in-the-wild-image, do the alignment and delete the original in-the-wild image.

"""Download Flickr-Face-HQ-Aging (FFHQ-Aging) dataset to current working directory."""

import os
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
import pydrive_utils
from collections import OrderedDict, defaultdict
from pdb import set_trace as st

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------

json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

license_specs = {
    'json':      dict(file_url='https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX', file_path='LICENSE.txt',                    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'images':    dict(file_url='https://drive.google.com/uc?id=1sP2qz8TzLkzG2gjwAa4chtdB31THska4', file_path='images1024x1024/LICENSE.txt',    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'thumbs':    dict(file_url='https://drive.google.com/uc?id=1iaL1S381LS10VVtqu-b2WfF9TiY75Kmj', file_path='thumbnails128x128/LICENSE.txt',  file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'wilds':     dict(file_url='https://drive.google.com/uc?id=1rsfFOEQvkd6_Z547qhpq5LhDl2McJEzw', file_path='in-the-wild-images/LICENSE.txt', file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'tfrecords': dict(file_url='https://drive.google.com/uc?id=1SYUmqKdLoTYq-kqsnPsniLScMhspvl5v', file_path='tfrecords/ffhq/LICENSE.txt',     file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
}

#----------------------------------------------------------------------------

def download_file(session, file_spec, stats, chunk_size=128, num_attempts=10):
    file_path = file_spec['file_path']
    file_url = file_spec['file_url']
    file_dir = os.path.dirname(file_path)
    tmp_path = file_path + '.tmp.' + uuid.uuid4().hex
    if file_dir:
        os.makedirs(file_dir, exist_ok=True)

    for attempts_left in reversed(range(num_attempts)):
        data_size = 0
        try:
            # Download.
            data_md5 = hashlib.md5()
            with session.get(file_url, stream=True) as res:
                res.raise_for_status()
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=chunk_size<<10):
                        f.write(chunk)
                        data_size += len(chunk)
                        data_md5.update(chunk)
                        with stats['lock']:
                            stats['bytes_done'] += len(chunk)

            # Validate.
            if 'file_size' in file_spec and data_size != file_spec['file_size']:
                raise IOError('Incorrect file size', file_path)
            if 'file_md5' in file_spec and data_md5.hexdigest() != file_spec['file_md5']:
                raise IOError('Incorrect file MD5', file_path)
            if 'pixel_size' in file_spec or 'pixel_md5' in file_spec:
                with PIL.Image.open(tmp_path) as image:
                    if 'pixel_size' in file_spec and list(image.size) != file_spec['pixel_size']:
                        raise IOError('Incorrect pixel size', file_path)
                    if 'pixel_md5' in file_spec and hashlib.md5(np.array(image)).hexdigest() != file_spec['pixel_md5']:
                        raise IOError('Incorrect pixel MD5', file_path)
            break

        except:
            with stats['lock']:
                stats['bytes_done'] -= data_size

            # Handle known failure cases.
            if data_size > 0 and data_size < 8192:
                with open(tmp_path, 'rb') as f:
                    data = f.read()
                data_str = data.decode('utf-8')

                # Google Drive virus checker nag.
                links = [html.unescape(link) for link in data_str.split('"') if 'export=download' in link]
                if len(links) == 1:
                    if attempts_left:
                        file_url = requests.compat.urljoin(file_url, links[0])
                        continue

                # Google Drive quota exceeded.
                if 'Google Drive - Quota exceeded' in data_str:
                    if not attempts_left:
                        raise IOError("Google Drive download quota exceeded -- please try again later")

            # Last attempt => raise error.
            if not attempts_left:
                raise

    # Rename temp file to the correct name.
    os.replace(tmp_path, file_path) # atomic
    # with stats['lock']:
    #     stats['files_done'] += 1

    # Attempt to clean up any leftover temps.
    for filename in glob.glob(file_path + '.tmp.*'):
        try:
            os.remove(filename)
        except:
            pass

#----------------------------------------------------------------------------

def choose_bytes_unit(num_bytes):
    b = int(np.rint(num_bytes))
    if b < (100 << 0): return 'B', (1 << 0)
    if b < (100 << 10): return 'kB', (1 << 10)
    if b < (100 << 20): return 'MB', (1 << 20)
    if b < (100 << 30): return 'GB', (1 << 30)
    return 'TB', (1 << 40)

#----------------------------------------------------------------------------

def format_time(seconds):
    s = int(np.rint(seconds))
    if s < 60: return '%ds' % s
    if s < 60 * 60: return '%dm %02ds' % (s // 60, s % 60)
    if s < 24 * 60 * 60: return '%dh %02dm' % (s // (60 * 60), (s // 60) % 60)
    if s < 100 * 24 * 60 * 60: return '%dd %02dh' % (s // (24 * 60 * 60), (s // (60 * 60)) % 24)
    return '>100d'

#----------------------------------------------------------------------------

def download_files(file_specs, dst_dir='.', output_size=256, drive=None, num_threads=32, status_delay=0.2, timing_window=50, **download_kwargs):

    # Determine which files to download.
    done_specs = {spec['file_path']: spec for spec in file_specs if os.path.isfile(spec['file_path'].replace('in-the-wild-images',dst_dir))}

    missing_specs = [spec for spec in file_specs if spec['file_path'] not in done_specs]
    files_total = len(file_specs)
    bytes_total = sum(spec['file_size'] for spec in file_specs)
    stats = dict(files_done=len(done_specs), bytes_done=sum(spec['file_size'] for spec in done_specs.values()), lock=threading.Lock())
    if len(done_specs) == files_total:
        print('All files already downloaded -- skipping.')
        return

    # Launch worker threads.
    spec_queue = queue.Queue()
    exception_queue = queue.Queue()
    for spec in missing_specs:
        spec_queue.put(spec)
    thread_kwargs = dict(spec_queue=spec_queue, exception_queue=exception_queue,
                         stats=stats, dst_dir=dst_dir, output_size=output_size,
                         drive=drive, download_kwargs=download_kwargs)
    for _thread_idx in range(min(num_threads, len(missing_specs))):
        threading.Thread(target=_download_thread, kwargs=thread_kwargs, daemon=True).start()

    # Monitor status until done.
    bytes_unit, bytes_div = choose_bytes_unit(bytes_total)
    spinner = '/-\\|'
    timing = []
    while True:
        spinner = spinner[1:] + spinner[:1]
        if drive != None:
            with stats['lock']:
                files_done = stats['files_done']

            print('\r{} done processing {}/{} files'.format(spinner[0], files_done, files_total),
                  end='', flush=True)
        else:
            with stats['lock']:
                files_done = stats['files_done']
                bytes_done = stats['bytes_done']
            timing = timing[max(len(timing) - timing_window + 1, 0):] + [(time.time(), bytes_done)]
            bandwidth = max((timing[-1][1] - timing[0][1]) / max(timing[-1][0] - timing[0][0], 1e-8), 0)
            bandwidth_unit, bandwidth_div = choose_bytes_unit(bandwidth)
            eta = format_time((bytes_total - bytes_done) / max(bandwidth, 1))

            print('\r%s %6.2f%% done processed %d/%d files  %-13s  %-10s  ETA: %-7s ' % (
                spinner[0],
                bytes_done / bytes_total * 100,
                files_done, files_total,
                'downloaded %.2f/%.2f %s' % (bytes_done / bytes_div, bytes_total / bytes_div, bytes_unit),
                '%.2f %s/s' % (bandwidth / bandwidth_div, bandwidth_unit),
                'done' if bytes_total == bytes_done else '...' if len(timing) < timing_window or bandwidth == 0 else eta,
            ), end='', flush=True)

        if files_done == files_total:
            print()
            break


        try:
            exc_info = exception_queue.get(timeout=status_delay)
            raise exc_info[1].with_traceback(exc_info[2])
        except queue.Empty:
            pass

def _download_thread(spec_queue, exception_queue, stats, dst_dir, output_size, drive, download_kwargs):
    with requests.Session() as session:
        while not spec_queue.empty():
            spec = spec_queue.get()
            try:
                if drive != None:
                    pydrive_utils.pydrive_download(drive, spec['file_url'], spec['file_path'])
                else:
                    download_file(session, spec, stats, **download_kwargs)

                if spec['file_path'].endswith('.png'):
                    align_in_the_wild_image(spec, dst_dir, output_size)
                    os.remove(spec['file_path'])

            except:
                exception_queue.put(sys.exc_info())

            with stats['lock']:
                stats['files_done'] += 1

#----------------------------------------------------------------------------

def align_in_the_wild_image(spec, dst_dir, output_size, transform_size=4096, enable_padding=True):
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copyfile('LICENSE.txt', os.path.join(dst_dir, 'LICENSE.txt'))

    item_idx = int(os.path.basename(spec['file_path'])[:-4])

    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = np.array(spec['face_landmarks'])
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 2.2) # This results in larger crops then the original FFHQ. For the original crops, replace 2.2 with 1.8
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    src_file = spec['file_path']
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    dst_subdir = os.path.join(dst_dir, '%05d' % (item_idx - item_idx % 1000))
    os.makedirs(dst_subdir, exist_ok=True)
    img.save(os.path.join(dst_subdir, '%05d.png' % item_idx))


#----------------------------------------------------------------------------

def run(resolution, debug, pydrive, cmd_auth, **download_kwargs):
    if pydrive:
        drive = pydrive_utils.create_drive_manager(cmd_auth)
    else:
        drive = None

    if not os.path.isfile(json_spec['file_path']) or not os.path.isfile('LICENSE.txt'):
        print('Downloading JSON metadata...')
        download_files([json_spec, license_specs['json']], drive=drive, **download_kwargs)

    print('Parsing JSON metadata...')
    with open(json_spec['file_path'], 'rb') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict)

    specs = [item['in_the_wild'] for item in json_data.values()] + [license_specs['wilds']]

    if len(specs):
        output_size = resolution
        dst_dir = 'ffhq_aging{}x{}'.format(output_size,output_size)
        np.random.shuffle(specs) # to make the workload more homogeneous
        if debug:
            specs = specs[:50] # to create images in multiple directories
        print('Downloading %d files...' % len(specs))
        download_files(specs, dst_dir, output_size, drive=drive, **download_kwargs)

    if os.path.isdir('in-the-wild-images'):
        shutil.rmtree('in-the-wild-images')

#----------------------------------------------------------------------------

def run_cmdline(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Download Flickr-Face-HQ-Aging (FFHQ-Aging) dataset to current working directory.')
    parser.add_argument('--debug',              help='activate debug mode, download 50 random images (default: False)', action='store_true')
    parser.add_argument('--pydrive',            help='use pydrive interface to download files. it overrides google drive quota limitation \
                                                      this requires google credentials (default: False)', action='store_true')
    parser.add_argument('--cmd_auth',           help='use command line google authentication when using pydrive interface (default: False)', action='store_true')
    parser.add_argument('--resolution',         help='final resolution of saved images (default: 256)', type=int, default=256, metavar='PIXELS')
    parser.add_argument('--num_threads',        help='number of concurrent download threads (default: 32)', type=int, default=32, metavar='NUM')
    parser.add_argument('--status_delay',       help='time between download status prints (default: 0.2)', type=float, default=0.2, metavar='SEC')
    parser.add_argument('--timing_window',      help='samples for estimating download eta (default: 50)', type=int, default=50, metavar='LEN')
    parser.add_argument('--chunk_size',         help='chunk size for each download thread (default: 128)', type=int, default=128, metavar='KB')
    parser.add_argument('--num_attempts',       help='number of download attempts per file (default: 10)', type=int, default=10, metavar='NUM')

    args = parser.parse_args()
    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_cmdline(sys.argv)

#----------------------------------------------------------------------------
