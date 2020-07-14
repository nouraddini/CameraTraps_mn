#
# ubc_to_json.py
#
# Convert the .csv file provided for the UBC data set to a 
# COCO-camera-traps .json file
#

#%% Constants and environment

from visualization import visualize_db
from data_management.databases import sanity_check_json_db
import pandas as pd
import os
import json
import uuid
import time
import humanfriendly
import numpy as np
from tqdm import tqdm

from path_utils import find_images

input_base = r'/mnt/blobfuse/wildlifeblobssc/'

output_base = r'/home/gramener/ubc/'
output_json_file = os.path.join(output_base,'ubc.json')
                        
os.makedirs(output_base,exist_ok=True)
mapped_fields = {"Survey.Name" : "survey_name",
                 "project_id": "survey_name",
                 "Camera.Name": "camera_name",
                 "station_id": "camera_name",
                 "Media.Filename": "filename",
                 "orig_file": "filename",
                 "timestamp_pst": "datetime",
                 "Date.Time": "datetime",
                 "Species": "species",
                 "latin_name": "species",
                 "common.name": "common_names",
                 "common_names": "common_names",
                 "Sighting.Quantity": "species_count"
                }
target_fields = ['common_names', 'species_count', 'group_count', 'behaviour']

#%% Create CCT dictionaries

images = []
annotations = []
categories = []

image_ids_to_images = {}

category_name_to_category = {}

# Force the empty category to be ID 0
empty_category = {}
empty_category['name'] = 'empty'
empty_category['id'] = 0
category_name_to_category['empty'] = empty_category
categories.append(empty_category)
next_category_id = 1

s_time = time.time()
for _folder in os.listdir(input_base):
    print(_folder)
    start_time = time.time()

    filenames_to_rows = {}
    filenames_with_multiple_annotations = []
    missing_images = []
    image_directory = os.path.join(input_base, _folder)
    files = os.listdir(image_directory)
    files = list(filter(lambda f: f.endswith('.csv'), files))
    input_metadata_file = os.path.join(input_base, _folder, files[0])
    assert(os.path.isfile(input_metadata_file))
    # Read Source Data
    input_metadata = pd.read_csv(input_metadata_file)
    input_metadata.rename(columns= mapped_fields, inplace= True)
    print('Read {} columns and {} rows from metadata file'.format(len(input_metadata.columns), len(input_metadata)))
    if _folder.startswith("SC_"):
        input_metadata['image_name'] = input_metadata[['camera_name', 'filename']].apply(lambda x: os.path.join(_folder, x[0].lower(), x[1].replace(".JPG", ".jpg")), axis = 1)
    else:
        input_metadata['image_name'] = input_metadata[['camera_name', 'filename']].apply(lambda x: os.path.join(_folder, x[0], x[1]), axis = 1)
    for i_row, fn in tqdm(enumerate(input_metadata['image_name']), total=len(input_metadata)):
        # print(fn)       
        if fn in filenames_to_rows:
            filenames_with_multiple_annotations.append(fn)
            filenames_to_rows[fn].append(i_row)
        else:
            filenames_to_rows[fn] = [i_row]
            image_path = os.path.join(input_base, fn)
            # print(image_path)
            if not os.path.isfile(image_path):
                missing_images.append(image_path)
        
    elapsed = time.time() - start_time
    print('Finished verifying image existence for {} files in {}, found {} filenames with multiple labels, {} missing images'.format(
        len(filenames_to_rows), humanfriendly.format_timespan(elapsed),
        len(filenames_with_multiple_annotations), len(missing_images)))
    
    #%% Check for images that aren't included in the metadata file

    # Enumerate all images
    image_full_paths = find_images(image_directory, bRecursive=True)

    unannotated_images = []

    for iImage, image_path in tqdm(enumerate(image_full_paths),total=len(image_full_paths)):
        relative_path = os.path.relpath(image_path,input_base)
        if relative_path not in filenames_to_rows:
            unannotated_images.append(relative_path)

    print('Finished checking {} images to make sure they\'re in the metadata, found {} unannotated images'.format(
            len(image_full_paths),len(unannotated_images)))
    print("Adding entries to CCT JSON file for Survey {0}".format(_folder))
    for image_name in tqdm(list(filenames_to_rows.keys())):
        img_id = image_name.replace('\\','/').replace('/','_').replace(' ','_')
        row_indices = filenames_to_rows[image_name]
        for i_row in row_indices:
            row = input_metadata.iloc[i_row]
            assert(row['image_name'] == image_name)
            timestamp = row['datetime']
            location = row['survey_name']+ " "+row['camera_name']
            if img_id in image_ids_to_images:
                im = image_ids_to_images[img_id]
                assert im['file_name'] == image_name
                assert im['location'] == location
            else:
                im = {}
                im['id'] = img_id
                im['file_name'] = image_name
                im['datetime'] = timestamp
                im['location'] = location

                image_ids_to_images[img_id] = im
                images.append(im)
            species = row['species']
        
            if (isinstance(species,float) or \
                (isinstance(species,str) and (len(species) == 0))):
                category_name = 'empty'
            else:
                category_name = species
            category_name = category_name.strip().lower()
            if category_name in category_name_to_category:
                category_id = category_name_to_category[category_name]['id'] 
            else:
                category_id = next_category_id
                category = {}
                category['id'] = category_id
                category['name'] = category_name
                category_name_to_category[category_name] = category
                categories.append(category)
                next_category_id += 1
            
            # Create an annotation
            ann = {}        
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = im['id']    
            ann['category_id'] = category_id
            
            # fieldname = list(mapped_fields.keys())[0]
            for target_field in target_fields:
                if target_field in input_metadata.columns:
                    val = row[target_field]
                    if isinstance(val,float) and np.isnan(val):
                        val = ''
                    else:
                        val = str(val).strip()
                    ann[target_field] = val
                
            annotations.append(ann)
e_time = time.time()
elapsed_time = e_time - s_time
print('Finished creating CCT dictionaries in {}'.format(
      humanfriendly.format_timespan(elapsed_time)))

#%% Create info struct

info = {}
info['year'] = 2020
info['version'] = 1
info['description'] = 'UBC Funnel'
info['contributor'] = 'UBC'


#%% Write output

json_data = {}
json_data['images'] = images
json_data['annotations'] = annotations
json_data['categories'] = categories
json_data['info'] = info
json.dump(json_data, open(output_json_file, 'w'), indent=2)

print('Finished writing .json file with {} images, {} annotations, and {} categories'.format(
    len(images), len(annotations), len(categories)))


#%% Validate output


options = sanity_check_json_db.SanityCheckOptions()
options.baseDir = input_base
options.bCheckImageSizes = False
options.bCheckImageExistence = False
options.bFindUnusedImages = False

sortedCategories, data = sanity_check_json_db.sanity_check_json_db(
    output_json_file, options)


#%% Preview labels


viz_options = visualize_db.DbVizOptions()
viz_options.num_to_visualize = 1000
viz_options.trim_to_images_with_bboxes = False
viz_options.add_search_links = True
viz_options.sort_by_filename = False
viz_options.parallelize_rendering = True
html_output_file, image_db = visualize_db.process_images(db_path=output_json_file,
                                                         output_dir=os.path.join(
                                                             output_base, 'preview'),
                                                         image_base_dir=input_base,
                                                         options=viz_options)
# os.startfile(html_output_file)
import subprocess, sys
opener ="open" if sys.platform == "darwin" else "xdg-open"
subprocess.call([opener, html_output_file])
