import os
import shutil
import json
import time
import uuid
import redis
import traceback
from PIL import Image
from datetime import datetime
from io import BytesIO

from flask import Flask, Response, jsonify, make_response, request
from requests_toolbelt.multipart.encoder import MultipartEncoder
import visualization.visualization_utils as viz_utils
import api_config

print('Creating application')
app = Flask(__name__)

db = redis.StrictRedis(host=api_config.REDIS_HOST, port=api_config.REDIS_PORT)

def _make_error_object(error_code, error_message):
    # here we make a dict that the request_processing_function can return to the endpoint function
    # to notify it of an error
    return {
        'error_message': error_message,
        'error_code': error_code
    }

def _make_error_response(error_code, error_message):
    return make_response(jsonify({'error': error_message}), error_code)

def check_posted_data(request):
 
    files = request.files
    params = request.args

    # check that the content uploaded is not too big
    # request.content_length is the length of the total payload
    # also will not proceed if cannot find content_length, hence in the else we exceed the max limit
    content_length = request.content_length
    if not content_length:
        return _make_error_object(411, 'No image(s) are sent, or content length cannot be determined.')
    if content_length > api_config.MAX_CONTENT_LENGTH_IN_MB * 1024 * 1024:
        return _make_error_object(413, ('Payload size {:.2f} MB exceeds the maximum allowed of {} MB. '
                    'Please upload fewer or more compressed images.').format(
            content_length / (1024 * 1024), api_config.MAX_CONTENT_LENGTH_IN_MB))

    render_boxes = True if params.get('render', '') in ['True', 'true'] else False

    # validate detection confidence value
    if 'confidence' in params:
        detection_confidence = float(params['confidence'])
        print('runserver, post_detect_sync, user specified detection confidence: ', detection_confidence)  # TODO delete
        if detection_confidence < 0.0 or detection_confidence > 1.0:
            return _make_error_object(400, 'Detection confidence {} is invalid. Needs to be between 0.0 and 1.0.'.format(
                detection_confidence))
    else:
        detection_confidence = api_config.DEFAULT_DETECTION_CONFIDENCE

    # check that the number of images is acceptable
    num_images = sum([1 if file.content_type in api_config.IMAGE_CONTENT_TYPES else 0 for file in files.values()])
    print('runserver, post_detect_sync, number of images received: ', num_images)

    if num_images > api_config.MAX_IMAGES_ACCEPTED:
        return _make_error_object(413, 'Too many images. Maximum number of images that can be processed in one call is {}.'.format(api_config.MAX_IMAGES_ACCEPTED))
    elif num_images == 0:
        return _make_error_object(400, 'No image(s) of accepted types (image/jpeg, image/png, application/octet-stream) received.')
    
    return {
        'render_boxes': render_boxes,
        'detection_confidence': detection_confidence
    }
   
@app.route(api_config.API_PREFIX + '/detect', methods = ['POST'])
def detect_sync():
    # check if the request_processing_function had an error while parsing user specified parameters
    post_data = check_posted_data(request)
    if post_data.get('error_code', None) is not None:
        return _make_error_response(post_data.get('error_code'), post_data.get('error_message'))

    render_boxes = post_data.get('render_boxes')
    detection_confidence = post_data.get('detection_confidence')
  
    redis_id = str(uuid.uuid4())
    d = {'id': redis_id, 'render_boxes': render_boxes, 'detection_confidence': detection_confidence}
    temp_direc = os.path.join(api_config.TEMP_FOLDER, redis_id)
    try:
        try:
            # TODO: convert to reading from memory instead
            os.makedirs(temp_direc,exist_ok=True)
            for name, file in request.files.items():
                if file.content_type in api_config.IMAGE_CONTENT_TYPES:
                    filename = request.files[name].filename
                    file.save(os.path.join(temp_direc, filename))
        
        except Exception as e:
            return _make_error_object(500, 'Error saving images: ' + str(e))
        
        db.rpush(api_config.REDIS_QUEUE, json.dumps(d))
        
        while True:
            result = db.get(redis_id)
            if result:
                result = json.loads(result.decode())
                if result['status'] == 200:
                    detection_results = result['detection_results']
                    filtered_results = result['filtered_results']
                    inference_time_detector = result['inference_time_detector']

                    db.delete(redis_id)
                
                else:
                    db.delete(redis_id)
                    print('Error performing detection on the images: ' + str(e))
                    return _make_error_response(500, 'Error performing detection on the images: ' + str(e))

                try:
                    print('runserver, post_detect_sync, rendering and sending images back...')
                    fields = {
                        'detection_result': ('detection_result', json.dumps(filtered_results), 'application/json'),
                    }

                    if render_boxes:
                         for result in detection_results:
                            image_name = result['file']
                            detections = result.get('detections', None)

                            image = Image.open(os.path.join(temp_direc, image_name))
                            viz_utils.render_detection_bounding_boxes(detections, image, confidence_threshold=detection_confidence)
                            output_img_stream = BytesIO()
                            image.save(output_img_stream, format='jpeg')
                            output_img_stream.seek(0)
                            fields[image_name] = (image_name, output_img_stream, 'image/jpeg')
                    
                    m = MultipartEncoder(fields=fields)
                    
                    if len(inference_time_detector) > 0:
                        mean_inference_time_detector = sum(inference_time_detector) / len(inference_time_detector)
                    else:
                        mean_inference_time_detector = -1
                    
                    # TODO: logging
                    # log.log_info('detector mean inference time', mean_inference_time_detector,
                    #  additionalProperties={
                    #      'detector mean inference time': str(mean_inference_time_detector),
                    #      # TODO 'classification mean inference time': str(''),
                    #      'num_images': len(image_names),
                    #      'render_boxes': render_boxes,
                    #      'detection_confidence': detection_confidence
                    #  })

                    shutil.rmtree(temp_direc)
                    return Response(m.to_string(), mimetype=m.content_type)

                except Exception as e:
                    print('Error returning result or rendering the detection boxes: ' + str(e))

            else:
                time.sleep(.25)

    except Exception as e:
        print(traceback.format_exc())
        return _make_error_object(500, 'Error processing images: ' + str(e))

if __name__ == '__main__':
    #app.run()

    #for testing without docker
    app.run(host='0.0.0.0', port=5050)

