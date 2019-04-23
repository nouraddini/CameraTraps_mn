import numpy as np
from PIL import Image, ImageFile, ImageFont, ImageDraw
ImageFile.LOAD_TRUNCATED_IMAGES = True

def open_image(input):
    """
    Opens an image in binary format using PIL.Image and convert to RGB mode.

    Args:
        input: an image in binary format read from the POST request's body or
            path to an image file (anything that PIL can open)

    Returns:
        an PIL image object in RGB mode
    """
    image = Image.open(input)
    if image.mode not in ('RGBA', 'RGB'):
        raise AttributeError('Input image not in RGBA or RGB mode and cannot be processed.')
    if image.mode == 'RGBA':
        # PIL.Image.convert() returns a converted copy of this image
        image = image.convert(mode='RGB')
    return image


def resize_image(image, targetWidth, targetHeight=-1):
    """
    Resizes a PIL image object to the specified width and height; does not resize
    in place.  If either width or height are -1, resizes with aspect ratio preservation.  
    If both are -1, returns the original image (does not copy in this case).
    """
    
    # Null operation
    if targetWidth == -1 and targetHeight == -1:
        
        return image    
    
    elif targetWidth == -1 or targetHeight == -1:
    
        # Aspect ratio as width over height
        aspectRatio = image.size[0] / image.size[1]
        
        if targetWidth != -1:
            # ar = w / h        
            # h = w / ar
            targetHeight = int(targetWidth / aspectRatio)
            
        else:
            # ar = w / h
            # w = ar * h
            targetWidth = int(aspectRatio * targetHeight)
            
    resizedImage = image.resize((targetWidth, targetHeight), Image.ANTIALIAS)
    return resizedImage


def render_iMerit_boxes(boxes, classes, image, label_map=None):
    """
    Renders bounding boxes and their category labels on a PIL image.

    Args:
        boxes: bounding box annotations from iMerit, format is [x_rel, y_rel, w_rel, h_rel] (rel = relative coords)
        image: PIL.Image object to annotate on
        label_map: optional dict mapping classes to a string for display

    Returns:
        image will be altered in place
    """
    display_boxes = []
    display_strs = []  # list of list, one list of strings for each bounding box (to accommodate multiple labels)
    for box, clss in zip(boxes, classes):
        x_rel, y_rel, w_rel, h_rel = box
        ymin, xmin = y_rel, x_rel
        ymax = ymin + h_rel
        xmax = xmin + w_rel

        display_boxes.append([ymin, xmin, ymax, xmax])

        if label_map:
            clss = label_map[int(clss)]
        display_strs.append([clss])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, display_str_list_list=display_strs)


def render_db_bounding_boxes(boxes, classes, image, original_size=None, label_map=None):
    """
    Render bounding boxes (with class labels) on [image].  This is a wrapper for
    draw_bounding_boxes_on_image, allowing the caller to operate on a resized image
    by providing the original size of the image; bboxes will be scaled accordingly.
    """
    display_boxes = []
    display_strs = []
    
    if original_size is not None:
        image_size = original_size
    else:
        image_size = image.size
        
    img_width, img_height = image_size
    for box, clss in zip(boxes, classes):
        x_min_abs, y_min_abs, width_abs, height_abs = box

        ymin = y_min_abs / img_height
        ymax = ymin + height_abs / img_height

        xmin = x_min_abs / img_width
        xmax = xmin + width_abs / img_width

        display_boxes.append([ymin, xmin, ymax, xmax])

        if label_map:
            clss = label_map[int(clss)]
        display_strs.append([clss])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, display_str_list_list=display_strs)


def render_detection_bounding_boxes(boxes_and_scores, image, label_map={}, confidence_threshold=0.5):
    """
    Renders bounding boxes, label and confidence on an image if confidence is above the threshold.
    This is works with the output of the detector batch processing API.

    Args:
        boxes, scores, classes:  outputs of generate_detections.
        image: PIL.Image object, output of generate_detections.
        label_map: optional, mapping the numerical label to a string name.
        confidence_threshold: threshold above which the bounding box is rendered.

    image is modified in place!
    """
    display_boxes = []
    display_strs = []  # list of list, one list of strings for each bounding box (to accommodate multiple labels)
    for detection in boxes_and_scores:
        score = detection[4]
        if score > confidence_threshold:
            display_boxes.append(detection[:4])
            clss = 1  # we just detect animals right now
            label = label_map[clss] if clss in label_map else str(clss)
            displayed_label = '{}: {}%'.format(label, round(100 * score))
            display_strs.append([displayed_label])

    display_boxes = np.array(display_boxes)
    draw_bounding_boxes_on_image(image, display_boxes, display_str_list_list=display_strs)


# the following two functions are from https://github.com/tensorflow/models/blob/master/research/object_detection/utils/visualization_utils.py

def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='red',
                                 thickness=1,
                                 display_str_list_list=()):
    """
    Draws bounding boxes on image.

    Args:
      image: a PIL.Image object.
      boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
             The coordinates are in normalized format between [0, 1].
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list_list: list of list of strings.
                             a list of strings for each bounding box.
                             The reason to pass a list of strings for a
                             bounding box is that it might contain
                             multiple labels.

    Raises:
      ValueError: if boxes is not a [N, 4] array
    """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        # print('Input must be of size [N, 4], but is ' + str(boxes_shape))
        return  # no object detection on this image, return
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
            draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                       boxes[i, 3], color, thickness, display_str_list)


def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

    Bounding box coordinates can be specified in either absolute (pixel) or
    normalized coordinates by setting the use_normalized_coordinates argument.

    Each string in display_str_list is displayed on a separate line above the
    bounding box in black text on a rectangle filled with the input 'color'.
    If the top of the bounding box extends to the edge of the image, the strings
    are displayed below the bounding box.

    Args:
      image: a PIL.Image object.
      ymin: ymin of bounding box.
      xmin: xmin of bounding box.
      ymax: ymax of bounding box.
      xmax: xmax of bounding box.
      color: color to draw bounding box. Default is red.
      thickness: line thickness. Default value is 4.
      display_str_list: list of strings to display in box
                        (each to be shown on its own line).
      use_normalized_coordinates: If True (default), treat coordinates
        ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
        coordinates as absolute.
    """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin
