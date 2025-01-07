import tensorflow as tf
import os
import json


DATA_DIRS = {
    "train": "/Users/peiyuanlee/Desktop/Fashion/train",
    "test": "/Users/peiyuanlee/Desktop/Fashion/test",
    "validation": "/Users/peiyuanlee/Desktop/Fashion/validation"
}

# Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Helper function to load annotations
def load_annotations(annos_dir):
    annotations = {}
    for file in os.listdir(annos_dir):

        if file.endswith('.json'):
            with open(os.path.join(annos_dir, file), 'r') as f:
                data = json.load(f)
                annotations[file] = data
    return annotations

# Function to load images and match with annotations
def load_images_and_annotations(images_dir, annos_dir):
    annotations = load_annotations(annos_dir)
    image_paths = []
    category_id = []
    category_name = []
    landmark = []
    segmentation = []
    occlusion = []
    style = []
    zoom = []
    viewpoint = []
    bounding_box = []
    scale = []
    source = []
    pair_id = []

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            filepath = os.path.join(images_dir, filename)
            f_name = filename[:-4] + '.json'

            # extract for each item
            number = 1
            while True:
                image = annotations.get(f_name, {})
                item = image.get('item' + str(number), None)
                if(item is None):
                    break
                image_paths.append(filepath)
                number += 1
                category_id.append(item.get("category_id", -1))
                category_name.append(item.get("category_name", -1))
                landmark.append(item.get("landmarks", -1))
                occlusion.append(item.get("occlusion", -1))
                style.append(item.get("style", -1))
                bounding_box.append(item.get("bounding_box", -1))
                zoom.append(item.get("zoom_in", -1))
                segmentation.append(item.get("segmentation", -1))
                viewpoint.append(item.get("view_point", -1))
                scale.append(item.get("scale", -1))
                source.append(image.get('source', -1))
                pair_id.append(image.get('pair_id', -1))



    return image_paths, category_id, category_name, landmark, occlusion, style, bounding_box, zoom, segmentation, viewpoint, source, pair_id

    
# Load data into TensorFlow Datasets
def load_dataset(images_dir, annos_dir):
    image_paths, category_id, category_name, landmark, occlusion, style, bounding_box, zoom, segmentation, viewpoint, source, pair_id = load_images_and_annotations(images_dir, annos_dir)
    #dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_paths, dtype=tf.string), tf.constant(category_id), tf.constant(category_name, dtype= tf.string), 
                                                  tf.ragged.constant(landmark), 
                                                  tf.constant(occlusion), 
                                                  tf.constant(style), 
                                                  tf.constant(bounding_box),
                                                    tf.constant(zoom), tf.ragged.constant(segmentation), tf.constant(viewpoint), tf.constant(source, dtype=tf.string), tf.constant(pair_id)))


    # Preprocessing: Load image and resize
    def process_image(file_path, category_id, category_name, landmark, occlusion, style, bounding_box, zoom, segmentation, viewpoint, source, pair_id):
        image = tf.io.read_file(file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, IMG_SIZE)
        image = image / 255.0  # Normalize to [0, 1]
        return image, category_id, category_name, landmark, occlusion, style, bounding_box, zoom, segmentation, viewpoint, source, pair_id

    dataset = dataset.map(process_image)
    return dataset

# Create datasets
train_dataset = load_dataset(os.path.join(DATA_DIRS["train"], "image"), os.path.join(DATA_DIRS["train"], "annos"))
val_dataset = load_dataset(os.path.join(DATA_DIRS["validation"], "image"), os.path.join(DATA_DIRS["validation"], "annos"))
test_dataset = load_dataset(os.path.join(DATA_DIRS["test"], "test/image"), os.path.join(DATA_DIRS["test"], "annos"))

