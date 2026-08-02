

# Workshop by hvzn: Object Detection Project

## Description
This repository provides a workflow for preparing datasets and exporting custom object detection models using the **TensorFlow Object Detection API**. It is specifically configured to detect and localize two snack brands: **Lays** and **Pocky**. 

The project includes utility scripts for:
- Converting CSV annotations into TensorFlow `TFRecord` format (required for TF Object Detection training).
- Freezing and exporting trained model checkpoints into optimized inference graphs and TensorFlow `SavedModel` formats.

## 📂 Project Structure
```
workshop-by-hvzn/
├── image/
│   └── anotation/          # PASCAL VOC XML annotations for training images
├── data/                   # Directory for CSV labels and generated TFRecords
├── generate_tfrecord.py    # Converts CSV annotations to TFRecord format
├── generate_train_test_record.py # Convenience script to generate train/test records
├── export_inference_graph.py # CLI tool to export trained models for inference
├── exporter.py             # Helper functions for graph freezing & model exporting
└── __init__.py
```

## 🛠 Prerequisites
- **Python 3.6+**
- **TensorFlow 1.15** (⚠️ This codebase uses TF 1.x APIs like `tf.app.flags`, `tf.contrib.slim`, and `tf.gfile`. For TF 2.x, you will need to add `tf.compat.v1` prefixes or migrate the code.)
- **Dependencies:** `pandas`, `Pillow`, `protobuf`
- **TensorFlow Object Detection API** installed and compiled (Proto files must be built).

## 🚀 Usage

### 1. Dataset Preparation
The TF Object Detection API requires data in `TFRecord` format. This repository provides scripts to convert your labeled data.

**Step 1: Prepare CSV files**
The `generate_tfrecord.py` script expects CSV files containing bounding box coordinates. You can convert the provided PASCAL VOC XML files (`image/anotation/*.xml`) to CSV format using a standard `xml_to_csv.py` script or your preferred tool. Ensure the CSV contains columns: `filename`, `xmin`, `ymin`, `xmax`, `ymax`, `class`.

**Step 2: Generate TFRecords**
Run the convenience script to generate both training and testing records:
```bash
python generate_train_test_record.py
```
Or run them individually:
```bash
python generate_tfrecord.py --csv_input=data/train.csv --output_path=data/train.record
python generate_tfrecord.py --csv_input=data/test.csv --output_path=data/test.record
```
> 💡 **Custom Classes:** Modify the `class_text_to_int()` function in `generate_tfrecord.py` if you add new object categories. Currently, it maps `lays` → `1` and `pocky` → `2`.

### 2. Exporting a Trained Model
After training your model using the TF Object Detection API, use the export script to freeze weights and generate inference-ready artifacts.

```bash
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path path/to/pipeline.config \
    --trained_checkpoint_prefix path/to/model.ckpt \
    --output_directory path/to/exported_model_directory
```

**Available Flags:**
| Flag | Description |
|------|-------------|
| `--input_type` | Type of input node: `image_tensor`, `encoded_image_string_tensor`, or `tf_example` (default: `image_tensor`) |
| `--input_shape` | (Optional) Fixed input shape for `image_tensor` as comma-separated integers, e.g., `300,300,3` |
| `--pipeline_config_path` | Path to the `.config` file used during training |
| `--trained_checkpoint_prefix` | Path to the trained checkpoint (e.g., `model.ckpt-XXXXX`) |
| `--output_directory` | Directory where the frozen graph (`frozen_inference_graph.pb`), checkpoint, and `saved_model` will be saved |

The export process will output:
- `frozen_inference_graph.pb`: Standalone graph with weights baked in.
- `saved_model/`: Directory compatible with TensorFlow Serving.
- Checkpoint files (`model.ckpt.*`) and text representation (`graph.pbtxt`).

## ⚠️ Important Notes
- **TensorFlow Version:** This repository is built for TensorFlow 1.x. Modern TF 2.x environments will require compatibility wrappers (`tf.compat.v1`) or API migration.
- **Image Directory:** `generate_tfrecord.py` expects images to be located in an `images/` folder relative to the script's working directory.
- **Labels:** Ensure your label map (`.pbtxt` file used in the pipeline config) matches the integer IDs defined in `class_text_to_int()`.

## 📄 License
This project contains code licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) (c) 2017 The TensorFlow Authors.
