# Walk Assist Segmentation Dataset

This dataset is a YOLO segmentation export for the annotated walk-assist classes:

- car
- crosswalk
- fence
- pedestrian_stop
- pedestrian_walk
- person
- pole
- vegetation

The original export mixed YOLO segmentation polygon rows with YOLO detection bbox rows. The bbox-only rows were converted into four-corner rectangle polygons so every non-empty annotation row is compatible with YOLO segmentation training.

## Contents

- `train/`, `valid/`, `test/`: images and YOLO segmentation labels
- `data.yaml`: dataset config
- `bbox_to_polygon_report.csv`: conversion report
- `annotation_sample_contact_sheet.jpg`: visual annotation QA samples

## Training

From the repository root:

```bash
yolo segment train data=model_training/datasets/walk_assist_segmentation/data.yaml model=yolo11n-seg.pt
```

## Conversion Notes

Existing polygon annotations were preserved unchanged. Bbox-only annotations were converted from:

```txt
class x_center y_center width height
```

to:

```txt
class x1 y1 x2 y1 x2 y2 x1 y2
```

The conversion script is available at `model_training/dataset_tools/convert_bboxes_to_segmentation.py`.
