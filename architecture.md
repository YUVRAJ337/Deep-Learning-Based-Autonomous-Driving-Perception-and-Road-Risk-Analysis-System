# Autonomous Driving Perception and Decision System

**Course:** Deep Learning\
**Platform:** macOS Apple Silicon (M4, PyTorch MPS)\
**Project type:** Video-based autonomous-driving perception prototype\
**Status:** Proposed implementation plan --- components are not
considered complete until tested and documented.

------------------------------------------------------------------------

## 1. Problem statement

The goal is to build a video-based perception prototype that analyses
road scenes, identifies relevant road users, estimates the
drivable-road/lane region, and produces a simple high-level driving
recommendation.

The initial prototype focuses on three outputs:

1.  **Object detection:** identify road users such as cars, buses,
    trucks, bicycles and pedestrians.
2.  **Road/lane segmentation:** identify road-related pixels or lane
    regions, depending on the selected dataset and model.
3.  **High-level decision:** produce a simple recommendation such as
    `CONTINUE`, `SLOW_DOWN` or `STOP` from the detected scene.

This is an educational simulation/prototype. It is not a road-certified
autonomous-driving system and must not control a real vehicle.

## 2. Proposed system architecture

``` mermaid
flowchart TD
    A[Input driving video] --> B[Frame extraction and preprocessing]
    B --> C[Deep-learning perception]
    C --> D[Object detector]
    C --> E[Road / lane segmentation]
    D --> F[Scene representation]
    E --> F
    F --> G[Decision module]
    G --> H[Safety rules and threshold checks]
    H --> I[CONTINUE / SLOW_DOWN / STOP]
    I --> J[Annotated video and evaluation report]

    subgraph Object detection model
      D1[ResNet-18 CNN backbone] --> D2[Feature Pyramid Network]
      D2 --> D3[RetinaNet detection head]
      D3 --> D
    end

    subgraph Segmentation model
      E1[Input frame] --> E2[U-Net encoder]
      E2 --> E3[Bottleneck]
      E3 --> E4[Decoder with skip connections]
      E4 --> E
    end
```

### Data flow in plain language

1.  A driving video is read frame by frame.
2.  Frames are resized and normalized to match the model's expected
    input.
3.  The object-detection model predicts object classes, bounding boxes
    and confidence scores.
4.  The segmentation model predicts a pixel-level road/lane mask.
5.  A scene-representation step combines the predictions into structured
    information.
6.  A simple, transparent decision module maps that information to a
    driving recommendation.
7.  The system draws detections, masks and the recommendation on the
    output video.

**Important:** the diagram describes the intended architecture. A block
should only be marked implemented after it has run successfully and its
output has been checked.

## 3. Deep-learning models and their roles

### 3.1 CNN and convolution

A convolutional neural network learns visual features such as edges,
textures, shapes and object parts.

For input image (X) and learnable kernel (K), a 2-D convolution can be
written as:

\[ Y(i,j)=`\sum`{=tex}\_m`\sum`{=tex}\_n X(i-m,j-n)K(m,n) \]

A bias and activation function are commonly applied:

\[ Z(i,j)=`\operatorname{ReLU}`{=tex}(Y(i,j)+b) \]

ReLU is:

\[ `\operatorname{ReLU}`{=tex}(x)=`\max`{=tex}(0,x) \]

Early CNN layers usually learn simple visual patterns; deeper layers
combine them into more meaningful features.

### 3.2 ResNet-18 backbone

ResNet uses residual/skip connections to make deeper networks easier to
optimize. Instead of learning a complete mapping (H(x)), a residual
block learns a residual function (F(x)):

\[ y=F(x,W)+x \]

If input and output dimensions differ, a projection shortcut may be
used:

\[ y=F(x,W)+W_sx \]

In this project, ResNet-18 is a candidate feature-extraction backbone
for the detector. Its role is to produce feature maps; it does not by
itself produce the final object boxes.

### 3.3 Feature Pyramid Network (FPN)

Objects appear at different sizes. FPN combines feature maps from
multiple backbone stages so the detector can use both high-level
semantic information and finer spatial detail.

A simplified top-down formulation is:

\[ P_l=`\operatorname{Conv}`{=tex}*{1`\times1`{=tex}}(C_l)+
`\operatorname{Upsample}`{=tex}(P*{l+1}) \]

where (C_l) is a backbone feature map and (P_l) is the corresponding
pyramid feature.

### 3.4 RetinaNet object detection

The RetinaNet-style detector uses classification and bounding-box
regression heads on pyramid features.

For a predicted box (B_p) and ground-truth box (B\_{gt}), intersection
over union (IoU) is:

\[ IoU=`\frac{|B_p\cap B_{gt}|}{|B_p\cup B_{gt}|}`{=tex} \]

A common classification loss for RetinaNet is focal loss:

\[ FL(p_t)=-`\alpha`{=tex}\_t(1-p_t)\^`\gamma`{=tex}`\log`{=tex}(p_t) \]

Focal loss reduces the contribution of easy examples and focuses
training more on hard examples. The parameters (`\alpha`{=tex}) and
(`\gamma`{=tex}) are hyperparameters.

The detector's outputs are typically: - predicted class, - confidence
score, - bounding-box coordinates.

### 3.5 U-Net road/lane segmentation

U-Net has an encoder-decoder structure. The encoder extracts
increasingly abstract features; the decoder upsamples them to produce a
pixel-level prediction. Skip connections transfer spatial detail from
encoder stages to corresponding decoder stages.

For binary segmentation, a sigmoid output gives a probability per pixel:

\[ `\hat `{=tex}y_i=`\sigma`{=tex}(z_i)=`\frac{1}{1+e^{-z_i}}`{=tex} \]

Binary cross-entropy loss is:

\[ L\_{BCE}=-`\frac{1}{N}`{=tex}`\sum`{=tex}\_{i=1}\^{N}
\[y_i`\log`{=tex}(`\hat `{=tex}y_i)+(1-y_i)`\log`{=tex}(1-`\hat `{=tex}y_i)\]
\]

Dice score measures overlap between predicted and ground-truth masks:

\[ Dice=`\frac{2|P\cap G|+\epsilon}{|P|+|G|+\epsilon}`{=tex} \]

A common Dice loss is:

\[ L\_{Dice}=1-Dice \]

The exact number of segmentation classes must match the dataset labels.
If the chosen dataset only labels road area, the project must call this
road segmentation rather than claiming lane-marking detection.

## 4. Scene representation and decision logic

The perception models produce visual predictions. A separate step
converts them into a compact scene representation, for example:

``` json
{
  "objects": [
    {
      "class": "pedestrian",
      "confidence": 0.93,
      "bbox": [120, 80, 210, 360]
    }
  ],
  "road_region_detected": true,
  "lane_region_detected": true
}
```

The coordinates above are illustrative only. A 2-D bounding box alone
does not provide reliable physical distance or prove that a pedestrian
is crossing the vehicle's path.

### Initial decision baseline

For the first prototype, use transparent rules based only on information
the system actually measures. For example:

-   `STOP` if a detected pedestrian overlaps a defined danger region in
    the image.
-   `SLOW_DOWN` if a relevant object enters a wider caution region.
-   `CONTINUE` if no configured hazard condition is triggered.

These are demonstration rules, not validated driving-safety rules. Image
regions are only rough proxies for distance and must not be described as
metric distance estimates.

### Future decision-model experiment

After the baseline works, the same structured scene representation can
be evaluated with a traditional ML classifier (for example, XGBoost)
and, if its interface and training data are suitable, a decision model
such as Laya or Jev.

The comparison should use the same scenarios and report: - decision
accuracy against labelled scenarios, - inference latency, - confusion
matrix, - per-class precision and recall, - failure cases.

Do not assume a decision model is safer or more accurate simply because
it is faster.

## 5. Dataset and data requirements

A model needs data that matches its task.

  -----------------------------------------------------------------------
  Task                                Required annotations
  ----------------------------------- -----------------------------------
  Object detection                    Object class and bounding boxes

  Road/lane segmentation              Pixel-level road/lane masks

  Decision classification             Scene features and a labelled
                                      action for each scenario
  -----------------------------------------------------------------------

The repository's `data/` directory was observed to be very small (about
108 KB) and no `.pt` or `.pth` checkpoints were found in the inspected
tree. Therefore, the project must not claim that the full dataset or
pretrained repository checkpoints are already available locally.

Before selecting a dataset, verify: 1. its licence and permitted
academic use; 2. its object classes and annotation format; 3. whether it
includes segmentation masks or only bounding boxes; 4. whether its
images are compatible with the selected model; 5. the
train/validation/test split.

A pretrained model can be used for inference without training it.
Fine-tuning requires a labelled training set and a documented training
run. These are different claims and must be reported accurately.

## 6. Evaluation plan

### Object detection

-   Precision and recall
-   Mean Average Precision (mAP), with the IoU threshold stated
-   Qualitative inspection of correct detections and false positives

### Segmentation

-   Intersection over Union (IoU)
-   Dice score
-   Pixel accuracy, with class imbalance noted

### Decision layer

-   Accuracy and confusion matrix on labelled scenarios
-   Decision latency (milliseconds per decision)
-   False-stop and missed-hazard counts in the test scenarios

### Runtime

-   Frames per second (FPS)
-   End-to-end frame processing time
-   Device/backend used (Apple MPS or CPU)

Do not compare latency figures unless the hardware, input size, batch
size, and measurement method are held constant.

## 7. Development plan

### Phase 1 --- Environment and repository inspection

-   Create an isolated Python 3.11 environment.
-   Verify PyTorch and Apple MPS.
-   Inspect model definitions, configuration files and demo entry
    points.
-   Identify available pretrained weights and data.

### Phase 2 --- First working perception demo

-   Select a small, compatible driving-video sample.
-   Run object detection with a compatible pretrained checkpoint.
-   Save an annotated output video.
-   Record the model, checkpoint source and inference settings.

### Phase 3 --- Segmentation

-   Identify a compatible road/lane segmentation dataset and checkpoint,
    or train a small model if time and labels permit.
-   Generate and inspect segmentation overlays.
-   State exactly whether the output is road segmentation or
    lane-marking segmentation.

### Phase 4 --- Decision baseline

-   Convert predictions into structured scene features.
-   Implement simple, explainable rules.
-   Test the rules on labelled example scenarios.

### Phase 5 --- Optional comparisons

-   Add XGBoost as a traditional ML baseline if enough labelled
    scenarios are available.
-   Evaluate Laya/Jev only after confirming input format, model access,
    inference support and a suitable decision dataset.

### Phase 6 --- Documentation and presentation

-   Include architecture, dataset source, model details, metrics,
    limitations and sample outputs.
-   Separate completed work from planned extensions.

## 8. Current implementation status

  -----------------------------------------------------------------------
  Component                           Status
  ----------------------------------- -----------------------------------
  Repository cloned                   Completed

  Python 3.11 available               Completed

  PyTorch installed                   Present; verify inside project
                                      environment

  Apple MPS availability              Previously verified as available

  Local pretrained checkpoints        Not found in inspected repository
                                      tree

  Local dataset                       Not yet identified; `data/` was
                                      about 108 KB

  Object detection video demo         Not yet verified

  Road/lane segmentation demo         Not yet verified

  Decision baseline                   Planned

  XGBoost comparison                  Optional / planned

  Jev/Laya integration                Optional / planned

  Evaluation and final report         Pending
  -----------------------------------------------------------------------

## 9. Scope and limitations

-   This is a course-level prototype, not a production
    autonomous-driving system.
-   Camera-only image coordinates do not automatically provide accurate
    real-world distance.
-   Detection confidence is not the same as calibrated probability of
    danger.
-   A simple `STOP/SLOW_DOWN/CONTINUE` policy is not a complete motion
    planner.
-   No real vehicle should be controlled by this prototype.
-   Performance claims must be based on experiments actually run by the
    project team.

## 10. Suggested project summary

This project explores a modular autonomous-driving perception pipeline
using deep learning. A CNN-based visual backbone and object detector
identify road users, while a U-Net-style segmentation model estimates
road or lane regions. The resulting predictions are converted into a
structured scene representation and passed to an explainable decision
baseline that produces high-level driving recommendations. The system is
designed to be evaluated through detection/segmentation metrics and
runtime measurements, with traditional ML and fast decision models
considered as optional comparisons.

------------------------------------------------------------------------

**Attribution note:** This project began from an existing public GitHub
repository. Retain the original repository link and licence, acknowledge
reused code and pretrained weights, and document which components were
independently modified, implemented, trained or evaluated.
