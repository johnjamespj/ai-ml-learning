# Lesson 72: Multimodal learning

Multimodal models combine information from different modalities such as:
- text
- images
- audio
- video
- sensor signals

## Fusion strategies
- early fusion: combine raw/low-level features
- intermediate fusion: combine learned representations
- late fusion: combine model decisions

## Alignment
Contrastive learning can align representations from paired modalities.

## Exercise
Create a toy problem with two information sources. Train:
1. model using modality A only;
2. model using modality B only;
3. fused model.

Measure whether fusion actually contributes complementary information.

## Sensor connection
Radar, EO/IR, ADS-B, weather and metadata can be treated as multimodal or multisensor inputs.
