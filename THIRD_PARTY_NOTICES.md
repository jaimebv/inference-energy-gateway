# Third-Party Notices

## TensorFlow Lite Micro person detection model

`src/person_detect_model_data.cpp` and `src/person_detect_model_data.h` contain
a TensorFlow Lite Micro person-detection model converted into a C array. Those
files retain their original TensorFlow Authors copyright and Apache License 2.0
header.

## Espressif managed components

The public repository should not commit `managed_components/`. PlatformIO and
ESP-IDF component management download Espressif dependencies from
`src/idf_component.yml` during the build. Each downloaded component carries its
own upstream license.
