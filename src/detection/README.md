## Training

### Prepare detection training images

```jsx
python prepare_detect_images.py
```

```jsx
bash train_track1.sh
```

Or you can download the best trained model checkpoints from [here](https://drive.google.com/drive/folders/1_0Ai-DOVLLaB1bMJKQAH9mcFHAY2kqbc?usp=share_link)

## Inference

### Prepare detection inference images

```jsx
python generate_image_list.py
```

### Inference synthetic images

```jsx
bash inference_track1_synthetic.sh
```

### Inference real images

```jsx
bash inference_track1_real.sh
```
